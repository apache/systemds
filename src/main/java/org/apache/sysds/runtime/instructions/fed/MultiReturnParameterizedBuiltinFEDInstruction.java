/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.instructions.fed;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.transform.encode.EncoderRecode;

public class MultiReturnParameterizedBuiltinFEDInstruction extends ComputationFEDInstruction {
	protected final ArrayList<CPOperand> _outputs;

	private MultiReturnParameterizedBuiltinFEDInstruction(Operator op, CPOperand input1, CPOperand input2,
		ArrayList<CPOperand> outputs, String opcode, String istr) {
		super(FEDType.MultiReturnParameterizedBuiltin, op, input1, input2, outputs.get(0), opcode, istr);
		_outputs = outputs;
	}

	public CPOperand getOutput(int i) {
		return _outputs.get(i);
	}

	public static MultiReturnParameterizedBuiltinFEDInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		ArrayList<CPOperand> outputs = new ArrayList<>();
		String opcode = parts[0];

		if(opcode.equalsIgnoreCase("transformencode")) {
			// one input and two outputs
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			outputs.add(new CPOperand(parts[3], Types.ValueType.FP64, Types.DataType.MATRIX));
			outputs.add(new CPOperand(parts[4], Types.ValueType.STRING, Types.DataType.FRAME));
			return new MultiReturnParameterizedBuiltinFEDInstruction(null, in1, in2, outputs, opcode, str);
		}
		else {
			throw new DMLRuntimeException("Invalid opcode in MultiReturnBuiltin instruction: " + opcode);
		}

	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		// obtain and pin input frame
		FrameObject fin = ec.getFrameObject(input1.getName());
		String spec = ec.getScalarInput(input2).getStringValue();

		Map<FederatedRange, FederatedData> fedMapping = fin.getFedMapping();

		// first we use the spec to construct a meta frame which will provide us with info about the encodings
		List<Pair<FederatedRange, Future<FederatedResponse>>> metaFutures = new ArrayList<>();
		for(Map.Entry<FederatedRange, FederatedData> entry : fedMapping.entrySet()) {
			Future<FederatedResponse> response = entry.getValue().executeFederatedOperation(new FederatedRequest(
				FederatedRequest.FedMethod.ENCODE_META, spec, entry.getKey().getBeginDimsInt()[1] + 1), true);
			metaFutures.add(new ImmutablePair<>(entry.getKey(), response));
		}

		// TODO support encodings other than recode
		// the combined mappings for the frame columns (because we only support recode)
		Map<String, Long>[] combinedRecodeMaps = new HashMap[(int) fin.getNumColumns()];
		try {
			for(Pair<FederatedRange, Future<FederatedResponse>> pair : metaFutures) {
				FederatedRange range = pair.getKey();
				FederatedResponse federatedResponse = pair.getValue().get();
				if(federatedResponse.isSuccessful()) {
					FrameBlock fb = (FrameBlock) federatedResponse.getData()[0];
					combineRecodeMaps(combinedRecodeMaps, fb, range.getBeginDimsInt()[1]);
				}
			}
		}
		catch(InterruptedException | ExecutionException e) {
			throw new DMLRuntimeException("Federated meta frame creation failed: " + e.getMessage());
		}

		// construct a single meta frameblock out of the multiple HashMaps with the recodings
		FrameBlock meta = frameBlockFromRecodeMaps(combinedRecodeMaps);

		// actually encode the frame block and construct an encoded matrix block at worker
		List<Pair<Map.Entry<FederatedRange, FederatedData>, Future<FederatedResponse>>> encodedFutures = new ArrayList<>();
		for(Map.Entry<FederatedRange, FederatedData> entry : fedMapping.entrySet()) {
			FederatedRange fedRange = entry.getKey();
			int columnStart = (int) fedRange.getBeginDims()[1];
			int columnEnd = (int) fedRange.getEndDims()[1];
			
			// Slice out relevant meta part
			// range is inclusive
			FrameBlock slicedMeta = meta.slice(0, meta.getNumRows() - 1, columnStart, columnEnd - 1, null);
			
			Future<FederatedResponse> response = entry.getValue().executeFederatedOperation(new FederatedRequest(
				FederatedRequest.FedMethod.FRAME_ENCODE, slicedMeta, spec, columnStart + 1), true);
			encodedFutures.add(new ImmutablePair<>(entry, response));
		}
		
		// construct a federated matrix with the encoded data
		MatrixObject transformedMat = ec.getMatrixObject(getOutput(0));
		transformedMat.getDataCharacteristics().set(fin.getDataCharacteristics());
		Map<FederatedRange, FederatedData> transformedFedMapping = new HashMap<>();
		try {
			for(Pair<Map.Entry<FederatedRange, FederatedData>, Future<FederatedResponse>> data : encodedFutures) {
				FederatedResponse federatedResponse = data.getValue().get();
				if(federatedResponse.isSuccessful()) {
					FederatedRange federatedRange = data.getKey().getKey();
					FederatedData federatedData = data.getKey().getValue();
					long varId = (long) federatedResponse.getData()[0];
					
					transformedFedMapping.put(federatedRange, new FederatedData(federatedData, varId));
				}
			}
		}
		catch(InterruptedException | ExecutionException e) {
			throw new DMLRuntimeException("Federated transform apply failed: " + e.getMessage());
		}
		// set the federated mapping for the matrix
		transformedMat.setFedMapping(transformedFedMapping);

		// release input and outputs
		ec.setFrameOutput(getOutput(1).getName(), meta);
	}

	private FrameBlock frameBlockFromRecodeMaps(Map<String, Long>[] combinedRecodeMaps) {
		int rows = 0;
		for(Map<String, Long> map : combinedRecodeMaps) {
			if(map != null) {
				rows = Integer.max(rows, map.size());
			}
		}
		FrameBlock fb = new FrameBlock(combinedRecodeMaps.length, Types.ValueType.STRING);
		fb.ensureAllocatedColumns(rows);

		// find maximum number of elements needed for a column
		int c = -1;
		for(Map<String, Long> map : combinedRecodeMaps) {
			c++;
			if(map == null) {
				continue;
			}
			int r = 0;
			for(Map.Entry<String, Long> entry : map.entrySet()) {
				fb.set(r++, c, EncoderRecode.constructRecodeMapEntry(entry.getKey(), entry.getValue()));
			}
		}
		return fb;
	}

	private void combineRecodeMaps(Map<String, Long>[] combinedRecodeMaps, FrameBlock frameBlock, int startColumn) {
		for(int c = 0; c < frameBlock.getNumColumns(); c++) {
			HashMap<String, Long> recodeMap = frameBlock.getRecodeMap(c);
			int columnCombined = startColumn + c;

			if(recodeMap.isEmpty()) {
				// no values present so no values needed in combinedRecodeMaps
				continue;
			}
			else if(combinedRecodeMaps[columnCombined] == null) {
				// the combined map was not yet needed for this column and therefore is not allocated yet
				// we can just copy the map of the current frameblock
				combinedRecodeMaps[columnCombined] = new HashMap<>(recodeMap);
				continue;
			}

			// check if any keys are not yet in the combined mapping
			Map<String, Long> combinedColumnRecodeMap = combinedRecodeMaps[columnCombined];
			boolean keysMissing = !combinedColumnRecodeMap.keySet().containsAll(recodeMap.keySet());

			if(keysMissing) {
				// add new recode values
				Set<String> allKeys = new HashSet<>(combinedColumnRecodeMap.keySet());
				allKeys.addAll(recodeMap.keySet());

				combinedColumnRecodeMap.clear();
				long mapping = 1;
				for(String key : allKeys)
					combinedColumnRecodeMap.put(key, mapping++);
			}
		}
	}
}
