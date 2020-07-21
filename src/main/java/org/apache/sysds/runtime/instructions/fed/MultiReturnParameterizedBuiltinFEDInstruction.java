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
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;

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
import org.apache.sysds.runtime.transform.encode.Encoder;
import org.apache.sysds.runtime.transform.encode.EncoderComposite;
import org.apache.sysds.runtime.transform.encode.EncoderPassThrough;
import org.apache.sysds.runtime.transform.encode.EncoderRecode;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class MultiReturnParameterizedBuiltinFEDInstruction extends ComputationFEDInstruction {
	protected final ArrayList<CPOperand> _outputs;

	private MultiReturnParameterizedBuiltinFEDInstruction(Operator op, CPOperand input1, CPOperand input2,
		ArrayList<CPOperand> outputs, String opcode, String istr) {
		super(FEDType.MultiReturnParameterizedBuiltin, op, input1, input2, null, opcode, istr);
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

		// the encoder in which the complete encoding information will be aggregated
		EncoderComposite globalEncoder = new EncoderComposite(
			Arrays.asList(new EncoderRecode(), new EncoderPassThrough()));
		// first create encoders at the federated workers, then collect them and aggregate them to a single large
		// encoder
		CommonThreadPool pool = new CommonThreadPool(CommonThreadPool.get(fedMapping.size()));
		ArrayList<FederatedCreateEncoderTask> createTasks = new ArrayList<>();
		for(Map.Entry<FederatedRange, FederatedData> fedMap : fedMapping.entrySet())
			createTasks.add(new FederatedCreateEncoderTask(fedMap.getKey(), fedMap.getValue(), spec, globalEncoder));
		try {
			pool.invokeAll(createTasks);
		}
		catch(InterruptedException e) {
			throw new DMLRuntimeException("Federated Creation of encoders failed: " + e.getMessage());
		}

		Map<FederatedRange, FederatedData> transformedFedMapping = new HashMap<>();
		ArrayList<FederatedEncodeTask> encodeTasks = new ArrayList<>();
		for(Map.Entry<FederatedRange, FederatedData> fedMap : fedMapping.entrySet())
			encodeTasks
				.add(new FederatedEncodeTask(fedMap.getKey(), fedMap.getValue(), globalEncoder, transformedFedMapping));
		CommonThreadPool.invokeAndShutdown(pool, encodeTasks);

		// construct a federated matrix with the encoded data
		MatrixObject transformedMat = ec.getMatrixObject(getOutput(0));
		transformedMat.getDataCharacteristics().set(fin.getDataCharacteristics());
		// set the federated mapping for the matrix
		transformedMat.setFedMapping(transformedFedMapping);

		// release input and outputs
		ec.setFrameOutput(getOutput(1).getName(),
			globalEncoder.getMetaData(new FrameBlock(globalEncoder.getNumCols(), Types.ValueType.STRING)));
	}

	private static class FederatedCreateEncoderTask implements Callable<Void> {
		private final FederatedRange _range;
		private final FederatedData _data;
		private final String _spec;
		private final Encoder _result;

		public FederatedCreateEncoderTask(FederatedRange range, FederatedData data, String spec, Encoder result) {
			_range = range;
			_data = data;
			_spec = spec;
			_result = result;
		}

		@Override
		public Void call() throws Exception {
			int columnOffset = (int) _range.getBeginDims()[1] + 1;

			// create an encoder with the given spec. The columnOffset (which is 1 based) has to be used to
			// tell the federated worker how much the indexes in the spec have to be offset.
			Future<FederatedResponse> response = _data.executeFederatedOperation(
				new FederatedRequest(FederatedRequest.FedMethod.CREATE_ENCODER, _spec, columnOffset),
				true);
			// collect responses with encoders
			try {
				Encoder encoder = (Encoder) response.get().getData()[0];
				// merge this encoder into a composite encoder
				synchronized(_result) {
					_result.mergeAt(encoder, columnOffset);
				}
			}
			catch(Exception e) {
				throw new DMLRuntimeException("Federated encoder creation failed: " + e.getMessage());
			}
			return null;
		}
	}

	private static class FederatedEncodeTask implements Callable<Void> {
		private final FederatedRange _range;
		private final FederatedData _data;
		private final Encoder _globalEncoder;
		private final Map<FederatedRange, FederatedData> _resultMapping;

		public FederatedEncodeTask(FederatedRange range, FederatedData data, Encoder globalEncoder,
			Map<FederatedRange, FederatedData> resultMapping) {
			_range = range;
			_data = data;
			_globalEncoder = globalEncoder;
			_resultMapping = resultMapping;
		}

		@Override
		public Void call() throws Exception {
			int colStart = (int) _range.getBeginDims()[1] + 1;
			int colEnd = (int) _range.getEndDims()[1] + 1;
			// get the encoder segment that is relevant for this federated worker
			Encoder encoder = _globalEncoder.subRangeEncoder(colStart, colEnd);

			FederatedResponse response = _data
				.executeFederatedOperation(new FederatedRequest(FederatedRequest.FedMethod.FRAME_ENCODE, encoder), true)
				.get();
			long varId = (long) response.getData()[0];
			synchronized(_resultMapping) {
				_resultMapping.put(new FederatedRange(_range), new FederatedData(_data, varId));
			}
			return null;
		}
	}
}
