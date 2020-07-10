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
import java.util.List;
import java.util.Map;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.ImmutableTriple;
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

		List<Encoder> encoderList = new ArrayList<>();
		encoderList.add(new EncoderRecode());
		encoderList.add(new EncoderPassThrough());
		EncoderComposite compositeEncoder = new EncoderComposite(encoderList);
		// TODO check parallel stream problems and performance. Run in the CommonThreadPool?
		// first create encoders at the federated workers, then collect them and aggregate them to a single large
		// encoder
		EncoderComposite globalEncoder = fedMapping.entrySet().parallelStream().map(entry -> {
			int columnOffset = (int) entry.getKey().getBeginDims()[1] + 1;

			// create an encoder with the given spec. The columnOffset (which is 1 based) has to be used to
			// tell the federated worker how much the indexes in the spec have to be offset.
			Future<FederatedResponse> response = entry.getValue().executeFederatedOperation(
				new FederatedRequest(FederatedRequest.FedMethod.CREATE_ENCODER, spec, columnOffset),
				true);
			// for aggregation column offset and response is needed
			return new ImmutablePair<>(columnOffset, response);
		}).reduce(compositeEncoder,
			(compEncoder, pair) -> {
				// collect responses with encoders
				try {
					FederatedResponse federatedResponse = pair.getRight().get();

					Encoder encoder = (Encoder) federatedResponse.getData()[0];
					// merge this encoder into a composite encoder
					compEncoder.mergeAt(encoder, pair.getLeft());
					return compEncoder;
				}
				catch(Exception e) {
					throw new DMLRuntimeException("Federated encoder creation failed: " + e.getMessage());
				}
			},
			(encL, encR) -> {
				// combine partial results (two composite encoders)
				encL.mergeAt(encR, 1);
				return encL;
			});

		// redistribute sub ranges of encoders, which are globally valid with their encodings
		Map<FederatedRange, FederatedData> transformedFedMapping = fedMapping.entrySet().stream().map(entry -> {
			FederatedRange range = entry.getKey();
			FederatedData data = entry.getValue();

			int colStart = (int) range.getBeginDims()[1] + 1;
			int colEnd = (int) range.getEndDims()[1] + 1;
			Encoder encoder = globalEncoder.subRangeEncoder(colStart, colEnd);

			Future<FederatedResponse> response = data.executeFederatedOperation(
				new FederatedRequest(FederatedRequest.FedMethod.FRAME_ENCODE, encoder),
				true);
			return new ImmutableTriple<>(range, data, response);
		}).collect(Collectors.toMap(ImmutableTriple::getLeft,
			triple -> {
				// collect the responses with data into a single federated mapping map
				try {
					FederatedResponse response = triple.getRight().get();
					long varId = (long) response.getData()[0];
					return new FederatedData(triple.getMiddle(), varId);
				}
				catch(Exception e) {
					throw new DMLRuntimeException("Federated encoder appliance failed: " + e.getMessage());
				}
			}));

		// construct a federated matrix with the encoded data
		MatrixObject transformedMat = ec.getMatrixObject(getOutput(0));
		transformedMat.getDataCharacteristics().set(fin.getDataCharacteristics());
		// set the federated mapping for the matrix
		transformedMat.setFedMapping(transformedFedMapping);

		// release input and outputs
		ec.setFrameOutput(getOutput(1).getName(),
			globalEncoder.getMetaData(new FrameBlock(globalEncoder.getNumCols(), Types.ValueType.STRING)));
	}
}
