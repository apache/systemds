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

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.common.Types;
import org.apache.sysds.lops.SortKeys;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedUDF;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class QuantileSortFEDInstruction extends UnaryFEDInstruction{

	private QuantileSortFEDInstruction(CPOperand in, CPOperand out, String opcode, String istr) {
		this(in, null, out, opcode, istr);
	}

	private QuantileSortFEDInstruction(CPOperand in1, CPOperand in2, CPOperand out, String opcode,
		String istr) {
		super(FEDInstruction.FEDType.QSort, null, in1, in2, out, opcode, istr);
	}

	public static QuantileSortFEDInstruction parseInstruction ( String str ) {
		CPOperand in1 = new CPOperand("", Types.ValueType.UNKNOWN, Types.DataType.UNKNOWN);
		CPOperand in2 = null;
		CPOperand out = new CPOperand("", Types.ValueType.UNKNOWN, Types.DataType.UNKNOWN);

		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		if ( opcode.equalsIgnoreCase(SortKeys.OPCODE) ) {
			if ( parts.length == 3 ) {
				// Example: sort:mVar1:mVar2 (input=mVar1, output=mVar2)
				parseUnaryInstruction(str, in1, out);
				return new QuantileSortFEDInstruction(in1, out, opcode, str);
			}
			else if ( parts.length == 4 ) {
				// Example: sort:mVar1:mVar2:mVar3 (input=mVar1, weights=mVar2, output=mVar3)
				in2 = new CPOperand("", Types.ValueType.UNKNOWN, Types.DataType.UNKNOWN);
				parseUnaryInstruction(str, in1, in2, out);
				return new QuantileSortFEDInstruction(in1, in2, out, opcode, str);
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		}
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a QuantileSortFEDInstruction: " + str);
		}
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		MatrixObject in = ec.getMatrixObject(input1);
		FederationMap fedMapping = in.getFedMapping();

		long varID = FederationUtils.getNextFedDataID();
		FederationMap sortedMapping = fedMapping.mapParallel(varID, (range, data) -> {
			try {
				MatrixBlock wtBlock = null;
				if (input2 != null) {
					wtBlock = ec.getMatrixInput(input2.getName());
				}

				FederatedResponse response = data
					.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF, -1,
						new GetSorted(data.getVarID(), varID, wtBlock))).get();
				if(!response.isSuccessful())
					response.throwExceptionFromResponse();
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
			return null;
		});


		MatrixObject sorted = ec.getMatrixObject(output);
		sorted.getDataCharacteristics().set(in.getDataCharacteristics());

		// set the federated mapping for the matrix
		sorted.setFedMapping(sortedMapping);
	}

	private static class GetSorted extends FederatedUDF {

		private static final long serialVersionUID = -1969015577260167645L;
		private final long _outputID;
		private final MatrixBlock _weights;

		protected GetSorted(long input, long outputID, MatrixBlock weights) {
			super(new long[] {input});
			_outputID = outputID;
			_weights = weights;
		}
		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixBlock mb = ((MatrixObject) data[0]).acquireReadAndRelease();

			MatrixBlock res = mb.sortOperations(_weights, new MatrixBlock());

			MatrixObject mout = ExecutionContext.createMatrixObject(res);

			// add it to the list of variables
			ec.setVariable(String.valueOf(_outputID), mout);
			// return schema
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS_EMPTY);
		}
		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}
}
