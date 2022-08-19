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
import org.apache.sysds.hops.fedplanner.FTypes.FType;
import org.apache.sysds.lops.SortKeys;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedUDF;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.controlprogram.federated.MatrixLineagePair;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.QuantileSortCPInstruction;
import org.apache.sysds.runtime.instructions.spark.QuantileSortSPInstruction;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class QuantileSortFEDInstruction extends UnaryFEDInstruction {
	int _numThreads;

	private QuantileSortFEDInstruction(CPOperand in, CPOperand out, String opcode, String istr, int k) {
		this(in, null, out, opcode, istr, k);
	}

	private QuantileSortFEDInstruction(CPOperand in1, CPOperand in2, CPOperand out, String opcode,
		String istr, int k) {
		super(FEDInstruction.FEDType.QSort, null, in1, in2, out, opcode, istr);
		_numThreads = k;
	}

	public static QuantileSortFEDInstruction parseInstruction(QuantileSortCPInstruction instr) {
		return new QuantileSortFEDInstruction(instr.input1, instr.input2, instr.output, instr.getOpcode(),
			instr.getInstructionString(), instr.getNumThreads());
	}

	public static QuantileSortFEDInstruction parseInstruction(QuantileSortSPInstruction instr) {
		return new QuantileSortFEDInstruction(instr.input1, instr.input2, instr.output, instr.getOpcode(),
				instr.getInstructionString(), 1);
	}

	private static void parseInstruction(String instr, CPOperand in1, CPOperand in2, CPOperand out) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(instr);

		out.split(parts[parts.length-2]);

		switch(parts.length) {
			case 4:
				in1.split(parts[1]);
				in2 = null;
				break;
			case 5:
				in1.split(parts[1]);
				in2.split(parts[2]);
				break;
			default:
				throw new DMLRuntimeException("Unexpected number of operands in the instruction: " + instr);
		}
	}

	public static QuantileSortFEDInstruction parseInstruction ( String str , boolean hasFedOut) {
		CPOperand in1 = new CPOperand("", Types.ValueType.UNKNOWN, Types.DataType.UNKNOWN);
		CPOperand in2 = null;
		CPOperand out = new CPOperand("", Types.ValueType.UNKNOWN, Types.DataType.UNKNOWN);

		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		boolean isSpark = str.startsWith("SPARK");
		int k;
		FederatedOutput fedOut;
		if ( hasFedOut){
			k = isSpark ? 1 : Integer.parseInt(parts[parts.length-2]);
			fedOut = FederatedOutput.valueOf(parts[parts.length-1]);
		} else {
			k = isSpark ? 1 : Integer.parseInt(parts[parts.length-1]);
			fedOut = FederatedOutput.NONE;
		}

		QuantileSortFEDInstruction inst;

		if ( opcode.equalsIgnoreCase(SortKeys.OPCODE) ) {
			int oneInputLength = isSpark ? 3 : 4;
			int twoInputLength = isSpark ? 4 : 5;
			if ( parts.length == oneInputLength ) {
				// Example: sort:mVar1:mVar2 (input=mVar1, output=mVar2)
				parseUnaryInstruction(str, in1, out);
				inst = new QuantileSortFEDInstruction(in1, out, opcode, str, k);
			}
			else if ( parts.length == twoInputLength ) {
				// Example: sort:mVar1:mVar2:mVar3 (input=mVar1, weights=mVar2, output=mVar3)
				in2 = new CPOperand("", Types.ValueType.UNKNOWN, Types.DataType.UNKNOWN);
				InstructionUtils.checkNumFields(str, twoInputLength-1);
				parseInstruction(str, in1, in2, out);
				inst = new QuantileSortFEDInstruction(in1, in2, out, opcode, str, k);
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		}
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a QuantileSortFEDInstruction: " + str);
		}
		inst._fedOut = fedOut;
		return inst;
	}
	@Override
	public void processInstruction(ExecutionContext ec) {
		if(ec.getMatrixObject(input1).isFederated(FType.COL) || ec.getMatrixObject(input1).isFederated(FType.FULL))
			processColumnQSort(ec);
		else
			processRowQSort(ec);
	}

	public void processRowQSort(ExecutionContext ec) {
		MatrixObject in = ec.getMatrixObject(input1);
		MatrixObject out = ec.getMatrixObject(output);

		// TODO make sure that qsort result is used by qpick only where the main operation happens
		if(input2 != null) {
			MatrixLineagePair weights = ec.getMatrixLineagePair(input2);
			String newInst = _numThreads > 1 ? InstructionUtils.stripThreadCount(instString) : instString;
			newInst = InstructionUtils.replaceOperand(newInst, 1, "append");
			newInst = InstructionUtils.concatOperands(newInst, "true");
			FederatedRequest[] fr1 = in.getFedMapping().broadcastSliced(weights, false);
			FederatedRequest fr2 = FederationUtils.callInstruction(newInst, output,
				new CPOperand[]{input1, input2}, new long[]{ in.getFedMapping().getID(), fr1[0].getID()});
			in.getFedMapping().execute(getTID(), true, fr1, fr2);
			out.getDataCharacteristics().set(in.getDataCharacteristics());
			out.getDataCharacteristics().setCols(2);
			out.setFedMapping(in.getFedMapping().copyWithNewID(fr2.getID(), 2));
		}
		else {
			// make a copy without sorting
			long id = FederationUtils.getNextFedDataID();
			out.getDataCharacteristics().set(in.getDataCharacteristics());
			out.setFedMapping(in.getFedMapping().identCopy(getTID(), id));
		}
	}

	public void processColumnQSort(ExecutionContext ec) {
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
						new GetSorted(data.getVarID(), varID, wtBlock, _numThreads))).get();
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
		private final int _numThreads;

		protected GetSorted(long input, long outputID, MatrixBlock weights, int k) {
			super(new long[] {input});
			_outputID = outputID;
			_weights = weights;
			_numThreads = k;
		}
		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixBlock mb = ((MatrixObject) data[0]).acquireReadAndRelease();

			MatrixBlock res = mb.sortOperations(_weights, new MatrixBlock(), _numThreads);

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
