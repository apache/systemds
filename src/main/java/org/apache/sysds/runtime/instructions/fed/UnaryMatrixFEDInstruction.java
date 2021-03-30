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

import java.util.concurrent.Future;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.matrix.data.LibCommonsMath;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

public class UnaryMatrixFEDInstruction extends UnaryFEDInstruction {
	protected UnaryMatrixFEDInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String instr) {
		super(FEDType.Unary, op, in, out, opcode, instr);
	}
	
	public static boolean isValidOpcode(String opcode) {
		return !LibCommonsMath.isSupportedUnaryOperation(opcode);
//			&& !opcode.startsWith("ucum"); //ucumk+ ucum* ucumk+* ucummin ucummax
	}

	public static UnaryMatrixFEDInstruction parseInstruction(String str) {
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);

		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode;
		opcode = parts[0];
		if( (opcode.equalsIgnoreCase("exp") || opcode.startsWith("ucum")) && parts.length == 5) {
			in.split(parts[1]);
			out.split(parts[2]);
			ValueFunction func = Builtin.getBuiltinFnObject(opcode);
			return new UnaryMatrixFEDInstruction(new UnaryOperator(func,
				Integer.parseInt(parts[3]),Boolean.parseBoolean(parts[4])), in, out, opcode, str);
		}
		opcode = parseUnaryInstruction(str, in, out);
		return new UnaryMatrixFEDInstruction(InstructionUtils.parseUnaryOperator(opcode), in, out, opcode, str);
	}
	
	@Override 
	public void processInstruction(ExecutionContext ec) {
		if(getOpcode().startsWith("ucum"))
			processCumulativeInstruction(ec);
		else {
			MatrixObject mo1 = ec.getMatrixObject(input1);

			//federated execution on arbitrary row/column partitions
			//(only assumption for sparse-unsafe: fed mapping covers entire matrix)
			FederatedRequest fr1 = FederationUtils.callInstruction(instString,
				output,
				new CPOperand[] {input1},
				new long[] {mo1.getFedMapping().getID()});
			mo1.getFedMapping().execute(getTID(), true, fr1);

			setOutputFedMapping(ec, mo1, fr1.getID());
		}
	}

	public void processCumulativeInstruction(ExecutionContext ec) {
		MatrixObject mo1 = ec.getMatrixObject(input1);
		Future<FederatedResponse>[] tmp;

		FederatedRequest fr1 = FederationUtils.callInstruction(instString, output,
			new CPOperand[] {input1}, new long[] {mo1.getFedMapping().getID()});
		if(mo1.isFederated(FederationMap.FType.ROW)) {
			FederatedRequest fr2 = new FederatedRequest(FederatedRequest.RequestType.GET_VAR, fr1.getID());
			FederatedRequest fr3 = mo1.getFedMapping().cleanup(getTID(), fr2.getID());
			tmp = mo1.getFedMapping().execute(getTID(), true, fr1, fr2, fr3);
		} else
			tmp = mo1.getFedMapping().execute(getTID(), true, fr1);

		if(mo1.isFederated(FederationMap.FType.ROW))
			aggPartialResults(ec, tmp, mo1, fr1);

		setOutputFedMapping(ec, mo1, fr1.getID());
	}

	private void aggPartialResults(ExecutionContext ec, Future<FederatedResponse>[] tmp, MatrixObject mo1, FederatedRequest fr) {
		int size = (int)mo1.getNumRows();
		MatrixBlock retBlock = new MatrixBlock(size, (int)mo1.getNumColumns(), 0.0);
		MatrixBlock res = new MatrixBlock(size, (int)mo1.getNumColumns(), 0.0);
		BinaryOperator bop = InstructionUtils.parseBinaryOperator("+");

		for(int i = 0; i < tmp.length - 1; i++)
			try {
				MatrixBlock curr = ((MatrixBlock) tmp[i].get().getData()[0]);
				curr = curr.slice(curr.getNumRows()-1,curr.getNumRows()-1);

				size = (int) (mo1.getNumRows() - mo1.getFedMapping().getFederatedRanges()[i].getEndDims()[0]);
				MatrixBlock mb = new MatrixBlock(size, (int) mo1.getNumColumns(),0.0)
					.binaryOperations(bop, curr, new MatrixBlock());

				int from = (int) mo1.getFedMapping().getFederatedRanges()[i+1].getBeginDims()[0];
				int to = (int) mo1.getFedMapping().getFederatedRanges()[tmp.length-1].getEndDims()[0]-1;
				retBlock.copy(from, to,0, 3, mb, false); //TODO 3
				res.binaryOperationsInPlace(bop, retBlock);
			}
			catch(Exception e) {
				throw new DMLRuntimeException("Federated Get data failed with exception on TernaryFedInstruction", e);
			}

		setOutputFedMapping(ec, mo1, fr.getID());

		MatrixObject moTmp = ExecutionContext.createMatrixObject(res);
		// add it to the list of variables
//		ec.setVariable(String.valueOf(_outputID), mo);

		String[] parts = instString.split(OPERAND_DELIM);
		parts[1] = getOpcode().equals("ucumk+") ? "+" : "*"; //TODO
		FederatedRequest fr1 = mo1.getFedMapping().broadcast(moTmp);
		FederatedRequest fr2 = FederationUtils.callInstruction(instString, output, new CPOperand[]{input1, input2},
			new long[]{mo1.getFedMapping().getID(), fr1.getID()});
		FederatedRequest fr3 = mo1.getFedMapping().cleanup(getTID(), fr1.getID());
		mo1.getFedMapping().execute(getTID(), true, fr1, fr2, fr3);

		setOutputFedMapping(ec, mo1, fr2.getID());
	}

	private void setOutputFedMapping(ExecutionContext ec, MatrixObject fedMapObj, long fedOutputID) {
		MatrixObject out = ec.getMatrixObject(output);
		out.getDataCharacteristics().set(fedMapObj.getDataCharacteristics());
		out.setFedMapping(fedMapObj.getFedMapping().copyWithNewID(fedOutputID));
	}
}
