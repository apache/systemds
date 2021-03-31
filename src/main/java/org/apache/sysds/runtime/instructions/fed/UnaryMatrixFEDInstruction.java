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
import org.apache.sysds.lops.Lop;
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
import org.apache.sysds.runtime.matrix.data.LibCommonsMath;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
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
			tmp = mo1.getFedMapping().execute(getTID(), true, fr1, fr2);
		} else
			tmp = mo1.getFedMapping().execute(getTID(), true, fr1);

		MatrixObject out = setOutputFedMapping(ec, mo1, fr1.getID());

		if(mo1.isFederated(FederationMap.FType.ROW)) {
			NewVariable tmpVar = getTmpVariable(ec, mo1, tmp);
			aggPartialResults(out, tmpVar);
		}
	}

	private void aggPartialResults(MatrixObject out, NewVariable tmpVar) {
		CPOperand operand = new CPOperand(String.valueOf(tmpVar._id), ValueType.FP64, DataType.MATRIX);
		modifyInstString(operand);

		FederatedRequest[] fr1 = out.getFedMapping().broadcastSliced(tmpVar._mo, false);
		FederatedRequest fr2 = FederationUtils.callInstruction(instString, output, out.getFedMapping().getID(),
			new CPOperand[]{output, operand},
			new long[]{out.getFedMapping().getID(), fr1[0].getID()});
		FederatedRequest fr4 = new FederatedRequest(FederatedRequest.RequestType.GET_VAR, fr2.getID()); // TODO remove
		FederatedRequest fr3 = out.getFedMapping().cleanup(getTID(), fr1[0].getID());
		Future<FederatedResponse>[] ffr = out.getFedMapping().execute(getTID(), true, fr1, fr2, fr3, fr4);

		for(int i = 0; i < ffr.length - 1; i++)
			try {
				MatrixBlock curr = ((MatrixBlock) ffr[i].get().getData()[0]);
			}
			catch(Exception e) {
				throw new DMLRuntimeException("Federated Get data failed with exception on UnaryMatrixFEDInstruction", e);
			}

		out.setFedMapping(out.getFedMapping().copyWithNewID(fr2.getID()));
	}

	private void modifyInstString(CPOperand operand) {
		String[] parts = instString.split(Lop.OPERAND_DELIMITOR);
		parts[1] = getOpcode().equals("ucumk+") ? "+" : "*"; //TODO
		instString = String.join(Lop.OPERAND_DELIMITOR, new String[] {parts[0], parts[1], parts[3], InstructionUtils.createOperand(operand), parts[3]});
	}

	// compute the difference to add an create MatrixObject
	private NewVariable getTmpVariable(ExecutionContext ec, MatrixObject mo1, Future<FederatedResponse>[] tmp) {
		int size = (int)mo1.getNumRows();
		MatrixBlock res = new MatrixBlock(size, (int)mo1.getNumColumns(), getOpcode().equals("ucumk+") ? 0.0 : 1.0); //TODO replace with boolean
		BinaryOperator bop = InstructionUtils.parseBinaryOperator(getOpcode().equals("ucumk+") ? "+" : "*");

		for(int i = 0; i < tmp.length - 1; i++)
			try {
				MatrixBlock curr = ((MatrixBlock) tmp[i].get().getData()[0]);
				curr = curr.slice(curr.getNumRows()-1,curr.getNumRows()-1);

				size = (int) (mo1.getNumRows() - mo1.getFedMapping().getFederatedRanges()[i].getEndDims()[0]);
				MatrixBlock mb = new MatrixBlock(size, (int) mo1.getNumColumns(),0.0)
					.binaryOperations(bop, curr, new MatrixBlock());

				int from = (int) mo1.getFedMapping().getFederatedRanges()[i+1].getBeginDims()[0];
				int to = (int) mo1.getFedMapping().getFederatedRanges()[tmp.length-1].getEndDims()[0]-1;
				MatrixBlock retBlock = new MatrixBlock((int) mo1.getNumRows(), (int)mo1.getNumColumns(), getOpcode().equals("ucumk+") ? 0.0 : 1.0);
				retBlock.copy(from, to,0, mb.getNumColumns()-1, mb, true); //TODO 1
				res.binaryOperationsInPlace(bop, retBlock);
			}
			catch(Exception e) {
				throw new DMLRuntimeException("Federated Get data failed with exception on UnaryMatrixFEDInstruction", e);
			}

		// add it to the list of variables
		MatrixObject moTmp = ExecutionContext.createMatrixObject(res);
		long varID = FederationUtils.getNextFedDataID();
		ec.setVariable(String.valueOf(varID), moTmp);

		return new NewVariable(varID, moTmp);
	}

	private static final class NewVariable {
		public long _id;
		public MatrixObject _mo;

		public NewVariable(long id, MatrixObject mo) {
			_id = id;
			_mo = mo;
		}
	}

	private MatrixObject setOutputFedMapping(ExecutionContext ec, MatrixObject fedMapObj, long fedOutputID) {
		MatrixObject out = ec.getMatrixObject(output);
		out.getDataCharacteristics().set(fedMapObj.getDataCharacteristics());
		out.setFedMapping(fedMapObj.getFedMapping().copyWithNewID(fedOutputID));
		return out;
	}
}
