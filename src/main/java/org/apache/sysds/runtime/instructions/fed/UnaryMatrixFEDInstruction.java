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

import java.util.Arrays;
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
			aggResults(ec, mo1, tmp, out);
		}
	}

	private void aggResults(ExecutionContext ec, MatrixObject mo1, Future<FederatedResponse>[] tmp, MatrixObject out) {
		NewVariable tmpVar;
		if(Arrays.asList("ucumk+", "ucum*").contains(getOpcode())) {
			tmpVar =  getSumOrProdVariable(ec, mo1, tmp, getOpcode().equals("ucumk+"));
			aggPartialResults(out, tmpVar);
		}
		else if(Arrays.asList("ucummin", "ucummax").contains(getOpcode())) {
			tmpVar = getMinOrMaxVariable(ec, mo1, tmp, getOpcode().equals("ucummin"));
			aggPartialResults(out, tmpVar);
		} else {
			// B11 = B11 + B12 ⊙ a
			MatrixBlock a = getSumprodVariable(ec, mo1, tmp);
			aggSumprod(ec, mo1, out, a);
		}
	}

	private void aggPartialResults(MatrixObject out, NewVariable tmpVar) {
		CPOperand operand = new CPOperand(String.valueOf(tmpVar._id), ValueType.FP64, DataType.MATRIX);
		modifyInstString(operand);

		FederatedRequest[] fr1 = out.getFedMapping().broadcastSliced(tmpVar._mo, false);
		FederatedRequest fr2 = FederationUtils.callInstruction(instString, output, out.getFedMapping().getID(),
			new CPOperand[]{output, operand},
			new long[]{out.getFedMapping().getID(), fr1[0].getID()});
		FederatedRequest fr3 = out.getFedMapping().cleanup(getTID(), fr1[0].getID());
		out.getFedMapping().execute(getTID(), true, fr1, fr2, fr3);

		out.setFedMapping(out.getFedMapping().copyWithNewID(fr2.getID()));
	}

	private void aggSumprod(ExecutionContext ec, MatrixObject mo1, MatrixObject out, MatrixBlock a) {
		MatrixBlock values = rightIndex(mo1);
		MatrixBlock agg = values.slice(0, values.getNumRows()-1,1, 1).binaryOperationsInPlace(InstructionUtils.parseBinaryOperator("*"), a);
		agg = agg.binaryOperationsInPlace(InstructionUtils.parseBinaryOperator("+"), values.slice(0,values.getNumRows()-1,0,0));
	}

	private MatrixBlock rightIndex(MatrixObject mo1) {
		long varID = FederationUtils.getNextFedDataID();
		CPOperand operand = new CPOperand(String.valueOf(varID), ValueType.FP64, DataType.MATRIX);

		String indexingInstString = constructRightIndexString(1, 1, 1, 2, operand);

		FederatedRequest fr1 = FederationUtils.callInstruction(indexingInstString, operand,
			new CPOperand[] {input1}, new long[] {mo1.getFedMapping().getID()});
		FederatedRequest fr2 = new FederatedRequest(FederatedRequest.RequestType.GET_VAR, fr1.getID());
		Future<FederatedResponse>[] tmp = mo1.getFedMapping().execute(getTID(), true, fr1, fr2);

		MatrixBlock ret = new MatrixBlock(tmp.length, 2, 0.0);
		for(int i = 0; i < tmp.length; i++) {
			try {
				ret.copy(i, i, 0, 1, ((MatrixBlock) tmp[i].get().getData()[0]), true);
			}
			catch(Exception e) {
				throw new DMLRuntimeException("Federated Get data failed with exception on UnaryMatrixFEDInstruction", e);
			}
		}

//		outSliced.setFedMapping(mo1.getFedMapping().copyWithNewID(fr1.getID()));

//		return new NewVariable(varID, outSliced);

		return ret;
	}

	private String constructRightIndexString(long rl, long ru, long cl, long cu, CPOperand operand) {
		String[] parts = instString.split(Lop.OPERAND_DELIMITOR);
		parts[1] = "rightIndex";
		return InstructionUtils.concatOperands(parts[0], parts[1], parts[2],
			InstructionUtils.createLiteralOperand(String.valueOf(rl), ValueType.INT64),
			InstructionUtils.createLiteralOperand(String.valueOf(ru), ValueType.INT64),
			InstructionUtils.createLiteralOperand(String.valueOf(cl), ValueType.INT64),
			InstructionUtils.createLiteralOperand(String.valueOf(cu), ValueType.INT64),
			InstructionUtils.createOperand(operand));
	}

	private void modifyInstString(CPOperand operand) {
		String[] parts = instString.split(Lop.OPERAND_DELIMITOR);
		String opcode = getOpcode();

		parts[1] = opcode.equalsIgnoreCase("ucumk+") ? "+" :
			opcode.equalsIgnoreCase("ucum*") ? "*" :
				opcode.equalsIgnoreCase("ucummin") ? "min" : "max";
		instString = InstructionUtils.concatOperands(parts[0], parts[1], parts[3], InstructionUtils.createOperand(operand), parts[3]);

		if(Arrays.asList("min", "max").contains(parts[1]))
			instString = InstructionUtils.concatOperands(instString, "16");
	}

	private NewVariable getMinOrMaxVariable(ExecutionContext ec, MatrixObject mo1, Future<FederatedResponse>[] tmp, boolean isMin) {
		MatrixBlock val = new MatrixBlock(1, (int) mo1.getNumColumns(), isMin ? Double.MAX_VALUE : -Double.MAX_VALUE);

		int size = (int)mo1.getNumRows();
		MatrixBlock retBlock = new MatrixBlock(size, (int)mo1.getNumColumns(), isMin ? Double.MAX_VALUE : -Double.MAX_VALUE);
		BinaryOperator bop = InstructionUtils.parseBinaryOperator(isMin ? "min" : "max");

		for(int i = 0; i < tmp.length - 1; i++)
			try {
				MatrixBlock curr = ((MatrixBlock) tmp[i].get().getData()[0]);
				curr = curr.slice(curr.getNumRows()-1,curr.getNumRows()-1);

				val = curr.binaryOperationsInPlace(bop, val);
				size = (int) (mo1.getNumRows() - mo1.getFedMapping().getFederatedRanges()[i].getEndDims()[0]);
				MatrixBlock mb = new MatrixBlock(size, (int) mo1.getNumColumns(), 0.0)
					.binaryOperations(InstructionUtils.parseBinaryOperator("+"), val, new MatrixBlock());

				int from = (int) mo1.getFedMapping().getFederatedRanges()[i+1].getBeginDims()[0];
				int to = (int) mo1.getFedMapping().getFederatedRanges()[tmp.length-1].getEndDims()[0]-1;
				retBlock.copy(from, to,0, mb.getNumColumns()-1, mb, true);
			}
			catch(Exception e) {
				throw new DMLRuntimeException("Federated Get data failed with exception on UnaryMatrixFEDInstruction", e);
			}

		// add it to the list of variables
		MatrixObject moTmp = ExecutionContext.createMatrixObject(retBlock);
		long varID = FederationUtils.getNextFedDataID();
		ec.setVariable(String.valueOf(varID), moTmp);

		return new NewVariable(varID, moTmp);
	}

	// compute the difference to add an create MatrixObject
	private NewVariable getSumOrProdVariable(ExecutionContext ec, MatrixObject mo1, Future<FederatedResponse>[] tmp, boolean isSum) {
		int size = (int)mo1.getNumRows();
		double initVal = isSum ? 0.0 : 1.0;
		MatrixBlock res = new MatrixBlock(size, (int)mo1.getNumColumns(), initVal);
		BinaryOperator bop = InstructionUtils.parseBinaryOperator(isSum ? "+" : "*");

		for(int i = 0; i < tmp.length - 1; i++)
			try {
				MatrixBlock curr = ((MatrixBlock) tmp[i].get().getData()[0]);
				curr = curr.slice(curr.getNumRows()-1,curr.getNumRows()-1);

				size = (int) (mo1.getNumRows() - mo1.getFedMapping().getFederatedRanges()[i].getEndDims()[0]);
				MatrixBlock mb = new MatrixBlock(size, (int) mo1.getNumColumns(), initVal)
					.binaryOperations(bop, curr, new MatrixBlock());

				int from = (int) mo1.getFedMapping().getFederatedRanges()[i+1].getBeginDims()[0];
				int to = (int) mo1.getFedMapping().getFederatedRanges()[tmp.length-1].getEndDims()[0]-1;
				MatrixBlock retBlock = new MatrixBlock((int) mo1.getNumRows(), (int)mo1.getNumColumns(), initVal);
				retBlock.copy(from, to,0, mb.getNumColumns()-1, mb, true);

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

	private MatrixBlock getSumprodVariable(ExecutionContext ec, MatrixObject mo1, Future<FederatedResponse>[] tmp) {
		MatrixBlock prod = getProd(mo1);
		int size = (int)mo1.getNumRows();
		MatrixBlock res = new MatrixBlock(size, 1, 0.0);

		for(int i = 0; i < tmp.length; i++)
			try {
				MatrixBlock curr = ((MatrixBlock) tmp[i].get().getData()[0]);
				prod.setValue(i, 0, curr.getValue(curr.getNumRows()-1, 0));
			}
			catch(Exception e) {
				throw new DMLRuntimeException("Federated Get data failed with exception on UnaryMatrixFEDInstruction", e);
			}

		// aggregate sumprod to get scalars
		UnaryOperator uop = new UnaryOperator(Builtin.getBuiltinFnObject("ucumk+*"));
		MatrixBlock scalars = prod.unaryOperations(uop, new MatrixBlock());

//		// create a for res = B11 + B12 * a
//		for(int i = 0; i < scalars.getNumRows() - 1; i++) {
//			size = (int) (mo1.getNumRows() - mo1.getFedMapping().getFederatedRanges()[i].getEndDims()[0]);
//			int from = (int) mo1.getFedMapping().getFederatedRanges()[i+1].getBeginDims()[0];
//			int to = (int) mo1.getFedMapping().getFederatedRanges()[tmp.length-1].getEndDims()[0]-1;
//			res.copy(from, to,0, 0, new MatrixBlock(size, 1, scalars.getValue(i, 0)), true);
//		}
//
//		// add it to the list of variables
//		MatrixObject moTmp = ExecutionContext.createMatrixObject(res);
//		long varID = FederationUtils.getNextFedDataID();
//		ec.setVariable(String.valueOf(varID), moTmp);
		return scalars;
	}

	private MatrixBlock getProd(MatrixObject mo1) {
		String tmpInstString = instString.replace("ucumk+*", "ucumk+");

		FederatedRequest fr1 = FederationUtils.callInstruction(tmpInstString, output,
			new CPOperand[] {input1}, new long[] {mo1.getFedMapping().getID()});
		FederatedRequest fr2 = new FederatedRequest(FederatedRequest.RequestType.GET_VAR, fr1.getID());
		Future<FederatedResponse>[] tmp = mo1.getFedMapping().execute(getTID(), true, fr1, fr2);

		// slice and construct prod matrix
		MatrixBlock ret = new MatrixBlock(tmp.length, 2, 0.0);
		for(int i = 0; i < tmp.length; i++)
			try {
				MatrixBlock curr = ((MatrixBlock) tmp[i].get().getData()[0]);
				ret.setValue(i, 1, curr.getValue(curr.getNumRows()-1, 1));
			}
			catch(Exception e) {
				throw new DMLRuntimeException("Federated Get data failed with exception on UnaryMatrixFEDInstruction",
					e);
			}
		return ret;
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
