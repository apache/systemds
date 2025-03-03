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

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.fedplanner.FTypes.FType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.controlprogram.federated.MatrixLineagePair;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.CumulativeOffsetSPInstruction;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;

public class CumulativeOffsetFEDInstruction extends BinaryFEDInstruction
{
	private UnaryOperator _uop = null;

	private CumulativeOffsetFEDInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, double init, boolean broadcast, String opcode, String istr) {
		super(FEDType.CumsumOffset, op, in1, in2, out, opcode, istr);

		if ("bcumoffk+".equals(opcode))
			_uop = new UnaryOperator(Builtin.getBuiltinFnObject("ucumk+"));
		else if ("bcumoff*".equals(opcode))
			_uop = new UnaryOperator(Builtin.getBuiltinFnObject("ucum*"));
		else if ("bcumoff+*".equals(opcode))
			_uop = new UnaryOperator(Builtin.getBuiltinFnObject("ucumk+*"));
		else if ("bcumoffmin".equals(opcode))
			_uop = new UnaryOperator(Builtin.getBuiltinFnObject("ucummin"));
		else if ("bcumoffmax".equals(opcode))
			_uop = new UnaryOperator(Builtin.getBuiltinFnObject("ucummax"));
	}

	public static CumulativeOffsetFEDInstruction parseInstruction(CumulativeOffsetSPInstruction instr) {
		return new CumulativeOffsetFEDInstruction(instr.getOperator(), instr.input1, instr.input2, instr.output,
			instr.getInitValue(), instr.getBroadcast(), instr.getOpcode(), instr.getInstructionString());
	}

	public static CumulativeOffsetFEDInstruction parseInstruction ( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType( str );
		InstructionUtils.checkNumFields(parts, 5);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand out = new CPOperand(parts[3]);
		double init = Double.parseDouble(parts[4]);
		boolean broadcast = Boolean.parseBoolean(parts[5]);
		return new CumulativeOffsetFEDInstruction(null, in1, in2, out, init, broadcast, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		MatrixObject mo1 = ec.getMatrixObject(input1);
		MatrixLineagePair mo2 = ec.getMatrixLineagePair(input2);
		if(getOpcode().startsWith("bcumoff") && mo1.isFederated(FType.ROW))
			processCumulativeInstruction(ec);
		else {
			//federated execution on arbitrary row/column partitions
			//(only assumption for sparse-unsafe: fed mapping covers entire matrix)
			FederatedRequest[] fr1 = mo1.getFedMapping().broadcastSliced(mo2, false);
			FederatedRequest fr2 = FederationUtils.callInstruction(instString, output,
				new CPOperand[] {input1, input2}, new long[] {mo1.getFedMapping().getID(), fr1[0].getID()});
			FederatedRequest fr3 = new FederatedRequest(FederatedRequest.RequestType.PUT_VAR, fr2.getID(), mo1.getDataCharacteristics(), mo1.getDataType());
			mo1.getFedMapping().execute(getTID(), true, fr1, fr3, fr2);

			setOutputFedMapping(ec, mo1, fr2.getID());
		}
	}

	public void processCumulativeInstruction(ExecutionContext ec) {
		MatrixObject mo1 = ec.getMatrixObject(input1.getName());
		MatrixLineagePair mo2 = ec.getMatrixLineagePair(input2);
		DataCharacteristics mcOut = ec.getDataCharacteristics(output.getName());

		long id = FederationUtils.getNextFedDataID();

		String opcode = getOpcode();
		MatrixObject out;

		if(opcode.equalsIgnoreCase("bcumoff+*")) {
			FederatedRequest fr3 = new FederatedRequest(FederatedRequest.RequestType.PUT_VAR, id, mcOut, mo1.getDataType());
			FederatedRequest fr4 = mo1.getFedMapping().broadcast(mo2);
			FederatedRequest fr1 = FederationUtils.callInstruction(instString, output, id,
				new CPOperand[] {input1, input2}, new long[] {mo1.getFedMapping().getID(), fr4.getID()}, Types.ExecType.SPARK, false);
			FederatedRequest fr2 = new FederatedRequest(FederatedRequest.RequestType.GET_VAR, fr1.getID());
			Future<FederatedResponse>[] tmp = mo1.getFedMapping().execute(getTID(), true, fr3, fr4, fr1, fr2);
			out = setOutputFedMapping(ec, mo1, fr1.getID());

			MatrixBlock scalingValues = getScalars(mo1, tmp);
			setScalingValues(ec, mo1, out, scalingValues);
		}
		else {
			String colAgg = opcode.replace("bcumoff", "uac");
			String agg2 = opcode.replace(opcode.contains("bcumoffk")? "bcumoffk" :"bcumoff", "");

			double init = opcode.equalsIgnoreCase("bcumoffk+") ? 0.0:
				opcode.equalsIgnoreCase("bcumoff*") ? 1.0 :
					opcode.equalsIgnoreCase("bcumoffmin") ? Double.MAX_VALUE : -Double.MAX_VALUE;

			Future<FederatedResponse>[] tmp = modifyAndGetInstruction(colAgg, mo1);
			MatrixBlock scalingValues = getResultBlock(tmp, (int)mo1.getNumColumns(), opcode, init, _uop);

			out = ec.getMatrixObject(output);
			setScalingValues(agg2, ec, mo1, out, scalingValues, init);
		}
		processCumulative(out, mo2);
	}

	private Future<FederatedResponse>[] modifyAndGetInstruction(String newInst, MatrixObject mo1) {
		String modifiedInstString = InstructionUtils.replaceOperand(instString, 1, newInst);
		modifiedInstString = InstructionUtils.removeOperand(modifiedInstString, 3);
		modifiedInstString = InstructionUtils.removeOperand(modifiedInstString, 4);
		modifiedInstString = InstructionUtils.removeOperand(modifiedInstString, 4);
		modifiedInstString = InstructionUtils.concatOperands(modifiedInstString, AggBinaryOp.SparkAggType.SINGLE_BLOCK.name());

		long id = FederationUtils.getNextFedDataID();
		FederatedRequest fr3 = new FederatedRequest(FederatedRequest.RequestType.PUT_VAR, id, new MatrixCharacteristics(-1, -1), mo1.getDataType());
		FederatedRequest fr1 = FederationUtils.callInstruction(modifiedInstString, output, id,
			new CPOperand[] {input1}, new long[] {mo1.getFedMapping().getID()}, Types.ExecType.SPARK, false);
		FederatedRequest fr2 = new FederatedRequest(FederatedRequest.RequestType.GET_VAR, fr1.getID());
		return mo1.getFedMapping().execute(getTID(), true, fr3, fr1, fr2);
	}

	private void processCumulative(MatrixObject out, MatrixLineagePair mo2) {
		String modifiedInstString = InstructionUtils.replaceOperand(instString, 2, InstructionUtils.createOperand(output));

		FederatedRequest fr3 = out.getFedMapping().broadcast(mo2);
		FederatedRequest fr4 = FederationUtils.callInstruction(modifiedInstString, output, out.getFedMapping().getID(),
			new CPOperand[] {output, input2}, new long[] {out.getFedMapping().getID(), fr3.getID()}, Types.ExecType.SPARK, false);
		out.getFedMapping().execute(getTID(), true, fr3, fr4);
		out.setFedMapping(out.getFedMapping().copyWithNewID(fr4.getID()));

		// modify fed ranges since ucumk+* output is always nx1
		if(getOpcode().equalsIgnoreCase("bcumoff+*")) {
			out.getDataCharacteristics().set(out.getNumRows(), 1L, out.getBlocksize());
			for(int i = 0; i < out.getFedMapping().getFederatedRanges().length; i++)
				out.getFedMapping().getFederatedRanges()[i].setEndDim(1, 1);
		} else {
			out.getDataCharacteristics().set(out.getNumRows(), out.getNumColumns(), out.getBlocksize());
		}
	}

	private static MatrixBlock getResultBlock(Future<FederatedResponse>[] tmp, int cols, String opcode, double init, UnaryOperator uop) {
		//TODO perf simple rbind, as the first row (init) is anyway not transferred

		//collect row vectors into local matrix
		MatrixBlock res = new MatrixBlock(tmp.length, cols, init);
		for(int i = 0; i < tmp.length-1; i++)
			try {
				res.copy(i+1, i+1, 0, cols-1, ((MatrixBlock) tmp[i].get().getData()[0]), true);
			}
			catch(Exception e) {
				throw new DMLRuntimeException("Federated Get data failed with exception on CumulativeOffsetFEDInstruction", e);
			}

		//local cumulative aggregate
		return res.unaryOperations(
			uop,
			new MatrixBlock());
	}

	private MatrixBlock getScalars(MatrixObject mo1, Future<FederatedResponse>[] tmp) {
		MatrixBlock[] aggRes = getAggMatrices(mo1);
		MatrixBlock prod = aggRes[0];
		MatrixBlock firstValues = aggRes[1];
		for(int i = 0; i < tmp.length; i++)
			try {
				MatrixBlock curr = ((MatrixBlock) tmp[i].get().getData()[0]);
				prod.set(i, 0, curr.get(curr.getNumRows()-1, 0));
			}
			catch(Exception e) {
				throw new DMLRuntimeException("Federated Get data failed with exception on CumulativeOffsetFEDInstruction", e);
			}

		// aggregate sumprod to get scalars
		MatrixBlock a = new MatrixBlock(tmp.length, 1, 0.0);
		a.copy(1, a.getNumRows()-1, 0, 0,
			prod.unaryOperations(new UnaryOperator(Builtin.getBuiltinFnObject("ucumk+*")), new MatrixBlock())
				.slice(0, prod.getNumRows()-2), true);

		// compute  B11 = B11 + B12 ⊙ a
		MatrixBlock B = firstValues.slice(0, firstValues.getNumRows()-1,1, 1)
			.binaryOperations(InstructionUtils.parseBinaryOperator(Opcodes.MULT.toString()), a, new MatrixBlock());
		return B.binaryOperationsInPlace(InstructionUtils.parseBinaryOperator(Opcodes.PLUS.toString()), firstValues.slice(0,firstValues.getNumRows()-1,0,0));
	}

	private MatrixBlock[] getAggMatrices(MatrixObject mo1) {
		Future<FederatedResponse>[] tmp = modifyAndGetInstruction("ucum*", mo1);

		// slice and return prod and first value
		MatrixBlock prod = new MatrixBlock(tmp.length, 2, 0.0);
		MatrixBlock firstValues = new MatrixBlock(tmp.length, 2, 0.0);
		for(int i = 0; i < tmp.length; i++)
			try {
				MatrixBlock curr = ((MatrixBlock) tmp[i].get().getData()[0]);
				prod.set(i, 1, curr.get(curr.getNumRows()-1, 1));
				firstValues.copy(i, i, 0,1, curr.slice(0, 0), true);
			}
			catch(Exception e) {
				throw new DMLRuntimeException("Federated Get data failed with exception on CumulativeOffsetFEDInstruction", e);
			}
		return new MatrixBlock[] {prod, firstValues};
	}

	private void setScalingValues(ExecutionContext ec, MatrixObject mo1, MatrixObject out, MatrixBlock scalingValues) {
		MatrixBlock condition = new MatrixBlock((int) mo1.getNumRows(), (int) mo1.getNumColumns(), 1.0);
		MatrixBlock mb2 = new MatrixBlock((int) mo1.getNumRows(), (int) mo1.getNumColumns(), 0.0);

		for(int i = 0; i < scalingValues.getNumRows()-1; i++) {
			int step = (int) mo1.getFedMapping().getFederatedRanges()[i + 1].getBeginDims()[0];
			condition.set(step, 0, 0.0);
			mb2.set(step, 0, scalingValues.get(i + 1, 0));
		}

		MatrixObject cond = ExecutionContext.createMatrixObject(condition);
		long condID = FederationUtils.getNextFedDataID();
		ec.setVariable(String.valueOf(condID), cond);

		MatrixObject mo2 = ExecutionContext.createMatrixObject(mb2);
		long varID2 = FederationUtils.getNextFedDataID();
		ec.setVariable(String.valueOf(varID2), mo2);

		CPOperand opCond = new CPOperand(String.valueOf(condID), Types.ValueType.FP64, Types.DataType.MATRIX);
		CPOperand op2 = new CPOperand(String.valueOf(varID2), Types.ValueType.FP64, Types.DataType.MATRIX);

		String ternaryInstString = InstructionUtils.constructTernaryString(instString, opCond, input1, op2, output);

		FederatedRequest[] fr1 = mo1.getFedMapping().broadcastSliced(cond, false);
		FederatedRequest[] fr2 = mo1.getFedMapping().broadcastSliced(mo2, false);
		FederatedRequest fr3 = FederationUtils.callInstruction(ternaryInstString, output,
			new CPOperand[] {input1, opCond, op2}, new long[] {mo1.getFedMapping().getID(), fr1[0].getID(), fr2[0].getID()});
		//TODO perf no need to execute here, we can piggyback the requests onto the final cumagg
		mo1.getFedMapping().execute(getTID(), true, fr1, fr2, fr3);

		out.setFedMapping(mo1.getFedMapping().copyWithNewID(fr3.getID()));

		ec.removeVariable(opCond.getName());
		ec.removeVariable(op2.getName());
	}

	private void setScalingValues(String opcode, ExecutionContext ec, MatrixObject mo1, MatrixObject out, MatrixBlock scalingValues, double init) {
		//TODO perf improvement (currently this creates a sliced broadcast in the size of the original matrix
		//but sparse w/ strategically placed offsets, but would need to be dense for dense prod/cumsum)

		//allocated large matrix of init value and placed offset rows in first row of every partition
		MatrixBlock mb2 = new MatrixBlock((int) mo1.getNumRows(), (int) mo1.getNumColumns(), init);
		for(int i = 1; i < scalingValues.getNumRows(); i++) {
			int step = (int) mo1.getFedMapping().getFederatedRanges()[i].getBeginDims()[0];
			mb2.copy(step, step, 0, (int)(mo1.getNumColumns()-1), scalingValues.slice(i, i), true);
		}

		MatrixObject mo2 = ExecutionContext.createMatrixObject(mb2);
		long varID2 = FederationUtils.getNextFedDataID();
		ec.setVariable(String.valueOf(varID2), mo2);
		CPOperand op2 = new CPOperand(String.valueOf(varID2), Types.ValueType.FP64, Types.DataType.MATRIX);

		String modifiedInstString = InstructionUtils.constructBinaryInstString(instString, opcode, input1, op2, output);

		long id = FederationUtils.getNextFedDataID();
		FederatedRequest fr3 = new FederatedRequest(FederatedRequest.RequestType.PUT_VAR, id, new MatrixCharacteristics(-1, -1), Types.DataType.MATRIX);
		FederatedRequest[] fr1 = mo1.getFedMapping().broadcastSliced(mo2, false);
		FederatedRequest fr2 = FederationUtils.callInstruction(modifiedInstString, output, id,
			new CPOperand[] {input1, op2}, new long[] {mo1.getFedMapping().getID(), fr1[0].getID()}, Types.ExecType.SPARK, false);
		mo1.getFedMapping().execute(getTID(), true, fr1, fr3, fr2);

		out.setFedMapping(mo1.getFedMapping().copyWithNewID(fr2.getID()));
		ec.removeVariable(op2.getName());
	}

	private MatrixObject setOutputFedMapping(ExecutionContext ec, MatrixObject fedMapObj, long fedOutputID) {
		MatrixObject out = ec.getMatrixObject(output);
		out.getDataCharacteristics().set(fedMapObj.getDataCharacteristics());
		out.setFedMapping(fedMapObj.getFedMapping().copyWithNewID(fedOutputID));
		return out;
	}
}
