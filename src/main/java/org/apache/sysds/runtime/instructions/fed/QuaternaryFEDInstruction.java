/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *	 http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.	See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.instructions.fed;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.hops.fedplanner.FTypes.FType;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.WeightedCrossEntropy;
import org.apache.sysds.lops.WeightedCrossEntropy.WCeMMType;
import org.apache.sysds.lops.WeightedDivMM;
import org.apache.sysds.lops.WeightedDivMM.WDivMMType;
import org.apache.sysds.lops.WeightedDivMMR;
import org.apache.sysds.lops.WeightedSigmoid;
import org.apache.sysds.lops.WeightedSigmoid.WSigmoidType;
import org.apache.sysds.lops.WeightedSquaredLoss;
import org.apache.sysds.lops.WeightedSquaredLoss.WeightsType;
import org.apache.sysds.lops.WeightedSquaredLossR;
import org.apache.sysds.lops.WeightedUnaryMM;
import org.apache.sysds.lops.WeightedUnaryMM.WUMMType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.QuaternaryCPInstruction;
import org.apache.sysds.runtime.instructions.spark.QuaternarySPInstruction;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.QuaternaryOperator;

public abstract class QuaternaryFEDInstruction extends ComputationFEDInstruction {
	protected CPOperand _input4 = null;

	protected QuaternaryFEDInstruction(FEDInstruction.FEDType type, Operator operator, CPOperand in1, CPOperand in2,
		CPOperand in3, CPOperand out, String opcode, String instruction_str) {
		super(type, operator, in1, in2, in3, out, opcode, instruction_str);
	}

	protected QuaternaryFEDInstruction(FEDInstruction.FEDType type, Operator operator, CPOperand in1, CPOperand in2,
		CPOperand in3, CPOperand in4, CPOperand out, String opcode, String instruction_str) {
		super(type, operator, in1, in2, in3, out, opcode, instruction_str);
		_input4 = in4;
	}

	public static QuaternaryFEDInstruction parseInstruction(QuaternaryCPInstruction inst, ExecutionContext ec) {
		Data data = ec.getVariable(inst.input1);
		if(data instanceof MatrixObject && ((MatrixObject) data).isFederatedExcept(FType.BROADCAST))
			return QuaternaryFEDInstruction.parseInstruction(inst);
		return null;
	}

	public static QuaternaryFEDInstruction parseInstruction(QuaternarySPInstruction inst, ExecutionContext ec) {
		Data data = ec.getVariable(inst.input1);
		if(data instanceof MatrixObject && ((MatrixObject) data).isFederated())
			return QuaternaryFEDInstruction.parseInstruction(inst);
		return null;
	}

	private static QuaternaryFEDInstruction parseInstruction(QuaternaryCPInstruction instr) {
		QuaternaryOperator qop = (QuaternaryOperator) instr.getOperator();
		if(qop.wtype1 != null)
			return QuaternaryWSLossFEDInstruction.parseInstruction(instr);
		else if(qop.wtype2 != null)
			return QuaternaryWSigmoidFEDInstruction.parseInstruction(instr);
		else if(qop.wtype3 != null)
			return QuaternaryWDivMMFEDInstruction.parseInstruction(instr);
		else if(qop.wtype4 != null)
			return QuaternaryWCeMMFEDInstruction.parseInstruction(instr);
		else if(qop.wtype5 != null)
			return QuaternaryWUMMFEDInstruction.parseInstruction(instr);
		// unreachable
		return null;
	}

	private static QuaternaryFEDInstruction parseInstruction(QuaternarySPInstruction instr) {
		QuaternaryOperator qop = (QuaternaryOperator) instr.getOperator();
		if(qop.wtype1 != null)
			return QuaternaryWSLossFEDInstruction.parseInstruction(instr);
		else if(qop.wtype2 != null)
			return QuaternaryWSigmoidFEDInstruction.parseInstruction(instr);
		else if(qop.wtype3 != null)
			return QuaternaryWDivMMFEDInstruction.parseInstruction(instr);
		else if(qop.wtype4 != null)
			return QuaternaryWCeMMFEDInstruction.parseInstruction(instr);
		else if(qop.wtype5 != null)
			return QuaternaryWUMMFEDInstruction.parseInstruction(instr);
		// unreachable
		return null;
	}

	public static QuaternaryFEDInstruction parseInstruction(String str) {
		if(str.startsWith(ExecType.SPARK.name())) {
			// rewrite the spark instruction to a cp instruction
			str = rewriteSparkInstructionToCP(str);
		}

		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		int addInput4 = (opcode.equals(WeightedCrossEntropy.OPCODE_CP) || opcode.equals(WeightedSquaredLoss.OPCODE_CP) ||
			opcode.equals(WeightedDivMM.OPCODE_CP)) ? 1 : 0;
		int addUOpcode = (opcode.equals(WeightedUnaryMM.OPCODE_CP) ? 1 : 0);

		InstructionUtils.checkNumFields(parts, 6 + addInput4 + addUOpcode);

		CPOperand in1 = new CPOperand(parts[1 + addUOpcode]);
		CPOperand in2 = new CPOperand(parts[2 + addUOpcode]);
		CPOperand in3 = new CPOperand(parts[3 + addUOpcode]);
		CPOperand out = new CPOperand(parts[4 + addInput4 + addUOpcode]);

		checkDataTypes(DataType.MATRIX, in1, in2, in3);

		QuaternaryOperator qop = null;
		if(addInput4 == 1) // wcemm, wsloss, wdivmm
		{
			CPOperand in4 = new CPOperand(parts[4]);

			if(opcode.equals(WeightedCrossEntropy.OPCODE_CP)) {
				final WCeMMType wcemm_type = WCeMMType.valueOf(parts[6]);
				if(wcemm_type.hasFourInputs())
					checkDataTypes(new DataType[] {DataType.SCALAR, DataType.MATRIX}, in4);
				qop = (wcemm_type.hasFourInputs() ? new QuaternaryOperator(wcemm_type,
					Double.parseDouble(in4.getName())) : new QuaternaryOperator(wcemm_type));
				return new QuaternaryWCeMMFEDInstruction(qop, in1, in2, in3, in4, out, opcode, str);
			}
			else if(opcode.equals(WeightedDivMM.OPCODE_CP)) {
				final WDivMMType wdivmm_type = WDivMMType.valueOf(parts[6]);
				if(wdivmm_type.hasFourInputs())
					checkDataTypes(new DataType[] {DataType.SCALAR, DataType.MATRIX}, in4);
				qop = new QuaternaryOperator(wdivmm_type);
				return new QuaternaryWDivMMFEDInstruction(qop, in1, in2, in3, in4, out, opcode, str);
			}
			else if(opcode.equals(WeightedSquaredLoss.OPCODE_CP)) {
				final WeightsType weights_type = WeightsType.valueOf(parts[6]);
				if(weights_type.hasFourInputs())
					checkDataTypes(DataType.MATRIX, in4);
				qop = new QuaternaryOperator(weights_type);
				return new QuaternaryWSLossFEDInstruction(qop, in1, in2, in3, in4, out, opcode, str);
			}
		}
		else if(opcode.equals(WeightedSigmoid.OPCODE_CP)) {
			final WSigmoidType wsigmoid_type = WSigmoidType.valueOf(parts[5]);
			qop = new QuaternaryOperator(wsigmoid_type);
			return new QuaternaryWSigmoidFEDInstruction(qop, in1, in2, in3, out, opcode, str);
		}
		else if(opcode.equals(WeightedUnaryMM.OPCODE_CP)) {
			final WUMMType wumm_type = WUMMType.valueOf(parts[6]);
			String uopcode = parts[1];
			qop = new QuaternaryOperator(wumm_type, uopcode);
			return new QuaternaryWUMMFEDInstruction(qop, in1, in2, in3, out, opcode, str);
		}

		throw new DMLRuntimeException("Unsupported opcode (" + opcode + ") for QuaternaryFEDInstruction.");
	}

	protected static void checkDataTypes(DataType data_type, CPOperand... cp_operands) {
		checkDataTypes(new DataType[] {data_type}, cp_operands);
	}

	protected static void checkDataTypes(DataType[] data_types, CPOperand... cp_operands) {
		for(CPOperand cpo : cp_operands) {
			if(!checkDataType(data_types, cpo)) {
				throw new DMLRuntimeException(
					"Federated quaternary operations " + "only supported with matrix inputs and scalar epsilon.");
			}
		}
	}

	private static boolean checkDataType(DataType[] data_types, CPOperand cp_operand) {
		for(DataType dt : data_types) {
			if(cp_operand.getDataType() == dt)
				return true;
		}
		return false;
	}

	protected static String rewriteSparkInstructionToCP(String inst_str) {
		// TODO: don't perform replacement over the whole instruction string, possibly changing string literals,
		// instead only at positions of ExecType and Opcode
		// rewrite the spark instruction to a cp instruction
		inst_str = inst_str.replace(ExecType.SPARK.name(), ExecType.CP.name());
		if(inst_str.contains(WeightedCrossEntropy.OPCODE))
			inst_str = inst_str.replace(WeightedCrossEntropy.OPCODE, WeightedCrossEntropy.OPCODE_CP);
		else if(inst_str.contains(WeightedDivMM.OPCODE))
			inst_str = inst_str.replace(WeightedDivMM.OPCODE, WeightedDivMM.OPCODE_CP);
		else if(inst_str.contains(WeightedSigmoid.OPCODE))
			inst_str = inst_str.replace(WeightedSigmoid.OPCODE, WeightedSigmoid.OPCODE_CP);
		else if(inst_str.contains(WeightedSquaredLoss.OPCODE))
			inst_str = inst_str.replace(WeightedSquaredLoss.OPCODE, WeightedSquaredLoss.OPCODE_CP);
		else if(inst_str.contains(WeightedUnaryMM.OPCODE))
			inst_str = inst_str.replace(WeightedUnaryMM.OPCODE, WeightedUnaryMM.OPCODE_CP);
		else if(inst_str.contains(WeightedDivMMR.OPCODE) || inst_str.contains(WeightedSquaredLossR.OPCODE)) {
			inst_str = inst_str.replace(WeightedDivMMR.OPCODE, WeightedDivMM.OPCODE_CP);
			inst_str = inst_str.replace(WeightedSquaredLossR.OPCODE, WeightedSquaredLoss.OPCODE_CP);
			// remove booleans which indicate cacheU and cacheV for redwsloss
			inst_str = inst_str.replace(Lop.OPERAND_DELIMITOR + "true", "");
			inst_str = inst_str.replace(Lop.OPERAND_DELIMITOR + "false", "");
		}
		inst_str += Lop.OPERAND_DELIMITOR + "1"; // num threads

		return inst_str;
	}

	protected void setOutputDataCharacteristics(MatrixObject X, MatrixObject U, MatrixObject V, ExecutionContext ec) {
		long rows = X.getNumRows() > 1 ? X.getNumRows() : U.getNumRows();
		long cols = X.getNumColumns() > 1 ? X
			.getNumColumns() : (U.getNumColumns() == V.getNumRows() ? V.getNumColumns() : V.getNumRows());
		MatrixObject out = ec.getMatrixObject(output);
		out.getDataCharacteristics().set(rows, cols, (int) X.getBlocksize());
	}
}
