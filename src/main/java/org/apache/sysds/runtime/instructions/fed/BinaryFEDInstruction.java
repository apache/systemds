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

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.hops.fedplanner.FTypes.FType;
import org.apache.sysds.lops.BinaryM.VectorType;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.AppendCPInstruction;
import org.apache.sysds.runtime.instructions.cp.BinaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.BinaryMatrixMatrixCPInstruction;
import org.apache.sysds.runtime.instructions.cp.BinaryMatrixScalarCPInstruction;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.CovarianceCPInstruction;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.QuantilePickCPInstruction;
import org.apache.sysds.runtime.instructions.spark.AggregateBinarySPInstruction;
import org.apache.sysds.runtime.instructions.spark.AppendGAlignedSPInstruction;
import org.apache.sysds.runtime.instructions.spark.AppendGSPInstruction;
import org.apache.sysds.runtime.instructions.spark.AppendMSPInstruction;
import org.apache.sysds.runtime.instructions.spark.AppendRSPInstruction;
import org.apache.sysds.runtime.instructions.spark.AppendSPInstruction;
import org.apache.sysds.runtime.instructions.spark.BinaryMatrixMatrixSPInstruction;
import org.apache.sysds.runtime.instructions.spark.BinaryMatrixScalarSPInstruction;
import org.apache.sysds.runtime.instructions.spark.BinarySPInstruction;
import org.apache.sysds.runtime.instructions.spark.CovarianceSPInstruction;
import org.apache.sysds.runtime.instructions.spark.CpmmSPInstruction;
import org.apache.sysds.runtime.instructions.spark.CumulativeOffsetSPInstruction;
import org.apache.sysds.runtime.instructions.spark.MapmmSPInstruction;
import org.apache.sysds.runtime.instructions.spark.QuantilePickSPInstruction;
import org.apache.sysds.runtime.instructions.spark.RmmSPInstruction;
import org.apache.sysds.runtime.matrix.operators.Operator;

public abstract class BinaryFEDInstruction extends ComputationFEDInstruction {

	protected BinaryFEDInstruction(FEDInstruction.FEDType type, Operator op, CPOperand in1, CPOperand in2, CPOperand out,
		String opcode, String istr, FederatedOutput fedOut) {
		super(type, op, in1, in2, out, opcode, istr, fedOut);
	}

	protected BinaryFEDInstruction(FEDInstruction.FEDType type, Operator op, CPOperand in1, CPOperand in2, CPOperand out,
		String opcode, String istr) {
		this(type, op, in1, in2, out, opcode, istr, FederatedOutput.NONE);
	}

	public BinaryFEDInstruction(FEDInstruction.FEDType type, Operator op, CPOperand in1, CPOperand in2, CPOperand in3,
		CPOperand out, String opcode, String istr) {
		super(type, op, in1, in2, in3, out, opcode, istr);
	}

	public static BinaryFEDInstruction parseInstruction(BinaryCPInstruction inst, ExecutionContext ec) {
		if((inst.input1.isMatrix() && ec.getMatrixObject(inst.input1).isFederatedExcept(FType.BROADCAST)) ||
			(inst.input2 != null && inst.input2.isMatrix() &&
				ec.getMatrixObject(inst.input2).isFederatedExcept(FType.BROADCAST))) {
			if(inst instanceof AppendCPInstruction)
				return AppendFEDInstruction.parseInstruction((AppendCPInstruction) inst);
			else if(inst instanceof QuantilePickCPInstruction)
				return QuantilePickFEDInstruction.parseInstruction((QuantilePickCPInstruction) inst);
			else if(inst instanceof CovarianceCPInstruction && (ec.getMatrixObject(inst.input1).isFederated(FType.ROW) ||
				ec.getMatrixObject(inst.input2).isFederated(FType.ROW)))
				return CovarianceFEDInstruction.parseInstruction((CovarianceCPInstruction) inst);
			else if(inst instanceof BinaryMatrixMatrixCPInstruction)
				return BinaryMatrixMatrixFEDInstruction.parseInstruction((BinaryMatrixMatrixCPInstruction) inst);
			else if(inst instanceof BinaryMatrixScalarCPInstruction)
				return BinaryMatrixScalarFEDInstruction.parseInstruction((BinaryMatrixScalarCPInstruction) inst);
		}
		return null;
	}

	public static BinaryFEDInstruction parseInstruction(BinarySPInstruction inst, ExecutionContext ec) {
		if(inst instanceof MapmmSPInstruction || inst instanceof CpmmSPInstruction || inst instanceof RmmSPInstruction) {
			Data data = ec.getVariable(inst.input1);
			if(data instanceof MatrixObject && ((MatrixObject) data).isFederatedExcept(FType.BROADCAST)) {
				return MMFEDInstruction.parseInstruction((AggregateBinarySPInstruction) inst);
			}
		}
		else if(inst instanceof QuantilePickSPInstruction) {
			QuantilePickSPInstruction qinstruction = (QuantilePickSPInstruction) inst;
			Data data = ec.getVariable(qinstruction.input1);
			if(data instanceof MatrixObject && ((MatrixObject) data).isFederatedExcept(FType.BROADCAST))
				return QuantilePickFEDInstruction.parseInstruction(qinstruction);
		}
		else if(inst instanceof AppendGAlignedSPInstruction || inst instanceof AppendGSPInstruction ||
			inst instanceof AppendMSPInstruction || inst instanceof AppendRSPInstruction) {
			BinarySPInstruction ainstruction = (BinarySPInstruction) inst;
			Data data1 = ec.getVariable(ainstruction.input1);
			Data data2 = ec.getVariable(ainstruction.input2);
			if((data1 instanceof MatrixObject && ((MatrixObject) data1).isFederatedExcept(FType.BROADCAST)) ||
				(data2 instanceof MatrixObject && ((MatrixObject) data2).isFederatedExcept(FType.BROADCAST))) {
				return AppendFEDInstruction.parseInstruction((AppendSPInstruction) inst);
			}
		}
		else if(inst instanceof BinaryMatrixScalarSPInstruction) {
			Data data = ec.getVariable(inst.input1);
			if(data instanceof MatrixObject && ((MatrixObject) data).isFederatedExcept(FType.BROADCAST)) {
				return BinaryMatrixScalarFEDInstruction.parseInstruction((BinaryMatrixScalarSPInstruction) inst);
			}
		}
		else if(inst instanceof BinaryMatrixMatrixSPInstruction) {
			Data data = ec.getVariable(inst.input1);
			if(data instanceof MatrixObject && ((MatrixObject) data).isFederatedExcept(FType.BROADCAST)) {
				return BinaryMatrixMatrixFEDInstruction.parseInstruction((BinaryMatrixMatrixSPInstruction) inst);
			}
		}
		else if((inst.input1.isMatrix() && ec.getCacheableData(inst.input1).isFederatedExcept(FType.BROADCAST)) ||
			(inst.input2.isMatrix() && ec.getMatrixObject(inst.input2).isFederatedExcept(FType.BROADCAST))) {
			if(inst instanceof CovarianceSPInstruction && (ec.getMatrixObject(inst.input1).isFederated(FType.ROW) ||
				ec.getMatrixObject(inst.input2).isFederated(FType.ROW)))
				return CovarianceFEDInstruction.parseInstruction((CovarianceSPInstruction) inst);
			else if(inst instanceof CumulativeOffsetSPInstruction) {
				return CumulativeOffsetFEDInstruction.parseInstruction((CumulativeOffsetSPInstruction) inst);
			}
			else
				return BinaryFEDInstruction.parseInstruction(InstructionUtils.concatOperands(inst.getInstructionString(),
					FEDInstruction.FederatedOutput.NONE.name()));
		}
		return null;
	}

	public static BinaryFEDInstruction parseInstruction(String str) {
		// TODO remove
		if(str.startsWith(ExecType.SPARK.name())) {
			// rewrite the spark instruction to a cp instruction
			str = rewriteSparkInstructionToCP(str);
		}

		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 3, 4, 5, 6);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand out = new CPOperand(parts[3]);
		FederatedOutput fedOut = FederatedOutput.valueOf(parts[parts.length - 1]);

		checkOutputDataType(in1, in2, out);
		Operator operator = InstructionUtils.parseBinaryOrBuiltinOperator(opcode, in1, in2);

		// Operator operator = InstructionUtils.parseBinaryOrBuiltinOperator(opcode, in1, in2);
		// TODO different binary instructions
		if(in1.getDataType() == DataType.SCALAR && in2.getDataType() == DataType.SCALAR)
			throw new DMLRuntimeException("Federated binary scalar scalar operations not yet supported");
		else if(in1.getDataType() == DataType.MATRIX && in2.getDataType() == DataType.MATRIX)
			return new BinaryMatrixMatrixFEDInstruction(operator, in1, in2, out, opcode, str, fedOut);
		else if(in1.getDataType() == DataType.TENSOR && in2.getDataType() == DataType.TENSOR)
			throw new DMLRuntimeException("Federated binary tensor tensor operations not yet supported");
		else if(in1.isMatrix() && in2.isScalar() || in2.isMatrix() && in1.isScalar())
			return new BinaryMatrixScalarFEDInstruction(operator, in1, in2, out, opcode, str, fedOut);
		else
			throw new DMLRuntimeException("Federated binary operations not yet supported:" + opcode);
	}

	protected static String parseBinaryInstruction(String instr, CPOperand in1, CPOperand in2, CPOperand out) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(instr);
		InstructionUtils.checkNumFields(parts, 3, 4);
		String opcode = parts[0];
		in1.split(parts[1]);
		in2.split(parts[2]);
		out.split(parts[3]);
		return opcode;
	}

	protected static String parseBinaryInstruction(String instr, CPOperand in1, CPOperand in2, CPOperand in3,
		CPOperand out) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(instr);
		InstructionUtils.checkNumFields(parts, 4);
		String opcode = parts[0];
		in1.split(parts[1]);
		in2.split(parts[2]);
		in3.split(parts[3]);
		out.split(parts[4]);
		return opcode;
	}

	protected static void checkOutputDataType(CPOperand in1, CPOperand in2, CPOperand out) {
		// check for valid data type of output
		if((in1.getDataType() == DataType.MATRIX || in2.getDataType() == DataType.MATRIX) &&
			out.getDataType() != DataType.MATRIX)
			throw new DMLRuntimeException("Element-wise matrix operations between variables " + in1.getName() + " and "
				+ in2.getName() + " must produce a matrix, which " + out.getName() + " is not");
	}

	protected static String rewriteSparkInstructionToCP(String inst_str) {
		// rewrite the spark instruction to a cp instruction
		inst_str = inst_str.replace(ExecType.SPARK.name(), ExecType.CP.name());
		inst_str = inst_str.replace(Lop.OPERAND_DELIMITOR + "map", Lop.OPERAND_DELIMITOR);
		inst_str = inst_str.replace(Lop.OPERAND_DELIMITOR + "RIGHT", "");
		inst_str = inst_str.replace(Lop.OPERAND_DELIMITOR + VectorType.ROW_VECTOR.name(), "");
		inst_str = inst_str.replace(Lop.OPERAND_DELIMITOR + VectorType.COL_VECTOR.name(), "");
		return inst_str;
	}
}
