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
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.WeightedCrossEntropy.WCeMMType;
import org.apache.sysds.lops.WeightedSquaredLoss.WeightsType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.fed.QuaternaryWCeMMFEDInstruction;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.QuaternaryOperator;

public abstract class QuaternaryFEDInstruction extends ComputationFEDInstruction
{
	protected CPOperand _input4 = null;

	protected QuaternaryFEDInstruction(FEDInstruction.FEDType type, Operator operator,
		CPOperand in1, CPOperand in2, CPOperand in3, CPOperand in4, CPOperand out, String opcode, String instruction_str)
	{
		super(type, operator, in1, in2, in3, out, opcode, instruction_str);
		_input4 = in4;
	}

	public static QuaternaryFEDInstruction parseInstruction(String str)
	{
		if(str.startsWith(ExecType.SPARK.name())) {
			// rewrite the spark instruction to a cp instruction
			str = str.replace(ExecType.SPARK.name(), ExecType.CP.name());
      str = str.replace("mapwcemm", "wcemm");
			str = str.replace("mapwsloss", "wsloss");
			str += Lop.OPERAND_DELIMITOR + "1"; //num threads
		}

		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand in3 = new CPOperand(parts[3]);
		CPOperand out = new CPOperand(parts[5]);

		InstructionUtils.checkNumFields(parts, 7);

		if(opcode.equals("wcemm") || opcode.equals("wsloss"))
		{
			CPOperand in4 = new CPOperand(parts[4]);
			checkDataTypes(in1, in2, in3, in4, opcode.equals("wcemm"));

			QuaternaryOperator quaternary_operator = null;

			if(opcode.equals("wcemm"))
			{
			  final WCeMMType wcemm_type = WCeMMType.valueOf(parts[6]);
			  quaternary_operator = (wcemm_type.hasFourInputs() ? new QuaternaryOperator(wcemm_type, Double.parseDouble(in4.getName())) : new QuaternaryOperator(wcemm_type));
			  return new QuaternaryWCeMMFEDInstruction(quaternary_operator, in1, in2, in3, in4, out, opcode, str);
			}
			else if(opcode.equals("wsloss"))
			{
			  final WeightsType weights_type = WeightsType.valueOf(parts[6]);
			  quaternary_operator = new QuaternaryOperator(weights_type);
			  return new QuaternaryWSLossFEDInstruction(quaternary_operator, in1, in2, in3, in4, out, opcode, str);
			}
		}

		throw new DMLRuntimeException("Unsupported opcode (" + opcode + ") for QuaternaryFEDInstruction.");
	}

	protected static void checkDataTypes(CPOperand in1, CPOperand in2, CPOperand in3, CPOperand in4, boolean in4_scalar_flag) {
		if(in1.getDataType() != DataType.MATRIX || in2.getDataType() != DataType.MATRIX
			|| in3.getDataType() != DataType.MATRIX
			|| !((in4.getDataType() == DataType.SCALAR && in4_scalar_flag) || in4.getDataType() == DataType.MATRIX)) {
			throw new DMLRuntimeException("Federated quaternary operations "
				+ "only supported with matrix inputs and scalar epsilon.");
		}
	}
}
