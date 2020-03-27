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

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.LibFederatedAgg;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;

public class AggregateUnaryFEDInstruction extends UnaryFEDInstruction {
	
	private AggregateUnaryFEDInstruction(AggregateUnaryOperator auop, AggregateOperator aop, CPOperand in,
			CPOperand out, String opcode, String istr) {
		super(FEDType.AggregateUnary, auop, in, out, opcode, istr);
	}
	
	public static AggregateUnaryFEDInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		String aopcode = InstructionUtils.deriveAggregateOperatorOpcode(opcode);
		Types.CorrectionLocationType corrLoc = InstructionUtils
			.deriveAggregateOperatorCorrectionLocation(opcode);
		AggregateUnaryOperator aggun = InstructionUtils.parseBasicAggregateUnaryOperator(opcode);
		AggregateOperator aop = InstructionUtils.parseAggregateOperator(aopcode, corrLoc.toString());
		return new AggregateUnaryFEDInstruction(aggun, aop, in1, out, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		String output_name = output.getName();
		String opcode = getOpcode();
		
		AggregateUnaryOperator au_op = (AggregateUnaryOperator) _optr;
		MatrixObject matrixObject;
		if (input1.getDataType() == DataType.MATRIX &&
				(matrixObject = ec.getMatrixObject(input1.getName())).isFederated()) {
			MatrixBlock outMatrix = LibFederatedAgg.aggregateUnaryMatrix(matrixObject, au_op);
			
			if (output.getDataType() == DataType.SCALAR) {
				DoubleObject ret = new DoubleObject(outMatrix.getValue(0, 0));
				ec.setScalarOutput(output_name, ret);
			}
			else {
				ec.setMatrixOutput(output_name, outMatrix);
				ec.getMatrixObject(output_name).getDataCharacteristics()
					.setBlocksize(ConfigurationManager.getBlocksize());
			}
		}
		else {
			throw new DMLRuntimeException(opcode + " only supported on federated matrix.");
		}
	}
}
