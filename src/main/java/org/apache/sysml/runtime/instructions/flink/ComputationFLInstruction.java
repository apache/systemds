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

package org.apache.sysml.runtime.instructions.flink;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.operators.Operator;

public abstract class ComputationFLInstruction extends FLInstruction {

	public CPOperand output;
	public CPOperand input1, input2, input3;

	public ComputationFLInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode,
									String istr) {
		super(op, opcode, istr);
		input1 = in1;
		input2 = in2;
		input3 = null;
		output = out;
	}

	public ComputationFLInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out,
									String opcode, String istr) {
		super(op, opcode, istr);
		input1 = in1;
		input2 = in2;
		input3 = in3;
		output = out;
	}

	public String getOutputVariableName() {
		return output.getName();
	}

	/**
	 * @param ec
	 * @throws DMLRuntimeException
	 */
	// TODO this is the same code as in ComputationSPInstruction --> move to common place
	protected void updateBinaryOutputMatrixCharacteristics(ExecutionContext ec)
			throws DMLRuntimeException {
		MatrixCharacteristics mcIn1 = ec.getMatrixCharacteristics(input1.getName());
		MatrixCharacteristics mcIn2 = ec.getMatrixCharacteristics(input2.getName());
		MatrixCharacteristics mcOut = ec.getMatrixCharacteristics(output.getName());
		boolean outer = (mcIn1.getRows() > 1 && mcIn1.getCols() == 1 && mcIn2.getRows() == 1 && mcIn2.getCols() > 1);

		if (!mcOut.dimsKnown()) {
			if (!mcIn1.dimsKnown())
				throw new DMLRuntimeException(
						"The output dimensions are not specified and cannot be inferred from input:" + mcIn1.toString() + " " + mcIn2.toString() + " " + mcOut.toString());
			else if (outer)
				ec.getMatrixCharacteristics(output.getName()).set(mcIn1.getRows(), mcIn2.getCols(),
						mcIn1.getRowsPerBlock(), mcIn2.getColsPerBlock());
			else
				ec.getMatrixCharacteristics(output.getName()).set(mcIn1.getRows(), mcIn1.getCols(),
						mcIn1.getRowsPerBlock(), mcIn1.getRowsPerBlock());
		}
	}

}
