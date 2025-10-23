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

package org.apache.sysds.runtime.instructions.ooc;

import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.operators.Operator;

public abstract class ComputationOOCInstruction extends OOCInstruction {
	public CPOperand output;
	public CPOperand input1, input2, input3;

	protected ComputationOOCInstruction(OOCType type, Operator op, CPOperand in1, CPOperand out, String opcode, String istr) {
		super(type, op, opcode, istr);
		input1 = in1;
		input2 = null;
		input3 = null;
		output = out;
	}
	
	protected ComputationOOCInstruction(OOCType type, Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr) {
		super(type, op, opcode, istr);
		input1 = in1;
		input2 = in2;
		input3 = null;
		output = out;
	}

	protected ComputationOOCInstruction(OOCType type, Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, String opcode, String istr) {
		super(type, op, opcode, istr);
		input1 = in1;
		input2 = in2;
		input3 = in3;
		output = out;
	}

	public String getOutputVariableName() {
		return output.getName();
	}
}
