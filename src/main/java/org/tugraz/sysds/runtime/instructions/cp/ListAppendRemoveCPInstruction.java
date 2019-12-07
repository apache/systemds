/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.runtime.instructions.cp;

import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.matrix.operators.Operator;

public final class ListAppendRemoveCPInstruction extends AppendCPInstruction {

	protected ListAppendRemoveCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out,
			AppendType type, String opcode, String istr) {
		super(op, in1, in2, out, type, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		//get input list and data
		ListObject lo = ec.getListObject(input1);
		Data dat2 = ec.getVariable(input2);
		
		//list append instruction
		if( getOpcode().equals("append") ) {
			//copy on write and append unnamed argument
			ListObject tmp = lo.copy().append(dat2);
			//set output variable
			ec.setVariable(output.getName(), tmp);
		}
		else {
			throw new DMLRuntimeException("Unsupported list operation: "+getOpcode());
		}
	}
}
