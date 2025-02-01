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

package org.apache.sysds.runtime.instructions.cp;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.matrix.operators.Operator;

public final class ListAppendRemoveCPInstruction extends AppendCPInstruction {

	private CPOperand output2 = null;
	
	protected ListAppendRemoveCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out,
			AppendType type, String opcode, String istr) {
		super(op, in1, in2, out, type, opcode, istr);
		if( opcode.equals(Opcodes.REMOVE.toString()) )
			output2 = new CPOperand(InstructionUtils.getInstructionPartsWithValueType(istr)[4]);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		//get input list and data
		ListObject lo = ec.getListObject(input1);
		
		//list append instruction
		if( getOpcode().equals("append") ) {
			//copy on write and append unnamed argument
			Data dat2 = ec.getVariable(input2);
			LineageItem li = DMLScript.LINEAGE ? ec.getLineage().get(input2) : null;
			ListObject tmp = null;
			if( dat2 instanceof ListObject && ((ListObject)dat2).getLength() == 1 ) {
				//add unfolded elements for lists of size 1 (e.g., named)
				ListObject lo2 = (ListObject) dat2;
				tmp = lo.copy().add(lo2.getName(0), lo2.getData(0), li);
			}
			else {
				tmp = lo.copy().add(dat2, li);
			}
			//set output variable
			ec.setVariable(output.getName(), tmp);
		}
		//list remove instruction
		else if( getOpcode().equals(Opcodes.REMOVE.toString()) ) {
			//copy on write and remove by position
			ScalarObject dat2 = ec.getScalarInput(input2);
			ListObject tmp1 = lo.copy();
			ListObject tmp2 = tmp1.remove((int)dat2.getLongValue()-1);
			
			//set output variables
			ec.setVariable(output.getName(), tmp1);
			ec.setVariable(output2.getName(), tmp2);
		}
		else {
			throw new DMLRuntimeException("Unsupported list operation: "+getOpcode());
		}
	}

	public CPOperand getOutput2(){
		return output2;
	}
}
