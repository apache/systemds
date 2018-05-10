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

package org.apache.sysml.runtime.instructions.cp;

import org.apache.sysml.lops.LeftIndex;
import org.apache.sysml.lops.RightIndex;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;

public final class ListIndexingCPInstruction extends IndexingCPInstruction {

	protected ListIndexingCPInstruction(CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl,
			CPOperand cu, CPOperand out, String opcode, String istr) {
		super(in, rl, ru, cl, cu, out, opcode, istr);
	}

	protected ListIndexingCPInstruction(CPOperand lhsInput, CPOperand rhsInput, CPOperand rl,
			CPOperand ru, CPOperand cl, CPOperand cu, CPOperand out, String opcode, String istr) {
		super(lhsInput, rhsInput, rl, ru, cl, cu, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		String opcode = getOpcode();
		ScalarObject rl = ec.getScalarInput(rowLower.getName(), rowLower.getValueType(), rowLower.isLiteral());
		ScalarObject ru = ec.getScalarInput(rowUpper.getName(), rowUpper.getValueType(), rowUpper.isLiteral());
		
		//right indexing
		if( opcode.equalsIgnoreCase(RightIndex.OPCODE) ) {
			ListObject list = (ListObject) ec.getVariable(input1.getName());
			
			//execute right indexing operation and set output
			if( rl.getValueType()==ValueType.STRING || ru.getValueType()==ValueType.STRING ) {
				ec.setVariable(output.getName(),
					list.slice(rl.getStringValue(), ru.getStringValue()));
			}
			else {
				ec.setVariable(output.getName(),
					list.slice((int)rl.getLongValue()-1, (int)ru.getLongValue()-1));
			}
		}
		//left indexing
		else if ( opcode.equalsIgnoreCase(LeftIndex.OPCODE)) {
//			FrameBlock lin = ec.getFrameInput(input1.getName());
//			FrameBlock out = null;
//			
//			if(input2.getDataType() == DataType.FRAME) { //FRAME<-FRAME
//				FrameBlock rin = ec.getFrameInput(input2.getName());
//				out = lin.leftIndexingOperations(rin, ixrange, new FrameBlock());
//				ec.releaseFrameInput(input2.getName());
//			}
//			else { //FRAME<-SCALAR 
//				if(!ixrange.isScalar())
//					throw new DMLRuntimeException("Invalid index range of scalar leftindexing: "+ixrange.toString()+"." );
//				ScalarObject scalar = ec.getScalarInput(input2.getName(), input2.getValueType(), input2.isLiteral());
//				out = new FrameBlock(lin);
//				out.set((int)ixrange.rowStart, (int)ixrange.colStart, scalar.getStringValue());
//			}
//
//			//unpin lhs input
//			ec.releaseFrameInput(input1.getName());
//			
//			//unpin output
//			ec.setFrameOutput(output.getName(), out);
		}
		else
			throw new DMLRuntimeException("Invalid opcode (" + opcode +") encountered in ListIndexingCPInstruction.");
	}
}
