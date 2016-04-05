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

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.operators.Operator;

public final class FrameAppendCPInstruction extends AppendCPInstruction
{	
	public FrameAppendCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, AppendType type, String opcode, String istr) {
		super(op, in1, in2, in3, out, type, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec)
		throws DMLRuntimeException 
	{
		//get inputs
		FrameBlock fin1 = ec.getFrameInput(input1.getName());
		FrameBlock fin2 = ec.getFrameInput(input2.getName());
	
		//check input dimensions
		if( _type == AppendType.CBIND && fin1.getNumRows() != fin2.getNumRows() ) {
			throw new DMLRuntimeException("Append-cbind is not possible for input matrices " + input1.getName() + " and " + input2.getName()
					+ " with different number of rows: "+fin1.getNumRows()+" vs "+fin2.getNumRows());
		}
		else if( _type == AppendType.RBIND && fin1.getNumColumns() != fin2.getNumColumns()) {
			throw new DMLRuntimeException("Append-rbind is not possible for input matrices " + input1.getName() + " and " + input2.getName()
					+ " with different number of columns: "+fin1.getNumColumns()+" vs "+fin2.getNumColumns());
		} 
			
		//execute append operations (append both inputs to initially empty output)
		FrameBlock ret = fin1.appendOperations(fin2, new FrameBlock(), _type==AppendType.CBIND);
		
		//set output and release inputs 
		ec.setFrameOutput(output.getName(), ret);
		ec.releaseFrameInput(input1.getName());
		ec.releaseFrameInput(input2.getName());
	}
}
