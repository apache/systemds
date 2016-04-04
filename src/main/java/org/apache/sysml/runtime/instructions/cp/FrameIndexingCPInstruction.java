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

import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.util.IndexRange;

public final class FrameIndexingCPInstruction extends IndexingCPInstruction
{	
	public FrameIndexingCPInstruction(Operator op, CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu, CPOperand out, String opcode, String istr){
		super(op, in, rl, ru, cl, cu, out, opcode, istr);
	}
	
	public FrameIndexingCPInstruction(Operator op, CPOperand lhsInput, CPOperand rhsInput, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu, CPOperand out, String opcode, String istr){
		super(op, lhsInput, rhsInput, rl, ru, cl, cu, out, opcode, istr);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{	
		String opcode = getOpcode();
		IndexRange ixrange = getIndexRange(ec);
		
		//right indexing
		if( opcode.equalsIgnoreCase("rangeReIndex") )
		{
			//execute right indexing operation
			FrameBlock in = ec.getFrameInput(input1.getName());
			FrameBlock out = in.sliceOperations(ixrange, new FrameBlock());	
				
			//unpin rhs input
			ec.releaseFrameInput(input1.getName());
			
			//unpin output
			ec.setFrameOutput(output.getName(), out);
		}
		//left indexing
		else if ( opcode.equalsIgnoreCase("leftIndex"))
		{
			FrameBlock lin = ec.getFrameInput(input1.getName());
			FrameBlock out = null;
			
			if(input2.getDataType() == DataType.FRAME) { //FRAME<-FRAME
				FrameBlock rin = ec.getFrameInput(input2.getName());
				out = lin.leftIndexingOperations(rin, ixrange, new FrameBlock());
				ec.releaseFrameInput(input2.getName());
			}
			else { //FRAME<-SCALAR 
				if(!ixrange.isScalar())
					throw new DMLRuntimeException("Invalid index range of scalar leftindexing: "+ixrange.toString()+"." );
				ScalarObject scalar = ec.getScalarInput(input2.getName(), ValueType.DOUBLE, input2.isLiteral());
				out = new FrameBlock(lin);
				out.set((int)ixrange.rowStart, (int)ixrange.colStart, scalar.getStringValue());
			}

			//unpin lhs input
			ec.releaseFrameInput(input1.getName());
			
			//unpin output
			ec.setFrameOutput(output.getName(), out);
		}
		else
			throw new DMLRuntimeException("Invalid opcode (" + opcode +") encountered in FrameIndexingCPInstruction.");		
	}
}
