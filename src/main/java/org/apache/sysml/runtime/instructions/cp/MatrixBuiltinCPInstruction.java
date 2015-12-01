/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.runtime.instructions.cp;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.matrix.data.LibCommonsMath;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.UnaryOperator;


public class MatrixBuiltinCPInstruction extends BuiltinUnaryCPInstruction
{
	public MatrixBuiltinCPInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String instr){
		super(op, in, out, 1, opcode, instr);
	}

	@Override 
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{	
		UnaryOperator u_op = (UnaryOperator) _optr;
		String output_name = output.getName();
		
		String opcode = getOpcode();
		if(LibCommonsMath.isSupportedUnaryOperation(opcode)) {
			MatrixBlock retBlock = LibCommonsMath.unaryOperations((MatrixObject)ec.getVariable(input1.getName()),getOpcode());
			ec.setMatrixOutput(output_name, retBlock);
		}
		else {
			MatrixBlock inBlock = ec.getMatrixInput(input1.getName());
			MatrixBlock retBlock = (MatrixBlock) (inBlock.unaryOperations(u_op, new MatrixBlock()));
		
			ec.releaseMatrixInput(input1.getName());
			
			// Ensure right dense/sparse output representation (guarded by released input memory)
			if( checkGuardedRepresentationChange(inBlock, retBlock) ) {
	 			retBlock.examSparsity();
	 		}
			
			ec.setMatrixOutput(output_name, retBlock);
		}		
	}
}
