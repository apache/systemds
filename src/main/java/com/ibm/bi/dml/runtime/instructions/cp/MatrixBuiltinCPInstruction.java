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

package com.ibm.bi.dml.runtime.instructions.cp;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.matrix.data.LibCommonsMath;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.UnaryOperator;


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
		MatrixBlock resultBlock = null;
		
		String opcode = getOpcode();
		if(LibCommonsMath.isSupportedUnaryOperation(opcode)) {
			resultBlock = LibCommonsMath.unaryOperations((MatrixObject)ec.getVariable(input1.getName()),getOpcode());
			ec.setMatrixOutput(output_name, resultBlock);
		}
		else {
			MatrixBlock matBlock = ec.getMatrixInput(input1.getName());
			resultBlock = (MatrixBlock) (matBlock.unaryOperations(u_op, new MatrixBlock()));
			
			ec.setMatrixOutput(output_name, resultBlock);
			ec.releaseMatrixInput(input1.getName());
		}		
	}
}
