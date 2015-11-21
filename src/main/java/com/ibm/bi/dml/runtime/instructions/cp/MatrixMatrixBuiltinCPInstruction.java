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
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class MatrixMatrixBuiltinCPInstruction extends BuiltinBinaryCPInstruction
{
	
	public MatrixMatrixBuiltinCPInstruction(Operator op, 
											   CPOperand in1, 
											   CPOperand in2, 
											   CPOperand out, 
											   String opcode,
											   String istr){
		super(op, in1, in2, out, 2, opcode, istr);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		String opcode = getOpcode();
        
        if ( LibCommonsMath.isSupportedMatrixMatrixOperation(opcode) ) {
        	MatrixBlock solution = LibCommonsMath.matrixMatrixOperations((MatrixObject)ec.getVariable(input1.getName()), (MatrixObject)ec.getVariable(input2.getName()), opcode);
    		ec.setMatrixOutput(output.getName(), solution);
        	return;
        }
		
        /* Default behavior of this instruction */
		String output_name = output.getName();
		BinaryOperator bop = (BinaryOperator) _optr;
		
		MatrixBlock inBlock1 = ec.getMatrixInput(input1.getName());
		MatrixBlock inBlock2 = ec.getMatrixInput(input2.getName());
		
		MatrixBlock retBlock = (MatrixBlock) inBlock1.binaryOperations(bop, inBlock2, new MatrixBlock());
	
		ec.releaseMatrixInput(input1.getName());
		ec.releaseMatrixInput(input2.getName());
		
		// Ensure right dense/sparse output representation (guarded by released input memory)
		if( checkGuardedRepresentationChange(inBlock1, inBlock2, retBlock) ) {
 			retBlock.examSparsity();
 		}
        
		ec.setMatrixOutput(output_name, retBlock);
	}
}