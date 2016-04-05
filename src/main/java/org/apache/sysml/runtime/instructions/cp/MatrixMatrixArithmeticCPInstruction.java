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
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;


public class MatrixMatrixArithmeticCPInstruction extends ArithmeticBinaryCPInstruction
{
	
	public MatrixMatrixArithmeticCPInstruction(Operator op, 
											   CPOperand in1, 
											   CPOperand in2, 
											   CPOperand out, 
											   String opcode,
											   String istr){
		super(op, in1, in2, out, opcode, istr);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException
	{
		// Read input matrices
        MatrixBlock inBlock1 = ec.getMatrixInput(input1.getName());
        MatrixBlock inBlock2 = ec.getMatrixInput(input2.getName());
		
		// Perform computation using input matrices, and produce the result matrix
		BinaryOperator bop = (BinaryOperator) _optr;
		MatrixBlock retBlock = (MatrixBlock) (inBlock1.binaryOperations (bop, inBlock2, new MatrixBlock()));
		
		// Release the memory occupied by input matrices
		ec.releaseMatrixInput(input1.getName());
		ec.releaseMatrixInput(input2.getName());
		
		// Ensure right dense/sparse output representation (guarded by released input memory)
		if( checkGuardedRepresentationChange(inBlock1, inBlock2, retBlock) ) {
			retBlock.examSparsity();
		}
		
		// Attach result matrix with MatrixObject associated with output_name
		ec.setMatrixOutput(output.getName(), retBlock);
	}
}