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
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.SimpleOperator;
import org.apache.sysml.runtime.util.IndexRange;

public class MatrixIndexingCPInstruction extends UnaryCPInstruction
{
	
	/*
	 * This class implements the matrix indexing functionality inside CP.  
	 * Example instructions: 
	 *     rangeReIndex:mVar1:Var2:Var3:Var4:Var5:mVar6
	 *         input=mVar1, output=mVar6, 
	 *         bounds = (Var2,Var3,Var4,Var5)
	 *         rowindex_lower: Var2, rowindex_upper: Var3 
	 *         colindex_lower: Var4, colindex_upper: Var5
	 *     leftIndex:mVar1:mVar2:Var3:Var4:Var5:Var6:mVar7
	 *         triggered by "mVar1[Var3:Var4, Var5:Var6] = mVar2"
	 *         the result is stored in mVar7
	 *  
	 */
	protected CPOperand rowLower, rowUpper, colLower, colUpper;
	
	public MatrixIndexingCPInstruction(Operator op, CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu, CPOperand out, String opcode, String istr){
		super(op, in, out, opcode, istr);
		rowLower = rl;
		rowUpper = ru;
		colLower = cl;
		colUpper = cu;
	}
	
	public MatrixIndexingCPInstruction(Operator op, CPOperand lhsInput, CPOperand rhsInput, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu, CPOperand out, String opcode, String istr){
		super(op, lhsInput, rhsInput, out, opcode, istr);
		rowLower = rl;
		rowUpper = ru;
		colLower = cl;
		colUpper = cu;
	}
	
	public static MatrixIndexingCPInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException {
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		if ( opcode.equalsIgnoreCase("rangeReIndex") ) {
			if ( parts.length == 7 ) {
				// Example: rangeReIndex:mVar1:Var2:Var3:Var4:Var5:mVar6
				CPOperand in, rl, ru, cl, cu, out;
				in = new CPOperand();
				rl = new CPOperand();
				ru = new CPOperand();
				cl = new CPOperand();
				cu = new CPOperand();
				out = new CPOperand();
				in.split(parts[1]);
				rl.split(parts[2]);
				ru.split(parts[3]);
				cl.split(parts[4]);
				cu.split(parts[5]);
				out.split(parts[6]);
				return new MatrixIndexingCPInstruction(new SimpleOperator(null), in, rl, ru, cl, cu, out, opcode, str);
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		} 
		else if ( opcode.equalsIgnoreCase("leftIndex")) {
			if ( parts.length == 8 ) {
				// Example: leftIndex:mVar1:mvar2:Var3:Var4:Var5:Var6:mVar7
				CPOperand lhsInput, rhsInput, rl, ru, cl, cu, out;
				lhsInput = new CPOperand();
				rhsInput = new CPOperand();
				rl = new CPOperand();
				ru = new CPOperand();
				cl = new CPOperand();
				cu = new CPOperand();
				out = new CPOperand();
				lhsInput.split(parts[1]);
				rhsInput.split(parts[2]);
				rl.split(parts[3]);
				ru.split(parts[4]);
				cl.split(parts[5]);
				cu.split(parts[6]);
				out.split(parts[7]);
				return new MatrixIndexingCPInstruction(new SimpleOperator(null), lhsInput, rhsInput, rl, ru, cl, cu, out, opcode, str);
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		}
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a MatrixIndexingCPInstruction: " + str);
		}
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{	
		String opcode = getOpcode();
		
		//get indexing range
		int rl = (int)(ec.getScalarInput(rowLower.getName(), rowLower.getValueType(), rowLower.isLiteral()).getLongValue()-1);
		int ru = (int)(ec.getScalarInput(rowUpper.getName(), rowUpper.getValueType(), rowUpper.isLiteral()).getLongValue()-1);
		int cl = (int)(ec.getScalarInput(colLower.getName(), colLower.getValueType(), colLower.isLiteral()).getLongValue()-1);
		int cu = (int)(ec.getScalarInput(colUpper.getName(), colUpper.getValueType(), colUpper.isLiteral()).getLongValue()-1);
		
		//get original matrix
		MatrixObject mo = (MatrixObject)ec.getVariable(input1.getName());
		
		//right indexing
		if( opcode.equalsIgnoreCase("rangeReIndex") )
		{
			MatrixBlock resultBlock = null;
			
			if( mo.isPartitioned() ) //via data partitioning
				resultBlock = mo.readMatrixPartition( new IndexRange(rl+1,ru+1,cl+1,cu+1) );
			else //via slicing the in-memory matrix
			{
				//execute right indexing operation
				MatrixBlock matBlock = ec.getMatrixInput(input1.getName());
				resultBlock = matBlock.sliceOperations(rl, ru, cl, cu, new MatrixBlock());	
				
				//unpin rhs input
				ec.releaseMatrixInput(input1.getName());
				
				//ensure correct sparse/dense output representation
				//(memory guarded by release of input)
				resultBlock.examSparsity();
			}	
			
			//unpin output
			ec.setMatrixOutput(output.getName(), resultBlock);
		}
		//left indexing
		else if ( opcode.equalsIgnoreCase("leftIndex"))
		{
			boolean inplace = mo.isUpdateInPlaceEnabled();
			MatrixBlock matBlock = ec.getMatrixInput(input1.getName());
			MatrixBlock resultBlock = null;
			
			if(input2.getDataType() == DataType.MATRIX) //MATRIX<-MATRIX
			{
				MatrixBlock rhsMatBlock = ec.getMatrixInput(input2.getName());
				resultBlock = matBlock.leftIndexingOperations(rhsMatBlock, rl, ru, cl, cu, new MatrixBlock(), inplace);
				ec.releaseMatrixInput(input2.getName());
			}
			else //MATRIX<-SCALAR 
			{
				if(!(rl==ru && cl==cu))
					throw new DMLRuntimeException("Invalid index range of scalar leftindexing: ["+rl+":"+ru+","+cl+":"+cu+"]." );
				ScalarObject scalar = ec.getScalarInput(input2.getName(), ValueType.DOUBLE, input2.isLiteral());
				resultBlock = (MatrixBlock) matBlock.leftIndexingOperations(scalar, rl, cl, new MatrixBlock(), inplace);
			}

			//unpin lhs input
			ec.releaseMatrixInput(input1.getName());
			
			//ensure correct sparse/dense output representation
			//(memory guarded by release of input)
			resultBlock.examSparsity();
			
			//unpin output
			ec.setMatrixOutput(output.getName(), resultBlock, inplace);
		}
		else
			throw new DMLRuntimeException("Invalid opcode (" + opcode +") encountered in MatrixIndexingCPInstruction.");		
	}
}
