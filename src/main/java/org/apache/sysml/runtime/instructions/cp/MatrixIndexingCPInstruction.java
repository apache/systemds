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

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.util.IndexRange;
import org.apache.sysml.utils.Statistics;

public final class MatrixIndexingCPInstruction extends IndexingCPInstruction
{	
	public MatrixIndexingCPInstruction(Operator op, CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu, CPOperand out, String opcode, String istr){
		super(op, in, rl, ru, cl, cu, out, opcode, istr);
	}
	
	public MatrixIndexingCPInstruction(Operator op, CPOperand lhsInput, CPOperand rhsInput, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu, CPOperand out, String opcode, String istr){
		super(op, lhsInput, rhsInput, rl, ru, cl, cu, out, opcode, istr);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{	
		String opcode = getOpcode();
		IndexRange ixrange = getIndexRange(ec);
		
		//get original matrix
		MatrixObject mo = (MatrixObject)ec.getVariable(input1.getName());
		
		//right indexing
		if( opcode.equalsIgnoreCase("rangeReIndex") )
		{
			MatrixBlock resultBlock = null;
			
			if( mo.isPartitioned() ) //via data partitioning
				resultBlock = mo.readMatrixPartition(ixrange.add(1));
			else //via slicing the in-memory matrix
			{
				//execute right indexing operation
				MatrixBlock matBlock = ec.getMatrixInput(input1.getName());
				resultBlock = matBlock.sliceOperations(ixrange, new MatrixBlock());	
				
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
			if(DMLScript.STATISTICS)
			{
				if(inplace)
					Statistics.incrementTotalLixUIP();
				Statistics.incrementTotalLix();
			}
			
			MatrixBlock matBlock = ec.getMatrixInput(input1.getName());
			MatrixBlock resultBlock = null;
			
			if(input2.getDataType() == DataType.MATRIX) //MATRIX<-MATRIX
			{
				MatrixBlock rhsMatBlock = ec.getMatrixInput(input2.getName());
				resultBlock = matBlock.leftIndexingOperations(rhsMatBlock, ixrange, new MatrixBlock(), inplace);
				ec.releaseMatrixInput(input2.getName());
			}
			else //MATRIX<-SCALAR 
			{
				if(!ixrange.isScalar())
					throw new DMLRuntimeException("Invalid index range of scalar leftindexing: "+ixrange.toString()+"." );
				ScalarObject scalar = ec.getScalarInput(input2.getName(), ValueType.DOUBLE, input2.isLiteral());
				resultBlock = (MatrixBlock) matBlock.leftIndexingOperations(scalar, 
						(int)ixrange.rowStart, (int)ixrange.colStart, new MatrixBlock(), inplace);
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
