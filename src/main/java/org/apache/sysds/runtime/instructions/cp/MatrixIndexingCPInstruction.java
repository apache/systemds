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

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.IndexRange;
import org.apache.sysds.utils.Statistics;

public final class MatrixIndexingCPInstruction extends IndexingCPInstruction {

	public MatrixIndexingCPInstruction(CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu,
			CPOperand out, String opcode, String istr) {
		super(in, rl, ru, cl, cu, out, opcode, istr);
	}

	protected MatrixIndexingCPInstruction(CPOperand lhsInput, CPOperand rhsInput, CPOperand rl,
			CPOperand ru, CPOperand cl, CPOperand cu, CPOperand out, String opcode, String istr) {
		super(lhsInput, rhsInput, rl, ru, cl, cu, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		String opcode = getOpcode();
		IndexRange ix = getIndexRange(ec);
		
		MatrixObject mo = ec.getMatrixObject(input1.getName());
		boolean inRange = ix.rowStart < mo.getNumRows() && ix.colStart < mo.getNumColumns();
		
		//right indexing
		if( opcode.equalsIgnoreCase(Opcodes.RIGHT_INDEX.toString()) )
		{
			if( output.isScalar() && inRange ) { //SCALAR out
				MatrixBlock matBlock = mo.acquireReadAndRelease();
				ec.setScalarOutput(output.getName(),
					new DoubleObject(matBlock.get((int)ix.rowStart, (int)ix.colStart)));
			}
			else { //MATRIX out
				MatrixBlock resultBlock = null;
				
				if( mo.isPartitioned() ) //via data partitioning
					resultBlock = mo.readMatrixPartition(ix.add(1));
				else if( ix.isScalar() && inRange ) {
					MatrixBlock matBlock = mo.acquireReadAndRelease();
					resultBlock = new MatrixBlock(
						matBlock.get((int)ix.rowStart, (int)ix.colStart));
				}
				else //via slicing the in-memory matrix
				{
					//execute right indexing operation (with shallow row copies for range
					//of entire sparse rows, which is safe due to copy on update)
					MatrixBlock matBlock = mo.acquireRead();
					resultBlock = matBlock.slice((int)ix.rowStart, (int)ix.rowEnd, 
						(int)ix.colStart, (int)ix.colEnd, false, new MatrixBlock());
					
					//unpin rhs input
					ec.releaseMatrixInput(input1.getName());
					
					//ensure correct sparse/dense output representation
					if( checkGuardedRepresentationChange(matBlock, resultBlock) )
						resultBlock.examSparsity();
				}
				
				//unpin output
				ec.setMatrixOutput(output.getName(), resultBlock);
			}
		}
		//left indexing
		else if ( opcode.equalsIgnoreCase(Opcodes.LEFT_INDEX.toString()))
		{
			UpdateType updateType = mo.getUpdateType();
			if(DMLScript.STATISTICS) {
				if( updateType.isInPlace() )
					Statistics.incrementTotalLixUIP();
				Statistics.incrementTotalLix();
			}
			
			MatrixBlock matBlock = ec.getMatrixInput(input1.getName());
			MatrixBlock resultBlock = null;
			
			if(input2.getDataType() == DataType.MATRIX) { //MATRIX<-MATRIX
				MatrixBlock rhsMatBlock = ec.getMatrixInput(input2.getName());
				resultBlock = matBlock.leftIndexingOperations(rhsMatBlock, ix, new MatrixBlock(), updateType);
				ec.releaseMatrixInput(input2.getName());
			}
			else { //MATRIX<-SCALAR 
				if(!ix.isScalar())
					throw new DMLRuntimeException("Invalid index range of scalar leftindexing: "+ix.toString()+"." );
				ScalarObject scalar = ec.getScalarInput(input2.getName(), ValueType.FP64, input2.isLiteral());
				resultBlock = matBlock.leftIndexingOperations(scalar, 
					(int)ix.rowStart, (int)ix.colStart, new MatrixBlock(), updateType);
			}

			//unpin lhs input
			ec.releaseMatrixInput(input1.getName());
			
			//ensure correct sparse/dense output representation
			//(memory guarded by release of input)
			resultBlock.examSparsity();
			
			//unpin output
			ec.setMatrixOutput(output.getName(), resultBlock, updateType);
		}
		else
			throw new DMLRuntimeException("Invalid opcode (" + opcode +") encountered in MatrixIndexingCPInstruction.");
	}
	
	@Override
	public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
		return Pair.of(output.getName(), new LineageItem(getOpcode(),
			LineageItemUtils.getLineage(ec, input1,input2,input3,rowLower,rowUpper,colLower,colUpper)));
	}
}
