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

package org.apache.sysml.runtime.compress;

import java.util.Arrays;

import org.apache.sysml.runtime.compress.utils.DblArray;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.SparseRow;

/**
 * Used to extract the values at certain indexes from each row in a sparse
 * matrix
 * 
 * Keeps returning all-zeros arrays until reaching the last possible index. The
 * current compression algorithm treats the zero-value in a sparse matrix like
 * any other value.
 */
public class ReaderColumnSelectionSparse extends ReaderColumnSelection 
{
	private final DblArray ZERO_DBL_ARRAY;
	private DblArray nonZeroReturn;

	// reusable return
	private DblArray reusableReturn;
	private double[] reusableArr;

	// current sparse row positions
	private SparseRow[] sparseCols = null;
	private int[] sparsePos = null;
	
	public ReaderColumnSelectionSparse(MatrixBlock data, int[] colIndexes, boolean skipZeros) 
	{
		super(colIndexes, CompressedMatrixBlock.TRANSPOSE_INPUT ? 
				data.getNumColumns() : data.getNumRows(), skipZeros);
		ZERO_DBL_ARRAY = new DblArray(new double[colIndexes.length], true);
		reusableArr = new double[colIndexes.length];
		reusableReturn = new DblArray(reusableArr);
		
		if( !CompressedMatrixBlock.TRANSPOSE_INPUT ){
			throw new RuntimeException("SparseColumnSelectionReader should not be used without transposed input.");
		}
		
		sparseCols = new SparseRow[colIndexes.length];
		sparsePos = new int[colIndexes.length];
		if( data.getSparseBlock()!=null )
		for( int i=0; i<colIndexes.length; i++ )
			sparseCols[i] = data.getSparseBlock().get(colIndexes[i]);
		Arrays.fill(sparsePos, 0);
	}

	@Override
	public DblArray nextRow() {
		if(_skipZeros) {
			while ((nonZeroReturn = getNextRow()) != null
				&& nonZeroReturn == ZERO_DBL_ARRAY);
			return nonZeroReturn;
		} else {
			return getNextRow();
		}
	}

	/**
	 * 
	 * @return
	 */
	private DblArray getNextRow() 
	{
		if(_lastRow == _numRows-1)
			return null;
		_lastRow++;
		
		if( !CompressedMatrixBlock.TRANSPOSE_INPUT ){
			throw new RuntimeException("SparseColumnSelectionReader should not be used without transposed input.");
		}
		
		//move pos to current row if necessary (for all columns)
		for( int i=0; i<_colIndexes.length; i++ )
			if( sparseCols[i] != null && (sparseCols[i].indexes().length<=sparsePos[i] 
				|| sparseCols[i].indexes()[sparsePos[i]]<_lastRow) )
			{
				sparsePos[i]++;
			}
		
		//extract current values
		Arrays.fill(reusableArr, 0);
		boolean zeroResult = true;
		for( int i=0; i<_colIndexes.length; i++ ) 
			if( sparseCols[i] != null && sparseCols[i].indexes().length>sparsePos[i]
				&&sparseCols[i].indexes()[sparsePos[i]]==_lastRow )
			{
				reusableArr[i] = sparseCols[i].values()[sparsePos[i]];
				zeroResult = false;
			}

		return zeroResult ? ZERO_DBL_ARRAY : reusableReturn;
	}
}
