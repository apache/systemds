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

import org.apache.sysml.runtime.compress.utils.DblArray;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

public class ReaderColumnSelectionDense extends ReaderColumnSelection 
{
	protected MatrixBlock _data;
	
	// reusable return
	private DblArray nonZeroReturn;
	private DblArray reusableReturn;
	private double[] reusableArr;

	public ReaderColumnSelectionDense(MatrixBlock data, int[] colIndices, boolean skipZeros) {
		super(colIndices, CompressedMatrixBlock.TRANSPOSE_INPUT ? 
				data.getNumColumns() : data.getNumRows(), skipZeros);
		_data = data;
		reusableArr = new double[colIndices.length];
		reusableReturn = new DblArray(reusableArr);
	}

	@Override
	public DblArray nextRow() {
		if( _skipZeros) {
			while ((nonZeroReturn = getNextRow()) != null
				&& DblArray.isZero(nonZeroReturn)); 
			return nonZeroReturn;
		} else {
			return getNextRow();
		}
	}

	/**
	 * 
	 * @return
	 */
	private DblArray getNextRow() {
		if(_lastRow == _numRows-1)
			return null;
		_lastRow++;
		for (int i = 0; i < _colIndexes.length; i++) {
			reusableArr[i] = CompressedMatrixBlock.TRANSPOSE_INPUT ? 
					_data.quickGetValue( _colIndexes[i], _lastRow ) : 
					_data.quickGetValue( _lastRow, _colIndexes[i] );
		}
		return reusableReturn;
	}
}
