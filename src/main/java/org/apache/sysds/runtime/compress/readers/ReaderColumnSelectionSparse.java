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

package org.apache.sysds.runtime.compress.readers;

import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Used to extract the values at certain indexes from each row in a sparse matrix
 * 
 * Keeps returning all-zeros arrays until reaching the last possible index. The current compression algorithm treats the
 * zero-value in a sparse matrix like any other value.
 */
public class ReaderColumnSelectionSparse extends ReaderColumnSelection {

	// reusable return
	private DblArray reusableReturn;
	private double[] reusableArr;

	// an empty array to return if the entire row was 0.
	private DblArray empty = new DblArray();

	private SparseBlock a;

	/**
	 * Reader of sparse matrix blocks for compression.
	 * 
	 * This reader should not be used if the input data is not sparse
	 * 
	 * @param data         The transposed and sparse matrix
	 * @param colIndexes   The column indexes to combine
	 */
	public ReaderColumnSelectionSparse(MatrixBlock data, int[] colIndexes) {
		super(colIndexes, data.getNumRows());
		reusableArr = new double[colIndexes.length];
		reusableReturn = new DblArray(reusableArr);
		a = data.getSparseBlock();
	}

	protected DblArray getNextRow() {
		if(_lastRow == _numRows - 1) {
			return null;
		}

		_lastRow++;

		boolean zeroResult = true;

		if(a != null && !a.isEmpty(_lastRow)) {

			int apos = a.pos(_lastRow);
			int alen = a.size(_lastRow) + apos;
			int[] aix = a.indexes(_lastRow);
			double[] avals = a.values(_lastRow);
			int skip = 0;
			int j = apos;

			while(skip < _colIndexes.length && j < alen) {
				if(_colIndexes[skip] == aix[j]) {
					reusableArr[skip++] = avals[j++];
					zeroResult = false;
				}
				else if(_colIndexes[skip] > aix[j]) {
					j++;
				}
				else {
					reusableArr[skip++] = 0;
				}

			}
		}

		return zeroResult ? empty : reusableReturn;
	}
}
