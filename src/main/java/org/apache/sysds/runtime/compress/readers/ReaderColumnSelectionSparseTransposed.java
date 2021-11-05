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
public class ReaderColumnSelectionSparseTransposed extends ReaderColumnSelection {

	private SparseBlock a;
	// current sparse skip positions.
	private int[] sparsePos = null;


	protected ReaderColumnSelectionSparseTransposed(MatrixBlock data, int[] colIndexes, int rl, int ru) {
		super(colIndexes, rl, Math.min(ru, data.getNumColumns()));
		sparsePos = new int[colIndexes.length];

		a = data.getSparseBlock();
		// Use -1 to indicate that this column is done.
		for(int i = 0; i < colIndexes.length; i++) {
			if(a.isEmpty(_colIndexes[i]))
				sparsePos[i] = -1;
			else
				sparsePos[i] = a.pos(_colIndexes[i]);
		}
	}

	protected DblArray getNextRow() {
		if(_rl == _ru - 1)
			return null;

		_rl++;

		boolean zeroResult = true;
		boolean allDone = true;
		for(int i = 0; i < _colIndexes.length; i++) {
			int colidx = _colIndexes[i];
			if(sparsePos[i] != -1) {
				allDone = false;
				final int alen = a.size(colidx) + a.pos(colidx);
				final int[] aix = a.indexes(colidx);
				final double[] avals = a.values(colidx);
				while(sparsePos[i] < alen && aix[sparsePos[i]] < _rl)
					sparsePos[i] += 1;

				if(sparsePos[i] >= alen) {
					// Mark this column as done.
					sparsePos[i] = -1;
					reusableArr[i] = 0;
				}
				else if(aix[sparsePos[i]] == _rl) {
					reusableArr[i] = avals[sparsePos[i]];
					zeroResult = false;
				}
				else
					reusableArr[i] = 0;
			}
		}
		if(allDone)
			_rl = _ru - 1;
		return zeroResult ? emptyReturn : reusableReturn;

	}
}
