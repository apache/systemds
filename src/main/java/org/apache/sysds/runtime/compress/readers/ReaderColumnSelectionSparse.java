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

	private SparseBlock a;

	protected ReaderColumnSelectionSparse(MatrixBlock data, int[] colIndexes, int rl, int ru) {
		super(colIndexes, rl, Math.min(ru, data.getNumRows()));
		a = data.getSparseBlock();
	}

	protected final DblArray getNextRow() {
		if(_rl == _ru - 1) {
			return null;
		}

		_rl++;

		if(a.isEmpty(_rl))
			return emptyReturn;

		final int apos = a.pos(_rl);
		final int alen = a.size(_rl) + apos;
		final int[] aix = a.indexes(_rl);

		if(aix[alen - 1] < _colIndexes[0] || aix[apos] > _colIndexes[_colIndexes.length - 1])
			return emptyReturn;

		return nextRow(apos, alen, aix, a.values(_rl));
	}

	private final DblArray nextRow(final int apos, final int alen, final int[] aix, final double[] avals) {
		boolean zeroResult = true;

		int skip = 0;
		int j = apos;
		while(aix[j] < _colIndexes[0])
			j++;
		while(skip < _colIndexes.length && j < alen) {
			if(_colIndexes[skip] == aix[j]) {
				reusableArr[skip] = avals[j];
				zeroResult = false;
				skip++;
				j++;
			}
			else if(_colIndexes[skip] > aix[j])
				j++;
			else
				reusableArr[skip++] = 0;
		}
		while(skip < _colIndexes.length)
			reusableArr[skip++] = 0;

		return zeroResult ? emptyReturn : reusableReturn;
	}
}
