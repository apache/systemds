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

import java.util.Arrays;

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

	private final SparseBlock a;

	protected ReaderColumnSelectionSparse(MatrixBlock data, int[] colIndexes, int rl, int ru) {
		super(colIndexes, rl, Math.min(ru, data.getNumRows()) - 1);
		a = data.getSparseBlock();
	}

	protected final DblArray getNextRow() {
		while(_rl < _ru) {
			_rl++;
			if(a.isEmpty(_rl))
				continue; // if empty easy skip

			final boolean zeroResult = processInRange(_rl);

			if(zeroResult)
				continue; // skip if no values found were in my cols

			return reusableReturn;
		}
		return null;

	}

	final boolean processInRange(final int r) {
		boolean zeroResult = true;
		final int apos = a.pos(r);
		final int alen = a.size(r) + apos;
		final int[] aix = a.indexes(r);
		final double[] avals = a.values(r);
		int skip = 0;
		int j = Arrays.binarySearch(aix, apos, alen, _colIndexes[0]);
		if(j < 0)
			j = Math.abs(j+1);

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
		if(zeroResult)
			return true; // skip if no values found were in my cols

		while(skip < _colIndexes.length)
			reusableArr[skip++] = 0;

		return false;
	}
}
