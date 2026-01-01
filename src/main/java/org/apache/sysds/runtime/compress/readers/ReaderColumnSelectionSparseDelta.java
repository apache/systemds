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

import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class ReaderColumnSelectionSparseDelta extends ReaderColumnSelection {

	private final SparseBlock _a;
	private final double[] _previousRow;
	private boolean _isFirstRow;

	protected ReaderColumnSelectionSparseDelta(MatrixBlock data, IColIndex colIndexes, int rl, int ru) {
		super(colIndexes, rl, Math.min(ru, data.getNumRows()) - 1);
		_a = data.getSparseBlock();
		_previousRow = new double[colIndexes.size()];
		_isFirstRow = true;
	}

	protected final DblArray getNextRow() {
		_rl++;
		for(int i = 0; i < _colIndexes.size(); i++)
			reusableArr[i] = 0.0;

		if(!_a.isEmpty(_rl))
			processInRange(_rl);

		if(_isFirstRow) {
			for(int i = 0; i < _colIndexes.size(); i++)
				_previousRow[i] = reusableArr[i];
			_isFirstRow = false;
		}
		else {
			for(int i = 0; i < _colIndexes.size(); i++) {
				final double currentVal = reusableArr[i];
				reusableArr[i] = currentVal - _previousRow[i];
				_previousRow[i] = currentVal;
			}
		}

		return reusableReturn;
	}

	final void processInRange(final int r) {
		final int apos = _a.pos(r);
		final int alen = _a.size(r) + apos;
		final int[] aix = _a.indexes(r);
		final double[] avals = _a.values(r);
		int skip = 0;
		int j = Arrays.binarySearch(aix, apos, alen, _colIndexes.get(0));
		if(j < 0)
			j = Math.abs(j + 1);

		while(skip < _colIndexes.size() && j < alen) {
			if(_colIndexes.get(skip) == aix[j]) {
				reusableArr[skip] = avals[j];
				skip++;
				j++;
			}
			else if(_colIndexes.get(skip) > aix[j])
				j++;
			else
				skip++;
		}
	}
}



