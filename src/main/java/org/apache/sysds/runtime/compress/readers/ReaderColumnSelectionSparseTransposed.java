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
public class ReaderColumnSelectionSparseTransposed extends ReaderColumnSelection {

	// sparse block to iterate through
	private final SparseBlock a;
	// current sparse skip positions.
	private final int[] sparsePos;

	private boolean atEnd = false;

	protected ReaderColumnSelectionSparseTransposed(MatrixBlock data, int[] colIndexes, int rl, int ru) {
		super(colIndexes, rl, Math.min(ru, data.getNumColumns()));
		sparsePos = new int[colIndexes.length];
		a = data.getSparseBlock();
		_rl = _rl + 1; // correct row since this iterator use the exact row

		for(int i = 0; i < colIndexes.length; i++) {
			final int c = _colIndexes[i];
			if(a.isEmpty(c)) {
				atEnd = true;
				sparsePos[i] = -1;
			}
			else {
				final int[] aIdx = a.indexes(c);
				final int pos = a.pos(c);
				final int len = a.size(c) + pos;
				final int spa = Arrays.binarySearch(aIdx, pos, len, _rl);
				if(spa >= 0) {
					// it should never happen that if the value of _rl is found that the end is out of range.
					sparsePos[i] = spa;
				}
				else { // spa < 0 or larger.
					final int spaC = Math.abs(spa + 1);
					if(spaC < len && aIdx[spaC] < _ru)
						sparsePos[i] = spaC;
					else {
						atEnd = true;
						sparsePos[i] = -1;
					}
				}
			}
		}
	}

	protected DblArray getNextRow() {
		if(!atEnd)
			return getNextRowBeforeEnd();
		else
			return getNextRowAtEnd();
	}

	protected DblArray getNextRowBeforeEnd() {
		skipToRow();
		if(_rl >= _ru) { // if done return null
			_rl = _ru;
			return null;
		}
		for(int i = 0; i < _colIndexes.length; i++) {
			final int c = _colIndexes[i];
			final int sp = sparsePos[i];
			final int[] aix = a.indexes(c);
			if(aix[sp] == _rl) {
				final double[] avals = a.values(c);
				reusableArr[i] = avals[sp];
				final int spa = sparsePos[i]++;
				final int len = a.size(c) + a.pos(c) - 1;
				if(spa >= len || aix[spa] >= _ru) {
					sparsePos[i] = -1;
					atEnd = true;
				}
			}
			else
				reusableArr[i] = 0;
		}

		return reusableReturn;
	}

	private void skipToRow() {
		_rl = a.indexes(_colIndexes[0])[sparsePos[0]];
		for(int i = 1; i < _colIndexes.length; i++)
			_rl = Math.min(a.indexes(_colIndexes[i])[sparsePos[i]], _rl);
	}

	protected DblArray getNextRowAtEnd() {
		// at end
		skipToRowAtEnd();

		if(_rl == _ru) { // if done return null
			_rl = _ru;
			return null;
		}

		for(int i = 0; i < _colIndexes.length; i++) {
			int c = _colIndexes[i];
			final int sp = sparsePos[i];
			if(sp != -1) {
				final int[] aix = a.indexes(c);
				if(aix[sp] == _rl) {
					final double[] avals = a.values(c);
					reusableArr[i] = avals[sp];
					if(++sparsePos[i] >= a.size(c) + a.pos(c))
						sparsePos[i] = -1;
				}
				else
					reusableArr[i] = 0;
			}
		}
		return reusableReturn;
	}

	private void skipToRowAtEnd() {
		boolean allDone = true;
		int mr = _ru;
		for(int i = 0; i < _colIndexes.length; i++) {
			final int sp = sparsePos[i];
			if(sp != -1) {
				allDone = false;
				mr = Math.min(a.indexes(_colIndexes[i])[sp], mr);
			}
			else
				reusableArr[i] = 0;
		}
		_rl = mr;
		if(allDone)
			_rl = _ru;
	}
}
