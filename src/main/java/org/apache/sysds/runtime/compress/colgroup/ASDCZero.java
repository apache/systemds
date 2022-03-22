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

package org.apache.sysds.runtime.compress.colgroup;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public abstract class ASDCZero extends APreAgg {
	private static final long serialVersionUID = -69266306137398807L;
	
	/** Sparse row indexes for the data */
	protected AOffset _indexes;

	protected ASDCZero(int numRows) {
		super(numRows);
	}

	protected ASDCZero(int[] colIndices, int numRows, ADictionary dict, AOffset offsets, int[] cachedCounts) {
		super(colIndices, numRows, dict, cachedCounts);
		_indexes = offsets;
		_zeros = true;
	}

	@Override
	public final void leftMultByMatrixNoPreAgg(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {
		final AIterator it = _indexes.getIterator(cl);
		if(it == null)
			return;
		else if(it.value() > cu)
			_indexes.cacheIterator(it, cu); // cache this iterator.
		else if(rl == ru - 1)
			leftMultByMatrixNoPreAggSingleRow(matrix, result, rl, cl, cu, it);
		else
			leftMultByMatrixNoPreAggRows(matrix, result, rl, ru, cl, cu, it);
	}

	protected final void leftMultByMatrixNoPreAggSingleRow(MatrixBlock mb, MatrixBlock result, int r, int cl, int cu,
		AIterator it) {
		final double[] resV = result.getDenseBlockValues();
		final int nCols = result.getNumColumns();
		final int offRet = nCols * r;
		if(mb.isInSparseFormat()) {
			final SparseBlock sb = mb.getSparseBlock();
			if(cl != 0 && cu != _numRows)
				throw new NotImplementedException();
			leftMultByMatrixNoPreAggSingleRowSparse(sb, resV, offRet, r, it);
		}
		else {
			final DenseBlock db = mb.getDenseBlock();
			final double[] mV = db.values(r);
			final int off = db.pos(r);
			leftMultByMatrixNoPreAggSingleRowDense(mV, off, resV, offRet, r, cl, cu, it);
		}
	}

	protected final void leftMultByMatrixNoPreAggSingleRowDense(double[] mV, int off, double[] resV, int offRet, int r,
		int cl, int cu, AIterator it) {
		final int last = _indexes.getOffsetToLast();
		while(it.isNotOver(cu)) {
			multiplyScalar(mV[off + it.value()], resV, offRet, it);
			if(it.value() < last)
				it.next();
			else
				break;
		}
		_indexes.cacheIterator(it, cu);
	}

	protected synchronized final void leftMultByMatrixNoPreAggSingleRowSparse(SparseBlock sb, double[] resV, int offRet,
		int r, AIterator it) {
		if(sb.isEmpty(r))
			return;
		final int last = _indexes.getOffsetToLast();
		int apos = sb.pos(r); // use apos as the pointer
		final int alen = sb.size(r) + apos;
		final int[] aix = sb.indexes(r);
		final double[] aval = sb.values(r);

		int v = it.value();
		if(aix[alen - 1] < last) {
			while(apos < alen) {
				if(aix[apos] == v) {
					multiplyScalar(aval[apos++], resV, offRet, it);
					v = it.next();
				}
				else if(aix[apos] < v)
					apos++;
				else
					v = it.next();
			}
		}
		else {
			while(v < last) {
				if(aix[apos] == v) {
					multiplyScalar(aval[apos++], resV, offRet, it);
					v = it.next();
				}
				else if(aix[apos] < v)
					apos++;
				else
					v = it.next();
			}
			while(aix[apos] < last && apos < alen)
				apos++;

			if(last == aix[apos])
				multiplyScalar(aval[apos], resV, offRet, it);

		}

	}

	protected final void leftMultByMatrixNoPreAggRows(MatrixBlock mb, MatrixBlock result, int rl, int ru, int cl, int cu,
		AIterator it) {
		final double[] resV = result.getDenseBlockValues();
		final int nCols = result.getNumColumns();
		if(mb.isInSparseFormat()) {
			final SparseBlock sb = mb.getSparseBlock();
			leftMultByMatrixNoPreAggRowsSparse(sb, resV, nCols, rl, ru, cl, cu, it);
		}
		else
			leftMultByMatrixNoPreAggRowsDense(mb, resV, nCols, rl, ru, cl, cu, it);

	}

	protected final void leftMultByMatrixNoPreAggRowsSparse(SparseBlock sb, double[] resV, int nCols, int rl, int ru,
		int cl, int cu, AIterator it) {
		if(cl != 0 && cu != _numRows)
			throw new NotImplementedException();
		for(int r = rl; r < ru; r++) {
			final int offRet = nCols * r;
			leftMultByMatrixNoPreAggSingleRowSparse(sb, resV, offRet, r, it.clone());
		}

	}

	protected final void leftMultByMatrixNoPreAggRowsDense(MatrixBlock mb, double[] resV, int nCols, int rl, int ru,
		int cl, int cu, AIterator it) {
		final DenseBlock db = mb.getDenseBlock();
		for(int r = rl; r < ru; r++) {
			final double[] mV = db.values(r);
			final int off = db.pos(r);
			final int offRet = nCols * r;
			leftMultByMatrixNoPreAggSingleRowDense(mV, off, resV, offRet, r, cl, cu, it.clone());
		}
	}

	/**
	 * Multiply the scalar v with the tuple entry inside the column group at the current index the iterator says.
	 * 
	 * @param v      The value to multiply
	 * @param resV   The result matrix to put it into
	 * @param offRet The offset into the result matrix to consider start of the row
	 * @param it     The iterator containing the index the tuple in the dictionary have.
	 */
	protected abstract void multiplyScalar(double v, double[] resV, int offRet, AIterator it);

	public void decompressToDenseBlockDenseDictionary(DenseBlock db, int rl, int ru, int offR, int offC, AIterator it) {
		decompressToDenseBlockDenseDictionary(db, rl, ru, offR, offC, _dict.getValues(), it);
	}

	public abstract void decompressToDenseBlockDenseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values, AIterator it);

	public AIterator getIterator(int row) {
		return _indexes.getIterator(row);
	}

	protected abstract int getIndexesSize();
}
