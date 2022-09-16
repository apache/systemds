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

import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public abstract class ASDCZero extends APreAgg {
	private static final long serialVersionUID = -69266306137398807L;

	/** Sparse row indexes for the data */
	protected AOffset _indexes;
	final protected int _numRows;

	protected ASDCZero(int[] colIndices, int numRows, ADictionary dict, AOffset offsets, int[] cachedCounts) {
		super(colIndices, dict, cachedCounts);
		_indexes = offsets;
		_numRows = numRows;
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

	private final void leftMultByMatrixNoPreAggSingleRow(MatrixBlock mb, MatrixBlock result, int r, int cl, int cu,
		AIterator it) {
		final double[] resV = result.getDenseBlockValues();
		final int nCols = result.getNumColumns();
		final int offRet = nCols * r;
		if(mb.isInSparseFormat()) {
			final SparseBlock sb = mb.getSparseBlock();
			leftMultByMatrixNoPreAggSingleRowSparse(sb, resV, offRet, r, cu, it);
		}
		else {
			final DenseBlock db = mb.getDenseBlock();
			final double[] mV = db.values(r);
			final int off = db.pos(r);
			leftMultByMatrixNoPreAggSingleRowDense(mV, off, resV, offRet, r, cl, cu, it);
		}
	}

	private final void leftMultByMatrixNoPreAggSingleRowDense(double[] mV, int off, double[] resV, int offRet, int r,
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

	private final void leftMultByMatrixNoPreAggSingleRowSparse(final SparseBlock sb, final double[] resV,
		final int offRet, final int r, final int cu, final AIterator it) {
		if(sb.isEmpty(r))
			return;
		final int last = _indexes.getOffsetToLast();
		int apos = sb.pos(r); // use apos as the pointer
		final int alen = sb.size(r) + apos;
		final int[] aix = sb.indexes(r);
		final double[] aval = sb.values(r);
		int v = it.value();
		while(apos < alen && aix[apos] < v)
			apos++; // go though sparse block until offset start.
		if(cu < last) {
			while(v < cu && apos < alen) {
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
		else if(aix[alen - 1] < last) {
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

	private final void leftMultByMatrixNoPreAggRows(MatrixBlock mb, MatrixBlock result, int rl, int ru, int cl, int cu,
		AIterator it) {
		final double[] resV = result.getDenseBlockValues();
		final int nCols = result.getNumColumns();
		if(mb.isInSparseFormat())
			leftMultByMatrixNoPreAggRowsSparse(mb.getSparseBlock(), resV, nCols, rl, ru, cl, cu, it);
		else
			leftMultByMatrixNoPreAggRowsDense(mb, resV, nCols, rl, ru, cl, cu, it);

	}

	private final void leftMultByMatrixNoPreAggRowsSparse(SparseBlock sb, double[] resV, int nCols, int rl, int ru,
		int cl, int cu, AIterator it) {
		for(int r = rl; r < ru; r++) {
			final int offRet = nCols * r;
			leftMultByMatrixNoPreAggSingleRowSparse(sb, resV, offRet, r, cu, it.clone());
		}
	}

	private final void leftMultByMatrixNoPreAggRowsDense(MatrixBlock mb, double[] resV, int nCols, int rl, int ru,
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

	public void decompressToDenseBlock(DenseBlock db, int rl, int ru, int offR, int offC, AIterator it) {
		if(_dict instanceof MatrixBlockDictionary) {
			final MatrixBlockDictionary md = (MatrixBlockDictionary) _dict;
			final MatrixBlock mb = md.getMatrixBlock();
			// The dictionary is never empty.
			if(mb.isInSparseFormat())
				// TODO make sparse decompression where the iterator is known in argument
				decompressToDenseBlockSparseDictionary(db, rl, ru, offR, offC, mb.getSparseBlock());
			else
				decompressToDenseBlockDenseDictionaryWithProvidedIterator(db, rl, ru, offR, offC, mb.getDenseBlockValues(),
					it);
		}
		else
			decompressToDenseBlockDenseDictionaryWithProvidedIterator(db, rl, ru, offR, offC, _dict.getValues(), it);
	}

	public void decompressToDenseBlockDenseDictionary(DenseBlock db, int rl, int ru, int offR, int offC, AIterator it) {
		decompressToDenseBlockDenseDictionaryWithProvidedIterator(db, rl, ru, offR, offC, _dict.getValues(), it);
	}

	public abstract void decompressToDenseBlockDenseDictionaryWithProvidedIterator(DenseBlock db, int rl, int ru,
		int offR, int offC, double[] values, AIterator it);

	public AIterator getIterator(int row) {
		return _indexes.getIterator(row);
	}
}
