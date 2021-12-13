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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

/**
 * Column group that sparsely encodes the dictionary values. The idea is that all values is encoded with indexes except
 * the most common one. the most common one can be inferred by not being included in the indexes. If the values are very
 * sparse then the most common one is zero.
 * 
 * This column group is handy in cases where sparse unsafe operations is executed on very sparse columns. Then the zeros
 * would be materialized in the group without any overhead.
 */
public class ColGroupSDCSingleZeros extends APreAgg {
	private static final long serialVersionUID = 8033235615964315078L;
	
	/** Sparse row indexes for the data */
	protected transient AOffset _indexes;

	/**
	 * Constructor for serialization
	 * 
	 * @param numRows Number of rows contained
	 */
	protected ColGroupSDCSingleZeros(int numRows) {
		super(numRows);
	}

	protected ColGroupSDCSingleZeros(int[] colIndices, int numRows, ADictionary dict, AOffset offsets,
		int[] cachedCounts) {
		super(colIndices, numRows, dict, cachedCounts);
		_indexes = offsets;
		_zeros = true;
	}

	@Override
	public CompressionType getCompType() {
		return CompressionType.SDC;
	}

	@Override
	public ColGroupType getColGroupType() {
		return ColGroupType.SDCSingleZeros;
	}

	@Override
	protected void decompressToDenseBlockDenseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values) {

		final AIterator it = _indexes.getIterator(rl);
		if(it == null)
			return;
		else if(it.value() >= ru)
			_indexes.cacheIterator(it, ru);
		else if(ru > _indexes.getOffsetToLast()) {
			final int maxOff = _indexes.getOffsetToLast();
			final int nCol = _colIndexes.length;
			while(true) {
				final int row = offR + it.value();
				final double[] c = db.values(row);
				final int off = db.pos(row);
				for(int j = 0; j < nCol; j++)
					c[off + _colIndexes[j] + offC] += values[j];
				if(it.value() < maxOff)
					it.next();
				else
					break;
			}
		}
		else {
			final int nCol = _colIndexes.length;
			while(it.isNotOver(ru)) {
				final int row = offR + it.value();
				final double[] c = db.values(row);
				final int off = db.pos(row);
				for(int j = 0; j < nCol; j++)
					c[off + _colIndexes[j] + offC] += values[j];

				it.next();
			}
			_indexes.cacheIterator(it, ru);
		}

	}

	@Override
	protected void decompressToDenseBlockSparseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		SparseBlock sb) {
		final AIterator it = _indexes.getIterator(rl);
		if(it == null)
			return;
		else if(it.value() >= ru)
			_indexes.cacheIterator(it, ru);
		else if(ru > _indexes.getOffsetToLast()) {
			throw new NotImplementedException();
		}
		else {
			final int apos = sb.pos(0);
			final int alen = sb.size(0) + apos;
			final int[] aix = sb.indexes(0);
			final double[] avals = sb.values(0);
			while(it.isNotOver(ru)) {
				final int row = offR + it.value();
				final double[] c = db.values(row);
				final int off = db.pos(row);
				for(int j = apos; j < alen; j++)
					c[off + _colIndexes[aix[j]] + offC] += avals[j];
				it.next();
			}
			_indexes.cacheIterator(it, ru);
		}
	}

	@Override
	protected void decompressToSparseBlockSparseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		SparseBlock sb) {
		final AIterator it = _indexes.getIterator(rl);
		if(it == null)
			return;
		else if(it.value() >= ru)
			_indexes.cacheIterator(it, ru);
		else if(ru > _indexes.getOffsetToLast()) {
			throw new NotImplementedException();
		}
		else {
			final int apos = sb.pos(0);
			final int alen = sb.size(0) + apos;
			final int[] aix = sb.indexes(0);
			final double[] avals = sb.values(0);
			while(it.isNotOver(ru)) {
				final int row = offR + it.value();
				for(int j = apos; j < alen; j++)
					ret.append(row, _colIndexes[aix[j]] + offC, avals[j]);

				it.next();
			}
			_indexes.cacheIterator(it, ru);
		}
	}

	@Override
	protected void decompressToSparseBlockDenseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		double[] values) {
		final AIterator it = _indexes.getIterator(rl);
		if(it == null)
			return;
		else if(it.value() >= ru)
			_indexes.cacheIterator(it, ru);
		else if(ru > _indexes.getOffsetToLast()) {
			final int nCol = _colIndexes.length;
			final int lastOff = _indexes.getOffsetToLast();
			while(true) {
				final int row = offR + it.value();
				for(int j = 0; j < nCol; j++)
					ret.append(row, _colIndexes[j] + offC, values[j]);
				if(it.value() == lastOff)
					return;
				it.next();
			}
		}
		else {
			final int nCol = _colIndexes.length;
			while(it.isNotOver(ru)) {
				final int row = offR + it.value();
				for(int j = 0; j < nCol; j++)
					ret.append(row, _colIndexes[j] + offC, values[j]);

				it.next();
			}
			_indexes.cacheIterator(it, ru);
		}
	}

	@Override
	public double getIdx(int r, int colIdx) {
		final AIterator it = _indexes.getIterator(r);
		if(it == null || it.value() != r)
			return 0;
		return _dict.getValue(colIdx);
	}

	@Override
	protected void computeRowSums(double[] c, int rl, int ru, double[] preAgg) {
		computeRowSum(c, rl, ru, preAgg[0]);
	}

	protected void computeRowSum(double[] c, int rl, int ru, double def) {
		final AIterator it = _indexes.getIterator(rl);
		if(it == null)
			return;
		else if(it.value() > ru)
			_indexes.cacheIterator(it, ru);
		else if(ru > _indexes.getOffsetToLast()) {
			final int maxOff = _indexes.getOffsetToLast();
			while(true) {
				c[it.value()] += def;
				if(it.value() == maxOff)
					break;
				it.next();
			}
		}
		else {
			while(it.isNotOver(ru)) {
				c[it.value()] += def;
				it.next();
			}
			_indexes.cacheIterator(it, ru);
		}
	}

	@Override
	protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] preAgg) {
		ColGroupSDCSingle.computeRowMxx(c, builtin, rl, ru, _indexes, _numRows, 0, preAgg[0]);
	}

	@Override
	public int[] getCounts(int[] counts) {
		counts[0] = _indexes.getSize();
		counts[1] = _numRows - counts[0];
		return counts;
	}

	@Override
	public void preAggregate(MatrixBlock m, MatrixBlock preAgg, int rl, int ru) {
		if(m.isInSparseFormat())
			preAggregateSparse(m.getSparseBlock(), preAgg, rl, ru);
		else
			preAggregateDense(m, preAgg, rl, ru, 0, _numRows);
	}

	@Override
	public void preAggregateDense(MatrixBlock m, MatrixBlock preAgg, int rl, int ru, int cl, int cu) {
		final AIterator it = _indexes.getIterator(cl);
		final double[] preAV = preAgg.getDenseBlockValues();
		final double[] vals = m.getDenseBlockValues();
		final int nCol = m.getNumColumns();
		if(it == null)
			return;
		else if(it.value() > cu)
			_indexes.cacheIterator(it, cu);
		else if(cu < _indexes.getOffsetToLast() + 1) {
			while(it.value() < cu) {
				final int start = it.value() + nCol * rl;
				final int end = it.value() + nCol * ru;
				for(int offOut = 0, off = start; off < end; offOut ++, off += nCol)
					preAV[offOut] += vals[off];
				it.next();
			}
			_indexes.cacheIterator(it, cu);
		}
		else {
			int of = it.value();
			int start = of + nCol * rl;
			int end = of + nCol * ru;
			for(int offOut = 0, off = start; off < end; offOut ++, off += nCol)
				preAV[offOut] += vals[off];
			while(of < _indexes.getOffsetToLast()) {
				it.next();
				of = it.value();
				start = of + nCol * rl;
				end = of + nCol * ru;
				for(int offOut = 0, off = start; off < end; offOut ++, off += nCol)
					preAV[offOut] += vals[off];
			}
		}
	}

	private void preAggregateSparse(SparseBlock sb, MatrixBlock preAgg, int rl, int ru) {
		final AIterator it = _indexes.getIterator();
		if(rl == ru - 1) {
			final int apos = sb.pos(rl);
			final int alen = sb.size(rl) + apos;
			final int[] aix = sb.indexes(rl);
			final double[] avals = sb.values(rl);
			final int offsetToLast = _indexes.getOffsetToLast();

			double ret = 0;
			int j = apos;

			while(true) {
				final int idx = aix[j];

				if(idx == it.value()) {
					ret += avals[j++];
					if(j >= alen || it.value() >= offsetToLast)
						break;
					it.next();
				}
				else if(idx < it.value()) {
					j++;
					if(j >= alen)
						break;
				}
				else {
					if(it.value() >= offsetToLast)
						break;
					it.next();
				}
			}

			preAgg.setValue(0, 0, ret);
		}
		else
			throw new NotImplementedException();
	}

	@Override
	public long estimateInMemorySize() {
		long size = super.estimateInMemorySize();
		size += _indexes.getInMemorySize();
		return size;
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		double val0 = op.executeScalar(0);
		boolean isSparseSafeOp = op.sparseSafe || val0 == 0;
		if(isSparseSafeOp)
			return new ColGroupSDCSingleZeros(_colIndexes, _numRows, _dict.applyScalarOp(op), _indexes, getCachedCounts());
		else {
			ADictionary aDictionary = _dict.applyScalarOp(op, val0, getNumCols());// swapEntries();
			// ADictionary aDictionary = applyScalarOp(op, val0, getNumCols());
			return new ColGroupSDCSingle(_colIndexes, _numRows, aDictionary, _indexes, null);
		}
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		if(isRowSafe) {
			ADictionary ret = _dict.binOpLeft(op, v, _colIndexes);
			return new ColGroupSDCSingleZeros(_colIndexes, _numRows, ret, _indexes, getCachedCounts());
		}
		else {
			ADictionary ret = _dict.applyBinaryRowOpLeftAppendNewEntry(op, v, _colIndexes);
			return new ColGroupSDCSingle(_colIndexes, _numRows, ret, _indexes, getCachedCounts());
		}
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		if(isRowSafe) {
			ADictionary ret = _dict.binOpRight(op, v, _colIndexes);
			return new ColGroupSDCSingleZeros(_colIndexes, _numRows, ret, _indexes, getCachedCounts());
		}
		else {
			ADictionary ret = _dict.applyBinaryRowOpRightAppendNewEntry(op, v, _colIndexes);
			return new ColGroupSDCSingle(_colIndexes, _numRows, ret, _indexes, getCachedCounts());
		}
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		_indexes.write(out);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		super.readFields(in);
		_indexes = OffsetFactory.readIn(in);
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += _indexes.getExactSizeOnDisk();
		return ret;
	}

	@Override
	public boolean sameIndexStructure(AColGroupCompressed that) {
		if(that instanceof ColGroupSDCSingleZeros) {
			ColGroupSDCSingleZeros th = (ColGroupSDCSingleZeros) that;
			return th._indexes == _indexes;
		}
		else
			return false;
	}

	@Override
	public void preAggregateThatDDCStructure(ColGroupDDC that, Dictionary ret) {
		final AIterator itThis = _indexes.getIterator();
		final int nCol = that._colIndexes.length;
		final int finalOffThis = _indexes.getOffsetToLast();

		while(true) {
			final int fr = that._data.getIndex(itThis.value());
			that._dict.addToEntry(ret, fr, 0, nCol);
			if(itThis.value() >= finalOffThis)
				break;
			else
				itThis.next();
		}
	}

	@Override
	public void preAggregateThatSDCZerosStructure(ColGroupSDCZeros that, Dictionary ret) {
		final AIterator itThat = that._indexes.getIterator();
		final AIterator itThis = _indexes.getIterator();
		final int nCol = that._colIndexes.length;
		final int finalOffThis = _indexes.getOffsetToLast();
		final int finalOffThat = that._indexes.getOffsetToLast();

		while(true) {
			if(itThat.value() == itThis.value()) {
				that._dict.addToEntry(ret, that._data.getIndex(itThat.getDataIndex()), 0, nCol);
				if(itThat.value() >= finalOffThat)
					break;
				else
					itThat.next();
				if(itThis.value() >= finalOffThis)
					break;
				else
					itThis.next();
			}
			else if(itThat.value() < itThis.value()) {
				if(itThat.value() >= finalOffThat)
					break;
				else
					itThat.next();
			}
			else {
				if(itThis.value() >= finalOffThis)
					break;
				else
					itThis.next();
			}
		}
	}

	@Override
	public void preAggregateThatSDCSingleZerosStructure(ColGroupSDCSingleZeros that, Dictionary ret) {
		final int nCol = that._colIndexes.length;
		final AIterator itThat = that._indexes.getIterator();
		final AIterator itThis = _indexes.getIterator();
		final int finalOffThis = _indexes.getOffsetToLast();
		final int finalOffThat = that._indexes.getOffsetToLast();

		while(true) {
			if(itThat.value() == itThis.value()) {
				that._dict.addToEntry(ret, 0, 0, nCol);
				if(itThat.value() >= finalOffThat)
					break;
				else
					itThat.next();
				if(itThis.value() >= finalOffThis)
					break;
				else
					itThis.next();
			}
			else if(itThat.value() < itThis.value()) {
				if(itThat.value() >= finalOffThat)
					break;
				else
					itThat.next();
			}
			else {
				if(itThis.value() >= finalOffThis)
					break;
				else
					itThis.next();
			}

		}
	}

	@Override
	public int getPreAggregateSize(){
		return 1;
	}

	@Override
	public AColGroup replace(double pattern, double replace) {
		if(pattern == 0)
			return replaceZero(replace);
		ADictionary replaced = _dict.replace(pattern, replace, _colIndexes.length);
		return copyAndSet(replaced);
	}

	private AColGroup replaceZero(double replace) {
		ADictionary replaced = _dict.replaceZeroAndExtend(replace, _colIndexes.length);
		return new ColGroupSDCSingle(_colIndexes, _numRows, replaced, _indexes, getCachedCounts());
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s", "Indexes: "));
		sb.append(_indexes.toString());
		return sb.toString();
	}
}
