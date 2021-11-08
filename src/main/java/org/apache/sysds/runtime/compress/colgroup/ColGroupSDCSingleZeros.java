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
	/**
	 * Sparse row indexes for the data
	 */
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
		final int nCol = _colIndexes.length;
		final AIterator it = _indexes.getIterator(rl);
		while(it.hasNext() && it.value() < ru) {
			final int row = offR + it.value();
			final double[] c = db.values(row);
			final int off = db.pos(row) + offC;
			for(int j = 0; j < nCol; j++)
				c[off + _colIndexes[j]] += values[j];

			it.next();
		}
		_indexes.cacheIterator(it, ru);
	}

	@Override
	protected void decompressToDenseBlockSparseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		SparseBlock values) {
		throw new NotImplementedException();
		// final int offTCorr = offT - rl;
		// final DenseBlock db = target.getDenseBlock();
		// final int apos = values.pos(0);
		// final int alen = values.size(0) + apos;
		// final int[] aix = values.indexes(0);
		// final double[] avals = values.values(0);

		// AIterator it = _indexes.getIterator(rl);
		// while(it.hasNext() && it.value() < ru) {
		// final int idx = offTCorr + it.value();
		// final double[] c = db.values(idx);
		// final int off = db.pos(idx);

		// for(int j = apos; j < alen; j++)
		// c[off + _colIndexes[aix[j]]] += avals[j];

		// it.next();
		// }

		// _indexes.cacheIterator(it, ru);
	}

	@Override
	protected void decompressToSparseBlockSparseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		SparseBlock sb) {
		throw new NotImplementedException();
	}

	@Override
	protected void decompressToSparseBlockDenseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		double[] values) {
		final int nCol = _colIndexes.length;
		final AIterator it = _indexes.getIterator(rl);
		while(it.hasNext() && it.value() < ru) {
			final int row = offR + it.value();
			for(int j = 0; j < nCol; j++)
				ret.append(row, _colIndexes[j] + offC, values[j]);
			it.next();
		}
		_indexes.cacheIterator(it, ru);
	}

	@Override
	public double getIdx(int r, int colIdx) {
		final AIterator it = _indexes.getIterator(r);
		if(it.value() == r)
			return _dict.getValue(colIdx);
		else
			return 0.0;
	}

	@Override
	protected void computeRowSums(double[] c, boolean square, int rl, int ru) {
		final double vals = _dict.sumAllRowsToDouble(square, _colIndexes.length)[0];
		final AIterator it = _indexes.getIterator(rl);
		while(it.hasNext() && it.value() < ru) {
			c[it.value()] += vals;
			it.next();
		}

	}

	@Override
	protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru) {
		final double vals = _dict.aggregateTuples(builtin, _colIndexes.length)[0];
		final AIterator it = _indexes.getIterator(rl);
		int rix = rl;

		for(; rix < ru && it.hasNext(); rix++) {
			if(it.value() != rix)
				c[rix] = builtin.execute(c[rix], 0);
			else {
				c[rix] = builtin.execute(c[rix], vals);
				it.next();
			}
		}

		// cover remaining rows
		for(; rix < ru; rix++)
			c[rix] = builtin.execute(c[rix], 0);
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
			preAggregateDense(m, preAgg, rl, ru);
	}

	@Override
	public void preAggregateDense(MatrixBlock m, MatrixBlock preAgg, int rl, int ru, int cl, int cu) {
		final double[] mV = m.getDenseBlockValues();
		final double[] preAV = preAgg.getDenseBlockValues();
		final int numVals = getNumValues();
		final int blockSize = 2000;
		for(int block = cl; block < cu; block += blockSize) {
			final int blockEnd = Math.min(block + blockSize, cu);
			final AIterator itStart = _indexes.getIterator(block);
			AIterator it;
			for(int rowLeft = rl, offOut = 0; rowLeft < ru; rowLeft++, offOut += numVals) {
				final int offLeft = rowLeft * _numRows;
				it = itStart.clone();
				while(it.value() < blockEnd && it.hasNext()) {
					final int i = it.value();
					preAV[offOut] += mV[offLeft + i];
					it.next();
				}
			}
		}
	}

	private void preAggregateDense(MatrixBlock m, MatrixBlock preAgg, int rl, int ru) {
		final double[] preAV = preAgg.getDenseBlockValues();
		final double[] mV = m.getDenseBlockValues();
		final int numVals = getNumValues();
		for(int rowLeft = rl, offOut = 0; rowLeft < ru; rowLeft++, offOut += numVals) {
			final AIterator it = _indexes.getIterator();
			final int offLeft = rowLeft * _numRows;
			while(it.hasNext()) {
				final int i = it.value();
				preAV[offOut] += mV[offLeft + i];
				it.next();
			}
		}
	}

	private void preAggregateSparse(SparseBlock sb, MatrixBlock preAgg, int rl, int ru) {
		final double[] preAV = preAgg.getDenseBlockValues();
		final int numVals = getNumValues();
		for(int rowLeft = rl, offOut = 0; rowLeft < ru; rowLeft++, offOut += numVals) {
			if(sb.isEmpty(rowLeft))
				continue;
			final AIterator it = _indexes.getIterator();
			final int apos = sb.pos(rowLeft);
			final int alen = sb.size(rowLeft) + apos;
			final int[] aix = sb.indexes(rowLeft);
			final double[] avals = sb.values(rowLeft);
			int j = apos;
			while(it.hasNext() && j < alen) {
				final int index = aix[j];
				final int v = it.value();
				if(index < v)
					j++;
				else if(index == v) {
					preAV[offOut] += avals[j++];
					it.next();
				}
				else
					it.next();
			}
		}
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
			return new ColGroupSDCSingleZeros(_colIndexes, _numRows, applyScalarOp(op), _indexes, getCachedCounts());
		else {
			ADictionary aDictionary = applyScalarOp(op, val0, getNumCols());// swapEntries();
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
		while(itThis.hasNext()) {
			final int fr = that._data.getIndex(itThis.value());
			that._dict.addToEntry(ret, fr, 0, nCol);
			itThis.next();
		}
	}

	@Override
	public void preAggregateThatSDCZerosStructure(ColGroupSDCZeros that, Dictionary ret) {
		final AIterator itThat = that._indexes.getIterator();
		final AIterator itThis = _indexes.getIterator();
		final int nCol = that._colIndexes.length;

		while(itThat.hasNext() && itThis.hasNext()) {
			final int v = itThat.value();
			if(v == itThis.skipTo(v))
				that._dict.addToEntry(ret, that.getIndex(itThat.getDataIndex()), 0, nCol);

			itThat.next();
		}
	}

	@Override
	public void preAggregateThatSDCSingleZerosStructure(ColGroupSDCSingleZeros that, Dictionary ret) {
		final AIterator itThat = that._indexes.getIterator();
		final AIterator itThis = _indexes.getIterator();
		final int nCol = that._colIndexes.length;
		while(itThat.hasNext()) {
			final int v = itThat.value();
			if(v == itThis.skipTo(v))
				that._dict.addToEntry(ret, 0, 0, nCol);
			itThat.next();
		}
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
		sb.append(String.format("\n%15s ", "Indexes: "));
		sb.append(_indexes.toString());
		return sb.toString();
	}
}
