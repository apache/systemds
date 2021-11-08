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
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToByte;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToChar;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
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
 * the most common one. the most common one can be inferred by not being included in the indexes.
 * 
 * If the values are very sparse then the most common one is zero. This is the case for this column group, that
 * specifically exploits that the column contain lots of zero values.
 * 
 * This column group is handy in cases where sparse unsafe operations is executed on very sparse columns.
 */
public class ColGroupSDCZeros extends APreAgg {
	private static final long serialVersionUID = -3703199743391937991L;

	/**
	 * Sparse row indexes for the data
	 */
	protected transient AOffset _indexes;

	/**
	 * Pointers to row indexes in the dictionary. Note the dictionary has one extra entry.
	 */
	protected transient AMapToData _data;

	/**
	 * Constructor for serialization
	 * 
	 * @param numRows Number of rows contained
	 */
	protected ColGroupSDCZeros(int numRows) {
		super(numRows);
	}

	protected ColGroupSDCZeros(int[] colIndices, int numRows, ADictionary dict, AOffset offsets, AMapToData data) {
		super(colIndices, numRows, dict, null);
		_indexes = offsets;
		_data = data;
		_zeros = true;
	}

	protected ColGroupSDCZeros(int[] colIndices, int numRows, ADictionary dict, AOffset offsets, AMapToData data,
		int[] cachedCounts) {
		super(colIndices, numRows, dict, cachedCounts);
		_indexes = offsets;
		_data = data;
		_zeros = true;
	}

	@Override
	public CompressionType getCompType() {
		return CompressionType.SDC;
	}

	@Override
	public ColGroupType getColGroupType() {
		return ColGroupType.SDCZeros;
	}

	@Override
	protected void decompressToDenseBlockDenseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values) {
		final int nCol = _colIndexes.length;

		AIterator it = _indexes.getIterator(rl);
		while(it.hasNext() && it.value() < ru) {
			final int idx = offR + it.value();
			final double[] c = db.values(idx);
			final int off = db.pos(idx) + offC;
			final int offDict = getIndex(it.getDataIndexAndIncrement()) * nCol;
			for(int j = 0; j < nCol; j++)
				c[off + _colIndexes[j]] += values[offDict + j];

		}
		_indexes.cacheIterator(it, ru);
	}

	@Override
	protected void decompressToDenseBlockSparseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		SparseBlock sb) {
		AIterator it = _indexes.getIterator(rl);
		while(it.hasNext() && it.value() < ru) {
			final int idx = offR + it.value();
			final int dictIndex = getIndex(it.getDataIndexAndIncrement());
			if(sb.isEmpty(dictIndex))
				continue;

			final double[] c = db.values(idx);
			final int off = db.pos(idx) + offC;
			final int apos = sb.pos(dictIndex);
			final int alen = sb.size(dictIndex) + apos;
			final double[] avals = sb.values(dictIndex);
			final int[] aix = sb.indexes(dictIndex);
			for(int j = apos; j < alen; j++)
				c[off + _colIndexes[aix[j]]] += avals[j];
		}
		_indexes.cacheIterator(it, ru);
	}

	@Override
	protected void decompressToSparseBlockSparseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		SparseBlock sb) {
		AIterator it = _indexes.getIterator(rl);
		while(it.hasNext() && it.value() < ru) {
			final int row = offR + it.value();
			final int dictIndex = getIndex(it.getDataIndexAndIncrement());
			if(sb.isEmpty(dictIndex))
				continue;

			final int apos = sb.pos(dictIndex);
			final int alen = sb.size(dictIndex) + apos;
			final double[] avals = sb.values(dictIndex);
			final int[] aix = sb.indexes(dictIndex);
			for(int j = apos; j < alen; j++)
				ret.append(row, _colIndexes[aix[j]] + offC, avals[j] );
		}
		_indexes.cacheIterator(it, ru);
	}

	@Override
	protected void decompressToSparseBlockDenseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		double[] values) {
		final int nCol = _colIndexes.length;

		AIterator it = _indexes.getIterator(rl);
		while(it.hasNext() && it.value() < ru) {
			final int row = offR + it.value();
			final int offDict = getIndex(it.getDataIndexAndIncrement()) * nCol;
			for(int j = 0; j < nCol; j++)
				ret.append(row, _colIndexes[j] + offC, values[offDict + j]);
		}
		_indexes.cacheIterator(it, ru);
	}

	@Override
	public double getIdx(int r, int colIdx) {
		final AIterator it = _indexes.getIterator(r);
		final int nCol = _colIndexes.length;
		if(it.value() == r)
			return _dict.getValue(getIndex(it.getDataIndex()) * nCol + colIdx);
		else
			return 0.0;
	}

	@Override
	protected void computeRowSums(double[] c, boolean square, int rl, int ru) {
		final double[] vals = _dict.sumAllRowsToDouble(square, _colIndexes.length);
		final AIterator it = _indexes.getIterator(rl);
		while(it.hasNext() && it.value() < ru)
			c[it.value()] += vals[getIndex(it.getDataIndexAndIncrement())];
	}

	@Override
	protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru) {
		final double[] vals = _dict.aggregateTuples(builtin, _colIndexes.length);
		final AIterator it = _indexes.getIterator(rl);
		int rix = rl;

		for(; rix < ru && it.hasNext(); rix++) {
			if(it.value() != rix)
				c[rix] = builtin.execute(c[rix], 0);
			else
				c[rix] = builtin.execute(c[rix], vals[_data.getIndex(it.getDataIndexAndIncrement())]);
		}

		// cover remaining rows with default value
		for(; rix < ru; rix++)
			c[rix] = builtin.execute(c[rix], 0);
	}

	@Override
	public int[] getCounts(int[] counts) {
		final int nonDefaultLength = _data.size();
		// final AIterator it = _indexes.getIterator();
		final int zeros = _numRows - nonDefaultLength;
		for(int i = 0; i < nonDefaultLength; i++)
			counts[_data.getIndex(i)]++;

		counts[counts.length - 1] += zeros;

		return counts;
	}

	public int getIndex(int r) {
		return _data.getIndex(r);
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

		final int numVals = getNumValues();
		if(cl != 0 && cu != preAgg.getNumColumns())
			throw new NotImplementedException("Not implemented preAggregate of sub number of columns");
		if(_data instanceof MapToByte)
			preAggregateDenseByte(m, preAgg, ((MapToByte) _data).getBytes(), rl, ru, cl, cu, _numRows, numVals, _indexes);
		else if(_data instanceof MapToChar)
			preAggregateDenseChar(m, preAgg, ((MapToChar) _data).getChars(), rl, ru, cl, cu, _numRows, numVals, _indexes);
		else
			throw new DMLCompressionException("Unsupported map type:" + _data);

	}

	private static void preAggregateDenseByte(final MatrixBlock m, final MatrixBlock preAgg, final byte[] d,
		final int rl, final int ru, final int cl, final int cu, final int nRow, final int nVal, AOffset indexes) {
		final double[] preAV = preAgg.getDenseBlockValues();
		final double[] mV = m.getDenseBlockValues();
		// multi row iterator.
		final AIterator itStart = indexes.getIterator(cl);
		AIterator it = null;
		for(int rowLeft = rl, offOut = 0; rowLeft < ru; rowLeft++, offOut += nVal) {
			final int offLeft = rowLeft * nRow;
			it = itStart.clone();
			while(it.value() < cu && it.hasNext()) {
				int i = it.value();
				int index = d[it.getDataIndexAndIncrement()] & 0xFF;
				preAV[offOut + index] += mV[offLeft + i];
			}
		}
		if(it != null && cu < m.getNumColumns())
			indexes.cacheIterator(it, cu);
	}

	private static void preAggregateDenseChar(final MatrixBlock m, final MatrixBlock preAgg, final char[] d,
		final int rl, final int ru, final int cl, final int cu, final int nRow, final int nVal, AOffset indexes) {
		final double[] preAV = preAgg.getDenseBlockValues();
		final double[] mV = m.getDenseBlockValues();
		// multi row iterator.
		final AIterator itStart = indexes.getIterator(cl);
		AIterator it = null;
		for(int rowLeft = rl, offOut = 0; rowLeft < ru; rowLeft++, offOut += nVal) {
			final int offLeft = rowLeft * nRow;
			it = itStart.clone();
			while(it.value() < cu && it.hasNext()) {
				int i = it.value();
				int index = d[it.getDataIndexAndIncrement()];
				preAV[offOut + index] += mV[offLeft + i];
			}
		}
		if(it != null && cu < m.getNumColumns())
			indexes.cacheIterator(it, cu);
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
				final int val = it.value();
				if(index < val)
					j++;
				else if(index == val)
					preAV[offOut + _data.getIndex(it.getDataIndexAndIncrement())] += avals[j++];
				else
					it.next();
			}
		}
	}

	@Override
	public long estimateInMemorySize() {
		long size = super.estimateInMemorySize();
		size += _indexes.getInMemorySize();
		size += _data.getInMemorySize();
		return size;
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		double val0 = op.executeScalar(0);
		boolean isSparseSafeOp = op.sparseSafe || val0 == 0;
		if(isSparseSafeOp)
			return new ColGroupSDCZeros(_colIndexes, _numRows, applyScalarOp(op), _indexes, _data, getCachedCounts());
		else {
			ADictionary rValues = applyScalarOp(op, val0, getNumCols());
			return new ColGroupSDC(_colIndexes, _numRows, rValues, _indexes, _data, getCachedCounts());
		}
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		if(isRowSafe) {
			ADictionary ret = _dict.binOpLeft(op, v, _colIndexes);
			return new ColGroupSDCZeros(_colIndexes, _numRows, ret, _indexes, _data, getCachedCounts());
		}
		else {
			ADictionary ret = _dict.applyBinaryRowOpLeftAppendNewEntry(op, v, _colIndexes);
			return new ColGroupSDC(_colIndexes, _numRows, ret, _indexes, _data, getCachedCounts());
		}
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		if(isRowSafe) {
			ADictionary ret = _dict.binOpRight(op, v, _colIndexes);
			return new ColGroupSDCZeros(_colIndexes, _numRows, ret, _indexes, _data, getCachedCounts());
		}
		else {
			ADictionary ret = _dict.applyBinaryRowOpRightAppendNewEntry(op, v, _colIndexes);
			return new ColGroupSDC(_colIndexes, _numRows, ret, _indexes, _data, getCachedCounts());
		}
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		_indexes.write(out);
		_data.write(out);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		super.readFields(in);
		_indexes = OffsetFactory.readIn(in);
		_data = MapToFactory.readIn(in);
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += _indexes.getExactSizeOnDisk();
		ret += _data.getInMemorySize();
		return ret;
	}

	@Override
	public boolean sameIndexStructure(AColGroupCompressed that) {
		if(that instanceof ColGroupSDCZeros) {
			ColGroupSDCZeros th = (ColGroupSDCZeros) that;
			return th._indexes == _indexes && th._data == _data;
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
			final int to = getIndex(itThis.getDataIndexAndIncrement());
			that._dict.addToEntry(ret, fr, to, nCol);
		}
	}

	@Override
	public void preAggregateThatSDCZerosStructure(ColGroupSDCZeros that, Dictionary ret) {
		final AIterator itThat = that._indexes.getIterator();
		final AIterator itThis = _indexes.getIterator();
		final int nCol = that._colIndexes.length;
		while(itThat.hasNext() && itThis.hasNext()) {
			if(itThat.value() == itThis.value()) {
				final int fr = that.getIndex(itThat.getDataIndexAndIncrement());
				final int to = getIndex(itThis.getDataIndexAndIncrement());
				that._dict.addToEntry(ret, fr, to, nCol);
			}
			else if(itThat.value() < itThis.value())
				itThat.next();
			else
				itThis.next();
		}
	}

	@Override
	public void preAggregateThatSDCSingleZerosStructure(ColGroupSDCSingleZeros that, Dictionary ret) {
		final AIterator itThat = that._indexes.getIterator();
		final AIterator itThis = _indexes.getIterator();
		final int nCol = that._colIndexes.length;

		while(itThat.hasNext() && itThis.hasNext()) {
			if(itThat.value() == itThis.value()) {
				final int to = getIndex(itThis.getDataIndexAndIncrement());
				that._dict.addToEntry(ret, 0, to, nCol);
				itThat.next();
			}
			else if(itThat.value() < itThis.value())
				itThat.next();
			else
				itThis.next();
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
		return new ColGroupSDC(_colIndexes, _numRows, replaced, _indexes, _data, getCachedCounts());
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s ", "Indexes: "));
		sb.append(_indexes.toString());
		sb.append(String.format("\n%15s ", "Data: "));
		sb.append(_data);
		return sb.toString();
	}
}
