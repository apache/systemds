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

import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Plus;
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

	/** Sparse row indexes for the data */
	protected transient AOffset _indexes;

	/** Pointers to row indexes in the dictionary. Note the dictionary has one extra entry. */
	protected transient AMapToData _data;

	/**
	 * Constructor for serialization
	 * 
	 * @param numRows Number of rows contained
	 */
	protected ColGroupSDCZeros(int numRows) {
		super(numRows);
	}

	private ColGroupSDCZeros(int[] colIndices, int numRows, ADictionary dict, AOffset offsets, AMapToData data,
		int[] cachedCounts) {
		super(colIndices, numRows, dict, cachedCounts);
		_indexes = offsets;
		_data = data;
		_zeros = true;
	}

	protected static AColGroup create(int[] colIndices, int numRows, ADictionary dict, AOffset offsets, AMapToData data,
		int[] cachedCounts) {
		if(dict == null)
			return new ColGroupEmpty(colIndices);
		else
			return new ColGroupSDCZeros(colIndices, numRows, dict, offsets, data, cachedCounts);
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

		final AIterator it = _indexes.getIterator(rl);
		if(it == null)
			return;
		if(it.value() >= ru)
			_indexes.cacheIterator(it, ru);
		else if(db.isContiguous() && _colIndexes.length == 1) {
			if(ru > _indexes.getOffsetToLast())
				decompressToDenseBlockDenseDictionaryPostSingleColContiguous(db, rl, ru, offR, offC, values, it);
			else {
				if(db.getDim(1) == 1)
					decompressToDenseBlockDenseDictionaryPreSingleColOutContiguous(db, ru, offR, offC, values, it, _data);
				else
					decompressToDenseBlockDenseDictionaryPreSingleColContiguous(db, rl, ru, offR, offC, values, it);
				_indexes.cacheIterator(it, ru);
			}
		}
		else if(ru > _indexes.getOffsetToLast())
			decompressToDenseBlockDenseDictionaryPostGeneric(db, rl, ru, offR, offC, values, it);
		else {
			decompressToDenseBlockDenseDictionaryPreGeneric(db, rl, ru, offR, offC, values, it);
			_indexes.cacheIterator(it, ru);
		}
	}

	private void decompressToDenseBlockDenseDictionaryPostSingleColContiguous(DenseBlock db, int rl, int ru, int offR,
		int offC, double[] values, AIterator it) {
		final int lastOff = _indexes.getOffsetToLast() + offR;
		final int nCol = db.getDim(1);
		final double[] c = db.values(0);
		it.setOff(it.value() + offR);
		offC += _colIndexes[0];
		while(it.value() < lastOff) {
			final int off = it.value() * nCol + offC;
			c[off] += values[_data.getIndex(it.getDataIndex())];
			it.next();
		}
		final int off = it.value() * nCol + offC;
		c[off] += values[_data.getIndex(it.getDataIndex())];
		it.setOff(it.value() - offR);
	}

	private void decompressToDenseBlockDenseDictionaryPostGeneric(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values, AIterator it) {
		final int lastOff = _indexes.getOffsetToLast();
		final int nCol = _colIndexes.length;
		while(true) {
			final int idx = offR + it.value();
			final double[] c = db.values(idx);
			final int off = db.pos(idx) + offC;
			final int offDict = _data.getIndex(it.getDataIndex()) * nCol;
			for(int j = 0; j < nCol; j++)
				c[off + _colIndexes[j]] += values[offDict + j];
			if(it.value() == lastOff)
				return;
			it.next();
		}
	}

	private static void decompressToDenseBlockDenseDictionaryPreSingleColOutContiguous(DenseBlock db, int ru, int offR,
		int offC, double[] values, AIterator it, AMapToData m) {
		final double[] c = db.values(0);
		final int of = offR + offC;
		final int last = ru + of;
		it.setOff(it.value() + of);
		while(it.isNotOver(last)) {
			c[it.value()] += values[m.getIndex(it.getDataIndex())];
			it.next();
		}
		it.setOff(it.value() - of);
	}

	private void decompressToDenseBlockDenseDictionaryPreSingleColContiguous(DenseBlock db, int rl, int ru, int offR,
		int offC, double[] values, AIterator it) {
		final int last = ru + offR;
		final int nCol = db.getDim(1);
		final double[] c = db.values(0);
		it.setOff(it.value() + offR);
		offC += _colIndexes[0];
		while(it.isNotOver(last)) {
			final int off = it.value() * nCol + offC;
			c[off] += values[_data.getIndex(it.getDataIndex())];
			it.next();
		}
		it.setOff(it.value() - offR);
	}

	private void decompressToDenseBlockDenseDictionaryPreGeneric(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values, AIterator it) {
		final int nCol = _colIndexes.length;
		while(it.isNotOver(ru)) {
			final int idx = offR + it.value();
			final double[] c = db.values(idx);
			final int off = db.pos(idx) + offC;
			final int offDict = _data.getIndex(it.getDataIndex()) * nCol;
			for(int j = 0; j < nCol; j++)
				c[off + _colIndexes[j]] += values[offDict + j];

			it.next();
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
			final int lastOff = _indexes.getOffsetToLast();
			while(true) {
				final int idx = offR + it.value();
				final double[] c = db.values(idx);
				final int dx = it.getDataIndex();
				final int dictIndex = _data.getIndex(dx);
				if(sb.isEmpty(dictIndex)) {
					if(it.value() == lastOff)
						return;
					it.next();
					continue;
				}

				final int off = db.pos(idx) + offC;
				final int apos = sb.pos(dictIndex);
				final int alen = sb.size(dictIndex) + apos;
				final double[] avals = sb.values(dictIndex);
				final int[] aix = sb.indexes(dictIndex);
				for(int j = apos; j < alen; j++)
					c[off + _colIndexes[aix[j]]] += avals[j];
				if(it.value() == lastOff)
					return;
				it.next();
			}
		}
		else {
			while(it.isNotOver(ru)) {
				final int idx = offR + it.value();
				final int dx = it.getDataIndex();
				final int dictIndex = _data.getIndex(dx);
				if(sb.isEmpty(dictIndex)) {
					it.next();
					continue;
				}

				final double[] c = db.values(idx);
				final int off = db.pos(idx) + offC;
				final int apos = sb.pos(dictIndex);
				final int alen = sb.size(dictIndex) + apos;
				final double[] avals = sb.values(dictIndex);
				final int[] aix = sb.indexes(dictIndex);
				for(int j = apos; j < alen; j++)
					c[off + _colIndexes[aix[j]]] += avals[j];

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
			final int lastOff = _indexes.getOffsetToLast();
			while(true) {
				final int row = offR + it.value();
				final int dx = it.getDataIndex();
				final int dictIndex = _data.getIndex(dx);
				if(sb.isEmpty(dictIndex)) {
					if(it.value() == lastOff)
						return;
					it.next();
					continue;
				}

				final int apos = sb.pos(dictIndex);
				final int alen = sb.size(dictIndex) + apos;
				final double[] avals = sb.values(dictIndex);
				final int[] aix = sb.indexes(dictIndex);
				for(int j = apos; j < alen; j++)
					ret.append(row, _colIndexes[aix[j]] + offC, avals[j]);
				if(it.value() == lastOff)
					return;
				it.next();
			}
		}
		else {
			while(it.isNotOver(ru)) {
				final int row = offR + it.value();
				final int dx = it.getDataIndex();
				final int dictIndex = _data.getIndex(dx);
				if(sb.isEmpty(dictIndex)) {
					it.next();
					continue;
				}

				final int apos = sb.pos(dictIndex);
				final int alen = sb.size(dictIndex) + apos;
				final double[] avals = sb.values(dictIndex);
				final int[] aix = sb.indexes(dictIndex);
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
			final int lastOff = _indexes.getOffsetToLast();
			final int nCol = _colIndexes.length;
			while(true) {
				final int row = offR + it.value();
				final int dx = it.getDataIndex();
				final int offDict = _data.getIndex(dx) * nCol;
				for(int j = 0; j < nCol; j++)
					ret.append(row, _colIndexes[j] + offC, values[offDict + j]);
				if(it.value() == lastOff)
					return;
				it.next();
			}
		}
		else {

			final int nCol = _colIndexes.length;
			while(it.isNotOver(ru)) {
				final int row = offR + it.value();
				final int dx = it.getDataIndex();
				final int offDict = _data.getIndex(dx) * nCol;
				for(int j = 0; j < nCol; j++)
					ret.append(row, _colIndexes[j] + offC, values[offDict + j]);

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
		final int nCol = _colIndexes.length;
		return _dict.getValue(_data.getIndex(it.getDataIndex()) * nCol + colIdx);
	}

	@Override
	protected void computeRowSums(double[] c, int rl, int ru, double[] preAgg) {
		computeRowSums(c, rl, ru, preAgg, _data, _indexes, _numRows);
	}

	protected static final void computeRowSums(double[] c, int rl, int ru, double[] preAgg, AMapToData data,
		AOffset indexes, int nRows) {
		final AIterator it = indexes.getIterator(rl);
		if(it == null)
			return;
		else if(it.value() > ru)
			indexes.cacheIterator(it, ru);
		else if(ru >= indexes.getOffsetToLast()) {
			final int maxId = data.size() - 1;
			c[it.value()] += preAgg[data.getIndex(it.getDataIndex())];
			while(it.getDataIndex() < maxId) {
				it.next();
				c[it.value()] += preAgg[data.getIndex(it.getDataIndex())];
			}
		}
		else {
			while(it.isNotOver(ru)) {
				c[it.value()] += preAgg[data.getIndex(it.getDataIndex())];
				it.next();
			}
			indexes.cacheIterator(it, ru);
		}
	}

	@Override
	protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] preAgg) {
		ColGroupSDC.computeRowMxx(c, builtin, rl, ru, preAgg, _data, _indexes, _numRows, 0);
	}

	@Override
	public int[] getCounts(int[] counts) {
		return _data.getCounts(counts, _numRows);
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
		_data.preAggregateDense(m, preAgg.getDenseBlockValues(), rl, ru, cl, cu, _indexes);
	}

	private void preAggregateSparse(SparseBlock sb, MatrixBlock preAgg, int rl, int ru) {
		_data.preAggregateSparse(sb, preAgg.getDenseBlockValues(), rl, ru, _indexes);
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
			return create(_colIndexes, _numRows, _dict.applyScalarOp(op), _indexes, _data, getCachedCounts());
		else {
			ADictionary rValues = _dict.applyScalarOp(op, val0, getNumCols());
			return ColGroupSDC.create(_colIndexes, _numRows, rValues, _indexes, _data, getCachedCounts());
		}
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		if(isRowSafe) {
			ADictionary ret = _dict.binOpLeft(op, v, _colIndexes);
			return create(_colIndexes, _numRows, ret, _indexes, _data, getCachedCounts());
		}
		else if(op.fn instanceof Plus) {
			double[] def = ColGroupUtils.binaryDefRowLeft(op, v, _colIndexes);
			return ColGroupPFOR.create(_colIndexes, _numRows, _dict, _indexes, _data, getCachedCounts(), def);
		}
		else {
			ADictionary ret = _dict.applyBinaryRowOpLeftAppendNewEntry(op, v, _colIndexes);
			return ColGroupSDC.create(_colIndexes, _numRows, ret, _indexes, _data, getCachedCounts());
		}
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		if(isRowSafe) {
			ADictionary ret = _dict.binOpRight(op, v, _colIndexes);
			return create(_colIndexes, _numRows, ret, _indexes, _data, getCachedCounts());
		}
		else if(op.fn instanceof Plus) {
			double[] def = ColGroupUtils.binaryDefRowRight(op, v, _colIndexes);
			return ColGroupPFOR.create(_colIndexes, _numRows, _dict, _indexes, _data, getCachedCounts(), def);
		}
		else {
			ADictionary ret = _dict.applyBinaryRowOpRightAppendNewEntry(op, v, _colIndexes);
			return ColGroupSDC.create(_colIndexes, _numRows, ret, _indexes, _data, getCachedCounts());
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

		final int finalOffThis = _indexes.getOffsetToLast();
		while(true) {
			final int fr = that._data.getIndex(itThis.value());
			final int to = _data.getIndex(itThis.getDataIndex());
			that._dict.addToEntry(ret, fr, to, nCol);
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

		final int finalOffThis = _indexes.getOffsetToLast();
		final int finalOffThat = that._indexes.getOffsetToLast();

		final int nCol = that._colIndexes.length;
		while(true) {
			if(itThat.value() == itThis.value()) {
				final int fr = that._data.getIndex(itThat.getDataIndex());
				final int to = _data.getIndex(itThis.getDataIndex());
				that._dict.addToEntry(ret, fr, to, nCol);
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
		final AIterator itThat = that._indexes.getIterator();
		final AIterator itThis = _indexes.getIterator();
		final int nCol = that._colIndexes.length;

		final int finalOffThis = _indexes.getOffsetToLast();
		final int finalOffThat = that._indexes.getOffsetToLast();

		while(true) {
			if(itThat.value() == itThis.value()) {
				final int to = _data.getIndex(itThis.getDataIndex());
				that._dict.addToEntry(ret, 0, to, nCol);
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
	public AColGroup replace(double pattern, double replace) {
		if(pattern == 0)
			return replaceZero(replace);
		ADictionary replaced = _dict.replace(pattern, replace, _colIndexes.length);
		return copyAndSet(replaced);
	}

	private AColGroup replaceZero(double replace) {
		ADictionary replaced = _dict.replaceZeroAndExtend(replace, _colIndexes.length);
		return ColGroupSDC.create(_colIndexes, _numRows, replaced, _indexes, _data, getCachedCounts());
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s", "Indexes: "));
		sb.append(_indexes.toString());
		sb.append(String.format("\n%15s", "Data: "));
		sb.append(_data);
		return sb.toString();
	}
}
