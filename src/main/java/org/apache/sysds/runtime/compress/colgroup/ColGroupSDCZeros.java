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
import java.util.Arrays;

import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset.OffsetSliceInfo;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

/**
 * Column group that sparsely encodes the dictionary values. The idea is that all values is encoded with indexes except
 * the most common one. the most common one can be inferred by not being included in the indexes.
 * 
 * If the values are very sparse then the most common one is zero. This is the case for this column group, that
 * specifically exploits that the column contain lots of zero values.
 * 
 * This column group is handy in cases where sparse unsafe operations is executed on very sparse columns.
 */
public class ColGroupSDCZeros extends ASDCZero {
	private static final long serialVersionUID = -3703199743391937991L;

	/** Pointers to row indexes in the dictionary. Note the dictionary has one extra entry. */
	protected final AMapToData _data;

	private ColGroupSDCZeros(int[] colIndices, int numRows, ADictionary dict, AOffset indexes, AMapToData data,
		int[] cachedCounts) {
		super(colIndices, numRows, dict, indexes, cachedCounts);
		if(data.getUnique() != dict.getNumberOfValues(colIndices.length))
			throw new DMLCompressionException("Invalid construction of SDCZero group: number uniques: " + data.getUnique()
				+ " vs." + dict.getNumberOfValues(colIndices.length));
		_data = data;
	}

	public static AColGroup create(int[] colIndices, int numRows, ADictionary dict, AOffset offsets, AMapToData data,
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
		else if(it.value() >= ru)
			_indexes.cacheIterator(it, ru);
		else {
			decompressToDenseBlockDenseDictionaryWithProvidedIterator(db, rl, ru, offR, offC, values, it);
			_indexes.cacheIterator(it, ru);
		}
	}

	@Override
	public final void decompressToDenseBlockDenseDictionaryWithProvidedIterator(DenseBlock db, int rl, int ru, int offR,
		int offC, double[] values, AIterator it) {
		final int last = _indexes.getOffsetToLast();
		if(it == null || it.value() >= ru || rl > last)
			return;
		final boolean post = ru > last;
		final boolean contiguous = db.isContiguous();
		if(post) {
			if(contiguous && _colIndexes.length == 1)
				decompressToDenseBlockDenseDictionaryPostSingleColContiguous(db, rl, ru, offR, offC, values, it);
			else
				decompressToDenseBlockDenseDictionaryPostGeneric(db, rl, ru, offR, offC, values, it);
		}
		else if(contiguous && _colIndexes.length == 1) {
			if(db.getDim(1) == 1)
				decompressToDenseBlockDenseDictionaryPreSingleColOutContiguous(db, ru, offR, offC, values, it, _data);
			else
				decompressToDenseBlockDenseDictionaryPreSingleColContiguous(db, rl, ru, offR, offC, values, it);
		}
		else {
			if(_colIndexes.length == db.getDim(1))
				decompressToDenseBlockDenseDictionaryPreAllCols(db, rl, ru, offR, offC, values, it);
			else
				decompressToDenseBlockDenseDictionaryPreGeneric(db, rl, ru, offR, offC, values, it);
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

	private void decompressToDenseBlockDenseDictionaryPreAllCols(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values, AIterator it) {
		final int nCol = _colIndexes.length;
		while(it.isNotOver(ru)) {
			final int idx = offR + it.value();
			final double[] c = db.values(idx);
			final int off = db.pos(idx) + offC;
			final int offDict = _data.getIndex(it.getDataIndex()) * nCol;
			for(int j = 0; j < nCol; j++)
				c[off + j] += values[offDict + j];

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

		final int last = _indexes.getOffsetToLast();
		if(ru > last)
			decompressToDenseBlockSparseDictionaryPost(db, rl, ru, offR, offC, sb, it, last);
		else
			decompressToDenseBlockSparseDictionaryPre(db, rl, ru, offR, offC, sb, it);

	}

	private final void decompressToDenseBlockSparseDictionaryPost(DenseBlock db, int rl, int ru, int offR, int offC,
		SparseBlock sb, AIterator it, int last) {
		while(true) {
			final int idx = offR + it.value();
			final double[] c = db.values(idx);
			final int dx = it.getDataIndex();
			final int dictIndex = _data.getIndex(dx);
			if(sb.isEmpty(dictIndex)) {
				if(it.value() == last)
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
			if(it.value() == last)
				return;
			it.next();
		}
	}

	private final void decompressToDenseBlockSparseDictionaryPre(DenseBlock db, int rl, int ru, int offR, int offC,
		SparseBlock sb, AIterator it) {
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
	protected double[] preAggProductRows() {
		return _dict.productAllRowsToDoubleWithDefault(new double[_colIndexes.length]);
	}

	@Override
	protected void computeRowSums(double[] c, int rl, int ru, double[] preAgg) {
		computeRowSums(c, rl, ru, preAgg, _data, _indexes, _numRows);
	}

	@Override
	protected void computeRowProduct(double[] c, int rl, int ru, double[] preAgg) {
		ColGroupSDC.computeRowProduct(c, rl, ru, preAgg, _data, _indexes, _numRows);
	}

	protected static final void computeRowSums(double[] c, int rl, int ru, double[] preAgg, AMapToData data,
		AOffset indexes, int nRows) {
		final AIterator it = indexes.getIterator(rl);
		if(it == null)
			return;
		else if(it.value() > ru)
			indexes.cacheIterator(it, ru);
		else if(ru > indexes.getOffsetToLast()) {
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
		return _data.getCounts(counts);
	}

	@Override
	protected void multiplyScalar(double v, double[] resV, int offRet, AIterator it) {
		final int dx = _data.getIndex(it.getDataIndex());
		_dict.multiplyScalar(v, resV, offRet, dx, _colIndexes);
	}

	@Override
	public void preAggregateDense(MatrixBlock m, double[] preAgg, int rl, int ru, int cl, int cu) {
		_data.preAggregateDense(m, preAgg, rl, ru, cl, cu, _indexes);
	}

	@Override
	public void preAggregateSparse(SparseBlock sb, double[] preAgg, int rl, int ru) {
		_data.preAggregateSparse(sb, preAgg, rl, ru, _indexes);
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
		else if(op.fn instanceof Plus || (op.fn instanceof Minus && op instanceof RightScalarOperator)) {
			final double[] reference = ColGroupUtils.createReference(_colIndexes.length, val0);
			return ColGroupSDCFOR.create(_colIndexes, _numRows, _dict, _indexes, _data, getCachedCounts(), reference);
		}
		else {
			final ADictionary newDict = _dict.applyScalarOp(op);
			final double[] defaultTuple = ColGroupUtils.createReference(_colIndexes.length, val0);
			return ColGroupSDC.create(_colIndexes, _numRows, newDict, defaultTuple, _indexes, _data, getCachedCounts());
		}
	}

	@Override
	public AColGroup unaryOperation(UnaryOperator op) {
		final double val0 = op.fn.execute(0);
		final ADictionary nDict = _dict.applyUnaryOp(op);
		if(val0 == 0)
			return create(_colIndexes, _numRows, nDict, _indexes, _data, getCachedCounts());
		else {
			final double[] defaultTuple = new double[_colIndexes.length];
			Arrays.fill(defaultTuple, val0);
			return ColGroupSDC.create(_colIndexes, _numRows, nDict, defaultTuple, _indexes, _data, getCachedCounts());
		}
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		if(isRowSafe) {
			ADictionary newDict = _dict.binOpLeft(op, v, _colIndexes);
			return create(_colIndexes, _numRows, newDict, _indexes, _data, getCachedCounts());
		}
		else if(op.fn instanceof Plus) {
			double[] reference = ColGroupUtils.binaryDefRowLeft(op, v, _colIndexes);
			return ColGroupSDCFOR.create(_colIndexes, _numRows, _dict, _indexes, _data, getCachedCounts(), reference);
		}
		else {
			ADictionary newDict = _dict.binOpLeft(op, v, _colIndexes);
			double[] defaultTuple = new double[_colIndexes.length];
			for(int i = 0; i < _colIndexes.length; i++)
				defaultTuple[i] = op.fn.execute(v[_colIndexes[i]], 0);
			return ColGroupSDC.create(_colIndexes, _numRows, newDict, defaultTuple, _indexes, _data, getCachedCounts());
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
			return ColGroupSDCFOR.create(_colIndexes, _numRows, _dict, _indexes, _data, getCachedCounts(), def);
		}
		else {
			ADictionary newDict = _dict.binOpRight(op, v, _colIndexes);
			double[] defaultTuple = new double[_colIndexes.length];
			for(int i = 0; i < _colIndexes.length; i++)
				defaultTuple[i] = op.fn.execute(0, v[_colIndexes[i]]);
			return ColGroupSDC.create(_colIndexes, _numRows, newDict, defaultTuple, _indexes, _data, getCachedCounts());
		}
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		_indexes.write(out);
		_data.write(out);
	}

	public static ColGroupSDCZeros read(DataInput in, int nRows) throws IOException {
		int[] cols = readCols(in);
		ADictionary dict = DictionaryFactory.read(in);
		AOffset indexes = OffsetFactory.readIn(in);
		AMapToData data = MapToFactory.readIn(in);
		return new ColGroupSDCZeros(cols, nRows, dict, indexes, data, null);
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
		_data.preAggregateSDCZ_DDC(that._data, that._dict, _indexes, ret, that._colIndexes.length);
	}

	@Override
	public void preAggregateThatSDCZerosStructure(ColGroupSDCZeros that, Dictionary ret) {
		_data.preAggregateSDCZ_SDCZ(that._data, that._dict, that._indexes, _indexes, ret, that._colIndexes.length);
	}

	@Override
	public void preAggregateThatSDCSingleZerosStructure(ColGroupSDCSingleZeros that, Dictionary ret) {
		final AIterator itThat = that._indexes.getIterator();
		// _indexes.getOffsetIterator();
		final AIterator itThis = _indexes.getIterator();
		final int nCol = that._colIndexes.length;

		final int finalOffThis = _indexes.getOffsetToLast();
		final int finalOffThat = that._indexes.getOffsetToLast();
		final double[] v = ret.getValues();

		while(true) {
			if(itThat.value() == itThis.value()) {
				final int to = _data.getIndex(itThis.getDataIndex());
				that._dict.addToEntry(v, 0, to, nCol);
				if(itThat.value() >= finalOffThat)
					break;
				itThat.next();
				if(itThis.value() >= finalOffThis)
					break;
				itThis.next();
			}
			else if(itThat.value() < itThis.value()) {
				if(itThat.value() >= finalOffThat)
					break;
				itThat.next();
			}
			else {
				if(itThis.value() >= finalOffThis)
					break;
				itThis.next();
			}
		}
	}

	@Override
	protected void preAggregateThatRLEStructure(ColGroupRLE that, Dictionary ret) {
		final int finalOff = _indexes.getOffsetToLast();
		final double[] v = ret.getValues();
		final int nv = that.getNumValues();
		final int nCol = that._colIndexes.length;
		for(int k = 0; k < nv; k++) {
			final AIterator itThis = _indexes.getIterator();
			final int blen = that._ptr[k + 1];
			for(int apos = that._ptr[k], rs = 0, re = 0; apos < blen; apos += 2) {
				rs = re + that._data[apos];
				re = rs + that._data[apos + 1];
				// if index is later than run continue
				if(itThis.value() >= re || rs == re || rs > finalOff)
					continue;
				// while lower than run iterate through
				while(itThis.value() < rs && itThis.value() != finalOff)
					itThis.next();
				// process inside run
				for(int rix = itThis.value(); rix < re; rix = itThis.value()) { // nice skip inside runs
					that._dict.addToEntry(v, k, _data.getIndex(itThis.getDataIndex()), nCol);
					if(itThis.value() == finalOff) // break if final.
						break;
					itThis.next();
				}
			}
		}
	}

	@Override
	public AColGroup replace(double pattern, double replace) {
		ADictionary replaced = _dict.replace(pattern, replace, _colIndexes.length);
		if(pattern == 0) {
			double[] defaultTuple = new double[_colIndexes.length];
			for(int i = 0; i < _colIndexes.length; i++)
				defaultTuple[i] = replace;
			return ColGroupSDC.create(_colIndexes, _numRows, replaced, defaultTuple, _indexes, _data, getCachedCounts());
		}
		else
			return copyAndSet(replaced);
	}

	@Override
	protected void computeProduct(double[] c, int nRows) {
		c[0] = 0;
	}

	@Override
	protected void computeColProduct(double[] c, int nRows) {
		for(int i = 0; i < _colIndexes.length; i++)
			c[_colIndexes[i]] = 0;
	}

	@Override
	public double getCost(ComputationCostEstimator e, int nRows) {
		final int nVals = getNumValues();
		final int nCols = getNumCols();
		final int nRowsScanned = _data.size();
		return e.getCost(nRows, nRowsScanned, nCols, nVals, _dict.getSparsity());
	}

	@Override
	protected int numRowsToMultiply() {
		return _data.size();
	}

	@Override
	protected AColGroup allocateRightMultiplication(MatrixBlock right, int[] colIndexes, ADictionary preAgg) {
		if(colIndexes != null && preAgg != null)
			return create(colIndexes, _numRows, preAgg, _indexes, _data, getCachedCounts());
		else
			return null;
	}

	@Override
	protected double computeMxx(double c, Builtin builtin) {
		c = builtin.execute(c, 0);
		return _dict.aggregate(c, builtin);
	}

	@Override
	protected void computeColMxx(double[] c, Builtin builtin) {
		for(int x = 0; x < _colIndexes.length; x++)
			c[_colIndexes[x]] = builtin.execute(c[_colIndexes[x]], 0);

		_dict.aggregateCols(c, builtin, _colIndexes);
	}

	@Override
	public boolean containsValue(double pattern) {
		return (pattern == 0) || _dict.containsValue(pattern);
	}

	@Override
	public AColGroup sliceRows(int rl, int ru) {
		OffsetSliceInfo off = _indexes.slice(rl, ru);
		if(off.lIndex == -1)
			return null;
		AMapToData newData = _data.slice(off.lIndex, off.uIndex);
		return new ColGroupSDCZeros(_colIndexes, _numRows, _dict, off.offsetSlice, newData, null);
	}

	@Override
	protected AColGroup copyAndSet(int[] colIndexes, ADictionary newDictionary) {
		return create(colIndexes, _numRows, newDictionary, _indexes, _data, getCounts());
	}

	@Override
	public AColGroup append(AColGroup g) {
		return null;
	}

	@Override
	public AColGroup appendNInternal(AColGroup[] g) {
		return null;
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
