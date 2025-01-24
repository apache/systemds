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
import java.util.List;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUtils.P;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.PlaceHolderDict;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToByte;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToChar;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToUByte;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset.OffsetSliceInfo;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.estim.encoding.EncodingFactory;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
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
public class ColGroupSDCZeros extends ASDCZero implements IMapToDataGroup {
	private static final long serialVersionUID = -3703199743391937991L;

	/** Pointers to row indexes in the dictionary. Note the dictionary has one extra entry. */
	protected final AMapToData _data;

	private ColGroupSDCZeros(IColIndex colIndices, int numRows, IDictionary dict, AOffset indexes, AMapToData data,
		int[] cachedCounts) {
		super(colIndices, numRows, dict, indexes, cachedCounts);
		_data = data;
		if(CompressedMatrixBlock.debug) {
			if(data.getUnique() != dict.getNumberOfValues(colIndices.size()))
				throw new DMLCompressionException("Invalid construction of SDCZero group: number uniques: "
					+ data.getUnique() + " vs." + dict.getNumberOfValues(colIndices.size()));
			_data.verify();
			_indexes.verify(_data.size());
		}
	}

	public static AColGroup create(IColIndex colIndices, int numRows, IDictionary dict, AOffset offsets, AMapToData data,
		int[] cachedCounts) {
		if(dict == null)
			return new ColGroupEmpty(colIndices);
		else if(data.getUnique() == 1 && !(dict instanceof PlaceHolderDict)) {
			MatrixBlock mb = dict.getMBDict(colIndices.size()).getMatrixBlock().slice(0, 0);
			return ColGroupSDCSingleZeros.create(colIndices, numRows, MatrixBlockDictionary.create(mb), offsets, null);
		}
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
	public AMapToData getMapToData() {
		return _data;
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
			if(contiguous && _colIndexes.size() == 1 && db.getDim(1) == 1)
				decompressToDenseBlockDenseDictionaryPostSingleColContiguous(db, rl, ru, offR, offC, values, it);
			else if(contiguous && _colIndexes.size() == db.getDim(1)) // OffC == 0 implied
				decompressToDenseBlockDenseDictionaryPostAllCols(db, rl, ru, offR, values, it);
			else
				decompressToDenseBlockDenseDictionaryPostGeneric(db, rl, ru, offR, offC, values, it);
		}
		else if(contiguous && _colIndexes.size() == 1) {
			if(db.getDim(1) == 1)
				decompressToDenseBlockDenseDictionaryPreSingleColOutContiguous(db, ru, offR, offC, values, it, _data);
			else
				decompressToDenseBlockDenseDictionaryPreSingleColContiguous(db, rl, ru, offR, offC, values, it);
		}
		else {
			if(_colIndexes.size() == db.getDim(1))
				decompressToDenseBlockDenseDictionaryPreAllCols(db, rl, ru, offR, offC, values, it);
			else
				decompressToDenseBlockDenseDictionaryPreGeneric(db, rl, ru, offR, offC, values, it);
		}
	}

	private final void decompressToDenseBlockDenseDictionaryPostSingleColContiguous(DenseBlock db, int rl, int ru,
		int offR, int offC, double[] values, AIterator it) {
		final int lastOff = _indexes.getOffsetToLast() + offR;
		final int of = offR + offC;
		final double[] c = db.values(0);
		it.setOff(it.value() + of);
		decToDBDDSCP(c, values, it, _data, lastOff);
		it.setOff(it.value() - of);
	}

	private static void decToDBDDSCP(double[] c, double[] values, AIterator it, AMapToData m, int last) {
		decToDBDDSC(c, values, it, m, last);
		c[it.value()] += values[m.getIndex(it.getDataIndex())];
	}

	private final void decompressToDenseBlockDenseDictionaryPostAllCols(DenseBlock db, int rl, int ru, int offR,
		double[] values, AIterator it) {
		final int lastOff = _indexes.getOffsetToLast();
		final int nCol = _colIndexes.size();
		while(true) {
			final int idx = offR + it.value();
			final double[] c = db.values(idx);
			final int off = db.pos(idx);
			final int offDict = _data.getIndex(it.getDataIndex()) * nCol;
			for(int j = 0; j < nCol; j++)
				c[off + j] += values[offDict + j];
			if(it.value() == lastOff)
				return;
			it.next();
		}
	}

	private final void decompressToDenseBlockDenseDictionaryPostGeneric(DenseBlock db, int rl, int ru, int offR,
		int offC, double[] values, AIterator it) {
		final int lastOff = _indexes.getOffsetToLast();
		final int nCol = _colIndexes.size();
		while(true) {
			final int idx = offR + it.value();
			final double[] c = db.values(idx);
			final int off = db.pos(idx) + offC;
			final int offDict = _data.getIndex(it.getDataIndex()) * nCol;
			for(int j = 0; j < nCol; j++)
				c[off + _colIndexes.get(j)] += values[offDict + j];
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
		decToDBDDSC(c, values, it, m, last);
		it.setOff(it.value() - of);
	}

	private static void decToDBDDSC(double[] c, double[] values, AIterator it, AMapToData m, int last) {
		// JIT compile trick
		if(m instanceof MapToUByte)
			decToDBDDSC_UByte(c, values, it, (MapToUByte) m, last);
		else if(m instanceof MapToByte)
			decToDBDDSC_Byte(c, values, it, (MapToByte) m, last);
		else if(m instanceof MapToChar)
			decToDBDDSC_Char(c, values, it, (MapToChar) m, last);
		else
			decToDBDDSC_Generic(c, values, it, m, last);
	}

	private static void decToDBDDSC_Generic(double[] c, double[] values, AIterator it, AMapToData m, int last) {
		while(it.isNotOver(last)) {
			c[it.value()] += values[m.getIndex(it.getDataIndex())];
			it.next();
		}
	}

	private static void decToDBDDSC_UByte(double[] c, double[] values, AIterator it, MapToUByte m, int last) {
		while(it.isNotOver(last)) {
			c[it.value()] += values[m.getIndex(it.getDataIndex())];
			it.next();
		}
	}

	private static void decToDBDDSC_Byte(double[] c, double[] values, AIterator it, MapToByte m, int last) {
		while(it.isNotOver(last)) {
			c[it.value()] += values[m.getIndex(it.getDataIndex())];
			it.next();
		}
	}

	private static void decToDBDDSC_Char(double[] c, double[] values, AIterator it, MapToChar m, int last) {
		while(it.isNotOver(last)) {
			c[it.value()] += values[m.getIndex(it.getDataIndex())];
			it.next();
		}
	}

	private void decompressToDenseBlockDenseDictionaryPreSingleColContiguous(DenseBlock db, int rl, int ru, int offR,
		int offC, double[] values, AIterator it) {
		final int last = ru + offR;
		final int nCol = db.getDim(1);
		final double[] c = db.values(0);
		it.setOff(it.value() + offR);
		offC += _colIndexes.get(0);
		while(it.isNotOver(last)) {
			final int off = it.value() * nCol + offC;
			c[off] += values[_data.getIndex(it.getDataIndex())];
			it.next();
		}
		it.setOff(it.value() - offR);
	}

	private final void decompressToDenseBlockDenseDictionaryPreGeneric(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values, AIterator it) {
		final int nCol = _colIndexes.size();
		while(it.isNotOver(ru)) {
			decompressRowDenseDictionaryPreGeneric(db, nCol, offR, offC, values, it);
			it.next();
		}
	}

	private final void decompressRowDenseDictionaryPreGeneric(DenseBlock db, int nCol, int offR, int offC,
		double[] values, AIterator it) {
		final int idx = offR + it.value();
		final double[] c = db.values(idx);
		final int off = db.pos(idx) + offC;
		final int offDict = _data.getIndex(it.getDataIndex()) * nCol;
		for(int j = 0; j < nCol; j++)
			c[off + _colIndexes.get(j)] += values[offDict + j];
	}

	private void decompressToDenseBlockDenseDictionaryPreAllCols(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values, AIterator it) {
		final int nCol = _colIndexes.size();
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
				c[off + _colIndexes.get(aix[j])] += avals[j];
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
				c[off + _colIndexes.get(aix[j])] += avals[j];

			it.next();
		}
		// _indexes.cacheIterator(it, ru);
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
					ret.append(row, _colIndexes.get(aix[j]) + offC, avals[j]);
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
					ret.append(row, _colIndexes.get(aix[j]) + offC, avals[j]);
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
			final int nCol = _colIndexes.size();
			while(true) {
				final int row = offR + it.value();
				final int dx = it.getDataIndex();
				final int offDict = _data.getIndex(dx) * nCol;
				for(int j = 0; j < nCol; j++)
					ret.append(row, _colIndexes.get(j) + offC, values[offDict + j]);
				if(it.value() == lastOff)
					return;
				it.next();
			}
		}
		else {

			final int nCol = _colIndexes.size();
			while(it.isNotOver(ru)) {
				final int row = offR + it.value();
				final int dx = it.getDataIndex();
				final int offDict = _data.getIndex(dx) * nCol;
				for(int j = 0; j < nCol; j++)
					ret.append(row, _colIndexes.get(j) + offC, values[offDict + j]);

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
		final int nCol = _colIndexes.size();
		return _dict.getValue(_data.getIndex(it.getDataIndex()) * nCol + colIdx);
	}

	@Override
	protected double[] preAggProductRows() {
		return _dict.productAllRowsToDoubleWithDefault(new double[_colIndexes.size()]);
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
		_data.preAggregateDense(m.getDenseBlock(), preAgg, rl, ru, cl, cu, _indexes);
	}

	@Override
	public void leftMMIdentityPreAggregateDense(MatrixBlock that, MatrixBlock ret, int rl, int ru, int cl, int cu) {
		throw new NotImplementedException();
	}

	@Override
	public void preAggregateSparse(SparseBlock sb, double[] preAgg, int rl, int ru, int cl, int cu) {
		if(cl != 0 || cu < _indexes.getOffsetToLast()) {
			throw new NotImplementedException();
		}
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
			final double[] reference = ColGroupUtils.createReference(_colIndexes.size(), val0);
			return ColGroupSDCFOR.create(_colIndexes, _numRows, _dict, _indexes, _data, getCachedCounts(), reference);
		}
		else {
			final IDictionary newDict = _dict.applyScalarOp(op);
			final double[] defaultTuple = ColGroupUtils.createReference(_colIndexes.size(), val0);
			return ColGroupSDC.create(_colIndexes, _numRows, newDict, defaultTuple, _indexes, _data, getCachedCounts());
		}
	}

	@Override
	public AColGroup unaryOperation(UnaryOperator op) {
		final double val0 = op.fn.execute(0);
		final IDictionary nDict = _dict.applyUnaryOp(op);
		if(val0 == 0)
			return create(_colIndexes, _numRows, nDict, _indexes, _data, getCachedCounts());
		else {
			final double[] defaultTuple = new double[_colIndexes.size()];
			Arrays.fill(defaultTuple, val0);
			return ColGroupSDC.create(_colIndexes, _numRows, nDict, defaultTuple, _indexes, _data, getCachedCounts());
		}
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		if(isRowSafe) {
			IDictionary newDict = _dict.binOpLeft(op, v, _colIndexes);
			return create(_colIndexes, _numRows, newDict, _indexes, _data, getCachedCounts());
		}
		else if(op.fn instanceof Plus) {
			double[] reference = ColGroupUtils.binaryDefRowLeft(op, v, _colIndexes);
			return ColGroupSDCFOR.create(_colIndexes, _numRows, _dict, _indexes, _data, getCachedCounts(), reference);
		}
		else {
			IDictionary newDict = _dict.binOpLeft(op, v, _colIndexes);
			double[] defaultTuple = new double[_colIndexes.size()];
			for(int i = 0; i < _colIndexes.size(); i++)
				defaultTuple[i] = op.fn.execute(v[_colIndexes.get(i)], 0);
			return ColGroupSDC.create(_colIndexes, _numRows, newDict, defaultTuple, _indexes, _data, getCachedCounts());
		}
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		if(isRowSafe) {
			IDictionary ret = _dict.binOpRight(op, v, _colIndexes);
			return create(_colIndexes, _numRows, ret, _indexes, _data, getCachedCounts());
		}
		else if(op.fn instanceof Plus) {
			double[] def = ColGroupUtils.binaryDefRowRight(op, v, _colIndexes);
			return ColGroupSDCFOR.create(_colIndexes, _numRows, _dict, _indexes, _data, getCachedCounts(), def);
		}
		else {
			IDictionary newDict = _dict.binOpRight(op, v, _colIndexes);
			double[] defaultTuple = new double[_colIndexes.size()];
			for(int i = 0; i < _colIndexes.size(); i++)
				defaultTuple[i] = op.fn.execute(0, v[_colIndexes.get(i)]);
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
		IColIndex cols = ColIndexFactory.read(in);
		IDictionary dict = DictionaryFactory.read(in);
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
		else if(that instanceof ColGroupSDC) {
			ColGroupSDC th = (ColGroupSDC) that;
			return th._indexes == _indexes && th._data == _data;
		}
		else
			return false;
	}

	@Override
	public void preAggregateThatDDCStructure(ColGroupDDC that, Dictionary ret) {
		_data.preAggregateSDCZ_DDC(that._data, that._dict, _indexes, ret, that._colIndexes.size());
	}

	@Override
	public void preAggregateThatSDCZerosStructure(ColGroupSDCZeros that, Dictionary ret) {
		_data.preAggregateSDCZ_SDCZ(that._data, that._dict, that._indexes, _indexes, ret, that._colIndexes.size());
	}

	@Override
	public void preAggregateThatSDCSingleZerosStructure(ColGroupSDCSingleZeros that, Dictionary ret) {
		final AIterator itThat = that._indexes.getIterator();
		// _indexes.getOffsetIterator();
		final AIterator itThis = _indexes.getIterator();
		final int nCol = that._colIndexes.size();

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
		final int nCol = that._colIndexes.size();
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
		IDictionary replaced = _dict.replace(pattern, replace, _colIndexes.size());
		if(pattern == 0) {
			double[] defaultTuple = new double[_colIndexes.size()];
			for(int i = 0; i < _colIndexes.size(); i++)
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
		for(int i = 0; i < _colIndexes.size(); i++)
			c[_colIndexes.get(i)] = 0;
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
	protected AColGroup allocateRightMultiplication(MatrixBlock right, IColIndex colIndexes, IDictionary preAgg) {
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
		for(int x = 0; x < _colIndexes.size(); x++)
			c[_colIndexes.get(x)] = builtin.execute(c[_colIndexes.get(x)], 0);

		_dict.aggregateCols(c, builtin, _colIndexes);
	}

	@Override
	public boolean containsValue(double pattern) {
		return (pattern == 0) || _dict.containsValue(pattern);
	}

	@Override
	public AColGroup sliceRows(int rl, int ru) {
		if(ru > _numRows)
			throw new DMLRuntimeException("Invalid row range");
		OffsetSliceInfo off = _indexes.slice(rl, ru);
		if(off.lIndex == -1)
			return null;
		AMapToData newData = _data.slice(off.lIndex, off.uIndex);
		return create(_colIndexes, ru - rl, _dict, off.offsetSlice, newData, null);
	}

	@Override
	protected AColGroup copyAndSet(IColIndex colIndexes, IDictionary newDictionary) {
		return create(colIndexes, _numRows, newDictionary, _indexes, _data, getCachedCounts());
	}

	@Override
	public AColGroup append(AColGroup g) {
		return null;
	}

	@Override
	public AColGroup appendNInternal(AColGroup[] g, int blen, int rlen) {

		for(int i = 1; i < g.length; i++) {
			final AColGroup gs = g[i];
			if(!_colIndexes.equals(gs._colIndexes)) {
				LOG.warn("Not same columns therefore not appending \n" + _colIndexes + "\n\n" + g[i]._colIndexes);
				return null;
			}

			if(!(gs instanceof AOffsetsGroup)) {
				LOG.warn("Not valid OffsetGroup but " + gs.getClass().getSimpleName());
				return null;
			}

			if(gs instanceof ColGroupSDCZeros) {
				final ColGroupSDCZeros gc = (ColGroupSDCZeros) gs;
				if(!gc._dict.equals(_dict)) {
					LOG.warn("Not same Dictionaries therefore not appending \n" + _dict + "\n\n" + gc._dict);
					return null;
				}
			}
		}
		AMapToData nd = _data.appendN(Arrays.copyOf(g, g.length, IMapToDataGroup[].class));
		AOffset no = _indexes.appendN(Arrays.copyOf(g, g.length, AOffsetsGroup[].class), blen);

		return create(_colIndexes, rlen, _dict, no, nd, null);
	}

	@Override
	public AColGroup recompress() {
		return this;
	}

	@Override
	public int getNumberOffsets() {
		return _data.size();
	}

	@Override
	public IEncode getEncoding() {
		return EncodingFactory.create(_data, _indexes, _numRows);
	}

	@Override
	protected AColGroup fixColIndexes(IColIndex newColIndex, int[] reordering) {
		return ColGroupSDCZeros.create(newColIndex, getNumRows(), _dict.reorder(reordering), _indexes, _data,
			getCachedCounts());
	}

	@Override
	public AColGroup morph(CompressionType ct, int nRow) {
		// if(ct == getCompType())
		// return this;
		// else if (ct == CompressionType.SDCFOR)
		// return this; // it does not make sense to change to FOR.
		if(ct == CompressionType.DDC) {
			AMapToData retMap = MapToFactory.create(_numRows, _data.getUnique() + 1);
			IDictionary combinedDict = _dict.append(getDefaultTuple());
			retMap.fill(_data.getUnique());

			AIterator it = _indexes.getIterator();
			while(it.value() < _indexes.getOffsetToLast()) {
				retMap.set(it.value(), _data.getIndex(it.getDataIndex()));
				it.next();
			}
			retMap.set(it.value(), _data.getIndex(it.getDataIndex()));

			return ColGroupDDC.create(_colIndexes, combinedDict, retMap, null);

		}
		else
			return super.morph(ct, nRow);
	}


	@Override
	public void sparseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		final SparseBlock sr = ret.getSparseBlock();
		final int nCol = _colIndexes.size();
		final AIterator it = _indexes.getIterator();
		final int last = _indexes.getOffsetToLast();
		int c = 0;
		int of = it.value();

		while(of < last && c < points.length) {
			if(points[c].o == of) {
				c = processRowSparse(points, sr, nCol, c, of, _data.getIndex(it.getDataIndex()));
				of = it.next();
			}
			else if(points[c].o < of)
				c++;
			else
				of = it.next();
		}
		// increment the c pointer until it is pointing at least to last point or is done.
		while(c < points.length && points[c].o < last)
			c++;

		c = processRowSparse(points, sr, nCol, c, of, _data.getIndex(it.getDataIndex()));

	}

	@Override
	protected void denseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		final DenseBlock dr = ret.getDenseBlock();
		final int nCol = _colIndexes.size();
		final AIterator it = _indexes.getIterator();
		final int last = _indexes.getOffsetToLast();
		int c = 0;
		int of = it.value();

		while(of < last && c < points.length) {
			if(points[c].o == of) {
				c = processRowDense(points, dr, nCol, c, of, _data.getIndex(it.getDataIndex()));
				of = it.next();
			}
			else if(points[c].o < of)
					c++;
			else
				of = it.next();
			}
			// increment the c pointer until it is pointing at least to last point or is done.
			while(c < points.length && points[c].o < last)
				c++;
			c = processRowDense(points, dr, nCol, c, of, _data.getIndex(it.getDataIndex()));
	}

	private int processRowSparse(P[] points, final SparseBlock sr, final int nCol, int c, int of, final int did) {
		while(c < points.length && points[c].o == of) {
			_dict.putSparse(sr, did, points[c].r, nCol, _colIndexes);
			c++;
		}
		return c;
	}

	private int processRowDense(P[] points, final DenseBlock dr, final int nCol, int c, int of, final int did) {
		while(c < points.length && points[c].o == of) {
			_dict.putDense(dr, did, points[c].r, nCol, _colIndexes);
			c++;
		}
		return c;
	}

	protected void decompressToDenseBlockTransposedSparseDictionary(DenseBlock db, int rl, int ru, SparseBlock sb) {
		throw new NotImplementedException();
	}

	@Override
	protected void decompressToDenseBlockTransposedDenseDictionary(DenseBlock db, int rl, int ru, double[] dict) {
		throw new NotImplementedException();
	}

	@Override
	protected void decompressToSparseBlockTransposedSparseDictionary(SparseBlockMCSR db, SparseBlock sb, int nColOut) {
		throw new NotImplementedException();
	}

	@Override
	protected void decompressToSparseBlockTransposedDenseDictionary(SparseBlockMCSR db, double[] dict, int nColOut) {
		throw new NotImplementedException();
		// AIterator it = _indexes.getIterator();

		// final int rowOut = _colIndexes.size();
		// final int last = _indexes.getOffsetToLast();

		// int v = it.value();
		// while(v < last) {
		// final int di = _data.getIndex(it.getDataIndex());
		// for(int c = 0; c < rowOut; c++) {
		// db.append(_colIndexes.get(c), v, dict[di * rowOut + c]);
		// }
		// v = it.next();
		// }

		// // take last element.

		// final int di = _data.getIndex(it.getDataIndex());
		// for(int c = 0; c < rowOut; c++) {
		// db.append(_colIndexes.get(c), v, dict[di * rowOut + c]);
		// }

	}

	@Override
	public AColGroupCompressed combineWithSameIndex(int nRow, int nCol, AColGroup right) {
		ColGroupSDCZeros rightSDC = ((ColGroupSDCZeros) right);
		IDictionary b = rightSDC.getDictionary();
		IDictionary combined = DictionaryFactory.cBindDictionaries(_dict, b, this.getNumCols(), right.getNumCols());
		IColIndex combinedColIndex = _colIndexes.combine(right.getColIndices().shift(nCol));
		return new ColGroupSDCZeros(combinedColIndex, this.getNumRows(), combined, _indexes, _data, getCachedCounts());
	}

	@Override
	public AColGroupCompressed combineWithSameIndex(int nRow, int nCol, List<AColGroup> right) {
		final IDictionary combined = combineDictionaries(nCol, right);
		final IColIndex combinedColIndex = combineColIndexes(nCol, right);

		return new ColGroupSDCZeros(combinedColIndex, this.getNumRows(), combined, _indexes, _data, getCachedCounts());
	}

	@Override
	public AColGroup[] splitReshape(int multiplier, int nRow, int nColOrg) {
		IntArrayList[] splitOffs = new IntArrayList[multiplier];
		IntArrayList[] tmpMaps = new IntArrayList[multiplier];
		for(int i = 0; i < multiplier; i++) {
			splitOffs[i] = new IntArrayList();
			tmpMaps[i] = new IntArrayList();
		}

		AIterator it = _indexes.getIterator();
		final int last = _indexes.getOffsetToLast();

		while(it.value() != last) {
			final int v = it.value(); // offset
			final int d = it.getDataIndex(); // data index value
			final int m = _data.getIndex(d);

			final int outV = v / multiplier;
			final int outM = v % multiplier;

			tmpMaps[outM].appendValue(m);
			splitOffs[outM].appendValue(outV);

			it.next();
		}

		// last value
		final int v = it.value();
		final int d = it.getDataIndex();
		final int m = _data.getIndex(d);
		final int outV = v / multiplier;
		final int outM = v % multiplier;
		tmpMaps[outM].appendValue(m);
		splitOffs[outM].appendValue(outV);

		// iterate through all rows.

		AOffset[] offs = new AOffset[multiplier];
		AMapToData[] maps = new AMapToData[multiplier];
		for(int i = 0; i < multiplier; i++) {
			offs[i] = OffsetFactory.createOffset(splitOffs[i]);
			maps[i] = MapToFactory.create(_data.getUnique(), tmpMaps[i]);
		}

		// assign columns
		AColGroup[] res = new AColGroup[multiplier];
		for(int i = 0; i < multiplier; i++) {
			final IColIndex ci = i == 0 ? _colIndexes : _colIndexes.shift(i * nColOrg);
			res[i] = create(ci, _numRows / multiplier, _dict, offs[i], maps[i], null);
		}
		return res;
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
