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

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset.OffsetSliceInfo;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffsetIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.colgroup.scheme.ICLAScheme;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

/**
 * Column group that sparsely encodes the dictionary values. The idea is that all values is encoded with indexes except
 * the most common one. the most common one can be inferred by not being included in the indexes. If the values are very
 * sparse then the most common one is zero.
 * 
 * This column group is handy in cases where sparse unsafe operations is executed on very sparse columns. Then the zeros
 * would be materialized in the group without any overhead.
 */
public class ColGroupSDCSingleZeros extends ASDCZero {
	private static final long serialVersionUID = 8033235615964315078L;

	private ColGroupSDCSingleZeros(int[] colIndices, int numRows, ADictionary dict, AOffset offsets,
		int[] cachedCounts) {
		super(colIndices, numRows, dict, offsets, cachedCounts);
		if(offsets.getSize() * 2 > numRows + 2)
			throw new DMLCompressionException("Wrong direction of SDCSingleZero compression should be other way " + numRows
				+ " vs " + _indexes + "\n" + this);
	}

	public static AColGroup create(int[] colIndices, int numRows, ADictionary dict, AOffset offsets,
		int[] cachedCounts) {
		if(dict == null)
			return new ColGroupEmpty(colIndices);
		else
			return new ColGroupSDCSingleZeros(colIndices, numRows, dict, offsets, cachedCounts);
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
		else {
			decompressToDenseBlockDenseDictionaryWithProvidedIterator(db, rl, ru, offR, offC, values, it);
			_indexes.cacheIterator(it, ru);
		}
	}

	@Override
	public void decompressToDenseBlockDenseDictionaryWithProvidedIterator(DenseBlock db, int rl, int ru, int offR,
		int offC, double[] values, AIterator it) {
		final int last = _indexes.getOffsetToLast();
		if(it == null || it.value() >= ru || rl > last)
			return;
		else if(ru > _indexes.getOffsetToLast())
			decompressToDenseBlockDenseDictionaryPost(db, rl, ru, offR, offC, values, it);
		else {
			if(_colIndexes.length == 1 && db.getDim(1) == 1)
				decompressToDenseBlockDenseDictionaryPreSingleColOutContiguous(db, rl, ru, offR, offC, values[0], it);
			else
				decompressToDenseBlockDenseDictionaryPre(db, rl, ru, offR, offC, values, it);
		}
	}

	private void decompressToDenseBlockDenseDictionaryPost(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values, AIterator it) {
		final int maxOff = _indexes.getOffsetToLast();
		final int nCol = _colIndexes.length;
		int row = offR + it.value();
		double[] c = db.values(row);
		int off = db.pos(row);
		for(int j = 0; j < nCol; j++)
			c[off + _colIndexes[j] + offC] += values[j];
		while(it.value() < maxOff) {
			it.next();
			row = offR + it.value();
			c = db.values(row);
			off = db.pos(row);
			for(int j = 0; j < nCol; j++)
				c[off + _colIndexes[j] + offC] += values[j];

		}
	}

	private void decompressToDenseBlockDenseDictionaryPreSingleColOutContiguous(DenseBlock db, int rl, int ru, int offR,
		int offC, double v, AIterator it) {
		final double[] c = db.values(0);
		int r = it.value();
		while(r < ru) {
			c[offR + r] += v;
			r = it.next();
		}
	}

	private void decompressToDenseBlockDenseDictionaryPre(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values, AIterator it) {
		final int nCol = _colIndexes.length;
		int r = it.value();
		while(r < ru) {
			final int row = offR + r;
			final double[] c = db.values(row);
			final int off = db.pos(row);
			for(int j = 0; j < nCol; j++)
				c[off + _colIndexes[j] + offC] += values[j];

			r = it.next();
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
		final int apos = sb.pos(0);
		final int alen = sb.size(0) + apos;
		final double[] avals = sb.values(0);
		final int[] aix = sb.indexes(0);
		while(true) {
			final int idx = offR + it.value();
			final double[] c = db.values(idx);

			final int off = db.pos(idx) + offC;
			for(int j = apos; j < alen; j++)
				c[off + _colIndexes[aix[j]]] += avals[j];
			if(it.value() == last)
				return;
			it.next();
		}
	}

	private final void decompressToDenseBlockSparseDictionaryPre(DenseBlock db, int rl, int ru, int offR, int offC,
		SparseBlock sb, AIterator it) {
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

	@Override
	protected void decompressToSparseBlockSparseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		SparseBlock sb) {
		final AIterator it = _indexes.getIterator(rl);
		final int last = _indexes.getOffsetToLast();
		if(it == null)
			return;
		else if(it.value() >= ru)
			_indexes.cacheIterator(it, ru);
		else if(ru > last) {
			final int apos = sb.pos(0);
			final int alen = sb.size(0) + apos;
			final int[] aix = sb.indexes(0);
			final double[] avals = sb.values(0);
			while(it.value() < last) {
				final int row = offR + it.value();
				for(int j = apos; j < alen; j++)
					ret.append(row, _colIndexes[aix[j]] + offC, avals[j]);
				it.next();
			}
			final int row = offR + it.value();
			for(int j = apos; j < alen; j++)
				ret.append(row, _colIndexes[aix[j]] + offC, avals[j]);
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
			int row = offR + it.value();
			for(int j = 0; j < nCol; j++)
				ret.append(row, _colIndexes[j] + offC, values[j]);
			while(it.value() < lastOff) {
				it.next();
				row = offR + it.value();
				for(int j = 0; j < nCol; j++)
					ret.append(row, _colIndexes[j] + offC, values[j]);
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
	protected void computeRowProduct(double[] c, int rl, int ru, double[] preAgg) {
		ColGroupSDCSingle.computeRowProduct(c, rl, ru, _indexes, _numRows, 0, preAgg[0]);
	}

	@Override
	public int[] getCounts(int[] counts) {
		counts[0] = _indexes.getSize();
		return counts;
	}

	@Override
	protected void multiplyScalar(double v, double[] resV, int offRet, AIterator it) {
		_dict.multiplyScalar(v, resV, offRet, 0, _colIndexes);
	}

	@Override
	public void preAggregateDense(MatrixBlock m, double[] preAgg, int rl, int ru, int cl, int cu) {
		if(!m.getDenseBlock().isContiguous())
			throw new NotImplementedException("Not implemented support for preAggregate non contiguous dense matrix");
		final AIterator it = _indexes.getIterator(cl);
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
				for(int offOut = 0, off = start; off < end; offOut++, off += nCol)
					preAgg[offOut] += vals[off];
				it.next();
			}
			_indexes.cacheIterator(it, cu);
		}
		else {
			int of = it.value();
			int start = of + nCol * rl;
			int end = of + nCol * ru;
			for(int offOut = 0, off = start; off < end; offOut++, off += nCol)
				preAgg[offOut] += vals[off];
			while(of < _indexes.getOffsetToLast()) {
				it.next();
				of = it.value();
				start = of + nCol * rl;
				end = of + nCol * ru;
				for(int offOut = 0, off = start; off < end; offOut++, off += nCol)
					preAgg[offOut] += vals[off];
			}
		}
	}

	@Override
	public void preAggregateSparse(SparseBlock sb, double[] preAgg, int rl, int ru) {
		final AOffsetIterator it = _indexes.getOffsetIterator();
		if(rl == ru - 1)
			preAggregateSparseSingleRow(sb, preAgg, rl, _indexes.getOffsetToLast(), it);
		else
			preAggregateSparseMultiRow(sb, preAgg, rl, ru, _indexes.getOffsetToLast(), it);
	}

	private static void preAggregateSparseSingleRow(final SparseBlock sb, final double[] preAgg, final int r,
		final int last, final AOffsetIterator it) {
		if(sb.isEmpty(r))
			return;

		final int apos = sb.pos(r);
		final int alen = sb.size(r) + apos;
		final int[] aix = sb.indexes(r);
		final double[] avals = sb.values(r);

		double ret = 0;
		int i = it.value();
		int j = apos;
		while(i < last && j < alen) {
			final int idx = aix[j];
			if(idx == i) {
				ret += avals[j++];
				i = it.next();
			}
			else if(idx < i)
				j++;
			else
				i = it.next();
		}

		while(j < alen && aix[j] < last)
			j++;

		if(j < alen && aix[j] == last)
			ret += avals[j];

		preAgg[0] = ret;
	}

	private static void preAggregateSparseMultiRow(final SparseBlock sb, final double[] preAgg, final int rl,
		final int ru, final int last, final AOffsetIterator it) {

		int i = it.value();
		final int[] aOffs = new int[ru - rl];

		// Initialize offsets for each row
		for(int r = rl; r < ru; r++)
			aOffs[r - rl] = sb.pos(r);

		while(i < last) { // while we are not done iterating
			for(int r = rl; r < ru; r++) {
				if(sb.isEmpty(r))
					continue;
				final int off = r - rl;
				int apos = aOffs[off]; // current offset
				final int alen = sb.size(r) + sb.pos(r);
				final int[] aix = sb.indexes(r);
				while(apos < alen && aix[apos] < i)// increment all pointers to offset
					apos++;

				if(apos < alen && aix[apos] == i)
					preAgg[off] += sb.values(r)[apos];
				aOffs[off] = apos;
			}
			i = it.next();
		}

		// process final element
		for(int r = rl; r < ru; r++) {
			if(sb.isEmpty(r))
				continue;
			final int off = r - rl;
			int apos = aOffs[off];
			final int alen = sb.size(r) + sb.pos(r);
			final int[] aix = sb.indexes(r);
			while(apos < alen && aix[apos] < last)
				apos++;

			if(apos < alen && aix[apos] == last)
				preAgg[off] += sb.values(r)[apos];
			aOffs[off] = apos;
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
		final double val0 = op.executeScalar(0);
		final boolean isSparseSafeOp = val0 == 0;
		final ADictionary nDict = _dict.applyScalarOp(op);
		if(isSparseSafeOp)
			return create(_colIndexes, _numRows, nDict, _indexes, getCachedCounts());
		else {
			final double[] defaultTuple = new double[_colIndexes.length];
			Arrays.fill(defaultTuple, val0);
			return ColGroupSDCSingle.create(_colIndexes, _numRows, nDict, defaultTuple, _indexes, getCachedCounts());
		}
	}

	@Override
	public AColGroup unaryOperation(UnaryOperator op) {
		final double val0 = op.fn.execute(0);
		final ADictionary nDict = _dict.applyUnaryOp(op);
		if(val0 == 0)
			return create(_colIndexes, _numRows, nDict, _indexes, getCachedCounts());
		else {
			final double[] defaultTuple = new double[_colIndexes.length];
			Arrays.fill(defaultTuple, val0);
			return ColGroupSDCSingle.create(_colIndexes, _numRows, nDict, defaultTuple, _indexes, getCachedCounts());
		}
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
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		if(isRowSafe) {
			ADictionary ret = _dict.binOpLeft(op, v, _colIndexes);
			return ColGroupSDCSingleZeros.create(_colIndexes, _numRows, ret, _indexes, getCachedCounts());
		}
		else {
			ADictionary newDict = _dict.binOpLeft(op, v, _colIndexes);
			double[] defaultTuple = new double[_colIndexes.length];
			for(int i = 0; i < _colIndexes.length; i++)
				defaultTuple[i] = op.fn.execute(v[_colIndexes[i]], 0);
			return ColGroupSDCSingle.create(_colIndexes, _numRows, newDict, defaultTuple, _indexes, getCachedCounts());
		}
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		if(isRowSafe) {
			ADictionary ret = _dict.binOpRight(op, v, _colIndexes);
			return ColGroupSDCSingleZeros.create(_colIndexes, _numRows, ret, _indexes, getCachedCounts());
		}
		else {
			ADictionary newDict = _dict.binOpRight(op, v, _colIndexes);
			double[] defaultTuple = new double[_colIndexes.length];
			for(int i = 0; i < _colIndexes.length; i++)
				defaultTuple[i] = op.fn.execute(0, v[_colIndexes[i]]);
			return ColGroupSDCSingle.create(_colIndexes, _numRows, newDict, defaultTuple, _indexes, getCachedCounts());
		}
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		_indexes.write(out);
	}

	public static ColGroupSDCSingleZeros read(DataInput in, int nRows) throws IOException {
		int[] cols = readCols(in);
		ADictionary dict = DictionaryFactory.read(in);
		AOffset indexes = OffsetFactory.readIn(in);
		return new ColGroupSDCSingleZeros(cols, nRows, dict, indexes, null);
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
		final AOffsetIterator itThis = _indexes.getOffsetIterator();
		final int nCol = that._colIndexes.length;
		final int finalOffThis = _indexes.getOffsetToLast();
		final double[] rV = ret.getValues();
		if(nCol == 1)
			preAggregateThatDDCStructureSingleCol(that, rV, itThis, finalOffThis);
		else
			preAggregateThatDDCStructureMultiCol(that, rV, itThis, finalOffThis, nCol);
	}

	private void preAggregateThatDDCStructureSingleCol(ColGroupDDC that, double[] rV, AOffsetIterator itThis,
		int finalOffThis) {
		double rv = 0;
		final double[] tV = that._dict.getValues();
		while(true) {
			final int v = itThis.value();
			rv += tV[that._data.getIndex(v)];
			if(v >= finalOffThis)
				break;
			itThis.next();
		}

		rV[0] += rv;
	}

	private void preAggregateThatDDCStructureMultiCol(ColGroupDDC that, double[] rV, AOffsetIterator itThis,
		int finalOffThis, int nCol) {
		while(true) {
			final int v = itThis.value();
			final int fr = that._data.getIndex(v);
			that._dict.addToEntry(rV, fr, 0, nCol);
			if(v >= finalOffThis)
				break;
			itThis.next();
		}
	}

	@Override
	public void preAggregateThatSDCZerosStructure(ColGroupSDCZeros that, Dictionary ret) {
		final AIterator itThat = that._indexes.getIterator();
		final AOffsetIterator itThis = _indexes.getOffsetIterator();
		final int nCol = that._colIndexes.length;
		final int finalOffThis = _indexes.getOffsetToLast();
		final int finalOffThat = that._indexes.getOffsetToLast();
		final double[] rV = ret.getValues();
		if(nCol == 1)
			preAggregateThatSDCZerosStructureSingleCol(that, rV, itThat, finalOffThat, itThis, finalOffThis);
		else
			preAggregateThatSDCZerosStructureMultiCol(that, rV, itThat, finalOffThat, itThis, finalOffThis, nCol);
	}

	private void preAggregateThatSDCZerosStructureSingleCol(ColGroupSDCZeros that, double[] rV, AIterator itThat,
		int finalOffThat, AOffsetIterator itThis, int finalOffThis) {
		double rv = 0;
		final double[] tV = that._dict.getValues();
		while(true) {
			final int tv = itThat.value();
			final int v = itThis.value();
			if(tv == v) {
				rv += tV[that._data.getIndex(itThat.getDataIndex())];
				if(tv >= finalOffThat || v >= finalOffThis)
					break;
				itThat.next();
				itThis.next();
			}
			else if(tv < v) {
				if(tv >= finalOffThat)
					break;
				itThat.next();
			}
			else {
				if(v >= finalOffThis)
					break;
				itThis.next();
			}
		}
		rV[0] += rv;
	}

	private void preAggregateThatSDCZerosStructureMultiCol(ColGroupSDCZeros that, double[] rV, AIterator itThat,
		int finalOffThat, AOffsetIterator itThis, int finalOffThis, int nCol) {
		while(true) {
			final int tv = itThat.value();
			final int v = itThis.value();
			if(tv == v) {
				that._dict.addToEntry(rV, that._data.getIndex(itThat.getDataIndex()), 0, nCol);
				if(tv >= finalOffThat || v >= finalOffThis)
					break;
				itThat.next();
				itThis.next();
			}
			else if(tv < v) {
				if(tv >= finalOffThat)
					break;
				itThat.next();
			}
			else {
				if(v >= finalOffThis)
					break;
				itThis.next();
			}
		}
	}

	@Override
	public void preAggregateThatSDCSingleZerosStructure(ColGroupSDCSingleZeros that, Dictionary ret) {
		final int nCol = that._colIndexes.length;
		final AOffsetIterator itThis = _indexes.getOffsetIterator();
		final AOffsetIterator itThat = that._indexes.getOffsetIterator();
		final int finalOffThis = _indexes.getOffsetToLast();
		final int finalOffThat = that._indexes.getOffsetToLast();
		int count = 0;
		int tv = itThat.value();
		int v = itThis.value();
		while(tv < finalOffThat && v < finalOffThis) {
			if(tv == v) {
				count++;
				tv = itThat.next();
				v = itThis.next();
			}
			else if(tv < v)
				tv = itThat.next();
			else
				v = itThis.next();
		}
		while(tv < finalOffThat && tv < v)
			tv = itThat.next();
		while(v < finalOffThis && v < tv)
			v = itThis.next();
		if(tv == v)
			count++;

		that._dict.addToEntry(ret.getValues(), 0, 0, nCol, count);

	}

	@Override
	protected void preAggregateThatRLEStructure(ColGroupRLE that, Dictionary ret) {
		final int finalOff = _indexes.getOffsetToLast();
		final double[] v = ret.getValues();
		final int nv = that.getNumValues();
		final int nCol = that._colIndexes.length;
		for(int k = 0; k < nv; k++) {
			final AOffsetIterator itThis = _indexes.getOffsetIterator();
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
					that._dict.addToEntry(v, k, 0, nCol);
					if(itThis.value() == finalOff) // break if final.
						break;
					itThis.next();
				}
			}
		}
	}

	@Override
	public int getPreAggregateSize() {
		return 1;
	}

	@Override
	public AColGroup replace(double pattern, double replace) {
		ADictionary replaced = _dict.replace(pattern, replace, _colIndexes.length);
		if(pattern == 0) {
			double[] defaultTuple = new double[_colIndexes.length];
			for(int i = 0; i < _colIndexes.length; i++)
				defaultTuple[i] = replace;
			return ColGroupSDCSingle.create(_colIndexes, _numRows, replaced, defaultTuple, _indexes, getCachedCounts());
		}
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
		final int nRowsScanned = getCounts()[0];
		return e.getCost(nRows, nRowsScanned, nCols, nVals, _dict.getSparsity());
	}

	@Override
	protected int numRowsToMultiply() {
		return getCounts()[0];
	}

	@Override
	protected AColGroup allocateRightMultiplication(MatrixBlock right, int[] colIndexes, ADictionary preAgg) {
		if(colIndexes != null && preAgg != null)
			return create(colIndexes, _numRows, preAgg, _indexes, getCachedCounts());
		else
			return null;
	}

	@Override
	public AColGroup sliceRows(int rl, int ru) {
		OffsetSliceInfo off = _indexes.slice(rl, ru);
		if(off.lIndex == -1)
			return null;
		return new ColGroupSDCSingleZeros(_colIndexes, _numRows, _dict, off.offsetSlice, null);
	}

	@Override
	protected AColGroup copyAndSet(int[] colIndexes, ADictionary newDictionary) {
		return create(colIndexes, _numRows, newDictionary, _indexes, getCachedCounts());
	}

	@Override
	public AColGroup append(AColGroup g) {
		return null;
	}

	@Override
	public AColGroup appendNInternal(AColGroup[] g) {
		int sumRows = 0;
		for(int i = 1; i < g.length; i++) {
			if(!Arrays.equals(_colIndexes, g[i]._colIndexes)) {
				LOG.warn("Not same columns therefore not appending \n" + Arrays.toString(_colIndexes) + "\n\n"
					+ Arrays.toString(g[i]._colIndexes));
				return null;
			}

			if(!(g[i] instanceof ColGroupSDCSingleZeros)) {
				LOG.warn("Not SDCFOR but " + g[i].getClass().getSimpleName());
				return null;
			}

			final ColGroupSDCSingleZeros gc = (ColGroupSDCSingleZeros) g[i];
			if(!gc._dict.eq(_dict)) {
				LOG.warn("Not same Dictionaries therefore not appending \n" + _dict + "\n\n" + gc._dict);
				return null;
			}
			sumRows += gc.getNumRows();
		}
		AOffset no = _indexes.appendN(Arrays.copyOf(g, g.length, AOffsetsGroup[].class), getNumRows());
		return create(_colIndexes, sumRows, _dict, no, null);

	}

	@Override
	public ICLAScheme getCompressionScheme() {
		return null;
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
