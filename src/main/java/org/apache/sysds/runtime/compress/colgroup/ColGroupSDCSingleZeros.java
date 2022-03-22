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
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffsetIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
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

	/**
	 * Constructor for serialization
	 * 
	 * @param numRows Number of rows contained
	 */
	protected ColGroupSDCSingleZeros(int numRows) {
		super(numRows);
	}

	private ColGroupSDCSingleZeros(int[] colIndices, int numRows, ADictionary dict, AOffset offsets,
		int[] cachedCounts) {
		super(colIndices, numRows, dict, offsets, cachedCounts);
		_zeros = true;
	}

	protected static AColGroup create(int[] colIndices, int numRows, ADictionary dict, AOffset offsets,
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
		else
			decompressToDenseBlockDenseDictionary(db, rl, ru, offR, offC, values, it);
	}

	@Override
	public void decompressToDenseBlockDenseDictionary(DenseBlock db, int rl, int ru, int offR, int offC, double[] values,
		AIterator it) {
		final int last = _indexes.getOffsetToLast();
		if(it == null || it.value() >= ru || rl > last)
			return;
		else if(ru > _indexes.getOffsetToLast())
			decompressToDenseBlockDenseDictionaryPost(db, rl, ru, offR, offC, values, it);
		else {
			if(_colIndexes.length == 1 && db.getDim(1) == 1)
				decompressToDenseBlockDenseDictionaryPreSingleColOutContiguous(db, rl, ru, offR, offC, values, it);
			else
				decompressToDenseBlockDenseDictionaryPre(db, rl, ru, offR, offC, values, it);
			_indexes.cacheIterator(it, ru);
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
		int offC, double[] values, AIterator it) {
		// final int nCol = _colIndexes.length;
		final double[] c = db.values(0);
		// final int off = db.pos(row);
		final double v = values[0];
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

			preAgg[0] = ret;
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
		final double val0 = op.executeScalar(0);
		final boolean isSparseSafeOp = op.sparseSafe || val0 == 0;
		final ADictionary nDict = _dict.applyScalarOp(op);
		if(isSparseSafeOp)
			return new ColGroupSDCSingleZeros(_colIndexes, _numRows, nDict, _indexes, getCachedCounts());
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
			return new ColGroupSDCSingleZeros(_colIndexes, _numRows, ret, _indexes, getCachedCounts());
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
	protected void computeRowProduct(double[] c, int rl, int ru, double[] preAgg) {
		for(int i = 0; i < c.length; i++)
			c[i] = 0;
	}

	@Override
	protected void computeColProduct(double[] c, int nRows) {
		for(int i = 0; i < c.length; i++)
			c[i] = 0;
	}

	@Override
	public double getCost(ComputationCostEstimator e, int nRows) {
		final int nVals = getNumValues();
		final int nCols = getNumCols();
		final int nRowsScanned = getCounts()[0];
		return e.getCost(nRows, nRowsScanned, nCols, nVals, _dict.getSparsity());
	}

	@Override 
	protected int getIndexesSize(){
		return getCounts()[0];
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
