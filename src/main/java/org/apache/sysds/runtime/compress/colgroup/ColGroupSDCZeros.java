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
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
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
public class ColGroupSDCZeros extends ColGroupValue {
	private static final long serialVersionUID = -32143916423465004L;

	/**
	 * Sparse row indexes for the data
	 */
	protected AOffset _indexes;

	/**
	 * Pointers to row indexes in the dictionary. Note the dictionary has one extra entry.
	 */
	protected AMapToData _data;

	/**
	 * Constructor for serialization
	 * 
	 * @param numRows Number of rows contained
	 */
	protected ColGroupSDCZeros(int numRows) {
		super(numRows);
	}

	protected ColGroupSDCZeros(int[] colIndices, int numRows, ADictionary dict, int[] indexes, AMapToData data,
		int[] cachedCounts) {
		super(colIndices, numRows, dict, cachedCounts);
		_indexes = OffsetFactory.create(indexes, numRows);
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
	protected void decompressToBlockUnSafeDenseDictionary(MatrixBlock target, int rl, int ru, int offT,
		double[] values) {
		final int nCol = _colIndexes.length;
		final int tCol = target.getNumColumns();
		final int offTCorrected = offT - rl;
		final double[] c = target.getDenseBlockValues();
		AIterator it = _indexes.getIterator(rl);
		offT = offT * tCol;
		while(it.hasNext() && it.value() < ru) {
			int rc = (offTCorrected + it.value()) * tCol;
			int offC = getIndex(it.getDataIndexAndIncrement()) * nCol;
			for(int j = 0; j < nCol; j++) {
				c[rc + _colIndexes[j]] += values[offC + j];
			}
		}
	}

	@Override
	protected void decompressToBlockUnSafeSparseDictionary(MatrixBlock target, int rl, int ru, int offT,
		SparseBlock sb) {

		final int tCol = target.getNumColumns();
		final int offTCorrected = offT - rl;
		final double[] c = target.getDenseBlockValues();
		AIterator it = _indexes.getIterator(rl);
		while(it.hasNext() && it.value() < ru) {
			final int rc = (offTCorrected + it.value()) * tCol;
			final int dictIndex = getIndex(it.getDataIndexAndIncrement());
			if(sb.isEmpty(dictIndex))
				continue;

			final int apos = sb.pos(dictIndex);
			final int alen = sb.size(dictIndex) + apos;
			final double[] avals = sb.values(dictIndex);
			final int[] aix = sb.indexes(dictIndex);
			for(int j = apos; j < alen; j++)
				c[rc + _colIndexes[aix[j]]] += avals[j];
		}
	}

	// @Override
	// public void decompressToBlock(MatrixBlock target, int[] colIndexTargets) {
	// throw new NotImplementedException();
	// }

	// @Override
	// public void decompressColumnToBlock(MatrixBlock target, int colpos) {
	// final double[] c = target.getDenseBlockValues();
	// final double[] values = getValues();
	// final AIterator it = _indexes.getIterator();
	// while(it.hasNext())
	// c[it.value()] += values[getIndex(it.getDataIndexAndIncrement()) * _colIndexes.length + colpos];
	// target.setNonZeros(getNumberNonZeros() / _colIndexes.length);
	// }

	// @Override
	// public void decompressColumnToBlock(MatrixBlock target, int colpos, int rl, int ru) {
	// throw new NotImplementedException();
	// }

	// @Override
	// public void decompressColumnToBlock(double[] c, int colpos, int rl, int ru) {
	// final double[] values = getValues();
	// final AIterator it = _indexes.getIterator(rl);
	// while(it.hasNext() && it.value() < ru)
	// c[it.value() - rl] += values[getIndex(it.getDataIndexAndIncrement()) * _colIndexes.length + colpos];
	// }

	@Override
	public double get(int r, int c) {
		int ix = Arrays.binarySearch(_colIndexes, c);
		if(ix < 0)
			throw new RuntimeException("Column index " + c + " not in group.");

		final AIterator it = _indexes.getIterator(r);
		if(it.value() == r)
			return _dict.getValue(getIndex(it.getDataIndex()) * _colIndexes.length + ix);
		else
			return 0.0;

	}

	@Override
	public void countNonZerosPerRow(int[] rnnz, int rl, int ru) {
		final int nCol = _colIndexes.length;
		final AIterator it = _indexes.getIterator(rl);
		while(it.hasNext() && it.value() < ru) {
			rnnz[it.value() - rl] += nCol;
			it.next();
		}
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

		for(; rix < ru; rix++) {
			c[rix] = builtin.execute(c[rix], 0);
		}
	}

	@Override
	public int[] getCounts(int[] counts) {
		return getCounts(0, _numRows, counts);
	}

	@Override
	public int[] getCounts(int rl, int ru, int[] counts) {

		int i = rl;
		final AIterator it = _indexes.getIterator(rl);

		int zeros = 0;
		while(it.hasNext() && it.value() < ru) {
			int oldI = i;
			i = it.value();
			zeros += i - oldI - 1;
			counts[_data.getIndex(it.getDataIndexAndIncrement())]++;
		}

		counts[counts.length - 1] += zeros + ru - i;

		return counts;
	}

	public int getIndex(int r) {
		return _data.getIndex(r);
	}

	// @Override
	// public double[] preAggregate(double[] a, int aRows) {
	// final double[] vals = allocDVector(getNumValues(), true);
	// final AIterator it = _indexes.getIterator();
	// if(aRows > 0) {
	// final int offT = _numRows * aRows;
	// while(it.hasNext()) {
	// final int i = it.value();
	// vals[getIndex(it.getDataIndexAndIncrement())] += a[i + offT];
	// }
	// }
	// else
	// while(it.hasNext()) {
	// final int i = it.value();
	// vals[getIndex(it.getDataIndexAndIncrement())] += a[i];
	// }

	// return vals;
	// }

	// @Override
	// public double[] preAggregateSparse(SparseBlock sb, int row) {
	// final double[] vals = allocDVector(getNumValues(), true);
	// final int[] sbIndexes = sb.indexes(row);
	// final double[] sparseV = sb.values(row);
	// final AIterator it = _indexes.getIterator();
	// final int sbEnd = sb.size(row) + sb.pos(row);

	// int sbP = sb.pos(row);

	// while(it.hasNext() && sbP < sbEnd) {
	// if(it.value() == sbIndexes[sbP])
	// vals[getIndex(it.getDataIndexAndIncrement())] += sparseV[sbP++];
	// if(sbP < sbEnd)
	// it.skipTo(sbIndexes[sbP]);
	// while(sbP < sbEnd && sbIndexes[sbP] < it.value())
	// sbP++;
	// }

	// return vals;
	// }

	@Override
	protected void preAggregate(MatrixBlock m, MatrixBlock preAgg, int rl, int ru) {
		if(m.isInSparseFormat())
			preAggregateSparse(m.getSparseBlock(), preAgg, rl, ru);
		else
			preAggregateDense(m, preAgg, rl, ru);
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
				preAV[offOut + getIndex(it.getDataIndexAndIncrement())] += mV[offLeft + i];
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
			for(int j = apos; j < alen; j++) {
				it.skipTo(aix[j]);
				if(it.value() == aix[j])
					preAV[offOut + _data.getIndex(it.getDataIndexAndIncrement())] += avals[j];
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
	public AColGroup binaryRowOp(BinaryOperator op, double[] v, boolean sparseSafe, boolean left) {
		if(sparseSafe)
			return new ColGroupSDCZeros(_colIndexes, _numRows, applyBinaryRowOp(op, v, sparseSafe, left), _indexes,
				_data, getCachedCounts());
		else
			return new ColGroupSDC(_colIndexes, _numRows, applyBinaryRowOp(op, v, sparseSafe, left), _indexes, _data,
				getCachedCounts());
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
	public boolean sameIndexStructure(ColGroupCompressed that) {
		return that instanceof ColGroupSDCZeros && ((ColGroupSDCZeros) that)._indexes == _indexes &&
			((ColGroupSDCZeros) that)._data == _data;
	}

	@Override
	public int getIndexStructureHash() {
		return _indexes.hashCode() + _data.hashCode();
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

	// @Override
	// public IPreAggregate preAggregateDDC(ColGroupDDC lhs) {
	// final int rhsNV = this.getNumValues();
	// final int nCol = lhs.getNumValues();
	// final int retSize = nCol * rhsNV;
	// final IPreAggregate ag = PreAggregateFactory.ag(retSize);
	// final AIterator it = _indexes.getIterator();

	// while(it.hasNext()) {
	// final int col = lhs._data.getIndex(it.value());
	// final int row = getIndex(it.getDataIndexAndIncrement());
	// ag.increment(col + row * nCol);
	// }
	// return ag;
	// }

	// @Override
	// public IPreAggregate preAggregateSDC(ColGroupSDC lhs) {
	// final int rhsNV = this.getNumValues();
	// final int nCol = lhs.getNumValues();

	// final int defL = nCol - 1;
	// final int retSize = nCol * rhsNV;

	// IPreAggregate ag = PreAggregateFactory.ag(retSize);

	// AIterator lIt = lhs._indexes.getIterator();
	// AIterator rIt = this._indexes.getIterator();

	// while(lIt.hasNext() && rIt.hasNext())
	// if(lIt.value() == rIt.value())
	// ag.increment(
	// lhs.getIndex(lIt.getDataIndexAndIncrement()) + getIndex(rIt.getDataIndexAndIncrement()) * nCol);
	// else if(lIt.value() > rIt.value())
	// ag.increment(defL + getIndex(rIt.getDataIndexAndIncrement()) * nCol);
	// else
	// lIt.next();

	// while(rIt.hasNext())
	// ag.increment(defL + getIndex(rIt.getDataIndexAndIncrement()) * nCol);

	// return ag;
	// }

	// @Override
	// public IPreAggregate preAggregateSDCSingle(ColGroupSDCSingle lhs) {
	// throw new NotImplementedException("Not supported pre aggregate of :" + lhs.getClass().getSimpleName() + " in "
	// + this.getClass().getSimpleName());
	// }

	// @Override
	// public IPreAggregate preAggregateSDCZeros(ColGroupSDCZeros lhs) {
	// final int rhsNV = this.getNumValues();
	// final int nCol = lhs.getNumValues();
	// final int retSize = nCol * rhsNV;

	// final IPreAggregate ag = PreAggregateFactory.ag(retSize);

	// final AIterator lIt = lhs._indexes.getIterator();
	// final AIterator rIt = _indexes.getIterator();

	// while(lIt.hasNext() && rIt.hasNext())
	// if(lIt.value() == rIt.value())
	// ag.increment(
	// lhs.getIndex(lIt.getDataIndexAndIncrement()) + getIndex(rIt.getDataIndexAndIncrement()) * nCol);
	// else if(lIt.value() < rIt.value())
	// lIt.next();
	// else
	// rIt.next();

	// return ag;
	// }

	// @Override
	// public IPreAggregate preAggregateSDCSingleZeros(ColGroupSDCSingleZeros lhs) {
	// final int rhsNV = this.getNumValues();
	// final int nCol = lhs.getNumValues();
	// final int retSize = nCol * rhsNV;
	// final IPreAggregate ag = PreAggregateFactory.ag(retSize);
	// final AIterator lIt = lhs._indexes.getIterator();
	// final AIterator rIt = _indexes.getIterator();

	// while(lIt.hasNext() && rIt.hasNext())
	// if(lIt.value() == rIt.value()) {
	// ag.increment(getIndex(rIt.getDataIndexAndIncrement()));
	// lIt.next();
	// }
	// else if(lIt.value() < rIt.value())
	// lIt.next();
	// else
	// rIt.next();

	// return ag;
	// }

	// @Override
	// public IPreAggregate preAggregateOLE(ColGroupOLE lhs) {
	// final int NVR = this.getNumValues();
	// final int NVL = lhs.getNumValues();
	// final int retSize = NVR * NVL;
	// final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
	// final IPreAggregate ag = PreAggregateFactory.ag(retSize);

	// for(int kl = 0; kl < NVL; kl++) {
	// final AIterator rIt = _indexes.getIterator();
	// final int bOffL = lhs._ptr[kl];
	// final int bLenL = lhs.len(kl);
	// for(int bixL = 0, offL = 0, sLenL = 0; rIt.hasNext() && bixL < bLenL; bixL += sLenL + 1, offL += blksz) {
	// sLenL = lhs._data[bOffL + bixL];
	// for(int i = 1; rIt.hasNext() && i <= sLenL; i++) {
	// final int col = offL + lhs._data[bOffL + bixL + i];
	// rIt.skipTo(col);
	// if(rIt.value() == col)
	// ag.increment(kl + getIndex(rIt.getDataIndexAndIncrement()) * NVL);

	// }
	// }
	// }
	// return ag;
	// }

	// @Override
	// public IPreAggregate preAggregateRLE(ColGroupRLE lhs) {
	// throw new NotImplementedException("Not supported pre aggregate of :" + lhs.getClass().getSimpleName() + " in "
	// + this.getClass().getSimpleName());
	// }

	@Override
	public Dictionary preAggregateThatDDCStructure(ColGroupDDC that, Dictionary ret) {
		final AIterator itThis = _indexes.getIterator();
		final int nCol = that._colIndexes.length;

		while(itThis.hasNext()) {
			final int fr = that._data.getIndex(itThis.value());
			final int to = getIndex(itThis.getDataIndexAndIncrement());
			that._dict.addToEntry(ret, fr, to, nCol);
		}

		return ret;
	}

	@Override
	public Dictionary preAggregateThatSDCStructure(ColGroupSDC that, Dictionary ret, boolean preModified) {
		throw new NotImplementedException();
	}

	@Override
	public Dictionary preAggregateThatSDCZerosStructure(ColGroupSDCZeros that, Dictionary ret) {
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
		return ret;
	}

	@Override
	public Dictionary preAggregateThatSDCSingleZerosStructure(ColGroupSDCSingleZeros that, Dictionary ret) {
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
		return ret;
	}

	@Override
	public Dictionary preAggregateThatSDCSingleStructure(ColGroupSDCSingle that, Dictionary re, boolean preModified) {
		throw new NotImplementedException();
	}

}
