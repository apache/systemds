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
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.colgroup.pre.IPreAggregate;
import org.apache.sysds.runtime.compress.colgroup.pre.PreAggregateFactory;
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
public class ColGroupSDCSingleZeros extends ColGroupValue {
	private static final long serialVersionUID = -32043916423425004L;

	/**
	 * Sparse row indexes for the data
	 */
	protected AOffset _indexes;

	/**
	 * Constructor for serialization
	 * 
	 * @param numRows Number of rows contained
	 */
	protected ColGroupSDCSingleZeros(int numRows) {
		super(numRows);
	}

	protected ColGroupSDCSingleZeros(int[] colIndices, int numRows, ADictionary dict, int[] indexes,
		int[] cachedCounts) {
		super(colIndices, numRows, dict, cachedCounts);
		_indexes = OffsetFactory.create(indexes, numRows);
		_zeros = true;
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
	public void decompressToBlockSafe(MatrixBlock target, int rl, int ru, int offT, double[] values) {
		decompressToBlockUnSafe(target, rl, ru, offT, values);
		target.setNonZeros(_indexes.getSize() * _colIndexes.length + target.getNonZeros());
	}

	@Override
	public void decompressToBlockUnSafe(MatrixBlock target, int rl, int ru, int offT, double[] values) {
		final int nCol = _colIndexes.length;
		final int tCol = target.getNumColumns();
		final int offTCorrected = offT - rl;
		final double[] c = target.getDenseBlockValues();

		AIterator it = _indexes.getIterator();
		it.skipTo(rl);

		while(it.hasNext() && it.value() < ru) {
			int rc = (offTCorrected + it.value()) * tCol;
			for(int j = 0; j < nCol; j++) {
				c[rc + _colIndexes[j]] += values[j];
			}
		}
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int[] colIndexTargets) {
		throw new NotImplementedException("Not Implemented");
	}

	@Override
	public void decompressColumnToBlock(MatrixBlock target, int colpos) {
		final double[] c = target.getDenseBlockValues();
		final double[] values = getValues();
		final AIterator it = _indexes.getIterator();
		while(it.hasNext()) {
			c[it.value()] += values[_colIndexes.length + colpos];
			it.next();
		}
		target.setNonZeros(getNumberNonZeros() / _colIndexes.length);
	}

	@Override
	public void decompressColumnToBlock(MatrixBlock target, int colpos, int rl, int ru) {
		throw new NotImplementedException("Not Implemented");
	}

	@Override
	public void decompressColumnToBlock(double[] c, int colpos, int rl, int ru) {
		final double[] values = getValues();
		final AIterator it = _indexes.getIterator();
		it.skipTo(rl);
		while(it.hasNext() && it.value() < ru) {
			c[it.value() - rl] += values[colpos];
			it.next();
		}
	}

	@Override
	public double get(int r, int c) {
		int ix = Arrays.binarySearch(_colIndexes, c);
		if(ix < 0)
			throw new RuntimeException("Column index " + c + " not in group.");

		final AIterator it = _indexes.getIterator();
		it.skipTo(r);
		if(it.value() == r)
			return _dict.getValue(ix);
		else
			return 0.0;

	}

	@Override
	public void countNonZerosPerRow(int[] rnnz, int rl, int ru) {
		final int nCol = _colIndexes.length;
		final AIterator it = _indexes.getIterator();
		it.skipTo(rl);
		while(it.hasNext() && it.value() < ru) {
			rnnz[it.value() - rl] += nCol;
			it.next();
		}
	}

	@Override
	protected void computeRowSums(double[] c, boolean square, int rl, int ru, boolean mean) {
		final double vals = _dict.sumAllRowsToDouble(square, _colIndexes.length)[0];
		final AIterator it = _indexes.getIterator();
		it.skipTo(rl);
		while(it.hasNext() && it.value() < ru) {
			c[it.value()] += vals;
			it.next();
		}

	}

	@Override
	protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru) {
		final double vals = _dict.aggregateTuples(builtin, _colIndexes.length)[0];
		final AIterator it = _indexes.getIterator();
		it.skipTo(rl);
		int rix = rl;
		for(; rix < ru && it.hasNext(); rix++) {
			if(it.value() != rix)
				c[rix] = builtin.execute(c[rix], 0);
			else {
				c[rix] = builtin.execute(c[rix], vals);
				it.next();
			}
		}

		for(; rix < ru; rix++) {
			c[rix] = builtin.execute(c[rix], 0);
		}
	}

	@Override
	public int[] getCounts(int[] counts) {
		counts[0] = _indexes.getSize();
		counts[1] = _numRows - counts[0];
		return counts;
	}

	@Override
	public int[] getCounts(int rl, int ru, int[] counts) {
		final AIterator it = _indexes.getIterator();
		it.skipTo(rl);

		while(it.hasNext() && it.value() < ru) {
			it.next();
			counts[0]++;
		}

		counts[1] = ru - rl - counts[0];

		return counts;
	}

	public double[] preAggregate(double[] a, int aRows) {
		final double[] vals = allocDVector(getNumValues(), true);
		final AIterator it = _indexes.getIterator();
		if(aRows > 0) {
			final int offT = _numRows * aRows;
			while(it.hasNext()) {
				final int i = it.value();
				vals[0] += a[i + offT];
				it.next();
			}
		}
		else
			while(it.hasNext()) {
				final int i = it.value();
				vals[0] += a[i];
				it.next();
			}

		return vals;
	}

	public double[] preAggregateSparse(SparseBlock sb, int row) {
		final double[] vals = allocDVector(getNumValues(), true);
		final int[] sbIndexes = sb.indexes(row);
		final double[] sparseV = sb.values(row);
		final AIterator it = _indexes.getIterator();
		final int sbEnd = sb.size(row) + sb.pos(row);

		int sbP = sb.pos(row);

		while(it.hasNext() && sbP < sbEnd) {
			if(it.value() == sbIndexes[sbP])
				vals[0] += sparseV[sbP++];
			if(sbP < sbEnd)
				it.skipTo(sbIndexes[sbP]);
			while(sbP < sbEnd && sbIndexes[sbP] < it.value())
				sbP++;
		}

		return vals;
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
			ADictionary aDictionary = swapEntries(applyScalarOp(op, val0, getNumCols()));
			// ADictionary aDictionary = applyScalarOp(op, val0, getNumCols());
			return new ColGroupSDCSingle(_colIndexes, _numRows, aDictionary, _indexes, null);
		}
	}

	@Override
	public AColGroup binaryRowOp(BinaryOperator op, double[] v, boolean sparseSafe, boolean left) {
		if(sparseSafe)
			return new ColGroupSDCSingleZeros(_colIndexes, _numRows, applyBinaryRowOp(op.fn, v, sparseSafe, left),
				_indexes, getCachedCounts());
		else {
			ADictionary aDictionary = applyBinaryRowOp(op.fn, v, sparseSafe, left);
			return new ColGroupSDCSingle(_colIndexes, _numRows, aDictionary, _indexes, getCachedCounts());
		}
	}

	private ADictionary swapEntries(ADictionary aDictionary) {
		double[] values = aDictionary.getValues().clone();
		double[] swap = new double[_colIndexes.length];
		System.arraycopy(values, 0, swap, 0, _colIndexes.length);
		System.arraycopy(values, _colIndexes.length, values, 0, _colIndexes.length);
		System.arraycopy(swap, 0, values, _colIndexes.length, _colIndexes.length);
		return new Dictionary(values);
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
	public boolean sameIndexStructure(ColGroupCompressed that) {
		return that instanceof ColGroupSDCSingleZeros && ((ColGroupSDCSingleZeros) that)._indexes == _indexes;
	}

	@Override
	public int getIndexStructureHash() {
		return _indexes.hashCode();
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s ", "Indexes: "));
		sb.append(_indexes.toString());
		return sb.toString();
	}

	@Override
	public IPreAggregate preAggregateDDC(ColGroupDDC lhs) {
		final int rhsNV = this.getNumValues();
		final int nCol = lhs.getNumValues();
		final int retSize = nCol * rhsNV;
		final IPreAggregate ag = PreAggregateFactory.ag(retSize);
		final AIterator it = _indexes.getIterator();

		while(it.hasNext()) {
			final int col = lhs._data.getIndex(it.value());
			ag.increment(col);
		}
		return ag;
	}

	@Override
	public IPreAggregate preAggregateSDC(ColGroupSDC lhs) {
		throw new NotImplementedException("Not supported pre aggregate of :" + lhs.getClass().getSimpleName() + " in "
			+ this.getClass().getSimpleName());
	}

	@Override
	public IPreAggregate preAggregateSDCSingle(ColGroupSDCSingle lhs) {
		throw new NotImplementedException("Not supported pre aggregate of :" + lhs.getClass().getSimpleName() + " in "
			+ this.getClass().getSimpleName());
	}

	@Override
	public IPreAggregate preAggregateSDCZeros(ColGroupSDCZeros lhs) {
		final int rhsNV = this.getNumValues();
		final int nCol = lhs.getNumValues();
		final int retSize = nCol * rhsNV;

		final IPreAggregate ag = PreAggregateFactory.ag(retSize);
		final AIterator lIt = lhs._indexes.getIterator();
		final AIterator rIt = this._indexes.getIterator();

		while(lIt.hasNext() && rIt.hasNext())
			if(lIt.value() == rIt.value()) {
				ag.increment(lhs.getIndex(lIt.getDataIndexAndIncrement()));
				rIt.next();
			}
			else if(lIt.value() < rIt.value())
				lIt.next();
			else
				rIt.next();

		return ag;
	}

	@Override
	public IPreAggregate preAggregateSDCSingleZeros(ColGroupSDCSingleZeros lhs) {
		// we always know that there is only one value in each column group.
		int[] ret = new int[1];
		final AIterator lIt = lhs._indexes.getIterator();
		final AIterator rIt = this._indexes.getIterator();
		while(lIt.hasNext() && rIt.hasNext())
			if(lIt.value() == rIt.value()) {
				ret[0]++;
				lIt.next();
				rIt.next();
			}
			else if(lIt.value() < rIt.value())
				lIt.next();
			else
				rIt.next();

		return PreAggregateFactory.ag(ret);
	}

	@Override
	public IPreAggregate preAggregateOLE(ColGroupOLE lhs) {
		throw new NotImplementedException("Not supported pre aggregate of :" + lhs.getClass().getSimpleName() + " in "
			+ this.getClass().getSimpleName());
	}

	@Override
	public IPreAggregate preAggregateRLE(ColGroupRLE lhs) {
		throw new NotImplementedException("Not supported pre aggregate of :" + lhs.getClass().getSimpleName() + " in "
			+ this.getClass().getSimpleName());
	}

	@Override
	public Dictionary preAggregateThatDDCStructure(ColGroupDDC that, Dictionary ret) {
		throw new NotImplementedException();
	}

	@Override
	public Dictionary preAggregateThatSDCStructure(ColGroupSDC that, Dictionary ret, boolean preModified) {
		throw new NotImplementedException();
	}

	@Override
	public Dictionary preAggregateThatSDCZerosStructure(ColGroupSDCZeros that, Dictionary ret) {
		throw new NotImplementedException();
	}

	@Override
	public Dictionary preAggregateThatSDCSingleZerosStructure(ColGroupSDCSingleZeros that, Dictionary ret) {
		final AIterator itThat = that._indexes.getIterator();
		final AIterator itThis = that._indexes.getIterator();
		final int nCol = that._colIndexes.length;

		while(itThat.hasNext()) {
			final int v = itThat.value();
			if(v == itThis.skipTo(v))
				that._dict.addToEntry(ret, 0, 0, nCol);
			itThat.next();
		}

		return ret;
	}

	@Override
	public Dictionary preAggregateThatSDCSingleStructure(ColGroupSDCSingle that, Dictionary ret, boolean preModified) {
		throw new NotImplementedException();
	}

	@Override
	public MatrixBlock preAggregate(MatrixBlock m, int rl, int ru) {
		throw new NotImplementedException();
	}
}
