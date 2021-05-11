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
public class ColGroupSDCSingle extends ColGroupValue {
	private static final long serialVersionUID = -32043916423465004L;

	/**
	 * Sparse row indexes for the data
	 */
	protected AOffset _indexes;

	/**
	 * Constructor for serialization
	 * 
	 * @param numRows Number of rows contained
	 */
	protected ColGroupSDCSingle(int numRows) {
		super(numRows);
	}

	protected ColGroupSDCSingle(int[] colIndices, int numRows, ADictionary dict, int[] indexes, int[] cachedCounts) {
		super(colIndices, numRows, dict, cachedCounts);
		_indexes = OffsetFactory.create(indexes, numRows);
		_zeros = false;
	}

	protected ColGroupSDCSingle(int[] colIndices, int numRows, ADictionary dict, AOffset offsets, int[] cachedCounts) {
		super(colIndices, numRows, dict, cachedCounts);
		_indexes = offsets;
		_zeros = false;
	}

	@Override
	public CompressionType getCompType() {
		return CompressionType.SDC;
	}

	@Override
	public ColGroupType getColGroupType() {
		return ColGroupType.SDCSingle;
	}

	@Override
	public void decompressToBlockSafe(MatrixBlock target, int rl, int ru, int offT, double[] values) {
		decompressToBlockUnSafe(target, rl, ru, offT, values);
		target.setNonZeros(_numRows * _colIndexes.length + target.getNonZeros());
	}

	@Override
	public void decompressToBlockUnSafe(MatrixBlock target, int rl, int ru, int offT, double[] values) {

		final int nCol = _colIndexes.length;
		final int tCol = target.getNumColumns();
		final int offsetToDefault = values.length - nCol;
		double[] c = target.getDenseBlockValues();
		offT = offT * tCol;
		int i = rl;
		AIterator it = _indexes.getIterator();
		it.skipTo(rl);
		for(; i < ru && it.hasNext(); i++, offT += tCol) {
			if(it.value() == i) {
				for(int j = 0; j < nCol; j++)
					c[offT + _colIndexes[j]] += values[offsetToDefault + j];
				it.next();
			}
			else
				for(int j = 0; j < nCol; j++)
					c[offT + _colIndexes[j]] += values[j];
		}

		for(; i < ru; i++, offT += tCol)
			for(int j = 0; j < nCol; j++)
				c[offT + _colIndexes[j]] += values[j];
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int[] colIndexTargets) {
		throw new NotImplementedException("Not Implemented");
	}

	@Override
	public void decompressColumnToBlock(MatrixBlock target, int colpos) {
		final double[] c = target.getDenseBlockValues();
		final double[] values = getValues();
		final int offsetToDefault = _colIndexes.length;
		final AIterator it = _indexes.getIterator();
		final double v1 = values[offsetToDefault + colpos];
		final double v2 = values[colpos];

		int i = 0;
		for(; i < _numRows && it.hasNext(); i++) {
			if(it.value() == i) {
				c[i] += v1;
				it.next();
			}
			else
				c[i] += v2;
		}
		for(; i < _numRows; i++)
			c[i] += v2;

		target.setNonZeros(getNumberNonZeros() / _colIndexes.length);
	}

	@Override
	public void decompressColumnToBlock(MatrixBlock target, int colpos, int rl, int ru) {
		throw new NotImplementedException("Not Implemented");
	}

	@Override
	public void decompressColumnToBlock(double[] c, int colpos, int rl, int ru) {
		final int nCol = _colIndexes.length;
		final double[] values = getValues();
		final int offsetToDefault = values.length - nCol;
		final AIterator it = _indexes.getIterator();

		int offT = 0;
		int i = rl;
		it.skipTo(rl);

		for(; i < ru && it.hasNext(); i++, offT++) {
			if(it.value() == i) {
				it.next();
				c[offT] += values[offsetToDefault + colpos];
			}
			else
				c[offT] += values[colpos];
		}

		for(; i < ru; i++, offT++)
			c[offT] += values[colpos];
	}

	@Override
	public double get(int r, int c) {
		// find local column index
		int ix = Arrays.binarySearch(_colIndexes, c);
		if(ix < 0)
			throw new RuntimeException("Column index " + c + " not in group.");

		AIterator it = _indexes.getIterator();
		it.skipTo(r);
		if(it.value() == r)
			return _dict.getValue(_colIndexes.length + ix);
		else
			return _dict.getValue(ix);

	}

	@Override
	public void countNonZerosPerRow(int[] rnnz, int rl, int ru) {
		throw new NotImplementedException("Not Implemented");
	}

	@Override
	protected void computeRowSums(double[] c, boolean square, int rl, int ru, boolean mean) {

		// // pre-aggregate nnz per value tuple
		final double[] vals = _dict.sumAllRowsToDouble(square, _colIndexes.length);
		final AIterator it = _indexes.getIterator();

		int rix = rl;
		it.skipTo(rl);
		for(; rix < ru && it.hasNext(); rix++) {
			if(it.value() != rix)
				c[rix] += vals[0];
			else {
				c[rix] += vals[1];
				it.next();
			}
		}
		for(; rix < ru; rix++) {
			c[rix] += vals[0];
		}
	}

	@Override
	protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru) {

		final double[] vals = _dict.aggregateTuples(builtin, _colIndexes.length);
		final AIterator it = _indexes.getIterator();

		it.skipTo(rl);
		int rix = rl;

		for(; rix < ru && it.hasNext(); rix++) {
			if(it.value() != rix)
				c[rix] = builtin.execute(c[rix], vals[0]);
			else {
				c[rix] = builtin.execute(c[rix], vals[1]);
				it.next();
			}
		}
		for(; rix < ru; rix++) {
			c[rix] = builtin.execute(c[rix], vals[0]);
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

	public double[] preAggregate(double[] a, int row) {
		final int numVals = getNumValues();
		final double[] vals = allocDVector(numVals, true);
		final AIterator it = _indexes.getIterator();

		int i = 0;

		if(row > 0) {
			int offA = _numRows * row;
			for(; i < _numRows && it.hasNext(); i++, offA++)
				if(it.value() == i)
					vals[1] += a[offA];
				else
					vals[0] += a[offA];
			for(; i < _numRows; i++, offA++)
				vals[0] += a[offA];
		}
		else {
			for(; i < _numRows && it.hasNext(); i++)
				if(it.value() == i)
					vals[1] += a[i];
				else
					vals[0] += a[i];
			for(; i < _numRows; i++)
				vals[0] += a[i];
		}

		return vals;
	}

	public double[] preAggregateSparse(SparseBlock sb, int row) {
		final int numVals = getNumValues();
		final double[] vals = allocDVector(numVals, true);
		final int[] indexes = sb.indexes(row);
		final double[] sparseV = sb.values(row);
		final AIterator it = _indexes.getIterator();

		for(int i = sb.pos(row); i < sb.size(row) + sb.pos(row); i++) {
			it.skipTo(indexes[i]);
			if(it.value() == indexes[i]) {
				vals[0] += sparseV[i];
				it.next();
			}
			else
				vals[1] += sparseV[i];
		}
		return vals;
	}

	@Override
	public long estimateInMemorySize() {
		long size = ColGroupSizes.estimateInMemorySizeGroupValue(_colIndexes.length, getNumValues(), isLossy());
		size += _indexes.getInMemorySize();
		return size;
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		return new ColGroupSDCSingle(_colIndexes, _numRows, applyScalarOp(op), _indexes, getCachedCounts());
	}

	@Override
	public AColGroup binaryRowOp(BinaryOperator op, double[] v, boolean sparseSafe, boolean left) {
		return new ColGroupSDCSingle(_colIndexes, _numRows, applyBinaryRowOp(op.fn, v, true, left), _indexes,
			getCachedCounts());
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
		return that instanceof ColGroupSDCSingle && ((ColGroupSDCSingle) that)._indexes == _indexes;
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

		int i = 0;

		int row;
		for(; i < this._numRows && it.hasNext(); i++) {
			int col = lhs._data.getIndex(i);
			if(it.value() == i) {
				row = 1;
				it.next();
			}
			else
				row = 0;
			if(col < lhs.getNumValues())
				ag.increment(col + row * nCol);
		}
		row = 0;
		for(; i < this._numRows; i++) {
			int col = lhs._data.getIndex(i);
			if(col < lhs.getNumValues())
				ag.increment(col + row * nCol);
		}

		return ag;
	}

	@Override
	public IPreAggregate preAggregateSDC(ColGroupSDC lhs) {
		final int lhsNV = lhs.getNumValues();
		final int rhsNV = this.getNumValues();
		final int retSize = lhsNV * rhsNV;
		final int nCol = lhs.getNumValues();
		final IPreAggregate ag = PreAggregateFactory.ag(retSize);
		final int defL = lhsNV - 1;
		final AIterator lIt = lhs._indexes.getIterator();
		final AIterator rIt = _indexes.getIterator();

		int i = 0;
		int col;
		int row;
		for(; i < this._numRows && lIt.hasNext() && rIt.hasNext(); i++) {
			if(lIt.value() == i)
				col = lhs.getIndex(lIt.getDataIndexAndIncrement());
			else
				col = defL;
			if(rIt.value() == i) {
				row = 1;
				rIt.next();
			}
			else
				row = 0;
			ag.increment(col + row * nCol);
		}

		if(lIt.hasNext()) {
			row = 0;
			for(; i < this._numRows && lIt.hasNext(); i++) {
				if(lIt.value() == i)
					col = lhs.getIndex(lIt.getDataIndexAndIncrement());
				else
					col = defL;

				ag.increment(col + row * nCol);
			}
		}

		if(rIt.hasNext()) {
			col = defL;
			for(; i < this._numRows && rIt.hasNext(); i++) {
				if(rIt.value() == i) {
					row = 1;
					rIt.next();
				}
				else
					row = 0;
				ag.increment(col + row * nCol);
			}
		}

		ag.increment(defL, this._numRows - i);

		return ag;
	}

	@Override
	public IPreAggregate preAggregateSDCSingle(ColGroupSDCSingle lhs) {
		final int lhsNV = lhs.getNumValues();
		final int rhsNV = this.getNumValues();
		final int retSize = lhsNV * rhsNV;
		final int nCol = lhs.getNumValues();
		IPreAggregate ag = PreAggregateFactory.ag(retSize);
		;
		final AIterator lIt = lhs._indexes.getIterator();
		final AIterator rIt = _indexes.getIterator();

		int i = 0;
		int col;
		int row;
		for(; i < this._numRows && lIt.hasNext() && rIt.hasNext(); i++) {
			if(lIt.value() == i) {
				col = 1;
				lIt.next();
			}
			else
				col = 0;
			if(rIt.value() == i) {
				row = 1;
				rIt.next();
			}
			else
				row = 0;
			ag.increment(col + row * nCol);
		}

		if(lIt.hasNext()) {
			row = 1;
			for(; i < _numRows && lIt.hasNext(); i++) {
				if(lIt.value() == i) {
					col = 1;
					lIt.next();
				}
				else
					col = 0;

				ag.increment(col + row * nCol);
			}
		}

		if(rIt.hasNext()) {
			col = 1;
			for(; i < _numRows && rIt.hasNext(); i++) {
				if(rIt.value() == i) {
					row = 1;
					rIt.next();
				}
				else
					row = 0;
				ag.increment(col + row * nCol);
			}
		}

		ag.increment(0, _numRows - i);
		return ag;
	}

	@Override
	public IPreAggregate preAggregateSDCZeros(ColGroupSDCZeros lhs) {
		throw new NotImplementedException("Not supported pre aggregate of :" + lhs.getClass().getSimpleName() + " in "
			+ this.getClass().getSimpleName());
	}

	@Override
	public IPreAggregate preAggregateSDCSingleZeros(ColGroupSDCSingleZeros lhs) {
		throw new NotImplementedException("Not supported pre aggregate of :" + lhs.getClass().getSimpleName() + " in "
			+ this.getClass().getSimpleName());
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
		// final AIterator it = _indexes.getIterator();
		// final int offsetToDefault = this.getNumValues() - 1;
		// final int nCol = that._colIndexes.length;

		// int i = 0;

		// for(; i < _numRows && it.hasNext(); i++) {
		// int to = (it.value() == i) ? 1 : 0;
		// that._dict.addToEntry(ret, that.getIndex(i), to, nCol);
		// }

		// for(; i < _numRows; i++)
		// that._dict.addToEntry(ret, that.getIndex(i), 0, nCol);

		// return ret;
	}

	@Override
	public Dictionary preAggregateThatSDCStructure(ColGroupSDC that, Dictionary ret, boolean preModified) {
		final AIterator itThat = that._indexes.getIterator();
		final AIterator itThis = _indexes.getIterator();
		final int nCol = that._colIndexes.length;
		// final int defThat = that.getNumValues() * nCol - nCol;

		if(preModified) {
			while(itThat.hasNext() && itThis.hasNext()) {
				if(itThat.value() == itThis.value()) {
					final int fr = that.getIndex(itThat.getDataIndexAndIncrement());
					that._dict.addToEntry(ret, fr, 1, nCol);
				}
				else if(itThat.value() < itThis.value())
					itThat.next();
				else{
					itThis.next();
					// that._dict.addToEntry(ret, defThat, 0, nCol);
				}
			}
		}
		else {
			throw new NotImplementedException();
		}
		return ret;
	}

	@Override
	public Dictionary preAggregateThatSDCZerosStructure(ColGroupSDCZeros that, Dictionary ret) {
		throw new NotImplementedException();
	}

	@Override
	public Dictionary preAggregateThatSDCSingleZerosStructure(ColGroupSDCSingleZeros that, Dictionary ret) {
		throw new NotImplementedException();
	}

	@Override
	public Dictionary preAggregateThatSDCSingleStructure(ColGroupSDCSingle that, Dictionary ret, boolean preModified) {
		final AIterator itThat = that._indexes.getIterator();
		final AIterator itThis = _indexes.getIterator();
		final int nCol = that._colIndexes.length;

		if(preModified) {

			while(itThat.hasNext() && itThis.hasNext()) {
				if(itThat.value() == itThis.value()) {
					itThat.next();
					itThis.next();
					that._dict.addToEntry(ret, 1, 0, nCol);
				}
				else if(itThat.value() < itThis.value()) {
					itThat.next();
					// that._dict.addToEntry(ret, 0, 1, nCol);
				}
				else
					itThis.next();
			}

			return ret;
		}
		else {
			int i = 0;
			for(; i < _numRows && itThat.hasNext() && itThis.hasNext(); i++) {
				int to = 0;
				if(itThis.value() == i) {
					itThis.next();
					to = 1;
				}
				int fr = 0;
				if(itThat.value() == i) {
					itThat.next();
					fr = 1;
				}
				that._dict.addToEntry(ret, fr, to, nCol);
			}

			if(itThat.hasNext()) {
				for(; i < _numRows && itThat.hasNext(); i++) {
					int fr = 0;
					if(itThat.value() == i) {
						itThat.next();
						fr = 1;
					}
					that._dict.addToEntry(ret, fr, 1, nCol);
				}
			}

			if(itThis.hasNext()) {
				for(; i < _numRows && itThis.hasNext(); i++) {
					int to = 0;
					if(itThis.value() == i) {
						itThis.next();
						to = 1;
					}
					that._dict.addToEntry(ret, 1, to, nCol);
				}
			}

			for(; i < _numRows; i++)
				that._dict.addToEntry(ret, 1, 1, nCol);

			return ret;
		}

	}

	@Override
	public MatrixBlock preAggregate(MatrixBlock m, int rl, int ru) {
		throw new NotImplementedException();
	}
}
