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
	protected int[] _indexes;

	// Helper Constructors
	protected ColGroupSDCSingleZeros() {
		super();
	}

	protected ColGroupSDCSingleZeros(int[] colIndices, int numRows, ADictionary dict, int[] indexes,
		int[] cachedCounts) {
		super(colIndices, numRows, dict, cachedCounts);
		_indexes = indexes;
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
		target.setNonZeros(_indexes.length * _colIndexes.length + target.getNonZeros());
	}

	@Override
	public void decompressToBlockUnSafe(MatrixBlock target, int rl, int ru, int offT, double[] values) {
		final int nCol = getNumCols();
		final int tCol = target.getNumColumns();
		final int offTCorrected = offT - rl;
		double[] c = target.getDenseBlockValues();
		int i = 0;
		while(i < _indexes.length && _indexes[i] < ru && _indexes[i] < rl) {
			i++;
		}
		for(; i < _indexes.length && _indexes[i] < ru; i++) {
			int rc = (offTCorrected + _indexes[i]) * tCol;
			for(int j = 0; j < nCol; j++)
				c[rc + _colIndexes[j]] += values[j];

		}

	}

	@Override
	public void decompressToBlock(MatrixBlock target, int[] colIndexTargets) {
		throw new NotImplementedException("Not Implemented");
	}

	@Override
	public void decompressColumnToBlock(MatrixBlock target, int colpos) {
		double[] c = target.getDenseBlockValues();
		double[] values = getValues();
		for(int i = 0; i < _indexes.length; i++)
			c[_indexes[i]] = values[colpos];
		target.setNonZeros(_indexes.length);
	}

	@Override
	public void decompressColumnToBlock(MatrixBlock target, int colpos, int rl, int ru) {
		throw new NotImplementedException("Not Implemented");
	}

	@Override
	public void decompressColumnToBlock(double[] c, int colpos, int rl, int ru) {
		double[] values = getValues();
		int off = 0;
		while(off < _indexes.length && _indexes[off] < rl)
			off++;
		for(; off < _indexes.length && _indexes[off] < ru; off++)
			c[_indexes[off] - rl] += values[colpos];
	}

	@Override
	public double get(int r, int c) {
		int ix = Arrays.binarySearch(_colIndexes, c);
		if(ix < 0)
			throw new RuntimeException("Column index " + c + " not in group.");

		int i = 0;
		while(i < _indexes.length && _indexes[i] < r)
			i++;
		if(i < _indexes.length && _indexes[i] == r)
			return _dict.getValue(ix);
		else
			return 0.0;

	}

	@Override
	public void countNonZerosPerRow(int[] rnnz, int rl, int ru) {
		int ncol = _colIndexes.length;
		// HACK!
		// since currently this method does not care if it returns 1 non zero or _numCols non zeros.
		// we return _num cols..

		int i = 0;
		while(_indexes[i] < rl) {
			i++;
		}
		for(; i < _indexes.length; i++) {
			rnnz[_indexes[i]] += ncol;
		}
	}

	@Override
	protected void computeRowSums(double[] c, boolean square, int rl, int ru, boolean mean) {
		double[] vals = _dict.sumAllRowsToDouble(square, _colIndexes.length);
		int i = 0;
		// TODO remove error correction from kahn.
		final int mult = (2 + (mean ? 1 : 0));
		while(_indexes[i] < rl) {
			i++;
		}
		for(; i < _indexes.length && _indexes[i] < ru; i++) {
			c[_indexes[i] * mult] += vals[0];
		}
	}

	@Override
	protected void computeRowMxx(MatrixBlock target, Builtin builtin, int rl, int ru) {
		double[] c = target.getDenseBlockValues();
		double[] vals = getValues();
		int ncol = getNumCols();
		int i = 0;
		while(_indexes[i] < rl) {
			i++;
		}
		for(; i < _indexes.length && _indexes[i] < ru; i++) {
			int idx = _indexes[i];
			for(int j = 0; j < ncol; j++)
				c[idx] = builtin.execute(c[idx], vals[j]);
		}
	}

	@Override
	public int[] getCounts(int[] counts) {
		counts[0] = _indexes.length;
		counts[1] = _numRows - _indexes.length;
		return counts;
	}

	@Override
	public int[] getCounts(int rl, int ru, int[] counts) {
		throw new NotImplementedException("Not Implemented");
	}

	@Override
	public void leftMultByMatrix(double[] a, double[] c, double[] values, int numRows, int numCols, int rl, int ru,
		int voff) {
		final int numVals = getNumValues();
		for(int i = rl, j = voff; i < ru; i++, j++) {
			double[] vals = preAggregate(a, j);
			postScaling(values, vals, c, numVals, i, numCols);
		}
	}

	@Override
	public void leftMultBySparseMatrix(SparseBlock sb, double[] c, double[] values, int numRows, int numCols, int row,
		double[] MaterializedRow) {
		final int numVals = getNumValues();
		double[] vals = preAggregateSparse(sb, row);
		postScaling(values, vals, c, numVals, row, numCols);
	}

	public double[] preAggregate(double[] a, int row) {
		// throw new NotImplementedException("Not Implemented");
		double[] vals = allocDVector(getNumValues(), true);
		if(row > 0) {
			int offT = _numRows * row;
			for(int i = 0; i < _indexes.length; i++)
				vals[0] += a[_indexes[i] + offT];
		}
		else
			for(int i = 0; i < _indexes.length; i++)
				vals[0] += a[_indexes[i]];

		return vals;
	}

	public double[] preAggregateSparse(SparseBlock sb, int row) {
		final int numVals = getNumValues();
		double[] vals = allocDVector(numVals, true);
		int[] sbIndexes = sb.indexes(row);
		double[] sparseV = sb.values(row);

		int sbP = sb.pos(row);
		int sbEnd = sb.size(row) + sb.pos(row);
		int i = 0;
		while(i < _indexes.length && sbP < sbEnd) {
			if(_indexes[i] == sbIndexes[sbP])
				vals[0] += sparseV[sbP++];
			if(sbP < sbEnd)
				while(i < _indexes.length && _indexes[i] < sbIndexes[sbP])
					i++;
			if(i < _indexes.length)
				while(sbP < sbEnd && sbIndexes[sbP] < _indexes[i])
					sbP++;
		}

		return vals;
	}

	@Override
	public long estimateInMemorySize() {
		return ColGroupSizes
			.estimateInMemorySizeSDCSingle(getNumCols(), _dict.size(), _numRows, _numRows - _indexes.length, isLossy());
	}

	@Override
	public void rightMultByVector(double[] vector, double[] c, int rl, int ru, double[] dictVals) {
		throw new NotImplementedException("Not Implemented Right Mult By Vector");
	}

	@Override
	public void rightMultByMatrix(int[] outputColumns, double[] preAggregatedB, double[] c, int thatNrColumns, int rl,
		int ru) {

		final int nCol = outputColumns.length;
		int i = 0;
		while(i < _indexes.length && _indexes[i] < ru && _indexes[i] < rl) {
			i++;
		}
		for(; i < _indexes.length && _indexes[i] < ru; i++) {
			int rc = _indexes[i] * thatNrColumns;
			for(int j = 0; j < nCol; j++)
				c[rc + _colIndexes[j]] += preAggregatedB[j];
		}

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
			ADictionary aDictionary = swapEntries(applyBinaryRowOp(op.fn, v, sparseSafe, left));
			return new ColGroupSDCSingle(_colIndexes, _numRows, aDictionary, _indexes, null);
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
		// write data
		out.writeInt(_indexes.length);
		for(int i = 0; i < _indexes.length; i++)
			out.writeInt(_indexes[i]);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		super.readFields(in);
		int length = in.readInt();
		_indexes = new int[length];
		for(int i = 0; i < length; i++)
			_indexes[i] = in.readInt();
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += 4;
		ret += _indexes.length * 4;
		return ret;
	}

	@Override
	public boolean sameIndexStructure(ColGroupValue that) {
		return that instanceof ColGroupSDCSingleZeros && ((ColGroupSDCSingleZeros) that)._indexes == _indexes;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s%5d ", "Indexes: ", this._indexes.length));
		sb.append(Arrays.toString(this._indexes));
		return sb.toString();
	}

	@Override
	public IPreAggregate preAggregateDDC(ColGroupDDC lhs) {
		IPreAggregate ag = PreAggregateFactory.ag(lhs.getNumValues());
		for(int i = 0; i < _indexes.length; i++) 
			ag.increment(lhs.getIndex(_indexes[i]));
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

		IPreAggregate ag = PreAggregateFactory.ag(retSize);
		int offL = 0;
		int offR = 0;
		final int[] l = lhs._indexes;
		final int[] r = this._indexes;
		while(offL < l.length && offR < r.length)
			if(l[offL] == r[offR]){
				ag.increment(lhs.getIndex(offL++));
				offR++;
			}
			else if(l[offL] < r[offR])
				offL++;
			else
				offR++;

		return ag;
	}

	@Override
	public IPreAggregate preAggregateSDCSingleZeros(ColGroupSDCSingleZeros lhs) {
		// we always know that there is only one value in each column group.
		int[] ret = new int[1];
		final int[] l = lhs._indexes;
		final int[] r = this._indexes;
		int offL = 0;
		int offR = 0;
		while(offL < l.length && offR < r.length)
			if(l[offL] == r[offR]) {
				ret[0]++;
				offL++;
				offR++;
			}
			else if(l[offL] < r[offR])
				offL++;
			else
				offR++;
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
}
