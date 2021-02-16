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
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.pre.IPreAggregate;
import org.apache.sysds.runtime.compress.colgroup.pre.PreAggregateFactory;
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
	protected int[] _indexes;

	/**
	 * Pointers to row indexes in the dictionary. Note the dictionary has one extra entry.
	 */
	protected char[] _data;

	// Helper Constructors
	protected ColGroupSDCZeros() {
		super();
	}

	protected ColGroupSDCZeros(int[] colIndices, int numRows, ADictionary dict, int[] indexes, char[] data,
		int[] cachedCounts) {
		super(colIndices, numRows, dict, cachedCounts);
		_indexes = indexes;
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
	public void decompressToBlockSafe(MatrixBlock target, int rl, int ru, int offT, double[] values) {
		decompressToBlockUnSafe(target, rl, ru, offT, values);
		target.setNonZeros(_indexes.length * _colIndexes.length + target.getNonZeros());
	}

	@Override
	public void decompressToBlockUnSafe(MatrixBlock target, int rl, int ru, int offT, double[] values) {
		final int nCol = _colIndexes.length;
		final int tCol = target.getNumColumns();
		final int offTCorrected = offT - rl;
		double[] c = target.getDenseBlockValues();
		int i = 0;
		while(i < _indexes.length && _indexes[i] < rl) {
			i++;
		}
		for(; i < _indexes.length && _indexes[i] < ru; i++) {
			int rc = (offTCorrected + _indexes[i]) * tCol;
			int offC = _data[i] * nCol;
			for(int j = 0; j < nCol; j++) {
				c[rc + _colIndexes[j]] += values[offC + j];
			}
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
			c[_indexes[i]] += values[_data[i] * _colIndexes.length + colpos];
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
			c[_indexes[off] - rl] += values[_data[off] * _colIndexes.length + colpos];
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
			return _dict.getValue(_data[i] * _colIndexes.length + ix);
		else
			return 0.0;

	}

	@Override
	public void countNonZerosPerRow(int[] rnnz, int rl, int ru) {
		int ncol = _colIndexes.length;
		int i = 0;
		while(i < _indexes.length && _indexes[i] < rl) {
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
		while(i < _indexes.length && _indexes[i] < rl) {
			i++;
		}
		for(; i < _indexes.length && _indexes[i] < ru; i++) {
			c[_indexes[i] * mult] += vals[_data[i]];
		}
	}

	@Override
	protected void computeRowMxx(MatrixBlock target, Builtin builtin, int rl, int ru) {
		// throw new NotImplementedException("Not Implemented Row Sums");
		double[] c = target.getDenseBlockValues();
		double[] vals = getValues();
		int i = 0;
		while(i < _indexes.length && _indexes[i] < rl) {
			i++;
		}
		for(; i < _indexes.length && _indexes[i] < ru; i++) {
			int idx = _indexes[i];
			int off = _data[i] * _colIndexes.length;
			for(int j = 0; j < _colIndexes.length; j++)
				c[idx] = builtin.execute(c[idx], vals[off + j]);
		}

	}

	@Override
	public int[] getCounts(int[] counts) {
		for(char v : _data) {
			counts[v]++;
		}
		counts[counts.length - 1] = _numRows - _data.length;
		return counts;
	}

	@Override
	public int[] getCounts(int rl, int ru, int[] counts) {
		throw new NotImplementedException("Not Implemented");
	}

	@Override
	public void leftMultByMatrix(double[] a, double[] c, double[] values, int numRows, int numCols, int rl, int ru,
		int voff) {
		final int numVals = values.length / _colIndexes.length;
		for(int i = rl, j = voff; i < ru; i++, j++) {
			double[] vals = preAggregate(a, j);
			postScaling(values, vals, c, numVals, i, numCols);
		}
	}

	@Override
	public void leftMultBySparseMatrix(SparseBlock sb, double[] c, double[] values, int numRows, int numCols, int row,
		double[] MaterializedRow) {
		final int numVals = values.length / _colIndexes.length;
		double[] vals = preAggregateSparse(sb, row);
		postScaling(values, vals, c, numVals, row, numCols);
	}

	@Override
	public double[] preAggregate(double[] a, int aRows) {
		double[] vals = allocDVector(getNumValues(), true);
		if(aRows > 0) {
			int offT = _numRows * aRows;
			for(int i = 0; i < _indexes.length; i++)
				vals[_data[i]] += a[_indexes[i] + offT];
		}
		else
			for(int i = 0; i < _indexes.length; i++)
				vals[_data[i]] += a[_indexes[i]];

		return vals;
	}

	@Override
	public double[] preAggregateSparse(SparseBlock sb, int row) {
		double[] vals = allocDVector(getNumValues(), true);
		int[] sbIndexes = sb.indexes(row);
		double[] sparseV = sb.values(row);

		int sbP = sb.pos(row);
		int sbEnd = sb.size(row) + sb.pos(row);
		int i = 0;
		while(i < _indexes.length && sbP < sbEnd) {
			if(_indexes[i] == sbIndexes[sbP])
				vals[_data[i++]] += sparseV[sbP++];
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
			.estimateInMemorySizeSDC(getNumCols(), _dict.size(), _numRows, _numRows - _data.length, isLossy());
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
			int offset = _data[i] * outputColumns.length;
			for(int j = 0; j < nCol; j++)
				c[rc + outputColumns[j]] += preAggregatedB[offset + j];
		}
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
			return new ColGroupSDCZeros(_colIndexes, _numRows, applyBinaryRowOp(op.fn, v, sparseSafe, left), _indexes,
				_data, getCachedCounts());
		else
			return new ColGroupSDC(_colIndexes, _numRows, applyBinaryRowOp(op.fn, v, sparseSafe, left), _indexes, _data,
				getCachedCounts());
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		// write data
		out.writeInt(_indexes.length);
		for(int i = 0; i < _indexes.length; i++)
			out.writeChar(_data[i]);
		for(int i = 0; i < _indexes.length; i++)
			out.writeInt(_indexes[i]);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		super.readFields(in);
		int length = in.readInt();
		_data = new char[length];
		_indexes = new int[length];
		for(int i = 0; i < length; i++)
			_data[i] = in.readChar();
		for(int i = 0; i < length; i++)
			_indexes[i] = in.readInt();
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += 4;
		ret += _data.length * 2;
		ret += _indexes.length * 4;
		return ret;
	}

	@Override
	public boolean sameIndexStructure(ColGroupValue that) {
		return that instanceof ColGroupSDCZeros && ((ColGroupSDCZeros) that)._indexes == _indexes;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s%5d ", "Indexes: ", this._indexes.length));
		sb.append(Arrays.toString(this._indexes));
		if(_data != null) {
			sb.append(String.format("\n%15s%5d ", "Data: ", this._data.length));
			sb.append("[");
			for(char c : this._data) {
				sb.append((int) c);
				sb.append(" ");
			}
			sb.append("]");
		}
		return sb.toString();
	}

	@Override
	public IPreAggregate preAggregateDDC(ColGroupDDC lhs) {
		final int rhsNV = this.getNumValues();
		final int nCol = lhs.getNumValues();
		final int retSize = nCol * rhsNV;

		IPreAggregate ag = PreAggregateFactory.ag(retSize);

		final int[] sdcIndexes = this._indexes;
		final char[] sdcData = this._data;
		for(int i = 0; i < sdcIndexes.length; i++) {
			int col = lhs.getIndex(sdcIndexes[i]);
			int row = sdcData[i];
			if(col < lhs.getNumValues())
				ag.increment(col + row * nCol);
		}
		return ag;
	}

	@Override
	public IPreAggregate preAggregateSDC(ColGroupSDC lhs) {
		final int rhsNV = this.getNumValues();
		final int nCol = lhs.getNumValues();

		final int defL = nCol - 1;
		final int retSize = nCol * rhsNV;

		IPreAggregate ag = PreAggregateFactory.ag(retSize);
		int offL = 0;
		int offR = 0;
		final int[] l = lhs._indexes;
		final int[] r = this._indexes;
		final char[] ld = lhs._data;
		final char[] rd = this._data;

		while(offL < l.length && offR < r.length)
			if(l[offL] == r[offR])
				ag.increment(ld[offL++] + rd[offR++] * nCol);
			else if(l[offL] > r[offR])
				ag.increment(defL + rd[offR++] * nCol);
			else
				offL++;

		while(offR < r.length)
			ag.increment(defL + rd[offR++] * nCol);

		return ag;
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
		final char[] ld = lhs._data;
		final char[] rd = this._data;
		while(offL < l.length && offR < r.length)
			if(l[offL] == r[offR])
				ag.increment(ld[offL++] + rd[offR++] * nCol);
			else if(l[offL] < r[offR])
				offL++;
			else
				offR++;

		return ag;
	}

	@Override
	public IPreAggregate preAggregateSDCSingleZeros(ColGroupSDCSingleZeros lhs) {
		final int rhsNV = this.getNumValues();
		final int nCol = lhs.getNumValues();
		final int retSize = nCol * rhsNV;

		IPreAggregate ag = PreAggregateFactory.ag(retSize);
		int offL = 0;
		int offR = 0;
		final int[] l = lhs._indexes;
		final int[] r = this._indexes;
		final char[] rd = this._data;

		while(offL < l.length && offR < r.length)
			if(l[offL] == r[offR])
				ag.increment(rd[offR++]);
			else if(l[offL] < r[offR])
				offL++;
			else
				offR++;

		return ag;
	}

	@Override
	public IPreAggregate preAggregateOLE(ColGroupOLE lhs) {
		final int NVR = this.getNumValues();
		final int NVL = lhs.getNumValues();
		final int retSize = NVR * NVL;
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		IPreAggregate ag = PreAggregateFactory.ag(retSize);
		final int[] r = this._indexes;
		final char[] rd = this._data;

		for(int kl = 0; kl < NVL; kl++) {
			int offR = 0;
			final int bOffL = lhs._ptr[kl];
			final int bLenL = lhs.len(kl);
			for(int bixL = 0, offL = 0, sLenL = 0; offR < r.length && bixL < bLenL; bixL += sLenL + 1, offL += blksz) {
				sLenL = lhs._data[bOffL + bixL];
				for(int i = 1; offR < r.length && i <= sLenL; i++) {
					final int col = offL + lhs._data[bOffL + bixL + i];
					while(offR < r.length && r[offR] < col)
						offR++;
					if(offR < r.length && r[offR] == col)
						ag.increment(kl + rd[offR++] * NVL);

				}
			}
		}
		return ag;
	}

	@Override
	public IPreAggregate preAggregateRLE(ColGroupRLE lhs) {
		throw new NotImplementedException("Not supported pre aggregate of :" + lhs.getClass().getSimpleName() + " in "
			+ this.getClass().getSimpleName());
	}
}
