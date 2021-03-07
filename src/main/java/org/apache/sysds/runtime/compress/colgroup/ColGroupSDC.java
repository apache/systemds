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
import org.apache.sysds.runtime.compress.colgroup.mapping.IMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
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
public class ColGroupSDC extends ColGroupValue {
	private static final long serialVersionUID = -12043916423465004L;

	/**
	 * Sparse row indexes for the data
	 */
	protected int[] _indexes;
	/**
	 * Pointers to row indexes in the dictionary. Note the dictionary has one extra entry.
	 */
	protected IMapToData _data;

	// Helper Constructors
	protected ColGroupSDC() {
		super();
	}

	protected ColGroupSDC(int[] colIndices, int numRows, ADictionary dict, int[] indexes, IMapToData data,
		int[] cachedCounts) {
		super(colIndices, numRows, dict, cachedCounts);
		_indexes = indexes;
		_data = data;
		_zeros = false;
	}

	@Override
	public CompressionType getCompType() {
		return CompressionType.SDC;
	}

	@Override
	public ColGroupType getColGroupType() {
		return ColGroupType.SDC;
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
		int off = 0;
		int i = rl;
		while(off < _indexes.length && _indexes[off] < rl)
			off++;
		for(; i < ru && off < _indexes.length; i++, offT += tCol) {
			if(_indexes[off] == i) {
				int offset = _data.getIndex(off) * nCol;
				for(int j = 0; j < nCol; j++)
					c[offT + _colIndexes[j]] += values[offset + j];
				off++;
			}
			else
				for(int j = 0; j < nCol; j++)
					c[offT + _colIndexes[j]] += values[offsetToDefault + j];
		}

		for(; i < ru; i++, offT += tCol)
			for(int j = 0; j < nCol; j++)
				c[offT + _colIndexes[j]] += values[offsetToDefault + j];

	}

	@Override
	public void decompressToBlock(MatrixBlock target, int[] colIndexTargets) {
		throw new NotImplementedException("Not Implemented");
	}

	@Override
	public void decompressColumnToBlock(MatrixBlock target, int colPos) {
		double[] c = target.getDenseBlockValues();
		double[] values = getValues();
		double defaultVal = values[values.length - _colIndexes.length + colPos];
		int off = 0;
		int i = 0;
		for(; i < _numRows && off < _indexes.length; i++) {
			if(_indexes[off] == i)
				c[i] += values[_data.getIndex(off++) * _colIndexes.length + colPos];
			else
				c[i] += defaultVal;
		}
		for(; i < _numRows; i++)
			c[i] += defaultVal;
		target.setNonZeros(_indexes.length);
	}

	@Override
	public void decompressColumnToBlock(MatrixBlock target, int colpos, int rl, int ru) {
		throw new NotImplementedException("Not Implemented");
	}

	@Override
	public void decompressColumnToBlock(double[] c, int colpos, int rl, int ru) {
		final int nCol = _colIndexes.length;
		final double[] values = getValues();
		final int offsetToDefault = values.length - nCol + colpos;
		int offT = 0;
		int off = 0;
		int i = rl;
		while(off < _indexes.length && _indexes[off] < rl)
			off++;
		for(; i < ru && off < _indexes.length; i++, offT++) {
			if(_indexes[off] == i) {
				int offset = _data.getIndex(off) * nCol;
				c[offT] += values[offset + colpos];
				off++;
			}
			else
				c[offT] += values[offsetToDefault];
		}

		for(; i < ru; i++, offT++)
			c[offT] += values[offsetToDefault];
	}

	@Override
	public double get(int r, int c) {
		// find local column index
		int ix = Arrays.binarySearch(_colIndexes, c);
		if(ix < 0)
			throw new RuntimeException("Column index " + c + " not in group.");

		// // get value
		int index = getIndex(r, 0);
		if(index < _indexes.length && _indexes[index] == r)
			return _dict.getValue(_data.getIndex(index) * _colIndexes.length + ix);
		else
			return _dict.getValue(getNumValues() * _colIndexes.length - _colIndexes.length + ix);
	}

	@Override
	public void countNonZerosPerRow(int[] rnnz, int rl, int ru) {
		throw new NotImplementedException("Not Implemented");
	}

	@Override
	protected void computeRowSums(double[] c, boolean square, int rl, int ru, boolean mean) {
		final int numVals = getNumValues();
		// // pre-aggregate nnz per value tuple
		double[] vals = _dict.sumAllRowsToDouble(square, _colIndexes.length);

		final int mult = (2 + (mean ? 1 : 0));
		int index = 0;
		for(int rix = rl; rix < ru; rix++) {
			index = getIndex(rix, index);
			if(index == _indexes.length || _indexes[index] != rix)
				c[rix * mult] += vals[numVals - 1];
			else
				c[rix * mult] += vals[_data.getIndex(index)];
		}
	}

	@Override
	protected void computeRowMxx(MatrixBlock c, Builtin builtin, int rl, int ru) {

		int ncol = getNumCols();
		double[] dictionary = getValues();
		double[] cVals = c.getDenseBlockValues();
		int index = 0;
		int offsetToDefault = dictionary.length - _colIndexes.length;
		if(_indexes.length == 0) {
			for(int i = rl; i < ru; i++)
				for(int j = 0; j < ncol; j++)
					cVals[i] = builtin.execute(cVals[i], dictionary[j]);
		}
		else if(_data == null)
			for(int i = rl; i < ru; i++) {
				index = getIndex(i, index);
				for(int j = 0; j < ncol; j++)
					if(index == _indexes.length || _indexes[index] != i)
						cVals[i] = builtin.execute(cVals[i], dictionary[offsetToDefault + j]);
					else
						cVals[i] = builtin.execute(cVals[i], dictionary[j]);
			}
		else
			for(int i = rl; i < ru; i++) {
				index = getIndex(i, index);
				for(int j = 0; j < ncol; j++)
					if(index == _indexes.length || _indexes[index] != i)
						cVals[i] = builtin.execute(cVals[i], dictionary[offsetToDefault + j]);
					else
						cVals[i] = builtin.execute(cVals[i], dictionary[getIndex(index) * _colIndexes.length + j]);
			}
	}

	@Override
	public int[] getCounts(int[] counts) {
		return getCounts(0, _numRows, counts);
	}

	@Override
	public int[] getCounts(int rl, int ru, int[] counts) {
		final int def = counts.length - 1;
		int off = 0;
		int i = rl;
		while(off < _indexes.length && _indexes[off] < i)
			off++;

		for(; i < ru && off < _indexes.length; i++) {
			if(off < _indexes.length && i == _indexes[off])
				counts[_data.getIndex(off++)]++;
			else
				counts[def]++;
		}

		if(i < ru)
			counts[def] += ru -i;

		return counts;
	}

	public int getIndex(int r){
		return _data.getIndex(r);
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
		final int numVals = getNumValues();
		double[] vals = allocDVector(numVals, true);
		int off = 0;
		int i = 0;
		final int def = numVals - 1;

		if(row > 0) {
			int offA = _numRows * row;
			for(; i < _numRows && off < _indexes.length; i++, offA++) {
				if(_indexes[off] == i) {
					vals[getIndex(off)] += a[offA];
					off++;
				}
				else
					vals[def] += a[offA];
			}
			for(; i < _numRows; i++, offA++)
				vals[def] += a[offA];
		}
		else {
			for(; i < _numRows && off < _indexes.length; i++)
				if(_indexes[off] == i) {
					vals[getIndex(off)] += a[i];
					off++;
				}
				else
					vals[def] += a[i];

			for(; i < _numRows; i++)
				vals[def] += a[i];
		}

		return vals;
	}

	private int getIndex(int row, int lastIndex) {
		if(lastIndex == _indexes.length) {
			return _indexes.length;
		}
		while(lastIndex < _indexes.length && row > _indexes[lastIndex]) {
			lastIndex++;
		}
		return lastIndex;
	}

	@Override
	public double[] preAggregateSparse(SparseBlock sb, int row) {
		final int numVals = getNumValues();
		double[] vals = allocDVector(numVals, true);
		int[] indexes = sb.indexes(row);
		double[] sparseV = sb.values(row);
		int index = 0;
		for(int i = sb.pos(row); i < sb.size(row) + sb.pos(row); i++) {
			index = getIndex(indexes[i], index);
			if(index == _indexes.length || _indexes[index] != indexes[i])
				vals[numVals - 1] += sparseV[i];
			else
				vals[getIndex(index)] += sparseV[i];
		}
		return vals;
	}

	@Override
	public long estimateInMemorySize() {
		return ColGroupSizes
			.estimateInMemorySizeSDC(getNumCols(), _dict.size(), _numRows, _numRows - _indexes.length, isLossy());
	}

	@Override
	public void rightMultByVector(double[] vector, double[] c, int rl, int ru, double[] dictVals) {
		throw new NotImplementedException("Not Implemented Right Mult By Vector");
	}

	@Override
	public void rightMultByMatrix(int[] outputColumns, double[] preAggregatedB, double[] c, int thatNrColumns, int rl,
		int ru) {
		final int nCol = outputColumns.length;
		int off = 0;
		int i = rl;
		while(off < _indexes.length && _indexes[off] < rl)
			off++;
		int offsetToDefault = getNumValues() * outputColumns.length - outputColumns.length;
		for(; i < ru && off < _indexes.length; i++) {
			int rc = i * thatNrColumns;
			if(_indexes[off] == i) {
				int offset = getIndex(off) * outputColumns.length;
				for(int j = 0; j < nCol; j++) {
					c[rc + outputColumns[j]] += preAggregatedB[offset + j];
				}
				off++;
			}
			else {
				for(int j = 0; j < nCol; j++) {
					c[rc + outputColumns[j]] += preAggregatedB[offsetToDefault + j];
				}
			}
		}

		for(; i < ru; i++) {
			int rc = i * thatNrColumns;
			for(int j = 0; j < nCol; j++) {
				c[rc + outputColumns[j]] += preAggregatedB[offsetToDefault + j];
			}
		}
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		return new ColGroupSDC(_colIndexes, _numRows, applyScalarOp(op), _indexes, _data, getCachedCounts());
	}

	@Override
	public AColGroup binaryRowOp(BinaryOperator op, double[] v, boolean sparseSafe, boolean left) {
		return new ColGroupSDC(_colIndexes, _numRows, applyBinaryRowOp(op.fn, v, true, left), _indexes, _data,
			getCachedCounts());
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		// write data
		out.writeInt(_indexes.length);
		_data.write(out);
		for(int i = 0; i < _indexes.length; i++)
			out.writeInt(_indexes[i]);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		super.readFields(in);
		int length = in.readInt();
		_indexes = new int[length];
		_data = MapToFactory.readIn(length, in, getNumValues());
		for(int i = 0; i < length; i++)
			_indexes[i] = in.readInt();
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += 4;
		ret += MapToFactory.getExactSizeOnDisk(_indexes.length, getNumValues());
		ret += _indexes.length * 4;
		return ret;
	}

	@Override
	public boolean sameIndexStructure(ColGroupValue that) {
		// TODO add such that if the column group switched from Zeros type it also matches.
		return that instanceof ColGroupSDC && ((ColGroupSDC) that)._indexes == _indexes && ((ColGroupSDC) that)._data == _data;
	}

	@Override
	public int getIndexStructureHash(){
		return _data.hashCode() + _indexes.hashCode();
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s%5d ", "Indexes: ", this._indexes.length));
		sb.append(Arrays.toString(this._indexes));
		sb.append(_data);
		return sb.toString();
	}

	@Override
	public IPreAggregate preAggregateDDC(ColGroupDDC lhs) {
		final int nCol = lhs.getNumValues();
		final int rhsNV = this.getNumValues();
		final int retSize = nCol * rhsNV;
		IPreAggregate ag = PreAggregateFactory.ag(retSize);

		final int[] sdcIndexes = this._indexes;
		final int offsetToDefault = this.getNumValues() - 1;

		int off = 0;
		int i = 0;

		int row;
		for(; i < this._numRows && off < sdcIndexes.length; i++) {
			int col = lhs.getIndex(i);
			if(sdcIndexes[off] == i) {
				row = getIndex(off);
				off++;
			}
			else
				row = offsetToDefault;
			if(col < lhs.getNumValues())
				ag.increment(col + row * nCol);
		}
		row = offsetToDefault;
		for(; i < this._numRows; i++) {
			int col = lhs.getIndex(i);
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
		IPreAggregate ag = PreAggregateFactory.ag(retSize);

		int offL = 0;
		int offR = 0;

		final int defL = lhsNV - 1;
		final int defR = rhsNV - 1;

		final int[] l = lhs._indexes;
		final int[] r = this._indexes;

		int i = 0;
		int col;
		int row;
		for(; i < this._numRows && offL < l.length && offR < r.length; i++) {
			if(l[offL] == i)
				col = lhs.getIndex(offL++);
			else
				col = defL;
			if(r[offR] == i)
				row = this.getIndex(offR++);
			else
				row = defR;
			ag.increment(col + row * nCol);
		}

		if(offL < l.length) {
			row = defR;
			for(; i < this._numRows && offL < l.length; i++) {
				if(l[offL] == i)
					col = lhs.getIndex(offL++);
				else
					col = defL;

				ag.increment(col + row * nCol);
			}
		}

		if(offR < r.length) {
			col = defL;
			for(; i < this._numRows && offR < r.length; i++) {
				if(r[offR] == i)
					row = this.getIndex(offR++);
				else
					row = defR;
				ag.increment(col + row * nCol);
			}
		}

		ag.increment(defL + defR * nCol, this._numRows - i);

		return ag;
	}

	@Override
	public IPreAggregate preAggregateSDCSingle(ColGroupSDCSingle lhs) {
		final int lhsNV = lhs.getNumValues();
		final int rhsNV = this.getNumValues();
		final int retSize = lhsNV * rhsNV;
		final int nCol = lhs.getNumValues();
		IPreAggregate ag = PreAggregateFactory.ag(retSize);

		int offL = 0;
		int offR = 0;

		final int defL = lhsNV - 1;
		final int defR = rhsNV - 1;

		final int[] l = lhs._indexes;
		final int[] r = this._indexes;

		int i = 0;
		int col;
		int row;
		for(; i < this._numRows && offL < l.length && offR < r.length; i++) {
			if(l[offL] == i) {
				col = defL;
				offL++;
			}
			else
				col = 0;
			if(r[offR] == i)
				row = this.getIndex(offR++);
			else
				row = defR;
			ag.increment(col + row * nCol);
		}

		if(offL < l.length) {
			row = defR;
			for(; i < this._numRows && offL < l.length; i++) {
				if(l[offL] == i) {
					col = defL;
					offL++;
				}
				else
					col = 0;

				ag.increment(col + row * nCol);
			}
		}

		if(offR < r.length) {
			for(; i < this._numRows && offR < r.length; i++) {
				if(r[offR] == i)
					row = this.getIndex(offR++);
				else
					row = defR;
				ag.increment(row * nCol);
			}
		}

		ag.increment(defR * nCol, this._numRows - i);

		return ag;
	}

	@Override
	public IPreAggregate preAggregateSDCZeros(ColGroupSDCZeros lhs) {
		final int rhsNV = this.getNumValues();
		final int nCol = lhs.getNumValues();

		final int defR = (rhsNV - 1) * nCol;
		final int retSize = nCol * rhsNV;

		IPreAggregate ag = PreAggregateFactory.ag(retSize);
		int offL = 0;
		int offR = 0;
		final int[] l = lhs._indexes;
		final int[] r = this._indexes;

		while(offL < l.length && offR < r.length)
			if(l[offL] == r[offR])
				ag.increment(lhs.getIndex(offL++) + this.getIndex(offR++) * nCol);
			else if(l[offL] > r[offR])
				offR++;
			else
				ag.increment(lhs.getIndex(offL++) + defR);

		while(offL < l.length)
			ag.increment(lhs.getIndex(offL++) + defR);
		return ag;
	}

	@Override
	public IPreAggregate preAggregateSDCSingleZeros(ColGroupSDCSingleZeros lhs) {
		throw new NotImplementedException("Not supported pre aggregate of :" + lhs.getClass().getSimpleName() + " in "
			+ this.getClass().getSimpleName());
	}

	@Override
	public IPreAggregate preAggregateOLE(ColGroupOLE lhs) {
		final int NVR = this.getNumValues();
		final int NVL = lhs.getNumValues();
		final int retSize = NVR * NVL;
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		IPreAggregate ag = PreAggregateFactory.ag(retSize);
		final int[] r = this._indexes;
		final int defR = (NVR - 1) * NVL;

		for(int kl = 0; kl < NVL; kl++) {
			int offR = 0;
			final int bOffL = lhs._ptr[kl];
			final int bLenL = lhs.len(kl);
			for(int bixL = 0, offL = 0, sLenL = 0; bixL < bLenL; bixL += sLenL + 1, offL += blksz) {
				sLenL = lhs._data[bOffL + bixL];
				for(int i = 1; i <= sLenL; i++) {
					final int col = offL + lhs._data[bOffL + bixL + i];
					while(offR < r.length &&  r[offR] < col)
						offR++;
					if(offR < r.length && r[offR] == col)
						ag.increment(kl + this.getIndex(offR++) * NVL);
					else 
						ag.increment(kl + defR);
					
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
