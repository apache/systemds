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
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
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
public class ColGroupSDC extends ColGroupValue {
	private static final long serialVersionUID = -12043916423465004L;

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
	protected ColGroupSDC(int numRows) {
		super(numRows);
	}

	protected ColGroupSDC(int[] colIndices, int numRows, ADictionary dict, int[] indexes, AMapToData data,
		int[] cachedCounts) {
		super(colIndices, numRows, dict, cachedCounts);
		_indexes = OffsetFactory.create(indexes, numRows);
		_data = data;
		_zeros = false;
	}

	protected ColGroupSDC(int[] colIndices, int numRows, ADictionary dict, AOffset offsets, AMapToData data,
		int[] cachedCounts) {
		super(colIndices, numRows, dict, cachedCounts);
		_indexes = offsets;
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
		target.setNonZeros(getNumberNonZeros());
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
				int offset = _data.getIndex(it.getDataIndexAndIncrement()) * nCol;
				for(int j = 0; j < nCol; j++)
					c[offT + _colIndexes[j]] += values[offset + j];
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
		final double[] c = target.getDenseBlockValues();
		final double[] values = getValues();
		final double defaultVal = values[values.length - _colIndexes.length + colPos];
		int i = 0;
		final AIterator it = _indexes.getIterator();
		for(; i < _numRows && it.hasNext(); i++) {
			if(it.value() == i)
				c[i] += values[_data.getIndex(it.getDataIndexAndIncrement()) * _colIndexes.length + colPos];
			else
				c[i] += defaultVal;
		}
		for(; i < _numRows; i++)
			c[i] += defaultVal;

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
		final int offsetToDefault = values.length - nCol + colpos;
		final AIterator it = _indexes.getIterator();

		int offT = 0;
		int i = rl;
		it.skipTo(rl);

		for(; i < ru && it.hasNext(); i++, offT++) {
			if(it.value() == i) {
				int offset = _data.getIndex(it.getDataIndexAndIncrement()) * nCol;
				c[offT] += values[offset + colpos];
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
		AIterator it = _indexes.getIterator();
		it.skipTo(r);
		if(it.value() == r)
			return _dict.getValue(_data.getIndex(it.getDataIndex()) * _colIndexes.length + ix);
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

		int rix = rl;
		AIterator it = _indexes.getIterator();
		it.skipTo(rl);
		for(; rix < ru && it.hasNext(); rix++) {
			if(it.value() != rix)
				c[rix] += vals[numVals - 1];
			else {
				c[rix] += vals[_data.getIndex(it.getDataIndexAndIncrement())];
			}
		}
		for(; rix < ru; rix++) {
			c[rix] += vals[numVals - 1];
		}

	}

	@Override
	protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru) {

		final int numVals = getNumValues();

		double[] vals = _dict.aggregateTuples(builtin, _colIndexes.length);

		AIterator it = _indexes.getIterator();
		it.skipTo(rl);

		int rix = rl;
		for(; rix < ru && it.hasNext(); rix++) {
			if(it.value() != rix)
				c[rix] = builtin.execute(c[rix], vals[numVals - 1]);
			else
				c[rix] = builtin.execute(c[rix], vals[_data.getIndex(it.getDataIndexAndIncrement())]);
		}

		for(; rix < ru; rix++) {
			c[rix] = builtin.execute(c[rix], vals[numVals - 1]);
		}

	}

	@Override
	public int[] getCounts(int[] counts) {
		return getCounts(0, _numRows, counts);
	}

	@Override
	public int[] getCounts(int rl, int ru, int[] counts) {
		final int def = counts.length - 1;

		int i = rl;
		AIterator it = _indexes.getIterator();
		it.skipTo(rl);

		for(; i < ru && it.hasNext(); i++) {
			if(i == it.value())
				counts[_data.getIndex(it.getDataIndexAndIncrement())]++;
			else
				counts[def]++;
		}

		if(i < ru)
			counts[def] += ru - i;

		return counts;
	}

	public int getIndex(int r) {
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
		final double[] vals = allocDVector(numVals, true);
		final AIterator it = _indexes.getIterator();
		final int def = numVals - 1;

		int i = 0;

		if(row > 0) {
			int offA = _numRows * row;
			for(; i < _numRows && it.hasNext(); i++, offA++)
				if(it.value() == i)
					vals[_data.getIndex(it.getDataIndexAndIncrement())] += a[offA];
				else
					vals[def] += a[offA];
			for(; i < _numRows; i++, offA++)
				vals[def] += a[offA];
		}
		else {
			for(; i < _numRows && it.hasNext(); i++)
				if(it.value() == i)
					vals[_data.getIndex(it.getDataIndexAndIncrement())] += a[i];
				else
					vals[def] += a[i];
			for(; i < _numRows; i++)
				vals[def] += a[i];
		}
		return vals;
	}

	@Override
	public double[] preAggregateSparse(SparseBlock sb, int row) {
		final int numVals = getNumValues();
		final double[] vals = allocDVector(numVals, true);
		final int[] indexes = sb.indexes(row);
		final double[] sparseV = sb.values(row);
		final AIterator it = _indexes.getIterator();

		for(int i = sb.pos(row); i < sb.size(row) + sb.pos(row); i++) {
			it.skipTo(indexes[i]);
			if(it.value() == indexes[i])
				vals[getIndex(it.getDataIndexAndIncrement())] += sparseV[i];
			else
				vals[numVals - 1] += sparseV[i];
		}
		return vals;
	}

	@Override
	public long estimateInMemorySize() {
		long size = ColGroupSizes.estimateInMemorySizeGroupValue(_colIndexes.length, getNumValues(), isLossy());
		size += _indexes.getInMemorySize();
		size += _data.getInMemorySize();
		return size;
	}

	@Override
	public void rightMultByVector(double[] vector, double[] c, int rl, int ru, double[] dictVals) {
		throw new NotImplementedException("Not Implemented Right Mult By Vector");
	}

	@Override
	public void rightMultByMatrix(int[] outputColumns, double[] preAggregatedB, double[] c, int thatNrColumns, int rl,
		int ru) {
		final int nCol = outputColumns.length;
		final int offsetToDefault = getNumValues() * outputColumns.length - outputColumns.length;
		final AIterator it = _indexes.getIterator();

		it.skipTo(rl);
		int i = rl;

		for(; i < ru && it.hasNext(); i++) {
			int rc = i * thatNrColumns;
			if(it.value() == i) {
				int offset = getIndex(it.getDataIndexAndIncrement()) * outputColumns.length;
				for(int j = 0; j < nCol; j++) {
					c[rc + outputColumns[j]] += preAggregatedB[offset + j];
				}
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
		_indexes.write(out);
		_data.write(out);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		super.readFields(in);
		_indexes = OffsetFactory.readIn(in);
		_data = MapToFactory.readIn(in, getNumValues());
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += _data.getExactSizeOnDisk();
		ret += _indexes.getExactSizeOnDisk();
		return ret;
	}

	@Override
	public boolean sameIndexStructure(ColGroupCompressed that) {
		// TODO add such that if the column group switched from Zeros type it also matches.
		return that instanceof ColGroupSDC && ((ColGroupSDC) that)._indexes == _indexes &&
			((ColGroupSDC) that)._data == _data;
	}

	@Override
	public int getIndexStructureHash() {
		return _data.hashCode() + _indexes.hashCode();
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s ", "Indexes: "));
		sb.append(_indexes.toString());
		sb.append(String.format("\n%15s ", "Data: "));
		sb.append(_data.toString());
		return sb.toString();
	}

	@Override
	public IPreAggregate preAggregateDDC(ColGroupDDC lhs) {
		final int nCol = lhs.getNumValues();
		final int rhsNV = this.getNumValues();
		final int retSize = nCol * rhsNV;
		final IPreAggregate ag = PreAggregateFactory.ag(retSize);
		final AIterator it = _indexes.getIterator();
		final int offsetToDefault = this.getNumValues() - 1;

		int i = 0;

		int row;
		for(; i < this._numRows && it.hasNext(); i++) {
			int col = lhs.getIndex(i);
			if(it.value() == i)
				row = getIndex(it.getDataIndexAndIncrement());
			else
				row = offsetToDefault;
			ag.increment(col + row * nCol);
		}
		row = offsetToDefault;
		for(; i < this._numRows; i++) {
			int col = lhs.getIndex(i);
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

		final int defL = lhsNV - 1;
		final int defR = rhsNV - 1;

		AIterator lIt = lhs._indexes.getIterator();
		AIterator rIt = _indexes.getIterator();

		int i = 0;
		int col;
		int row;
		for(; i < this._numRows && lIt.hasNext() && rIt.hasNext(); i++) {
			if(lIt.value() == i)
				col = lhs.getIndex(lIt.getDataIndexAndIncrement());
			else
				col = defL;
			if(rIt.value() == i)
				row = this.getIndex(rIt.getDataIndexAndIncrement());
			else
				row = defR;
			ag.increment(col + row * nCol);
		}

		if(lIt.hasNext()) {
			row = defR;
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
				if(rIt.value() == i)
					row = this.getIndex(rIt.getDataIndexAndIncrement());
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
		final IPreAggregate ag = PreAggregateFactory.ag(retSize);
		final int defR = rhsNV - 1;
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
			if(rIt.value() == i)
				row = this.getIndex(rIt.getDataIndexAndIncrement());
			else
				row = defR;
			ag.increment(col + row * nCol);
		}

		if(lIt.hasNext()) {
			row = defR;
			for(; i < this._numRows && lIt.hasNext(); i++) {
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
			for(; i < this._numRows && rIt.hasNext(); i++) {
				if(rIt.value() == i)
					row = this.getIndex(rIt.getDataIndexAndIncrement());
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
		final IPreAggregate ag = PreAggregateFactory.ag(retSize);
		final AIterator lIt = lhs._indexes.getIterator();
		final AIterator rIt = _indexes.getIterator();

		while(lIt.hasNext() && rIt.hasNext())
			if(lIt.value() == rIt.value())
				ag.increment(lhs.getIndex(lIt.getDataIndexAndIncrement()) +
					this.getIndex(rIt.getDataIndexAndIncrement()) * nCol);
			else if(lIt.value() > rIt.value())
				rIt.next();
			else
				ag.increment(lhs.getIndex(lIt.getDataIndexAndIncrement()) + defR);

		while(lIt.hasNext())
			ag.increment(lhs.getIndex(lIt.getDataIndexAndIncrement()) + defR);
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

		final int defR = (NVR - 1) * NVL;

		for(int kl = 0; kl < NVL; kl++) {
			AIterator it = _indexes.getIterator();
			final int bOffL = lhs._ptr[kl];
			final int bLenL = lhs.len(kl);
			for(int bixL = 0, offL = 0, sLenL = 0; bixL < bLenL; bixL += sLenL + 1, offL += blksz) {
				sLenL = lhs._data[bOffL + bixL];
				for(int i = 1; i <= sLenL; i++) {
					final int col = offL + lhs._data[bOffL + bixL + i];
					it.skipTo(col);
					if(it.value() == col)
						ag.increment(kl + this.getIndex(it.getDataIndexAndIncrement()) * NVL);
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

	@Override
	public Dictionary preAggregateThatDDCStructure(ColGroupDDC that, Dictionary ret) {

		final AIterator it = _indexes.getIterator();
		final int offsetToDefault = this.getNumValues() - 1;
		final int nCol = that._colIndexes.length;

		int i = 0;

		for(; i < _numRows && it.hasNext(); i++) {
			int to = (it.value() == i) ? getIndex(it.getDataIndexAndIncrement()) : offsetToDefault;
			that._dict.addToEntry(ret, that.getIndex(i), to, nCol);
		}

		for(; i < _numRows; i++)
			that._dict.addToEntry(ret, that.getIndex(i), offsetToDefault, nCol);

		return ret;
	}

	@Override
	public Dictionary preAggregateThatSDCStructure(ColGroupSDC that, Dictionary ret) {
		final AIterator itThat = that._indexes.getIterator();
		final AIterator itThis = _indexes.getIterator();
		final int offsetToDefaultThat = that.getNumValues() - 1;
		final int offsetToDefaultThis = getNumValues() - 1;
		final int nCol = that._colIndexes.length;

		int i = 0;

		for(; i < _numRows && itThat.hasNext() && itThis.hasNext(); i++) {
			int to = (itThis.value() == i) ? getIndex(itThis.getDataIndexAndIncrement()) : offsetToDefaultThis;
			int fr = (itThat.value() == i) ? that.getIndex(itThat.getDataIndexAndIncrement()) : offsetToDefaultThat;
			that._dict.addToEntry(ret, fr, to, nCol);
		}

		if(itThat.hasNext()) {
			for(; i < _numRows && itThat.hasNext(); i++) {
				int fr = (itThat.value() == i) ? that.getIndex(itThat.getDataIndexAndIncrement()) : offsetToDefaultThat;
				that._dict.addToEntry(ret, fr, offsetToDefaultThis, nCol);
			}
		}

		if(itThis.hasNext()) {
			for(; i < _numRows && itThis.hasNext(); i++) {
				int to = (itThis.value() == i) ? getIndex(itThis.getDataIndexAndIncrement()) : offsetToDefaultThis;
				that._dict.addToEntry(ret, offsetToDefaultThat, to, nCol);
			}
		}

		for(; i < _numRows; i++)
			that._dict.addToEntry(ret, offsetToDefaultThat, offsetToDefaultThis, nCol);
			
		return ret;
	}

	@Override
	public Dictionary preAggregateThatSDCZerosStructure(ColGroupSDCZeros that, Dictionary ret) {
		
		final AIterator itThat = that._indexes.getIterator();
		final AIterator itThis = _indexes.getIterator();
		final int nCol = that._colIndexes.length;
		final int defThis = this.getNumValues() * nCol - nCol;

		while(itThat.hasNext() && itThis.hasNext()) {
			if(itThat.value() == itThis.value()) {
				final int fr = that.getIndex(itThat.getDataIndexAndIncrement());
				final int to = getIndex(itThis.getDataIndexAndIncrement());
				that._dict.addToEntry(ret, fr, to, nCol);
			}
			else if(itThat.value() < itThis.value()){
				final int fr = that.getIndex(itThat.getDataIndexAndIncrement());
				that._dict.addToEntry(ret, fr, defThis, nCol);
			}
			else
				itThis.next();
		}

		while(itThat.hasNext()){
			final int fr = that.getIndex(itThat.getDataIndexAndIncrement());
			that._dict.addToEntry(ret, fr, defThis, nCol);
		}
		return ret;

	}

	@Override
	public Dictionary preAggregateThatSDCSingleZerosStructure(ColGroupSDCSingleZeros that, Dictionary ret){
		throw new NotImplementedException();
	}

}
