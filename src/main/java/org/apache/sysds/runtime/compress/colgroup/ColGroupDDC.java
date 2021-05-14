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

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.colgroup.pre.IPreAggregate;
import org.apache.sysds.runtime.compress.colgroup.pre.PreAggregateFactory;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.DenseBlockFP64;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

/**
 * Class to encapsulate information about a column group that is encoded with dense dictionary encoding (DDC).
 * 
 */
public class ColGroupDDC extends ColGroupValue {
	private static final long serialVersionUID = -3204391646123465004L;

	protected AMapToData _data;

	protected ColGroupDDC(int numRows) {
		super(numRows);
	}

	protected ColGroupDDC(int[] colIndices, int numRows, ADictionary dict, AMapToData data, int[] cachedCounts) {
		super(colIndices, numRows, dict, cachedCounts);
		_zeros = false;
		_data = data;
	}

	public CompressionType getCompType() {
		return CompressionType.DDC;
	}

	@Override
	public void decompressToBlockSafe(MatrixBlock target, int rl, int ru, int offT, double[] values) {
		decompressToBlockUnSafe(target, rl, ru, offT, values);
		target.setNonZeros(target.getNonZeros() + _numRows * _colIndexes.length);
	}

	@Override
	public void decompressToBlockUnSafe(MatrixBlock target, int rl, int ru, int offT, double[] values) {
		final int nCol = _colIndexes.length;
		final int tCol = target.getNumColumns();
		final double[] c = target.getDenseBlockValues();
		offT = offT * tCol;

		for(int i = rl; i < ru; i++, offT += tCol) {
			final int rowIndex = _data.getIndex(i) * nCol;
			for(int j = 0; j < nCol; j++)
				c[offT + _colIndexes[j]] += values[rowIndex + j];
		}

	}

	@Override
	public void decompressToBlock(MatrixBlock target, int[] colIndexTargets) {
		int ncol = getNumCols();
		double[] dictionary = getValues();
		for(int i = 0; i < _numRows; i++) {
			int rowIndex = _data.getIndex(i) * ncol;
			for(int colIx = 0; colIx < ncol; colIx++) {
				int origMatrixColIx = getColIndex(colIx);
				int col = colIndexTargets[origMatrixColIx];
				double cellVal = dictionary[rowIndex + colIx];
				target.quickSetValue(i, col, target.quickGetValue(i, col) + cellVal);
			}

		}
	}

	@Override
	public void decompressColumnToBlock(MatrixBlock target, int colpos) {
		int ncol = getNumCols();
		double[] c = target.getDenseBlockValues();
		double[] values = getValues();
		int nnz = 0;
		for(int i = 0; i < _numRows; i++) {
			int index = _data.getIndex(i);
			if(index < getNumValues())
				nnz += ((c[i] += values[(index) * ncol + colpos]) != 0) ? 1 : 0;
			else
				nnz++;

		}
		target.setNonZeros(nnz);
	}

	@Override
	public void decompressColumnToBlock(MatrixBlock target, int colpos, int rl, int ru) {
		int ncol = getNumCols();
		double[] c = target.getDenseBlockValues();
		double[] values = getValues();
		final int numValues = getNumValues();
		int nnz = 0;
		for(int i = 0, r = rl; i < ru - rl; i++, r++) {
			int index = _data.getIndex(r);
			if(index < numValues)
				nnz += ((c[i] += values[(index) * ncol + colpos]) != 0) ? 1 : 0;
			else
				nnz++;
		}
		target.setNonZeros(nnz);
	}

	@Override
	public void decompressColumnToBlock(double[] c, int colpos, int rl, int ru) {
		int ncol = getNumCols();
		double[] values = getValues();
		final int numValues = getNumValues();
		for(int i = 0, r = rl; i < ru - rl; i++, r++) {
			int index = _data.getIndex(r);
			if(index < numValues)
				c[i] += values[(index) * ncol + colpos];
		}
	}

	@Override
	public double get(int r, int c) {
		// find local column index
		int ix = Arrays.binarySearch(_colIndexes, c);
		if(ix < 0)
			throw new RuntimeException("Column index " + c + " not in DDC group.");

		// get value
		int index = _data.getIndex(r);
		if(index < getNumValues())
			return _dict.getValue(index * _colIndexes.length + ix);
		else
			return 0.0;

	}

	@Override
	public void countNonZerosPerRow(int[] rnnz, int rl, int ru) {
		int ncol = _colIndexes.length;
		final int numVals = getNumValues();
		double[] values = _dict.getValues();
		for(int i = rl; i < ru; i++) {
			int lnnz = 0;
			int index = _data.getIndex(i);
			if(index < numVals) {
				for(int colIx = index; colIx < ncol + index; colIx++) {
					lnnz += (values[colIx]) != 0 ? 1 : 0;
				}
			}
			rnnz[i - rl] += lnnz;
		}
	}

	@Override
	protected void computeRowSums(double[] c, boolean square, int rl, int ru, boolean mean) {
		double[] vals = _dict.sumAllRowsToDouble(square, _colIndexes.length);
		for(int rix = rl; rix < ru; rix++)
			c[rix] += vals[_data.getIndex(rix)];
	}

	@Override
	protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru) {
		final int nCol = getNumCols();
		double[] preAggregatedRows = _dict.aggregateTuples(builtin, nCol);
		for(int i = rl; i < ru; i++)
			c[i] = builtin.execute(c[i], preAggregatedRows[_data.getIndex(i)]);
	}

	@Override
	public int[] getCounts(int[] counts) {
		return getCounts(0, _numRows, counts);
	}

	@Override
	public int[] getCounts(int rl, int ru, int[] counts) {
		for(int i = rl; i < ru; i++) {
			int index = _data.getIndex(i);
			counts[index]++;
		}
		return counts;
	}

	@Override
	public double[] preAggregate(double[] a, int row) {
		double[] vals = allocDVector(getNumValues(), true);
		if(row > 0)
			for(int i = 0, off = _numRows * row; i < _numRows; i++, off++)
				vals[_data.getIndex(i)] += a[off];
		else
			for(int i = 0; i < _numRows; i++)
				vals[_data.getIndex(i)] += a[i];

		return vals;
	}

	@Override
	public double[] preAggregateSparse(SparseBlock sb, int row) {

		double[] vals = allocDVector(getNumValues(), true);
		int[] indexes = sb.indexes(row);
		double[] sparseV = sb.values(row);
		for(int i = sb.pos(row); i < sb.size(row) + sb.pos(row); i++)
			vals[_data.getIndex(indexes[i])] += sparseV[i];
		return vals;

	}

	@Override
	public MatrixBlock preAggregate(MatrixBlock m, int rl, int ru) {

		final int retCols = getNumValues();
		final int retRows = ru - rl;
		final double[] vals = allocDVector(retRows * retCols, true);
		final DenseBlock retB = new DenseBlockFP64(new int[] {retRows, retCols}, vals);
		final MatrixBlock ret = new MatrixBlock(retRows, retCols, retB);

		final double[] mV = m.getDenseBlockValues();

		ret.setNonZeros(retRows * retCols);
		for(int k = rl; k < ru; k++) {
			final int offT = ret.getNumColumns() * k;
			final int offM = m.getNumColumns() * k;
			for(int i = 0; i < _numRows; i++) {
				int index = _data.getIndex(i);
				vals[offT + index] += mV[offM + i];
			}
		}
		return ret;
	}

	/**
	 * Generic get value for byte-length-agnostic access to first column.
	 * 
	 * @param r      Global row index
	 * @param values The values contained in the column groups dictionary
	 * @return value
	 */
	protected double getData(int r, double[] values) {
		int index = _data.getIndex(r);
		return (index < values.length) ? values[index] : 0.0;
	}

	/**
	 * Generic get value for byte-length-agnostic access.
	 * 
	 * @param r      Global row index
	 * @param colIx  Local column index
	 * @param values The values contained in the column groups dictionary
	 * @return value
	 */
	protected double getData(int r, int colIx, double[] values) {
		int index = _data.getIndex(r) * _colIndexes.length + colIx;
		return (index < values.length) ? values[index] : 0.0;
	}

	/**
	 * Generic set value for byte-length-agnostic write of encoded value.
	 * 
	 * @param r    global row index
	 * @param code encoded value
	 */
	protected void setData(int r, int code) {
		_data.set(r, code);
	}

	@Override
	public IPreAggregate preAggregateDDC(ColGroupDDC lhs) {
		final int nCol = lhs.getNumValues();
		final int rhsNV = this.getNumValues();
		final int retSize = nCol * rhsNV;
		IPreAggregate ag = PreAggregateFactory.ag(retSize);
		// int[] m = _data.materializeMultiplied(nCol);
		for(int i = 0; i < this._numRows; i++)
			ag.increment(lhs._data.getIndex(i) + this._data.getIndex(i) * nCol);

		return ag;
	}

	@Override
	public IPreAggregate preAggregateSDC(ColGroupSDC lhs) {
		final int nCol = lhs.getNumValues();
		final int rhsNV = this.getNumValues();
		final int retSize = nCol * rhsNV;
		IPreAggregate ag = PreAggregateFactory.ag(retSize);

		AIterator lIt = lhs._indexes.getIterator();
		final int offsetToDefault = nCol - 1;

		int i = 0;

		int col;
		for(; i < this._numRows && lIt.hasNext(); i++) {
			int row = this._data.getIndex(i);
			if(lIt.value() == i)
				col = lhs._data.getIndex(lIt.getDataIndexAndIncrement());

			else
				col = offsetToDefault;
			ag.increment(col + row * nCol);
		}
		col = offsetToDefault;
		for(; i < this._numRows; i++) {
			int row = this._data.getIndex(i);
			ag.increment(col + row * nCol);
		}

		return ag;
	}

	@Override
	public IPreAggregate preAggregateSDCSingle(ColGroupSDCSingle lhs) {
		final int nCol = lhs.getNumValues();
		final int rhsNV = this.getNumValues();
		final int retSize = nCol * rhsNV;
		final IPreAggregate ag = PreAggregateFactory.ag(retSize);
		final AIterator lIt = lhs._indexes.getIterator();

		int i = 0;

		int col;
		for(; i < this._numRows && lIt.hasNext(); i++) {
			int row = this._data.getIndex(i);
			if(lIt.value() == i) {
				col = 1;
				lIt.next();
			}
			else
				col = 0;
			ag.increment(col + row * nCol);
		}

		for(; i < this._numRows; i++)
			ag.increment(this._data.getIndex(i) * nCol);

		return ag;
	}

	@Override
	public IPreAggregate preAggregateSDCZeros(ColGroupSDCZeros lhs) {
		final int nCol = lhs.getNumValues();
		final int rhsNV = this.getNumValues();
		final int retSize = nCol * rhsNV;
		final IPreAggregate ag = PreAggregateFactory.ag(retSize);
		final AIterator lIt = lhs._indexes.getIterator();

		while(lIt.hasNext()) {
			int row = this._data.getIndex(lIt.value());
			int col = lhs._data.getIndex(lIt.getDataIndexAndIncrement());
			ag.increment(col + row * nCol);
		}

		return ag;
	}

	@Override
	public IPreAggregate preAggregateSDCSingleZeros(ColGroupSDCSingleZeros lhs) {
		final int nCol = lhs.getNumValues();
		final int rhsNV = this.getNumValues();
		final int retSize = nCol * rhsNV;
		IPreAggregate ag = PreAggregateFactory.ag(retSize);

		final AIterator lIt = lhs._indexes.getIterator();

		while(lIt.hasNext()) {
			int row = this._data.getIndex(lIt.value());
			lIt.next();
			ag.increment(row);
		}

		return ag;
	}

	@Override
	public IPreAggregate preAggregateOLE(ColGroupOLE lhs) {
		final int NVR = this.getNumValues();
		final int NVL = lhs.getNumValues();
		final int retSize = NVR * NVL;
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		IPreAggregate ag = PreAggregateFactory.ag(retSize);

		for(int kl = 0; kl < NVL; kl++) {
			final int bOffL = lhs._ptr[kl];
			final int bLenL = lhs.len(kl);
			for(int bixL = 0, offL = 0, sLenL = 0; bixL < bLenL; bixL += sLenL + 1, offL += blksz) {
				sLenL = lhs._data[bOffL + bixL];
				for(int i = 1; i <= sLenL; i++) {
					int idx = this._data.getIndex(offL + lhs._data[bOffL + bixL + i]);
					ag.increment(kl + idx * NVL);
				}
			}
		}

		return ag;
	}

	@Override
	public IPreAggregate preAggregateRLE(ColGroupRLE lhs) {
		final int NVR = this.getNumValues();
		final int NVL = lhs.getNumValues();
		final int retSize = NVR * NVL;
		IPreAggregate ag = PreAggregateFactory.ag(retSize);

		for(int kl = 0; kl < NVL; kl++) {
			final int boffL = lhs._ptr[kl];
			final int blenL = lhs.len(kl);
			for(int bixL = 0, startL = 0, lenL = 0; bixL < blenL && startL < _numRows; startL += lenL, bixL += 2) {
				startL += lhs._data[boffL + bixL];
				lenL = lhs._data[boffL + bixL + 1];
				final int endL = startL + lenL;
				for(int i = startL; i < endL; i++) {
					int kr = _data.getIndex(i) * NVL;
					ag.increment(kl + kr);
				}
			}
		}
		return ag;
	}

	@Override
	public Dictionary preAggregateThatDDCStructure(ColGroupDDC that, Dictionary ret) {
		final int nCol = that._colIndexes.length;
		for(int r = 0; r < _numRows; r++)
			that._dict.addToEntry(ret, that._data.getIndex(r), this._data.getIndex(r), nCol);

		return ret;
	}

	@Override
	public Dictionary preAggregateThatSDCStructure(ColGroupSDC that, Dictionary ret, boolean preModified) {
		final AIterator itThat = that._indexes.getIterator();
		final int offsetToDefault = that.getNumValues() - 1;
		final int nCol = that._colIndexes.length;
		if(preModified) {
			while(itThat.hasNext()) {
				final int to = _data.getIndex(itThat.value());
				final int fr = that._data.getIndex(itThat.getDataIndexAndIncrement());
				that._dict.addToEntry(ret, fr, to, nCol);
			}
		}
		else {
			int i = 0;

			for(; i < _numRows && itThat.hasNext(); i++) {
				int fr = (itThat.value() == i) ? that._data
					.getIndex(itThat.getDataIndexAndIncrement()) : offsetToDefault;
				that._dict.addToEntry(ret, fr, this._data.getIndex(i), nCol);
			}

			for(; i < _numRows; i++)
				that._dict.addToEntry(ret, offsetToDefault, this._data.getIndex(i), nCol);
		}

		return ret;
	}

	@Override
	public Dictionary preAggregateThatSDCZerosStructure(ColGroupSDCZeros that, Dictionary ret) {
		final AIterator itThat = that._indexes.getIterator();
		final int nCol = that._colIndexes.length;

		while(itThat.hasNext()) {
			final int to = _data.getIndex(itThat.value());
			final int fr = that._data.getIndex(itThat.getDataIndexAndIncrement());
			that._dict.addToEntry(ret, fr, to, nCol);
		}

		return ret;
	}

	@Override
	public Dictionary preAggregateThatSDCSingleZerosStructure(ColGroupSDCSingleZeros that, Dictionary ret) {
		final AIterator itThat = that._indexes.getIterator();
		final int nCol = that._colIndexes.length;

		while(itThat.hasNext()) {
			final int to = _data.getIndex(itThat.value());
			itThat.next();
			that._dict.addToEntry(ret, 0, to, nCol);
		}

		return ret;
	}

	@Override
	public Dictionary preAggregateThatSDCSingleStructure(ColGroupSDCSingle that, Dictionary ret, boolean preModified) {
		final AIterator itThat = that._indexes.getIterator();
		final int nCol = that._colIndexes.length;
		if(preModified) {
			while(itThat.hasNext()) {
				final int to = _data.getIndex(itThat.value());
				itThat.next();
				that._dict.addToEntry(ret, 0, to, nCol);
			}
		}
		else {
			int i = 0;
			for(; i < _numRows && itThat.hasNext(); i++) {
				if(itThat.value() == i) {
					that._dict.addToEntry(ret, 0, this._data.getIndex(i), nCol);
					itThat.next();
				}
				else
					that._dict.addToEntry(ret, 1, this._data.getIndex(i), nCol);
			}

			for(; i < _numRows; i++)
				that._dict.addToEntry(ret, 1, this._data.getIndex(i), nCol);
		}

		return ret;
	}

	@Override
	public boolean sameIndexStructure(ColGroupCompressed that) {
		return that instanceof ColGroupDDC && ((ColGroupDDC) that)._data == _data;
	}

	@Override
	public int getIndexStructureHash() {
		return _data.hashCode();
	}

	@Override
	public ColGroupType getColGroupType() {
		return ColGroupType.DDC;
	}

	@Override
	public long estimateInMemorySize() {
		long size = super.estimateInMemorySize();
		size += _data.getInMemorySize();
		return size;
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		return new ColGroupDDC(_colIndexes, _numRows, applyScalarOp(op), _data, getCachedCounts());
	}

	@Override
	public AColGroup binaryRowOp(BinaryOperator op, double[] v, boolean sparseSafe, boolean left) {
		ADictionary aDict = applyBinaryRowOp(op.fn, v, true, left);
		return new ColGroupDDC(_colIndexes, _numRows, aDict, _data, getCachedCounts());
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		// write data
		_data.write(out);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		super.readFields(in);
		_data = MapToFactory.readIn(in, getNumValues());
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += _data.getExactSizeOnDisk();
		return ret;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s ", "Data: "));
		sb.append(_data);
		return sb.toString();
	}

}
