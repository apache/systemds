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

	// protected ColGroupDDC(int[] colIndices, int numRows, ABitmap ubm, CompressionSettings cs) {
	// super(colIndices, numRows, ubm, cs);
	// }

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
		double[] c = target.getDenseBlockValues();
		offT = offT * tCol;

		for(int i = rl; i < ru; i++, offT += tCol) {
			int rowIndex = getIndex(i) * nCol;
			for(int j = 0; j < nCol; j++)
				c[offT + _colIndexes[j]] += values[rowIndex + j];
		}

	}

	@Override
	public void decompressToBlock(MatrixBlock target, int[] colIndexTargets) {
		int ncol = getNumCols();
		double[] dictionary = getValues();
		for(int i = 0; i < _numRows; i++) {
			int rowIndex = getIndex(i) * ncol;
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
			int index = getIndex(i);
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
			int index = getIndex(r);
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
			int index = getIndex(r);
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
		int index = getIndex(r);
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
			int index = getIndex(i);
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
			c[rix] += vals[getIndex(rix)];
	}

	@Override
	protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru) {
		final int nCol = getNumCols();
		double[] preAggregatedRows = _dict.aggregateTuples(builtin, nCol);
		for(int i = rl; i < ru; i++)
			c[i] = builtin.execute(c[i], preAggregatedRows[getIndex(i)]);
	}

	@Override
	public int[] getCounts(int[] counts) {
		return getCounts(0, _numRows, counts);
	}

	@Override
	public int[] getCounts(int rl, int ru, int[] counts) {
		for(int i = rl; i < ru; i++) {
			int index = getIndex(i);
			counts[index]++;
		}
		return counts;
	}

	@Override
	public void leftMultByMatrix(double[] a, double[] c, double[] values, int numRows, int numCols, int rl, int ru,
		int voff) {

		int numVals = getNumValues();
		// if(8 * numVals < _numRows) {
		for(int i = rl, j = voff; i < ru; i++, j++) {
			double[] vals = preAggregate(a, j);
			postScaling(values, vals, c, numVals, i, numCols);
		}
		// }
		// else {
		// for(int i = rl, j = voff; i < ru; i++, j++) {
		// int offC = i * numCols;
		// numVals = numVals * _colIndexes.length;
		// for(int k = 0, aOff = j * _numRows; k < _numRows; k++, aOff++) {
		// double aval = a[aOff];
		// if(aval != 0) {
		// int valOff = getIndex(k) * _colIndexes.length;
		// if(valOff < numVals) {
		// for(int h = 0; h < _colIndexes.length; h++) {
		// int colIx = _colIndexes[h] + offC;
		// c[colIx] += aval * values[valOff + h];
		// }
		// }
		// }
		// }

		// }
		// }
	}

	@Override
	public void leftMultBySparseMatrix(SparseBlock sb, double[] c, double[] values, int numRows, int numCols, int row) {
		final int numVals = getNumValues();
		double[] vals = preAggregateSparse(sb, row);
		postScaling(values, vals, c, numVals, row, numCols);
		// LOG.error(Arrays.toString(c));
	}

	@Override
	public double[] preAggregate(double[] a, int row) {
		double[] vals = allocDVector(getNumValues(), true);
		if(row > 0)
			for(int i = 0, off = _numRows * row; i < _numRows; i++, off++)
				vals[getIndex(i)] += a[off];
		else
			for(int i = 0; i < _numRows; i++)
				vals[getIndex(i)] += a[i];

		return vals;
	}

	@Override
	public double[] preAggregateSparse(SparseBlock sb, int row) {

		// LOG.error(this);
		// LOG.error(sb);
		double[] vals = allocDVector(getNumValues(), true);
		int[] indexes = sb.indexes(row);
		double[] sparseV = sb.values(row);
		for(int i = sb.pos(row); i < sb.size(row) + sb.pos(row); i++)
			vals[getIndex(indexes[i])] += sparseV[i];

		// LOG.error(Arrays.toString(vals));
		return vals;
	}

	// @Override
	// public void leftMultByRowVector(double[] a, double[] c, int numVals, double[] values) {
	// // if(8 * numVals < _numRows) {
	// double[] vals = preAggregate(a);
	// postScaling(values, vals, c, numVals);
	// // }
	// // else {
	// // numVals = numVals * _colIndexes.length;
	// // // iterate over codes, compute all, and add to the result
	// // for(int i = 0; i < _numRows; i++) {
	// // double aval = a[i];
	// // if(aval != 0)
	// // for(int j = 0, valOff = getIndex(i) * _colIndexes.length; j < _colIndexes.length; j++)
	// // if(valOff < numVals) {
	// // c[_colIndexes[j]] += aval * values[valOff + j];
	// // }
	// // }
	// // }
	// }

	/**
	 * Generic get index in dictionary for value at row position.
	 * 
	 * @param r row position to get dictionary index for.
	 * @return The dictionary index
	 */
	protected int getIndex(int r) {
		return _data.getIndex(r);
	}

	/**
	 * Generic get index in dictionary for value at row, col position. If used consider changing to getIndex and
	 * precalculate offset to row
	 * 
	 * @param r     The row to find
	 * @param colIx the col index to find
	 * @return the index in the dictionary containing the specified value
	 */
	protected int getIndex(int r, int colIx) {
		return _data.getIndex(r) * getNumCols() + colIx;
	}

	/**
	 * Generic get value for byte-length-agnostic access to first column.
	 * 
	 * @param r      Global row index
	 * @param values The values contained in the column groups dictionary
	 * @return value
	 */
	protected double getData(int r, double[] values) {
		int index = getIndex(r);
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
		int index = getIndex(r, colIx);
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
			ag.increment(lhs.getIndex(i) + this.getIndex(i) * nCol);

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
			int row = this.getIndex(i);
			if(lIt.value() == i)
				col = lhs.getIndex(lIt.getDataIndexAndIncrement());

			else
				col = offsetToDefault;
			ag.increment(col + row * nCol);
		}
		col = offsetToDefault;
		for(; i < this._numRows; i++) {
			int row = this.getIndex(i);
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
			int row = this.getIndex(i);
			if(lIt.value() == i) {
				col = 1;
				lIt.next();
			}
			else
				col = 0;
			ag.increment(col + row * nCol);
		}

		for(; i < this._numRows; i++)
			ag.increment(this.getIndex(i) * nCol);

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
			int row = this.getIndex(lIt.value());
			int col = lhs.getIndex(lIt.getDataIndexAndIncrement());
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
			int row = this.getIndex(lIt.value());
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
					int idx = this.getIndex(offL + lhs._data[bOffL + bixL + i]);
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
					int kr = getIndex(i) * NVL;
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
			that._dict.addToEntry(ret, that.getIndex(r), this.getIndex(r), nCol);

		return ret;
	}

	@Override
	public Dictionary preAggregateThatSDCStructure(ColGroupSDC that, Dictionary ret) {
		final AIterator itThat = that._indexes.getIterator();
		final int offsetToDefault = that.getNumValues() - 1;
		final int nCol = that._colIndexes.length;

		int i = 0;

		for(; i < _numRows && itThat.hasNext(); i++) {
			int fr = (itThat.value() == i) ? that.getIndex(itThat.getDataIndexAndIncrement()) : offsetToDefault;
			that._dict.addToEntry(ret, fr, this.getIndex(i), nCol);
		}

		for(; i < _numRows; i++)
			that._dict.addToEntry(ret, offsetToDefault, this.getIndex(i), nCol);

		return ret;
	}

	@Override
	public Dictionary preAggregateThatSDCZerosStructure(ColGroupSDCZeros that, Dictionary ret) {
		final AIterator itThat = that._indexes.getIterator();
		final int nCol = that._colIndexes.length;

		while(itThat.hasNext()) {
			final int to = getIndex(itThat.value());
			final int fr = that.getIndex(itThat.getDataIndexAndIncrement());
			that._dict.addToEntry(ret, fr, to, nCol);
		}

		return ret;
	}

	@Override
	public Dictionary preAggregateThatSDCSingleZerosStructure(ColGroupSDCSingleZeros that, Dictionary ret) {
		final AIterator itThat = that._indexes.getIterator();
		final int nCol = that._colIndexes.length;

		while(itThat.hasNext()) {
			final int to = getIndex(itThat.value());
			itThat.next();
			that._dict.addToEntry(ret, 0, to, nCol);
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
		// if(_data instanceof MapToBit)
		// return ColGroupType.DDC0;
		// if(_data instanceof MapToByte)
		// return ColGroupType.DDC1;
		// else if(_data instanceof MapToChar)
		// return ColGroupType.DDC2;
		// else if(_data instanceof MapToInt)
		// return ColGroupType.DDC3;
		// throw new DMLCompressionException("Unknown ColGroupType for DDC");
	}

	@Override
	public long estimateInMemorySize() {
		return ColGroupSizes.estimateInMemorySizeDDC(getNumCols(), getNumValues(), _numRows, isLossy());

	}

	// @Override
	// public void rightMultByVector(double[] b, double[] c, int rl, int ru, double[] dictVals) {
	// 	final int numVals = getNumValues();
	// 	double[] vals = preaggValues(numVals, b, dictVals);
	// 	for(int i = rl; i < ru; i++)
	// 		c[i] += vals[_data.getIndex(i)];

	// }

	// @Override
	// public void rightMultByMatrix(int[] outputColumns, double[] preAggregatedB, double[] c, int thatNrColumns, int rl,
	// 	int ru) {
	// 	for(int j = rl, off = rl * thatNrColumns; j < ru; j++, off += thatNrColumns) {
	// 		int rowIdx = _data.getIndex(j);
	// 		for(int k = 0; k < outputColumns.length; k++)
	// 			c[off + outputColumns[k]] += preAggregatedB[rowIdx * outputColumns.length + k];
	// 	}
	// }

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
