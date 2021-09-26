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

import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToByte;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToChar;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToInt;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.data.DenseBlock;
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
	private static final long serialVersionUID = -5769772089913918987L;

	protected transient AMapToData _data;

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
	protected void decompressToBlockSparseDictionary(MatrixBlock target, int rl, int ru, int offT,
		SparseBlock sb) {
		final DenseBlock db = target.getDenseBlock();
		for(int i = rl; i < ru; i++, offT++) {
			final int rowIndex = _data.getIndex(i);
			if(sb.isEmpty(rowIndex))
				continue;
			final double[] c = db.values(offT);
			final int off = db.pos(offT);
			final int apos = sb.pos(rowIndex);
			final int alen = sb.size(rowIndex) + apos;
			final double[] avals = sb.values(rowIndex);
			final int[] aix = sb.indexes(rowIndex);
			for(int j = apos; j < alen; j++)
				c[off + _colIndexes[aix[j]]] += avals[j];
		}
	}

	@Override
	protected void decompressToBlockDenseDictionary(MatrixBlock target, int rl, int ru, int offT,
		double[] values) {
		final int nCol = _colIndexes.length;
		final DenseBlock db = target.getDenseBlock();
		for(int i = rl; i < ru; i++, offT++) {
			final double[] c = db.values(offT);
			final int off = db.pos(offT);
			final int rowIndex = _data.getIndex(i) * nCol;
			for(int j = 0; j < nCol; j++)
				c[off + _colIndexes[j]] += values[rowIndex + j];
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
	protected void computeRowSums(double[] c, boolean square, int rl, int ru) {
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
	public void preAggregate(final MatrixBlock m, final MatrixBlock preAgg, final int rl, final int ru) {
		if(m.isInSparseFormat())
			preAggregateSparse(m.getSparseBlock(), preAgg, rl, ru);
		else
			preAggregateDense(m, preAgg, rl, ru);
	}

	private void preAggregateDense(final MatrixBlock m, final MatrixBlock preAgg, final int rl, final int ru) {
		preAggregateDense(m, preAgg, rl, ru, 0, _numRows);
	}

	public void preAggregateDense(MatrixBlock m, MatrixBlock preAgg, int rl, int ru, int cl, int cu) {
		final double[] mV = m.getDenseBlockValues();
		final double[] preAV = preAgg.getDenseBlockValues();
		final int numVals = getNumValues();
		if(_data instanceof MapToByte)
			preAggregateDenseByte(mV, preAV, ((MapToByte) _data).getBytes(), rl, ru, cl, cu, _numRows, numVals);
		else if(_data instanceof MapToChar)
			preAggregateDenseChar(mV, preAV, ((MapToChar) _data).getChars(), rl, ru, cl, cu, _numRows, numVals);
		else if(_data instanceof MapToInt)
			preAggregateDenseInt(mV, preAV, ((MapToInt) _data).getInts(), rl, ru, cl, cu, _numRows, numVals);
		else {
			final int blockSize = 2000;
			for(int block = cl; block < cu; block += blockSize) {
				final int blockEnd = Math.min(block + blockSize, cu);
				for(int rowLeft = rl, offOut = 0; rowLeft < ru; rowLeft++, offOut += numVals) {
					final int offLeft = rowLeft * _numRows;
					for(int rc = block; rc < blockEnd; rc++) {
						final int idx = _data.getIndex(rc);
						preAV[offOut + idx] += mV[offLeft + rc];
					}
				}
			}
		}
	}

	private static void preAggregateDenseByte(final double[] mV, final double[] preAV, final byte[] d, final int rl,
		final int ru, final int cl, final int cu, final int nRow, final int nVal) {
		final int blockSize = 4000;
		for(int block = cl; block < cu; block += blockSize) {
			final int blockEnd = Math.min(block + blockSize, nRow);
			for(int rowLeft = rl, offOut = 0; rowLeft < ru; rowLeft++, offOut += nVal) {
				final int offLeft = rowLeft * nRow;
				for(int rc = block; rc < blockEnd; rc++) {
					final int idx = d[rc] & 0xFF;
					preAV[offOut + idx] += mV[offLeft + rc];
				}
			}
		}
	}

	private static void preAggregateDenseChar(final double[] mV, final double[] preAV, final char[] d, final int rl,
		final int ru, final int cl, final int cu, final int nRow, final int nVal) {
		final int blockSize = 4000;
		for(int block = cl; block < cu; block += blockSize) {
			final int blockEnd = Math.min(block + blockSize, nRow);
			for(int rowLeft = rl, offOut = 0; rowLeft < ru; rowLeft++, offOut += nVal) {
				final int offLeft = rowLeft * nRow;
				for(int rc = block; rc < blockEnd; rc++) {
					final int idx = d[rc];
					preAV[offOut + idx] += mV[offLeft + rc];
				}
			}
		}
	}

	private static void preAggregateDenseInt(final double[] mV, final double[] preAV, final int[] d, final int rl,
		final int ru, final int cl, final int cu, final int nRow, final int nVal) {
		final int blockSize = 2000;
		for(int block = cl; block < cu; block += blockSize) {
			final int blockEnd = Math.min(block + blockSize, nRow);
			for(int rowLeft = rl, offOut = 0; rowLeft < ru; rowLeft++, offOut += nVal) {
				final int offLeft = rowLeft * nRow;
				for(int rc = block; rc < blockEnd; rc++) {
					final int idx = d[rc];
					preAV[offOut + idx] += mV[offLeft + rc];
				}
			}
		}
	}

	private void preAggregateSparse(SparseBlock sb, MatrixBlock preAgg, int rl, int ru) {
		final double[] preAV = preAgg.getDenseBlockValues();
		final int numVals = getNumValues();
		for(int rowLeft = rl, offOut = 0; rowLeft < ru; rowLeft++, offOut += numVals) {
			if(sb.isEmpty(rowLeft))
				continue;
			final int apos = sb.pos(rowLeft);
			final int alen = sb.size(rowLeft) + apos;
			final int[] aix = sb.indexes(rowLeft);
			final double[] avals = sb.values(rowLeft);
			for(int j = apos; j < alen; j++) {
				preAV[offOut + _data.getIndex(aix[j])] += avals[j];
			}
		}
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
			that._dict.addToEntry(ret, 0, to, nCol);
			itThat.next();
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
				that._dict.addToEntry(ret, 0, to, nCol);
				itThat.next();
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
		ADictionary aDict = applyBinaryRowOp(op, v, true, left);
		return new ColGroupDDC(_colIndexes, _numRows, aDict, _data, getCachedCounts());
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		_data.write(out);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		super.readFields(in);
		_data = MapToFactory.readIn(in);
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
