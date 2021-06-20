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
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
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
	public static final long serialVersionUID = -3204391646123465004L;

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
	protected void decompressToBlockUnSafeSparseDictionary(MatrixBlock target, int rl, int ru, int offT,
		SparseBlock sb) {
		final int tCol = target.getNumColumns();
		final double[] c = target.getDenseBlockValues();
		offT = offT * tCol;
		for(int i = rl; i < ru; i++, offT += tCol) {
			final int rowIndex = _data.getIndex(i);
			if(sb.isEmpty(rowIndex))
				continue;
			final int apos = sb.pos(rowIndex);
			final int alen = sb.size(rowIndex) + apos;
			final double[] avals = sb.values(rowIndex);
			final int[] aix = sb.indexes(rowIndex);
			for(int j = apos; j < alen; j++)
				c[offT + _colIndexes[aix[j]]] += avals[j];

		}
	}

	@Override
	protected void decompressToBlockUnSafeDenseDictionary(MatrixBlock target, int rl, int ru, int offT,
		double[] values) {
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
			for(int rc = 0, offLeft = rowLeft * _numRows; rc < _numRows; rc++, offLeft++) {
				preAV[offOut + _data.getIndex(rc)] += mV[offLeft];
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
