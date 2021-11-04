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

import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

/**
 * Class to encapsulate information about a column group that is encoded with dense dictionary encoding (DDC).
 */
public class ColGroupDDC extends APreAgg {
	private static final long serialVersionUID = -5769772089913918987L;

	protected AMapToData _data;

	/**
	 * Constructor for serialization
	 * 
	 * @param numRows number of rows
	 */
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
	protected void decompressToBlockSparseDictionary(MatrixBlock target, int rl, int ru, int offT, SparseBlock sb) {
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
	protected void decompressToBlockDenseDictionary(MatrixBlock target, int rl, int ru, int offT, double[] values) {
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
	public double getIdx(int r, int colIdx) {
		return _dict.getValue(_data.getIndex(r) * _colIndexes.length + colIdx);
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
		for(int i = 0; i < _numRows; i++) {
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
			preAggregateDense(m, preAgg, rl, ru, 0, _numRows);
	}

	@Override
	public void preAggregateDense(MatrixBlock m, MatrixBlock preAgg, int rl, int ru, int cl, int cu) {
		_data.preAggregateDense(m, preAgg, rl, ru, cl, cu);
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
	public void preAggregateThatDDCStructure(ColGroupDDC that, Dictionary ret) {
		final int nCol = that._colIndexes.length;
		for(int r = 0; r < _numRows; r++)
			that._dict.addToEntry(ret, that._data.getIndex(r), this._data.getIndex(r), nCol);
	}

	@Override
	public void preAggregateThatSDCZerosStructure(ColGroupSDCZeros that, Dictionary ret) {
		final AIterator itThat = that._indexes.getIterator();
		final int nCol = that._colIndexes.length;

		while(itThat.hasNext()) {
			final int to = _data.getIndex(itThat.value());
			final int fr = that._data.getIndex(itThat.getDataIndexAndIncrement());
			that._dict.addToEntry(ret, fr, to, nCol);
		}
	}

	@Override
	public void preAggregateThatSDCSingleZerosStructure(ColGroupSDCSingleZeros that, Dictionary ret) {
		final AIterator itThat = that._indexes.getIterator();
		final int nCol = that._colIndexes.length;
		while(itThat.hasNext()) {
			final int to = _data.getIndex(itThat.value());
			that._dict.addToEntry(ret, 0, to, nCol);
			itThat.next();
		}
	}

	@Override
	public boolean sameIndexStructure(AColGroupCompressed that) {
		return that instanceof ColGroupDDC && ((ColGroupDDC) that)._data == _data;
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
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		ADictionary ret = _dict.binOpLeft(op, v, _colIndexes);
		return new ColGroupDDC(_colIndexes, _numRows, ret, _data, getCachedCounts());
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		ADictionary ret = _dict.binOpRight(op, v, _colIndexes);
		return new ColGroupDDC(_colIndexes, _numRows, ret, _data, getCachedCounts());
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
