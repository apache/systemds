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
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.functionobjects.Builtin;
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
public class ColGroupSDC extends AMorphingMMColGroup {
	private static final long serialVersionUID = 769993538831949086L;

	/** Sparse row indexes for the data */
	protected transient AOffset _indexes;
	/** Pointers to row indexes in the dictionary. Note the dictionary has one extra entry. */
	protected transient AMapToData _data;

	/**
	 * Constructor for serialization
	 * 
	 * @param numRows Number of rows contained
	 */
	protected ColGroupSDC(int numRows) {
		super(numRows);
	}

	private ColGroupSDC(int[] colIndices, int numRows, ADictionary dict, AOffset offsets, AMapToData data,
		int[] cachedCounts) {
		super(colIndices, numRows, dict, cachedCounts);
		_indexes = offsets;
		_data = data;
		_zeros = false;
	}

	protected static AColGroup create(int[] colIndices, int numRows, ADictionary dict, AOffset offsets, AMapToData data,
		int[] cachedCounts) {
		if(dict == null)
			return new ColGroupEmpty(colIndices);
		else
			return new ColGroupSDC(colIndices, numRows, dict, offsets, data, cachedCounts);
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
	public double getIdx(int r, int colIdx) {
		final AIterator it = _indexes.getIterator(r);
		final int rowOff = it == null || it.value() != r ? getNumValues() - 1 : _data.getIndex(it.getDataIndex());
		final int nCol = _colIndexes.length;
		return _dict.getValue(rowOff * nCol + colIdx);
	}

	@Override
	protected void computeRowSums(double[] c, int rl, int ru, double[] preAgg) {
		computeRowSums(c, rl, ru, preAgg, _data, _indexes, _numRows);
	}

	protected static final void computeRowSums(double[] c, int rl, int ru, double[] preAgg, AMapToData data,
		AOffset indexes, int nRows) {
		int r = rl;
		final AIterator it = indexes.getIterator(rl);
		final double def = preAgg[preAgg.length - 1];
		if(it != null && it.value() > ru)
			indexes.cacheIterator(it, ru);
		else if(it != null && ru >= indexes.getOffsetToLast()) {
			final int maxId = data.size() - 1;
			while(true) {
				if(it.value() == r) {
					c[r] += preAgg[data.getIndex(it.getDataIndex())];
					if(it.getDataIndex() < maxId)
						it.next();
					else {
						r++;
						break;
					}
				}
				else
					c[r] += def;
				r++;
			}
		}
		else if(it != null) {
			while(r < ru) {
				if(it.value() == r){
					c[r] += preAgg[data.getIndex(it.getDataIndex())];
					it.next();
				}
				else
					c[r] += def;
				r++;
			}
			indexes.cacheIterator(it, ru);
		}

		while(r < ru) {
			c[r] += def;
			r++;
		}
	}

	@Override
	protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] preAgg) {
		computeRowMxx(c, builtin, rl, ru, preAgg, _data, _indexes, _numRows, preAgg[preAgg.length - 1]);
	}

	protected static final void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] vals,
		AMapToData data, AOffset indexes, int nRows, double def) {
		int r = rl;
		final AIterator it = indexes.getIterator(rl);
		if(it != null && it.value() > ru)
			indexes.cacheIterator(it, ru);
		else if(it != null && ru >= indexes.getOffsetToLast()) {
			final int maxId = data.size() - 1;
			while(true) {
				if(it.value() == r) {
					c[r] = builtin.execute(c[r], vals[data.getIndex(it.getDataIndex())]);
					if(it.getDataIndex() < maxId)
						it.next();
					else {
						r++;
						break;
					}
				}
				else
					c[r] = builtin.execute(c[r], def);
				r++;
			}
		}
		else if(it != null) {
			while(r < ru) {
				if(it.value() == r){
					c[r] = builtin.execute(c[r], vals[data.getIndex(it.getDataIndex())]);
					it.next();
				}
				else
					c[r] = builtin.execute(c[r], def);
				r++;
			}
			indexes.cacheIterator(it, ru);
		}

		while(r < ru) {
			c[r] = builtin.execute(c[r], def);
			r++;
		}
	}

	@Override
	public int[] getCounts(int[] counts) {
		return _data.getCounts(counts, _numRows);
	}

	@Override
	public long estimateInMemorySize() {
		long size = super.estimateInMemorySize();
		size += _indexes.getInMemorySize();
		size += _data.getInMemorySize();
		return size;
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		return create(_colIndexes, _numRows, _dict.applyScalarOp(op), _indexes, _data, getCachedCounts());
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		ADictionary ret = _dict.binOpLeft(op, v, _colIndexes);
		return create(_colIndexes, _numRows, ret, _indexes, _data, getCachedCounts());
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		ADictionary ret = _dict.binOpRight(op, v, _colIndexes);
		return create(_colIndexes, _numRows, ret, _indexes, _data, getCachedCounts());
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
		_data = MapToFactory.readIn(in);
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += _data.getExactSizeOnDisk();
		ret += _indexes.getExactSizeOnDisk();
		return ret;
	}

	@Override
	public AColGroup extractCommon(double[] constV) {
		double[] commonV = _dict.getTuple(getNumValues() - 1, _colIndexes.length);
		if(commonV == null) // The common tuple was all zero. Therefore this column group should never have been SDC.
			return ColGroupSDCZeros.create(_colIndexes, _numRows, _dict, _indexes, _data, getCounts());

		for(int i = 0; i < _colIndexes.length; i++)
			constV[_colIndexes[i]] += commonV[i];

		ADictionary subtractedDict = _dict.subtractTuple(commonV);
		return ColGroupSDCZeros.create(_colIndexes, _numRows, subtractedDict, _indexes, _data, getCounts());
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s", "Indexes: "));
		sb.append(_indexes.toString());
		sb.append(String.format("\n%15s", "Data: "));
		sb.append(_data.toString());
		return sb.toString();
	}
}
