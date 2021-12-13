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
public class ColGroupSDCSingle extends AMorphingMMColGroup {
	private static final long serialVersionUID = 3883228464052204200L;
	/** Sparse row indexes for the data */
	protected transient AOffset _indexes;

	/**
	 * Constructor for serialization
	 * 
	 * @param numRows Number of rows contained
	 */
	protected ColGroupSDCSingle(int numRows) {
		super(numRows);
	}

	protected ColGroupSDCSingle(int[] colIndices, int numRows, ADictionary dict, AOffset offsets, int[] cachedCounts) {
		super(colIndices, numRows, dict, cachedCounts);
		_indexes = offsets;
		_zeros = false;
	}

	@Override
	public CompressionType getCompType() {
		return CompressionType.SDC;
	}

	@Override
	public ColGroupType getColGroupType() {
		return ColGroupType.SDCSingle;
	}

	@Override
	public double getIdx(int r, int colIdx) {
		final AIterator it = _indexes.getIterator(r);
		if(it == null || it.value() != r)
			return _dict.getValue(_colIndexes.length + colIdx);
		return _dict.getValue(colIdx);
	}

	@Override
	protected void computeRowSums(double[] c, int rl, int ru, double[] vals) {
		int r = rl;
		final AIterator it = _indexes.getIterator(rl);
		final double def = vals[1];
		final double norm = vals[0];
		if(it != null && it.value() > ru)
			_indexes.cacheIterator(it, ru);
		else if(it != null && ru >= _indexes.getOffsetToLast()) {
			final int maxOff = _indexes.getOffsetToLast();
			while(true) {
				if(it.value() == r) {
					c[r] += norm;
					if(it.value() < maxOff)
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
				if(it.value() == r)
					c[r] += norm;
				else
					c[r] += def;
				r++;
			}
			_indexes.cacheIterator(it, ru);
		}

		while(r < ru) {
			c[r] += def;
			r++;
		}
	}

	@Override
	protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] preAgg) {
		computeRowMxx(c, builtin, rl, ru, _indexes, _numRows, preAgg[1], preAgg[0]);
	}

	protected static final void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, AOffset indexes, int nRows,
		double def, double norm) {
		int r = rl;
		final AIterator it = indexes.getIterator(rl);
		if(it != null && it.value() > ru)
			indexes.cacheIterator(it, ru);
		else if(it != null && ru >= indexes.getOffsetToLast()) {
			final int maxOff = indexes.getOffsetToLast();
			while(true) {
				if(it.value() == r) {
					c[r] = builtin.execute(c[r], norm);
					if(it.value() < maxOff)
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
				if(it.value() == r) {
					c[r] = builtin.execute(c[r], norm);
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
		counts[0] = _indexes.getSize();
		counts[1] = _numRows - counts[0];
		return counts;
	}

	@Override
	public long estimateInMemorySize() {
		long size = super.estimateInMemorySize();
		size += _indexes.getInMemorySize();
		return size;
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		return new ColGroupSDCSingle(_colIndexes, _numRows, _dict.applyScalarOp(op), _indexes, getCachedCounts());
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		ADictionary ret = _dict.binOpLeft(op, v, _colIndexes);
		return new ColGroupSDCSingle(_colIndexes, _numRows, ret, _indexes, getCachedCounts());
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		ADictionary ret = _dict.binOpRight(op, v, _colIndexes);
		return new ColGroupSDCSingle(_colIndexes, _numRows, ret, _indexes, getCachedCounts());
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		_indexes.write(out);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		super.readFields(in);
		_indexes = OffsetFactory.readIn(in);
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += _indexes.getExactSizeOnDisk();
		return ret;
	}

	@Override
	public ColGroupSDCSingleZeros extractCommon(double[] constV) {
		double[] commonV = _dict.getTuple(getNumValues() - 1, _colIndexes.length);

		if(commonV == null) // The common tuple was all zero. Therefore this column group should never have been SDC.
			return new ColGroupSDCSingleZeros(_colIndexes, _numRows, _dict, _indexes, getCachedCounts());

		for(int i = 0; i < _colIndexes.length; i++)
			constV[_colIndexes[i]] += commonV[i];

		ADictionary subtractedDict = _dict.subtractTuple(commonV);
		return new ColGroupSDCSingleZeros(_colIndexes, _numRows, subtractedDict, _indexes, getCachedCounts());
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s", "Indexes: "));
		sb.append(_indexes.toString());
		return sb.toString();
	}
}
