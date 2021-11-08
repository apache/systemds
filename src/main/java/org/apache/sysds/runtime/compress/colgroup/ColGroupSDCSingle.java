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

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.data.DenseBlock;
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
public class ColGroupSDCSingle extends AColGroupValue {
	private static final long serialVersionUID = 3883228464052204200L;
	/**
	 * Sparse row indexes for the data
	 */
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
	protected void decompressToDenseBlockDenseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values) {
		final int nCol = _colIndexes.length;
		final int offsetToDefault = values.length - nCol;
		final AIterator it = _indexes.getIterator(rl);

		int offT = rl + offR;
		int i = rl;
		for(; i < ru && it.hasNext(); i++, offT++) {
			final double[] c = db.values(offT);
			final int off = db.pos(offT) + offC;
			if(it.value() == i) {
				for(int j = 0; j < nCol; j++)
					c[off + _colIndexes[j]] += values[j];
				it.next();
			}
			else
				for(int j = 0; j < nCol; j++)
					c[off + _colIndexes[j]] += values[offsetToDefault + j];
		}

		for(; i < ru; i++, offT++) {
			final double[] c = db.values(offT);
			final int off = db.pos(offT) + offC;
			for(int j = 0; j < nCol; j++)
				c[off + _colIndexes[j]] += values[offsetToDefault + j];
		}

		_indexes.cacheIterator(it, ru);
	}

	@Override
	protected void decompressToDenseBlockSparseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		SparseBlock values) {
		throw new NotImplementedException();
	}

	@Override
	protected void decompressToSparseBlockSparseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		SparseBlock sb) {
		throw new NotImplementedException();
	}

	@Override
	protected void decompressToSparseBlockDenseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		double[] values) {
		final int nCol = _colIndexes.length;
		final int offsetToDefault = values.length - nCol;
		final AIterator it = _indexes.getIterator(rl);

		int offT = rl + offR;
		int i = rl;
		for(; i < ru && it.hasNext(); i++, offT++) {
			if(it.value() == i) {
				for(int j = 0; j < nCol; j++)
					ret.append(offT, _colIndexes[j] + offC, values[j]);
				it.next();
			}
			else
				for(int j = 0; j < nCol; j++)
					ret.append(offT, _colIndexes[j] + offC, values[offsetToDefault + j]);
		}

		for(; i < ru; i++, offT++)
			for(int j = 0; j < nCol; j++)
				ret.append(offT, _colIndexes[j] + offC, values[offsetToDefault + j]);

		_indexes.cacheIterator(it, ru);
	}

	@Override
	public double getIdx(int r, int colIdx) {
		AIterator it = _indexes.getIterator(r);
		if(it.value() == r)
			return _dict.getValue(colIdx);
		else
			return _dict.getValue(_colIndexes.length + colIdx);
	}

	@Override
	protected void computeRowSums(double[] c, boolean square, int rl, int ru) {

		// // pre-aggregate nnz per value tuple
		final double[] vals = _dict.sumAllRowsToDouble(square, _colIndexes.length);
		final AIterator it = _indexes.getIterator();

		int rix = rl;
		it.skipTo(rl);
		for(; rix < ru && it.hasNext(); rix++) {
			if(it.value() != rix)
				c[rix] += vals[1];
			else {
				c[rix] += vals[0];
				it.next();
			}
		}
		for(; rix < ru; rix++) {
			c[rix] += vals[1];
		}
	}

	@Override
	protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru) {
		final double[] vals = _dict.aggregateTuples(builtin, _colIndexes.length);
		final AIterator it = _indexes.getIterator(rl);
		int rix = rl;

		for(; rix < ru && it.hasNext(); rix++) {
			if(it.value() != rix)
				c[rix] = builtin.execute(c[rix], vals[1]);
			else {
				c[rix] = builtin.execute(c[rix], vals[0]);
				it.next();
			}
		}

		// cover remaining rows with default value
		for(; rix < ru; rix++)
			c[rix] = builtin.execute(c[rix], vals[1]);
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
		return new ColGroupSDCSingle(_colIndexes, _numRows, applyScalarOp(op), _indexes, getCachedCounts());
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
	public void leftMultByMatrix(MatrixBlock matrix, MatrixBlock result, int rl, int ru) {
		// This method should not be called since if there is a matrix multiplication
		// the default value is transformed to be zero, and this column group would be allocated as a
		// SDC Zeros version
		throw new DMLCompressionException("This method should never be called");
	}

	@Override
	public void leftMultByAColGroup(AColGroup lhs, MatrixBlock result) {
		// This method should not be called since if there is a matrix multiplication
		// the default value is transformed to be zero, and this column group would be allocated as a
		// SDC Zeros version
		throw new DMLCompressionException("This method should never be called");
	}

	@Override
	public void tsmmAColGroup(AColGroup other, MatrixBlock result) {
		// This method should not be called since if there is a matrix multiplication
		// the default value is transformed to be zero, and this column group would be allocated as a
		// SDC Zeros version
		throw new DMLCompressionException("This method should never be called");
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s ", "Indexes: "));
		sb.append(_indexes.toString());
		return sb.toString();
	}
}
