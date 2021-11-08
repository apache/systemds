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
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
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
public class ColGroupSDC extends AColGroupValue {
	private static final long serialVersionUID = 769993538831949086L;
	/**
	 * Sparse row indexes for the data
	 */
	protected transient AOffset _indexes;
	/**
	 * Pointers to row indexes in the dictionary. Note the dictionary has one extra entry.
	 */
	protected transient AMapToData _data;

	/**
	 * Constructor for serialization
	 * 
	 * @param numRows Number of rows contained
	 */
	protected ColGroupSDC(int numRows) {
		super(numRows);
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
				int offset = _data.getIndex(it.getDataIndexAndIncrement()) * nCol;
				for(int j = 0; j < nCol; j++)
					c[off + _colIndexes[j]] += values[offset + j];
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
		SparseBlock sb) {
		throw new NotImplementedException();
		// final int offsetToDefault = sb.numRows() - 1;
		// final int defApos = sb.pos(offsetToDefault);
		// final int defAlen = sb.size(offsetToDefault) + defApos;
		// final double[] defAvals = sb.values(offsetToDefault);
		// final int[] defAix = sb.indexes(offsetToDefault);
		// final DenseBlock db = target.getDenseBlock();

		// int i = rl;
		// AIterator it = _indexes.getIterator(rl);
		// for(; i < ru && it.hasNext(); i++, offT++) {
		// final double[] c = db.values(offT);
		// final int off = db.pos(offT);
		// if(it.value() == i) {
		// int dictIndex = _data.getIndex(it.getDataIndexAndIncrement());
		// if(sb.isEmpty(dictIndex))
		// continue;
		// final int apos = sb.pos(dictIndex);
		// final int alen = sb.size(dictIndex) + apos;
		// final double[] avals = sb.values(dictIndex);
		// final int[] aix = sb.indexes(dictIndex);
		// for(int j = apos; j < alen; j++)
		// c[off + _colIndexes[aix[j]]] += avals[j];
		// }
		// else
		// for(int j = defApos; j < defAlen; j++)
		// c[off + _colIndexes[defAix[j]]] += defAvals[j];
		// }

		// for(; i < ru; i++, offT++) {
		// final double[] c = db.values(offT);
		// final int off = db.pos(offT);
		// for(int j = defApos; j < defAlen; j++)
		// c[off + _colIndexes[defAix[j]]] += defAvals[j];
		// }

		// _indexes.cacheIterator(it, ru);
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
			// final double[] c = db.values(offT);
			// final int off = db.pos(offT) + offC;
			if(it.value() == i) {
				int offset = _data.getIndex(it.getDataIndexAndIncrement()) * nCol;
				for(int j = 0; j < nCol; j++)
					ret.append(offT, _colIndexes[j] + offC, values[offset + j]);
				// c[off + _colIndexes[j]] += values[offset + j];
			}
			else
				for(int j = 0; j < nCol; j++)
					ret.append(offT, _colIndexes[j] + offC, values[offsetToDefault + j]);
			// c[off + _colIndexes[j]] += values[offsetToDefault + j];
		}

		for(; i < ru; i++, offT++) {
			// final double[] c = db.values(offT);
			// final int off = db.pos(offT) + offC;
			for(int j = 0; j < nCol; j++)
				ret.append(offT, _colIndexes[j] + offC, values[offsetToDefault + j]);
			// c[off + _colIndexes[j]] += values[offsetToDefault + j];
		}

		_indexes.cacheIterator(it, ru);
	}

	@Override
	public double getIdx(int r, int colIdx) {
		final AIterator it = _indexes.getIterator(r);
		final int nCol = _colIndexes.length;
		final int rowOff = it.value() == r ? getIndex(it.getDataIndex()) * nCol : getNumValues() * nCol - nCol;
		return _dict.getValue(rowOff + colIdx);
	}

	@Override
	protected void computeRowSums(double[] c, boolean square, int rl, int ru) {
		final int numVals = getNumValues();
		// // pre-aggregate nnz per value tuple
		double[] vals = _dict.sumAllRowsToDouble(square, _colIndexes.length);

		int rix = rl;
		AIterator it = _indexes.getIterator(rl);
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
		final double[] vals = _dict.aggregateTuples(builtin, _colIndexes.length);
		final AIterator it = _indexes.getIterator(rl);
		int rix = rl;

		for(; rix < ru && it.hasNext(); rix++) {
			if(it.value() != rix)
				c[rix] = builtin.execute(c[rix], vals[numVals - 1]);
			else
				c[rix] = builtin.execute(c[rix], vals[_data.getIndex(it.getDataIndexAndIncrement())]);
		}

		// cover remaining rows with default value
		for(; rix < ru; rix++)
			c[rix] = builtin.execute(c[rix], vals[numVals - 1]);
	}

	@Override
	public int[] getCounts(int[] counts) {
		final int nonDefaultLength = _data.size();
		// final AIterator it = _indexes.getIterator();
		final int defaults = _numRows - nonDefaultLength;
		for(int i = 0; i < nonDefaultLength; i++)
			counts[_data.getIndex(i)]++;

		counts[counts.length - 1] += defaults;

		return counts;
	}

	public int getIndex(int r) {
		return _data.getIndex(r);
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
		return new ColGroupSDC(_colIndexes, _numRows, applyScalarOp(op), _indexes, _data, getCachedCounts());
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		ADictionary ret = _dict.binOpLeft(op, v, _colIndexes);
		return new ColGroupSDC(_colIndexes, _numRows, ret, _indexes, _data, getCachedCounts());
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		ADictionary ret = _dict.binOpRight(op, v, _colIndexes);
		return new ColGroupSDC(_colIndexes, _numRows, ret, _indexes, _data, getCachedCounts());
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

	public ColGroupSDCZeros extractCommon(double[] constV) {
		double[] commonV = _dict.getTuple(getNumValues() - 1, _colIndexes.length);
		if(commonV == null) // The common tuple was all zero. Therefore this column group should never have been SDC.
			return new ColGroupSDCZeros(_colIndexes, _numRows, _dict, _indexes, _data, getCounts());

		for(int i = 0; i < _colIndexes.length; i++)
			constV[_colIndexes[i]] += commonV[i];

		ADictionary subtractedDict = _dict.subtractTuple(commonV);
		return new ColGroupSDCZeros(_colIndexes, _numRows, subtractedDict, _indexes, _data, getCounts());
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
		sb.append(String.format("\n%15s ", "Data: "));
		sb.append(_data.toString());
		return sb.toString();
	}
}
