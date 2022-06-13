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
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.instructions.cp.CM_COV_Object;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

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
	protected AOffset _indexes;
	/** The default value stored in this column group */
	protected double[] _defaultTuple;

	/**
	 * Constructor for serialization
	 * 
	 * @param numRows Number of rows contained
	 */
	protected ColGroupSDCSingle(int numRows) {
		super(numRows);
	}

	private ColGroupSDCSingle(int[] colIndices, int numRows, ADictionary dict, double[] defaultTuple, AOffset offsets,
		int[] cachedCounts) {
		super(colIndices, numRows, dict, cachedCounts);
		_indexes = offsets;
		_zeros = false;
		_defaultTuple = defaultTuple;

		if(_indexes == null)
			throw new NullPointerException("null indexes is invalid for SDCSingle");
	}

	protected static AColGroup create(int[] colIndexes, int numRows, ADictionary dict, double[] defaultTuple,
		AOffset offsets, int[] cachedCounts) {
		final boolean allZero = FORUtil.allZero(defaultTuple);
		if(dict == null && allZero)
			return new ColGroupEmpty(colIndexes);
		else if(dict == null) {
			ADictionary ADict = new Dictionary(new double[colIndexes.length]);
			return new ColGroupSDCSingle(colIndexes, numRows, ADict, defaultTuple, offsets, cachedCounts);
		}
		else if(allZero)
			return ColGroupSDCSingleZeros.create(colIndexes, numRows, dict, offsets, cachedCounts);
		else
			return new ColGroupSDCSingle(colIndexes, numRows, dict, defaultTuple, offsets, cachedCounts);
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
			return _defaultTuple[colIdx];
		else
			return _dict.getValue(colIdx);
	}

	@Override
	public ADictionary getDictionary() {
		throw new NotImplementedException(
			"Not implemented getting the dictionary out, and i think we should consider removing the option");
	}

	@Override
	protected double[] preAggSumRows() {
		return _dict.sumAllRowsToDoubleWithDefault(_defaultTuple);
	}

	@Override
	protected double[] preAggSumSqRows() {
		return _dict.sumAllRowsToDoubleSqWithDefault(_defaultTuple);
	}

	@Override
	protected double[] preAggProductRows() {
		throw new NotImplementedException("Should implement preAgg with extra cell");
	}

	@Override
	protected double[] preAggBuiltinRows(Builtin builtin) {
		return _dict.aggregateRowsWithDefault(builtin, _defaultTuple);
	}

	@Override
	protected double computeMxx(double c, Builtin builtin) {
		double ret = _dict.aggregate(c, builtin);
		for(int i = 0; i < _defaultTuple.length; i++)
			ret = builtin.execute(ret, _defaultTuple[i]);
		return ret;
	}

	@Override
	protected void computeColMxx(double[] c, Builtin builtin) {
		_dict.aggregateCols(c, builtin, _colIndexes);
		for(int x = 0; x < _colIndexes.length; x++)
			c[_colIndexes[x]] = builtin.execute(c[_colIndexes[x]], _defaultTuple[x]);
	}

	@Override
	protected void computeRowSums(double[] c, int rl, int ru, double[] preAgg) {
		computeRowSums(c, rl, ru, preAgg, _indexes, _numRows);
	}

	protected static final void computeRowSums(double[] c, int rl, int ru, double[] preAgg, AOffset indexes, int nRows) {
		int r = rl;
		final AIterator it = indexes.getIterator(rl);
		final double def = preAgg[1];
		final double norm = preAgg[0];
		if(it != null && it.value() > ru)
			indexes.cacheIterator(it, ru);
		else if(it != null && ru >= indexes.getOffsetToLast()) {
			final int maxOff = indexes.getOffsetToLast();
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
			indexes.cacheIterator(it, ru);
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
	protected void computeSum(double[] c, int nRows) {
		super.computeSum(c, nRows);
		int count = _numRows - getCounts()[0];
		for(int x = 0; x < _defaultTuple.length; x++)
			c[0] += _defaultTuple[x] * count;
	}

	@Override
	public void computeColSums(double[] c, int nRows) {
		super.computeColSums(c, nRows);
		int count = _numRows - getCounts()[0];
		for(int x = 0; x < _colIndexes.length; x++)
			c[_colIndexes[x]] += _defaultTuple[x] * count;
	}

	@Override
	protected void computeSumSq(double[] c, int nRows) {
		super.computeSumSq(c, nRows);
		int count = _numRows - getCounts()[0];
		for(int x = 0; x < _colIndexes.length; x++)
			c[0] += _defaultTuple[x] * _defaultTuple[x] * count;
	}

	@Override
	protected void computeColSumsSq(double[] c, int nRows) {
		super.computeColSumsSq(c, nRows);
		int count = _numRows - getCounts()[0];
		for(int x = 0; x < _colIndexes.length; x++)
			c[_colIndexes[x]] += _defaultTuple[x] * _defaultTuple[x] * count;
	}

	@Override
	protected void computeProduct(double[] c, int nRows) {
		final int count = _numRows - getCounts()[0];
		_dict.productWithDefault(c, getCounts(), _defaultTuple, count);
	}

	@Override
	protected void computeColProduct(double[] c, int nRows) {
		super.computeColProduct(c, nRows);
		int count = _numRows - getCounts()[0];
		for(int x = 0; x < _colIndexes.length; x++)
			c[_colIndexes[x]] *= Math.pow(_defaultTuple[x], count);
	}

	@Override
	protected void computeRowProduct(double[] c, int rl, int ru, double[] preAgg) {
		throw new NotImplementedException();
	}

	@Override
	public int[] getCounts(int[] counts) {
		counts[0] = _indexes.getSize();
		return counts;
	}

	@Override
	public long estimateInMemorySize() {
		long size = super.estimateInMemorySize();
		size += _indexes.getInMemorySize();
		size += 8 * _colIndexes.length;
		return size;
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		final double[] newDefaultTuple = new double[_defaultTuple.length];
		for(int i = 0; i < _defaultTuple.length; i++)
			newDefaultTuple[i] = op.executeScalar(_defaultTuple[i]);
		final ADictionary nDict = _dict.applyScalarOp(op);
		return create(_colIndexes, _numRows, nDict, newDefaultTuple, _indexes, getCachedCounts());
	}

	@Override
	public AColGroup unaryOperation(UnaryOperator op) {
		final double[] newDefaultTuple = new double[_defaultTuple.length];
		for(int i = 0; i < _defaultTuple.length; i++)
			newDefaultTuple[i] = op.fn.execute(_defaultTuple[i]);
		final ADictionary nDict = _dict.applyUnaryOp(op);
		return create(_colIndexes, _numRows, nDict, newDefaultTuple, _indexes, getCachedCounts());
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		final double[] newDefaultTuple = new double[_defaultTuple.length];
		for(int i = 0; i < _defaultTuple.length; i++)
			newDefaultTuple[i] = op.fn.execute(v[_colIndexes[i]], _defaultTuple[i]);
		final ADictionary newDict = _dict.binOpLeft(op, v, _colIndexes);
		return create(_colIndexes, _numRows, newDict, newDefaultTuple, _indexes, getCachedCounts());
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		final double[] newDefaultTuple = new double[_defaultTuple.length];
		for(int i = 0; i < _defaultTuple.length; i++)
			newDefaultTuple[i] = op.fn.execute(_defaultTuple[i], v[_colIndexes[i]]);
		final ADictionary newDict = _dict.binOpRight(op, v, _colIndexes);
		return new ColGroupSDCSingle(_colIndexes, _numRows, newDict, newDefaultTuple, _indexes, getCachedCounts());
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		_indexes.write(out);
		for(double d : _defaultTuple)
			out.writeDouble(d);

	}

	@Override
	public void readFields(DataInput in) throws IOException {
		super.readFields(in);
		_indexes = OffsetFactory.readIn(in);
		_defaultTuple = new double[_colIndexes.length];
		for(int i = 0; i < _colIndexes.length; i++)
			_defaultTuple[i] = in.readDouble();
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += _indexes.getExactSizeOnDisk();
		ret += 8 * _colIndexes.length; // _default tuple values.
		return ret;
	}

	@Override
	public AColGroup replace(double pattern, double replace) {
		ADictionary replaced = _dict.replace(pattern, replace, _colIndexes.length);
		double[] newDefaultTuple = new double[_defaultTuple.length];
		for(int i = 0; i < _defaultTuple.length; i++)
			newDefaultTuple[i] = _defaultTuple[i] == pattern ? replace : _defaultTuple[i];

		return create(_colIndexes, _numRows, replaced, newDefaultTuple, _indexes, getCachedCounts());
	}

	@Override
	public AColGroup extractCommon(double[] constV) {
		for(int i = 0; i < _colIndexes.length; i++)
			constV[_colIndexes[i]] += _defaultTuple[i];

		ADictionary subtractedDict = _dict.subtractTuple(_defaultTuple);
		return ColGroupSDCSingleZeros.create(_colIndexes, _numRows, subtractedDict, _indexes, getCachedCounts());
	}

	@Override
	public long getNumberNonZeros(int nRows) {
		long nnz = super.getNumberNonZeros(nRows);
		final int count = _numRows - getCounts()[0];
		for(int x = 0; x < _colIndexes.length; x++)
			nnz += _defaultTuple[x] != 0 ? count : 0;
		return nnz;
	}

	@Override
	public CM_COV_Object centralMoment(CMOperator op, int nRows) {
		CM_COV_Object ret = super.centralMoment(op, nRows);
		int count = _numRows - getCounts()[0];
		op.fn.execute(ret, _defaultTuple[0], count);
		return ret;
	}

	@Override
	public AColGroup rexpandCols(int max, boolean ignore, boolean cast, int nRows) {
		ADictionary d = _dict.rexpandCols(max, ignore, cast, _colIndexes.length);
		final double def = _defaultTuple[0];
		if(d == null) {
			if(def <= 0 || def > max)
				return ColGroupEmpty.create(max);
			else {
				double[] retDef = new double[max];
				retDef[((int) _defaultTuple[0]) - 1] = 1;
				return new ColGroupSDCSingle(Util.genColsIndices(max), nRows, new Dictionary(new double[max]), retDef,
					_indexes, null);
			}
		}
		else {
			if(def <= 0) {
				if(ignore)
					return ColGroupSDCSingleZeros.create(Util.genColsIndices(max), nRows, d, _indexes, getCachedCounts());
				else
					throw new DMLRuntimeException("Invalid content of zero in rexpand");
			}
			else if(def > max)
				return ColGroupSDCSingleZeros.create(Util.genColsIndices(max), nRows, d, _indexes, getCachedCounts());
			else {
				double[] retDef = new double[max];
				retDef[((int) _defaultTuple[0]) - 1] = 1;
				return new ColGroupSDCSingle(Util.genColsIndices(max), nRows, d, retDef, _indexes, getCachedCounts());
			}
		}
	}

	@Override
	public double getCost(ComputationCostEstimator e, int nRows) {
		final int nVals = getNumValues();
		final int nCols = getNumCols();
		final int nRowsScanned = getCounts()[0];
		return e.getCost(nRows, nRowsScanned, nCols, nVals, _dict.getSparsity());
	}

	@Override
	protected AColGroup sliceMultiColumns(int idStart, int idEnd, int[] outputCols) {
		ColGroupSDCSingle ret = (ColGroupSDCSingle) super.sliceMultiColumns(idStart, idEnd, outputCols);
		ret._defaultTuple = new double[idEnd - idStart];
		for(int i = idStart, j = 0; i < idEnd; i++, j++)
			ret._defaultTuple[j] = _defaultTuple[i];
		return ret;
	}

	@Override
	protected AColGroup sliceSingleColumn(int idx) {
		ColGroupSDCSingle ret = (ColGroupSDCSingle) super.sliceSingleColumn(idx);
		ret._defaultTuple = new double[1];
		ret._defaultTuple[0] = _defaultTuple[idx];
		return ret;
	}

	@Override
	public boolean containsValue(double pattern) {
		if(pattern == 0 && _zeros)
			return true;
		boolean ret = _dict.containsValue(pattern);
		if(ret == true)
			return ret;
		else {
			for(double v : _defaultTuple)
				if(v == pattern)
					return true;
			return false;
		}
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s", "Default: "));
		sb.append(Arrays.toString(_defaultTuple));
		sb.append(String.format("\n%15s", "Indexes: "));
		sb.append(_indexes.toString());
		return sb.toString();
	}
}
