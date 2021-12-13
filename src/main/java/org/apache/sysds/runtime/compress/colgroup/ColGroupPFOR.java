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
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

/**
 * ColGroup for Patched Frame Of Reference.
 * 
 * This column group fits perfectly into the collection of compression groups
 * 
 * It can be constructed when a SDCZeros group get a non zero default value. Then a natural extension is to transform
 * the group into a PFOR group, since the default value is then treated as an offset, and the dictionary can be copied
 * with no modifications.
 * 
 */
public class ColGroupPFOR extends AMorphingMMColGroup {

	private static final long serialVersionUID = 3883228464052204203L;

	/** Sparse row indexes for the data that is nonZero */
	protected AOffset _indexes;

	/** Pointers to row indexes in the dictionary. */
	protected transient AMapToData _data;

	/** Reference values in this column group */
	protected double[] _reference;

	/**
	 * Constructor for serialization
	 * 
	 * @param numRows Number of rows contained
	 */
	protected ColGroupPFOR(int numRows) {
		super(numRows);
	}

	private ColGroupPFOR(int[] colIndices, int numRows, ADictionary dict, AOffset indexes, AMapToData data,
		int[] cachedCounts, double[] reference) {
		super(colIndices, numRows, dict, cachedCounts);
		_data = data;
		_indexes = indexes;
		_zeros = allZero(reference);
		_reference = reference;
	}

	protected static AColGroup create(int[] colIndices, int numRows, ADictionary dict, AOffset indexes, AMapToData data,
		int[] cachedCounts, double[] reference) {
		if(dict == null) {
			// either ColGroupEmpty or const
			boolean allZero = true;
			for(double d : reference)
				if(d != 0) {
					allZero = false;
					break;
				}

			if(allZero)
				return new ColGroupEmpty(colIndices);
			else
				return ColGroupFactory.genColGroupConst(colIndices, reference);
		}
		return new ColGroupPFOR(colIndices, numRows, dict, indexes, data, cachedCounts, reference);
	}

	private final static boolean allZero(double[] in) {
		for(double v : in)
			if(v != 0)
				return false;
		return true;
	}

	@Override
	public CompressionType getCompType() {
		return CompressionType.PFOR;
	}

	@Override
	public ColGroupType getColGroupType() {
		return ColGroupType.PFOR;
	}

	@Override
	public int[] getCounts(int[] counts) {
		return _data.getCounts(counts, _numRows);
	}

	private final double refSum() {
		double ret = 0;
		for(double d : _reference)
			ret += d;
		return ret;
	}

	@Override
	protected void computeRowSums(double[] c, int rl, int ru, double[] preAgg) {
		ColGroupSDC.computeRowSums(c, rl, ru, preAgg, _data, _indexes, _numRows);
	}

	@Override
	protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] preAgg) {
		ColGroupSDC.computeRowMxx(c, builtin, rl, ru, preAgg, _data, _indexes, _numRows, preAgg[preAgg.length - 1]);
	}

	@Override
	public double getIdx(int r, int colIdx) {
		final AIterator it = _indexes.getIterator(r);
		final int nCol = _colIndexes.length;
		if(it == null || it.value() != r)
			return _reference[colIdx];
		final int rowOff = _data.getIndex(it.getDataIndex()) * nCol;
		return _dict.getValue(rowOff + colIdx) + _reference[colIdx];
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		final double[] newRef = new double[_reference.length];
		for(int i = 0; i < _reference.length; i++)
			newRef[i] = op.executeScalar(_reference[i]);
		if(op.fn instanceof Plus || op.fn instanceof Minus) {
			return create(_colIndexes, _numRows, _dict, _indexes, _data, getCachedCounts(), newRef);
		}
		else if(op.fn instanceof Multiply || op.fn instanceof Divide) {
			final ADictionary newDict = _dict.applyScalarOp(op);
			return create(_colIndexes, _numRows, newDict, _indexes, _data, getCachedCounts(), newRef);
		}
		else {
			final ADictionary newDict = _dict.applyScalarOp(op, _reference, newRef);
			return create(_colIndexes, _numRows, newDict, _indexes, _data, getCachedCounts(), newRef);
		}
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		final double[] newRef = new double[_reference.length];
		for(int i = 0; i < _reference.length; i++)
			newRef[i] = op.fn.execute(v[_colIndexes[i]], _reference[i]);

		if(op.fn instanceof Plus || op.fn instanceof Minus)
			return create(_colIndexes, _numRows, _dict, _indexes, _data, getCachedCounts(), newRef);
		else if(op.fn instanceof Multiply || op.fn instanceof Divide) {
			final ADictionary newDict = _dict.binOpLeft(op, v, _colIndexes);
			return create(_colIndexes, _numRows, newDict, _indexes, _data, getCachedCounts(), newRef);
		}
		else {
			final ADictionary newDict = _dict.binOpLeft(op, v, _colIndexes, _reference, newRef);
			return create(_colIndexes, _numRows, newDict, _indexes, _data, getCachedCounts(), newRef);
		}
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		final double[] newRef = new double[_reference.length];
		for(int i = 0; i < _reference.length; i++)
			newRef[i] = op.fn.execute(_reference[i], v[_colIndexes[i]]);
		if(op.fn instanceof Plus || op.fn instanceof Minus)
			return new ColGroupPFOR(_colIndexes, _numRows, _dict, _indexes, _data, getCachedCounts(), newRef);
		else if(op.fn instanceof Multiply || op.fn instanceof Divide) {
			final ADictionary newDict = _dict.binOpRight(op, v, _colIndexes);
			return new ColGroupPFOR(_colIndexes, _numRows, newDict, _indexes, _data, getCachedCounts(), newRef);
		}
		else {
			final ADictionary newDict = _dict.binOpRight(op, v, _colIndexes, _reference, newRef);
			return new ColGroupPFOR(_colIndexes, _numRows, newDict, _indexes, _data, getCachedCounts(), newRef);
		}
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		_indexes.write(out);
		_data.write(out);
		for(double d : _reference)
			out.writeDouble(d);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		super.readFields(in);
		_indexes = OffsetFactory.readIn(in);
		_data = MapToFactory.readIn(in);
		_reference = new double[_colIndexes.length];
		for(int i = 0; i < _colIndexes.length; i++)
			_reference[i] = in.readDouble();
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += _data.getExactSizeOnDisk();
		ret += _indexes.getExactSizeOnDisk();
		ret += 8 * _colIndexes.length; // reference values.
		return ret;
	}

	@Override
	public AColGroup replace(double pattern, double replace) {
		boolean patternInReference = false;
		for(double d : _reference)
			if(pattern == d) {
				patternInReference = true;
				break;
			}

		if(patternInReference) {
			throw new NotImplementedException("Not Implemented replace where a value in reference should be replaced");
			// _dict.replace(pattern, replace, _reference, _newReplace);
		}
		else {
			final ADictionary newDict = _dict.replace(pattern, replace, _reference);
			return create(_colIndexes, _numRows, newDict, _indexes, _data, getCachedCounts(), _reference);
		}

	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s", "Indexes: "));
		sb.append(_indexes.toString());
		sb.append(String.format("\n%15s", "Data: "));
		sb.append(_data);
		sb.append(String.format("\n%15s", "Reference:"));
		sb.append(Arrays.toString(_reference));
		return sb.toString();
	}

	@Override
	protected double computeMxx(double c, Builtin builtin) {
		return _dict.aggregate(c, builtin, _reference);
	}

	@Override
	protected void computeColMxx(double[] c, Builtin builtin) {
		_dict.aggregateCols(c, builtin, _colIndexes, _reference);
	}

	@Override
	protected void computeSum(double[] c, int nRows) {
		super.computeSum(c, nRows);
		final double refSum = refSum();
		c[0] += refSum * nRows;
	}

	@Override
	public void computeColSums(double[] c, int nRows) {
		super.computeColSums(c, nRows);
		for(int i = 0; i < _colIndexes.length; i++)
			c[_colIndexes[i]] += _reference[i] * nRows;
	}

	@Override
	protected void computeSumSq(double[] c, int nRows) {
		c[0] += _dict.sumSq(getCounts(), _reference);
	}

	@Override
	protected void computeColSumsSq(double[] c, int nRows) {
		_dict.colSumSq(c, getCounts(), _colIndexes, _reference);
	}

	@Override
	protected double[] preAggSumRows() {
		return _dict.sumAllRowsToDouble(_reference);
	}

	@Override
	protected double[] preAggSumSqRows() {
		return _dict.sumAllRowsToDoubleSq(_reference);
	}

	@Override
	protected double[] preAggProductRows() {
		throw new NotImplementedException();
	}

	@Override
	protected double[] preAggBuiltinRows(Builtin builtin) {
		return _dict.aggregateRows(builtin, _reference);
	}

	@Override
	protected void computeProduct(double[] c, int nRows) {
		throw new NotImplementedException("Not Implemented PFOR");
	}

	@Override
	protected void computeRowProduct(double[] c, int rl, int ru, double[] preAgg) {
		throw new NotImplementedException("Not Implemented PFOR");
	}

	@Override
	protected void computeColProduct(double[] c, int nRows) {
		throw new NotImplementedException("Not Implemented PFOR");
	}

	@Override
	protected AColGroup sliceSingleColumn(int idx) {
		ColGroupPFOR ret = (ColGroupPFOR) super.sliceSingleColumn(idx);
		// select values from double array.
		ret._reference = new double[1];
		ret._reference[0] = _reference[idx];
		return ret;
	}

	@Override
	protected AColGroup sliceMultiColumns(int idStart, int idEnd, int[] outputCols) {
		ColGroupPFOR ret = (ColGroupPFOR) super.sliceMultiColumns(idStart, idEnd, outputCols);
		final int len = idEnd - idStart;
		ret._reference = new double[len];
		for(int i = 0, ii = idStart; i < len; i++, ii++)
			ret._reference[i] = _reference[ii];

		return ret;
	}

	@Override
	public boolean containsValue(double pattern) {
		if(pattern == 0 && _zeros)
			return true;
		else if(Double.isNaN(pattern) || Double.isInfinite(pattern))
			return containsInfOrNan(pattern) || _dict.containsValue(pattern);
		else
			return _dict.containsValue(pattern, _reference);
	}

	private boolean containsInfOrNan(double pattern) {
		if(Double.isNaN(pattern)) {
			for(double d : _reference)
				if(Double.isNaN(d))
					return true;
			return false;
		}
		else {
			for(double d : _reference)
				if(Double.isInfinite(d))
					return true;
			return false;
		}
	}

	@Override
	public long getNumberNonZeros(int nRows) {
		int[] counts = getCounts();
		return (long) _dict.getNumberNonZeros(counts, _reference, nRows);
	}

	@Override
	public AColGroup extractCommon(double[] constV) {
		for(int i = 0; i < _colIndexes.length; i++)
			constV[_colIndexes[i]] += _reference[i];
		return ColGroupSDCZeros.create(_colIndexes, _numRows, _dict, _indexes, _data, getCounts());
	}
}
