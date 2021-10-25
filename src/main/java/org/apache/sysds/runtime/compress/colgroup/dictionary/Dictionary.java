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

package org.apache.sysds.runtime.compress.colgroup.dictionary;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.utils.MemoryEstimates;

/**
 * This dictionary class aims to encapsulate the storage and operations over unique floating point values of a column
 * group. The primary reason for its introduction was to provide an entry point for specialization such as shared
 * dictionaries, which require additional information.
 */
public class Dictionary extends ADictionary {

	private static final long serialVersionUID = -6517136537249507753L;

	private final double[] _values;

	public Dictionary(double[] values) {
		if(values == null || values.length == 0)
			throw new DMLCompressionException("Invalid construction of dictionary with null array");
		_values = values;
	}

	@Override
	public double[] getValues() {
		return _values;
	}

	@Override
	public double getValue(int i) {
		return (i >= size()) ? 0.0 : _values[i];
	}

	@Override
	public long getInMemorySize() {
		// object + values array + double
		return getInMemorySize(size());
	}

	protected static long getInMemorySize(int valuesCount) {
		// object + values array
		return 16 + (long) MemoryEstimates.doubleArrayCost(valuesCount);
	}

	@Override
	public double aggregate(double init, Builtin fn) {
		// full aggregate can disregard tuple boundaries
		double ret = init;
		for(int i = 0; i < _values.length; i++)
			ret = fn.execute(ret, _values[i]);
		return ret;
	}

	@Override
	public double[] aggregateTuples(Builtin fn, final int nCol) {
		if(nCol == 1)
			return _values;
		final int nRows = _values.length / nCol;
		double[] res = new double[nRows];
		for(int i = 0; i < nRows; i++) {
			final int off = i * nCol;
			res[i] = _values[off];
			for(int j = off + 1; j < off + nCol; j++)
				res[i] = fn.execute(res[i], _values[j]);
		}
		return res;
	}

	@Override
	public Dictionary inplaceScalarOp(ScalarOperator op) {
		// in-place modification of the dictionary
		int len = size();
		for(int i = 0; i < len; i++)
			_values[i] = op.executeScalar(_values[i]);
		return this;
	}

	@Override
	public Dictionary applyScalarOp(ScalarOperator op, double newVal, int numCols) {
		// allocate new array just once because we need to add the newVal.
		double[] values = new double[_values.length + numCols];
		for(int i = 0; i < _values.length; i++)
			values[i] = op.executeScalar(_values[i]);

		Arrays.fill(values, _values.length, _values.length + numCols, newVal);
		return new Dictionary(values);
	}

	@Override
	public Dictionary binOpRight(BinaryOperator op, double[] v, int[] colIndexes) {
		final ValueFunction fn = op.fn;
		final double[] retVals = new double[_values.length];
		final int len = size();
		final int lenV = colIndexes.length;
		for(int i = 0; i < len; i++)
			retVals[i] = fn.execute(_values[i], v[colIndexes[i % lenV]]);
		return new Dictionary(retVals);
	}

	@Override
	public final Dictionary binOpLeft(BinaryOperator op, double[] v, int[] colIndexes) {
		final ValueFunction fn = op.fn;
		final double[] retVals = new double[_values.length];
		final int len = size();
		final int lenV = colIndexes.length;
		for(int i = 0; i < len; i++)
			retVals[i] = fn.execute(v[colIndexes[i % lenV]], _values[i]);
		return new Dictionary(retVals);
	}

	@Override
	public Dictionary applyBinaryRowOpRightAppendNewEntry(BinaryOperator op, double[] v, int[] colIndexes) {
		ValueFunction fn = op.fn;
		final int len = size();
		final int lenV = colIndexes.length;
		final double[] values = new double[len + lenV];
		int i = 0;
		for(; i < len; i++)
			values[i] = fn.execute(_values[i], v[colIndexes[i % lenV]]);
		for(; i < len + lenV; i++)
			values[i] = fn.execute(0, v[colIndexes[i % lenV]]);
		return new Dictionary(values);
	}

	@Override
	public final Dictionary applyBinaryRowOpLeftAppendNewEntry(BinaryOperator op, double[] v, int[] colIndexes) {
		ValueFunction fn = op.fn;
		final int len = size();
		final int lenV = colIndexes.length;
		final double[] values = new double[len + lenV];
		int i = 0;
		for(; i < len; i++)
			values[i] = fn.execute(v[colIndexes[i % lenV]], _values[i]);
		for(; i < len + lenV; i++)
			values[i] = fn.execute(v[colIndexes[i % lenV]], 0);
		return new Dictionary(values);
	}

	@Override
	public Dictionary clone() {
		return new Dictionary(_values.clone());
	}

	@Override
	public Dictionary cloneAndExtend(int len) {
		double[] ret = Arrays.copyOf(_values, _values.length + len);
		return new Dictionary(ret);
	}

	public static Dictionary read(DataInput in) throws IOException {
		int numVals = in.readInt();
		// read distinct values
		double[] values = new double[numVals];
		for(int i = 0; i < numVals; i++)
			values[i] = in.readDouble();
		return new Dictionary(values);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(DictionaryFactory.Type.FP64_DICT.ordinal());
		out.writeInt(size());
		for(int i = 0; i < size(); i++)
			out.writeDouble(_values[i]);
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 4 + 8 * size();
	}

	private int size() {
		return _values.length;
	}

	@Override
	public int getNumberOfValues(int nCol) {
		return _values.length / nCol;
	}

	@Override
	public double[] sumAllRowsToDouble(boolean square, int nrColumns) {
		if(nrColumns == 1 && !square)
			return getValues(); // shallow copy of values

		// pre-aggregate value tuple
		final int numVals = getNumberOfValues(nrColumns);
		double[] ret = new double[numVals];
		for(int k = 0; k < numVals; k++) {
			ret[k] = sumRow(k, square, nrColumns);
		}

		return ret;
	}

	@Override
	public double sumRow(int k, boolean square, int nrColumns) {

		int valOff = k * nrColumns;
		double res = 0.0;
		if(!square) {
			for(int i = 0; i < nrColumns; i++) {
				res += _values[valOff + i];
			}
		}
		else {
			// kSquare
			for(int i = 0; i < nrColumns; i++)
				res += _values[valOff + i] * _values[valOff + i];
		}
		return res;
	}

	@Override
	public double[] colSum(int[] counts, int nCol) {
		final double[] res = new double[nCol];
		int idx = 0;
		for(int k = 0; k < _values.length / nCol; k++) {
			final int cntk = counts[k];
			for(int j = 0; j < nCol; j++)
				res[j] += _values[idx++] * cntk;
		}
		return res;
	}

	@Override
	public void colSum(double[] c, int[] counts, int[] colIndexes, boolean square) {
		for(int k = 0; k < _values.length / colIndexes.length; k++) {
			final int cntk = counts[k];
			for(int j = 0; j < colIndexes.length; j++) {
				double v = _values[k * colIndexes.length + j];
				if(square)
					c[colIndexes[j]] += v * v * cntk;
				else
					c[colIndexes[j]] += v * cntk;
			}
		}

	}

	@Override
	public double sum(int[] counts, int ncol) {
		double out = 0;
		int valOff = 0;
		for(int k = 0; k < _values.length / ncol; k++) {
			int countK = counts[k];
			for(int j = 0; j < ncol; j++) {
				out += getValue(valOff++) * countK;
			}
		}
		return out;
	}

	@Override
	public double sumsq(int[] counts, int ncol) {
		double out = 0;
		int valOff = 0;
		for(int k = 0; k < _values.length / ncol; k++) {
			int countK = counts[k];
			for(int j = 0; j < ncol; j++) {
				double val = getValue(valOff++);
				out += val * val * countK;
			}
		}
		return out;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();

		sb.append("Dictionary:");
		sb.append(Arrays.toString(_values));
		return sb.toString();
	}

	@Override
	public void addMaxAndMin(double[] ret, int[] colIndexes) {

		double[] mins = new double[colIndexes.length];
		double[] maxs = new double[colIndexes.length];
		for(int i = 0; i < colIndexes.length; i++) {
			mins[i] = _values[i];
			maxs[i] = _values[i];
		}
		for(int i = colIndexes.length; i < _values.length; i++) {
			int idx = i % colIndexes.length;
			mins[idx] = Math.min(_values[i], mins[idx]);
			maxs[idx] = Math.max(_values[i], maxs[idx]);
		}
		for(int i = 0; i < colIndexes.length; i++) {
			int idy = colIndexes[i] * 2;
			ret[idy] += mins[i];
			ret[idy + 1] += maxs[i];
		}
	}

	public String getString(int colIndexes) {
		StringBuilder sb = new StringBuilder();
		if(colIndexes == 1)
			sb.append(Arrays.toString(_values));
		else {
			sb.append("[\n\t");
			for(int i = 0; i < _values.length - 1; i++) {
				sb.append(_values[i]);
				sb.append((i) % (colIndexes) == colIndexes - 1 ? "\n\t" : ", ");
			}
			sb.append(_values[_values.length - 1]);
			sb.append("]");
		}
		return sb.toString();
	}

	public ADictionary sliceOutColumnRange(int idxStart, int idxEnd, int previousNumberOfColumns) {
		int numberTuples = getNumberOfValues(previousNumberOfColumns);
		int tupleLengthAfter = idxEnd - idxStart;
		double[] newDictValues = new double[tupleLengthAfter * numberTuples];
		int orgOffset = idxStart;
		int targetOffset = 0;
		for(int v = 0; v < numberTuples; v++) {
			for(int c = 0; c < tupleLengthAfter; c++, orgOffset++, targetOffset++) {
				newDictValues[targetOffset] = _values[orgOffset];
			}
			orgOffset += previousNumberOfColumns - idxEnd + idxStart;
		}
		return new Dictionary(newDictValues);
	}

	public ADictionary reExpandColumns(int max) {
		double[] newDictValues = new double[_values.length * max];

		for(int i = 0, offset = 0; i < _values.length; i++, offset += max) {
			int val = (int) Math.floor(_values[i]) - 1;
			newDictValues[offset + val] = 1;
		}

		return new Dictionary(newDictValues);
	}

	@Override
	public boolean containsValue(double pattern) {
		boolean NaNpattern = Double.isNaN(pattern);

		if(NaNpattern) {
			for(double v : _values)
				if(Double.isNaN(v))
					return true;
		}
		else {
			for(double v : _values)
				if(v == pattern)
					return true;
		}

		return false;
	}

	@Override
	public long getNumberNonZeros(int[] counts, int nCol) {
		long nnz = 0;
		final int nRow = _values.length / nCol;
		for(int i = 0; i < nRow; i++) {
			long rowCount = 0;
			final int off = i * nCol;
			for(int j = off; j < off + nCol; j++) {
				if(_values[j] != 0)
					rowCount++;
			}
			nnz += rowCount * counts[i];
		}
		return nnz;
	}

	@Override
	public void addToEntry(Dictionary d, int fr, int to, int nCol) {
		final int sf = nCol * fr; // start from
		final int ef = sf + nCol; // end from
		double[] v = d.getValues();
		for(int i = sf, j = nCol * to; i < ef; i++, j++) {
			v[j] += _values[i];
		}
	}

	@Override
	public boolean isLossy() {
		return false;
	}

	@Override
	public double[] getTuple(int index, int nCol) {

		final double[] tuple = new double[nCol];
		boolean allZero = true;
		for(int i = index * nCol, off = 0; i < (index + 1) * nCol && i < _values.length; i++, off++) {
			final double v = _values[i];
			if(v != 0) {
				tuple[off] = v;
				allZero = false;
			}
		}

		return allZero ? null : tuple;
	}

	@Override
	public ADictionary subtractTuple(double[] tuple) {
		double[] newValues = new double[_values.length];
		for(int i = 0; i < _values.length; i++) {
			newValues[i] = _values[i] - tuple[i % tuple.length];
		}
		return new Dictionary(newValues);
	}

	@Override
	public MatrixBlockDictionary getMBDict(int nCol) {
		return new MatrixBlockDictionary(_values, nCol);
	}

	@Override
	public void aggregateCols(double[] c, Builtin fn, int[] colIndexes) {
		int ncol = colIndexes.length;
		int vlen = size() / ncol;
		for(int k = 0; k < vlen; k++)
			for(int j = 0, valOff = k * ncol; j < ncol; j++)
				c[colIndexes[j]] = fn.execute(c[colIndexes[j]], getValue(valOff + j));

	}

	@Override
	public ADictionary scaleTuples(int[] scaling, int nCol) {
		final double[] scaledValues = new double[_values.length];
		int off = 0;
		for(int tuple = 0; tuple < _values.length / nCol; tuple++) {
			final int scale = scaling[tuple];
			for(int v = 0; v < nCol; v++) {
				scaledValues[off] = _values[off] * scale;
				off++;
			}
		}
		return new Dictionary(scaledValues);
	}

	@Override
	public Dictionary preaggValuesFromDense(int numVals, int[] colIndexes, int[] aggregateColumns, double[] b, int cut) {
		double[] ret = new double[numVals * aggregateColumns.length];
		for(int k = 0, off = 0; k < numVals * colIndexes.length; k += colIndexes.length, off += aggregateColumns.length) {
			for(int h = 0; h < colIndexes.length; h++) {
				int idb = colIndexes[h] * cut;
				double v = _values[k + h];
				if(v != 0)
					for(int i = 0; i < aggregateColumns.length; i++)
						ret[off + i] += v * b[idb + aggregateColumns[i]];
			}
		}
		return new Dictionary(ret);
	}

	@Override
	public ADictionary replace(double pattern, double replace, int nCol) {
		double[] retV = new double[_values.length];
		for(int i = 0; i < _values.length; i++) {
			final double v = _values[i];
			if(v == pattern)
				retV[i] = replace;
			else
				retV[i] = v;
		}
		return new Dictionary(retV);
	}

	@Override
	public ADictionary replaceZeroAndExtend(double replace, int nCol) {
		double[] retV = new double[_values.length + nCol];
		for(int i = 0; i < _values.length; i++) {
			final double v = _values[i];
			if(v == 0)
				retV[i] = replace;
			else
				retV[i] = v;
		}
		for(int i = _values.length; i < _values.length + nCol; i++)
			retV[i] = replace;

		return new Dictionary(retV);
	}

	@Override
	public double product(int[] counts, int nCol) {
		double ret = 1;
		final int len = _values.length / nCol;
		for(int i = 0; i < len; i++) {
			for(int j = i * nCol; j < (i + 1) * nCol; j++) {
				double v = _values[j];
				if(v != 0)
					ret *= Math.pow(v, counts[i]);
				else
					ret = 0;
			}
		}
		return ret;
	}

	@Override
	public void colProduct(double[] res, int[] counts, int[] colIndexes) {
		throw new NotImplementedException();
	}
}
