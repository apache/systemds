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
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.utils.MemoryEstimates;

/**
 * This dictionary class aims to encapsulate the storage and operations over unique floating point values of a column
 * group. The primary reason for its introduction was to provide an entry point for specialization such as shared
 * dictionaries, which require additional information.
 */
public class QDictionary extends ADictionary {

	private static final long serialVersionUID = 2100501253343438897L;

	protected double _scale;
	protected byte[] _values;

	protected QDictionary(byte[] values, double scale) {
		_values = values;
		_scale = scale;
	}

	@Override
	public double[] getValues() {
		if(_values == null) {
			return new double[0];
		}
		double[] res = new double[_values.length];
		for(int i = 0; i < _values.length; i++) {
			res[i] = getValue(i);
		}
		return res;
	}

	@Override
	public double getValue(int i) {
		return (i >= size()) ? 0.0 : _values[i] * _scale;
	}

	public byte getValueByte(int i) {
		return _values[i];
	}

	public byte[] getValuesByte() {
		return _values;
	}

	public double getScale() {
		return _scale;
	}

	@Override
	public long getInMemorySize() {
		// object + values array + double
		return getInMemorySize(size());
	}

	public static long getInMemorySize(int valuesCount) {
		// object + values array + double
		return 16 + MemoryEstimates.byteArrayCost(valuesCount) + 8;
	}

	@Override
	public double aggregate(double init, Builtin fn) {
		// full aggregate can disregard tuple boundaries
		int len = size();
		double ret = init;
		for(int i = 0; i < len; i++)
			ret = fn.execute(ret, getValue(i));
		return ret;
	}

	@Override
	public double aggregate(double init, Builtin fn, double[] reference) {
		throw new NotImplementedException();
	}

	@Override
	public double[] aggregateRows(Builtin fn, final int nCol) {
		if(nCol == 1)
			return getValues();
		final int nRows = _values.length / nCol;
		double[] res = new double[nRows];
		for(int i = 0; i < nRows; i++) {
			final int off = i * nCol;
			res[i] = _values[off];
			for(int j = off + 1; j < off + nCol; j++)
				res[i] = fn.execute(res[i], _values[j] * _scale);
		}
		return res;
	}

	@Override
	public double[] aggregateRows(Builtin fn, double[] reference) {
		throw new NotImplementedException();
	}

	@Override
	public QDictionary inplaceScalarOp(ScalarOperator op) {
		if(_values == null)
			return this;

		if(op.fn instanceof Multiply || op.fn instanceof Divide) {
			_scale = op.executeScalar(_scale);
			return this;
			// return new QDictionary(_values, op.executeScalar(_scale));
		}
		else if(op.fn instanceof Plus) {
			// TODO: find more operations that have the property of larges and smallest value producing the largest or
			// smallest value from operation
			double max = Math.max(Math.abs(op.executeScalar(-127 * _scale)), Math.abs(op.executeScalar(127 * _scale)));
			double oldScale = _scale;
			_scale = max / 127.0;

			for(int i = 0; i < _values.length; i++) {
				_values[i] = (byte) Math.round(op.executeScalar(_values[i] * oldScale) / _scale);
			}
		}
		else {
			double[] temp = new double[_values.length];
			double max = Math.abs(op.executeScalar(getValue(0)));
			for(int i = 0; i < _values.length; i++) {
				temp[i] = op.executeScalar(getValue(i));
				double absTemp = Math.abs(temp[i]);
				if(absTemp > max) {
					max = absTemp;
				}
			}
			_scale = max / (double) (Byte.MAX_VALUE);
			for(int i = 0; i < _values.length; i++) {
				_values[i] = (byte) Math.round(temp[i] / _scale);
			}
		}
		return this;
	}

	@Override
	public QDictionary applyScalarOp(ScalarOperator op) {
		throw new NotImplementedException();
	}

	@Override
	public QDictionary applyScalarOp(ScalarOperator op, double newVal, int numCols) {
		double[] temp = getValues();
		double max = Math.abs(newVal);
		for(int i = 0; i < size(); i++) {
			temp[i] = op.executeScalar(temp[i]);
			double absTemp = Math.abs(temp[i]);
			if(absTemp > max) {
				max = absTemp;
			}
		}
		double scale = max / (double) (Byte.MAX_VALUE);
		byte[] res = new byte[size() + numCols];
		for(int i = 0; i < size(); i++) {
			res[i] = (byte) Math.round(temp[i] / scale);
		}
		Arrays.fill(res, size(), size() + numCols, (byte) Math.round(newVal / scale));
		return new QDictionary(res, scale);
	}

	private int size() {
		return _values.length;
	}

	@Override
	public QDictionary clone() {
		return new QDictionary(_values.clone(), _scale);
	}

	@Override
	public QDictionary cloneAndExtend(int len) {
		byte[] ret = Arrays.copyOf(_values, _values.length + len);
		return new QDictionary(ret, _scale);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(DictionaryFactory.Type.INT8_DICT.ordinal());
		out.writeDouble(_scale);
		out.writeInt(_values.length);
		for(int i = 0; i < _values.length; i++)
			out.writeByte(_values[i]);
	}

	public static QDictionary read(DataInput in) throws IOException {
		double scale = in.readDouble();
		int numVals = in.readInt();
		byte[] values = new byte[numVals];
		for(int i = 0; i < numVals; i++) {
			values[i] = in.readByte();
		}
		return new QDictionary(values, scale);
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 8 + 4 + size();
	}

	@Override
	public int getNumberOfValues(int nCol) {
		return (_values == null) ? 0 : _values.length / nCol;
	}

	@Override
	public double[] sumAllRowsToDouble(int nrColumns) {
		if(nrColumns == 1)
			return getValues(); // shallow copy of values

		final int numVals = getNumberOfValues(nrColumns);
		double[] ret = new double[numVals];
		for(int k = 0; k < numVals; k++)
			ret[k] = sumRow(k, nrColumns);

		return ret;
	}

	@Override
	public double[] sumAllRowsToDoubleSq(int nrColumns) {
		final int numVals = getNumberOfValues(nrColumns);
		double[] ret = new double[numVals];
		for(int k = 0; k < numVals; k++)
			ret[k] = sumRowSq(k, nrColumns);
		return ret;
	}

	@Override
	public double[] sumAllRowsToDoubleSq(double[] reference) {
		throw new NotImplementedException();
	}

	@Override
	public double sumRow(int k, int nrColumns) {
		if(_values == null)
			return 0;
		int valOff = k * nrColumns;

		int res = 0;
		for(int i = 0; i < nrColumns; i++) {
			res += _values[valOff + i];
		}
		return res * _scale;

	}

	@Override
	public double sumRowSq(int k, int nrColumns) {
		if(_values == null)
			return 0;
		int valOff = k * nrColumns;
		double res = 0.0;
		for(int i = 0; i < nrColumns; i++)
			res += (int) (_values[valOff + i] * _values[valOff + i]) * _scale * _scale;
		return res;
	}

	@Override
	public double sumRowSq(int k, int nrColumns, double[] reference) {
		throw new NotImplementedException();
	}

	@Override
	public double[] colSum(int[] counts, int nCol) {
		throw new NotImplementedException("Not Implemented");
	}

	@Override
	public void colSum(double[] c, int[] counts, int[] colIndexes) {
		throw new NotImplementedException("Not Implemented");
	}

	@Override
	public void colSumSq(double[] c, int[] counts, int[] colIndexes) {
		throw new NotImplementedException("Not Implemented");
	}

	@Override
	public void colSumSq(double[] c, int[] counts, int[] colIndexes, double[] reference) {
		throw new NotImplementedException();
	}

	@Override
	public double sum(int[] counts, int ncol) {
		throw new NotImplementedException("Not Implemented");
	}

	@Override
	public double sumSq(int[] counts, int ncol) {
		throw new NotImplementedException("Not Implemented");
	}

	@Override
	public double sumSq(int[] counts, double[] reference) {
		throw new NotImplementedException("Not Implemented");
	}

	@Override
	public void addMaxAndMin(double[] ret, int[] colIndexes) {
		byte[] mins = new byte[colIndexes.length];
		byte[] maxs = new byte[colIndexes.length];
		for(int i = 0; i < colIndexes.length; i++) {
			mins[i] = _values[i];
			maxs[i] = _values[i];
		}
		for(int i = colIndexes.length; i < _values.length; i++) {
			int idx = i % colIndexes.length;
			mins[idx] = (byte) Math.min(_values[i], mins[idx]);
			maxs[idx] = (byte) Math.max(_values[i], maxs[idx]);
		}
		for(int i = 0; i < colIndexes.length; i++) {
			int idy = colIndexes[i] * 2;
			ret[idy] += mins[i] * _scale;
			ret[idy + 1] += maxs[i] * _scale;
		}
	}

	public String getString(int colIndexes) {
		StringBuilder sb = new StringBuilder();
		for(int i = 0; i < size(); i++) {
			sb.append(_values[i]);
			sb.append((i) % (colIndexes) == colIndexes - 1 ? "\n" : " ");
		}
		return sb.toString();
	}

	public Dictionary makeDoubleDictionary() {
		double[] doubleValues = getValues();
		return new Dictionary(doubleValues);
	}

	public ADictionary sliceOutColumnRange(int idxStart, int idxEnd, int previousNumberOfColumns) {
		int numberTuples = getNumberOfValues(previousNumberOfColumns);
		int tupleLengthAfter = idxEnd - idxStart;
		byte[] newDictValues = new byte[tupleLengthAfter * numberTuples];
		int orgOffset = idxStart;
		int targetOffset = 0;
		for(int v = 0; v < numberTuples; v++) {
			for(int c = 0; c < tupleLengthAfter; c++, orgOffset++, targetOffset++) {
				newDictValues[targetOffset] = _values[orgOffset];
			}
			orgOffset += previousNumberOfColumns - idxEnd + idxStart;
		}
		return new QDictionary(newDictValues, _scale);
	}

	public ADictionary reExpandColumns(int max) {
		byte[] newDictValues = new byte[_values.length * max];

		for(int i = 0, offset = 0; i < _values.length; i++, offset += max) {
			int val = _values[i] - 1;
			newDictValues[offset + val] = 1;
		}

		return new QDictionary(newDictValues, 1.0);
	}

	@Override
	public boolean containsValue(double pattern) {
		if(Double.isNaN(pattern) || Double.isInfinite(pattern))
			return false;
		throw new NotImplementedException("Not contains value on Q Dictionary");
	}

	@Override
	public boolean containsValue(double pattern, double[] reference) {
		throw new NotImplementedException();
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
	public long getNumberNonZeros(int[] counts, double[] reference, int nRows) {
		throw new NotImplementedException("not implemented yet");
	}

	@Override
	public void addToEntry(Dictionary d, int fr, int to, int nCol) {
		throw new NotImplementedException("Not implemented yet");
	}

	@Override
	public boolean isLossy() {
		return false;
	}

	@Override
	public double[] getTuple(int index, int nCol) {
		return null;
	}

	@Override
	public ADictionary subtractTuple(double[] tuple) {
		throw new NotImplementedException();
	}

	@Override
	public MatrixBlockDictionary getMBDict(int nCol) {
		throw new NotImplementedException();
	}

	@Override
	public void aggregateCols(double[] c, Builtin fn, int[] colIndexes) {
		throw new NotImplementedException();
	}

	@Override
	public void aggregateCols(double[] c, Builtin fn, int[] colIndexes, double[] reference) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary scaleTuples(int[] scaling, int nCol) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary preaggValuesFromDense(int numVals, int[] colIndexes, int[] aggregateColumns, double[] b,
		int cut) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary replace(double pattern, double replace, int nCol) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary replace(double pattern, double replace, double[] reference) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary replaceZeroAndExtend(double replace, int nCol) {
		throw new NotImplementedException();
	}

	@Override
	public double product(int[] counts, int nCol) {
		throw new NotImplementedException();
	}

	@Override
	public void colProduct(double[] res, int[] counts, int[] colIndexes) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary applyBinaryRowOpLeftAppendNewEntry(BinaryOperator op, double[] v, int[] colIndexes) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary binOpLeft(BinaryOperator op, double[] v, int[] colIndexes) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary binOpRight(BinaryOperator op, double[] v, int[] colIndexes) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary applyBinaryRowOpRightAppendNewEntry(BinaryOperator op, double[] v, int[] colIndexes) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary applyScalarOp(ScalarOperator op, double[] reference, double[] newReference) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary binOpLeft(BinaryOperator op, double[] v, int[] colIndexes, double[] reference,
		double[] newReference) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary binOpRight(BinaryOperator op, double[] v, int[] colIndexes, double[] reference,
		double[] newReference) {
		throw new NotImplementedException();
	}

	@Override
	public double[] sumAllRowsToDouble(double[] reference) {
		throw new NotImplementedException();
	}
}
