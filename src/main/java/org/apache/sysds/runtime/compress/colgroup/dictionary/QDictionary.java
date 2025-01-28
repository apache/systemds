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

import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.utils.MemoryEstimates;

/**
 * This dictionary class aims to encapsulate the storage and operations over unique floating point values of a column
 * group. The primary reason for its introduction was to provide an entry point for specialization such as shared
 * dictionaries, which require additional information.
 */
public class QDictionary extends ACachingMBDictionary {

	private static final long serialVersionUID = 2100501253343438897L;

	protected double _scale;
	protected byte[] _values;
	protected int _nCol;

	protected QDictionary(byte[] values, double scale, int nCol) {
		_values = values;
		_scale = scale;
		_nCol = nCol;
	}

	public static QDictionary create(byte[] values, double scale, int nCol, boolean check) {
		if(scale == 0)
			return null;
		if(check) {
			boolean containsOnlyZero = true;
			for(int i = 0; i < values.length && containsOnlyZero; i++) {
				if(values[i] != 0)
					containsOnlyZero = false;
			}
			if(containsOnlyZero)
				return null;
		}
		return new QDictionary(values, scale, nCol);
	}

	@Override
	public double[] getValues() {

		double[] res = new double[_values.length];
		for(int i = 0; i < _values.length; i++) {
			res[i] = getValue(i);
		}
		return res;
	}

	@Override
	public double getValue(int i) {
		return _values[i] * _scale;
	}

	@Override
	public final double getValue(int r, int c, int nCol) {
		return _values[r * nCol + c] * _scale;
	}

	@Override
	public long getInMemorySize() {
		// object + values array + double
		return getInMemorySize(size());
	}

	public static long getInMemorySize(int valuesCount) {
		// object + values array + double
		return 16 + (long) MemoryEstimates.byteArrayCost(valuesCount) + 8;
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

	private int size() {
		return _values.length;
	}

	@Override
	public QDictionary clone() {
		return new QDictionary(_values.clone(), _scale, _nCol);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(DictionaryFactory.Type.INT8_DICT.ordinal());
		out.writeDouble(_scale);
		out.writeInt(_values.length);
		for(int i = 0; i < _values.length; i++)
			out.writeByte(_values[i]);
		out.writeInt(_nCol);
	}

	public static QDictionary read(DataInput in) throws IOException {
		double scale = in.readDouble();
		int numVals = in.readInt();
		byte[] values = new byte[numVals];
		for(int i = 0; i < numVals; i++) {
			values[i] = in.readByte();
		}
		int nCol = in.readInt();
		return new QDictionary(values, scale, nCol);
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 8 + 4 + size() + 4;
	}

	@Override
	public int getNumberOfValues(int nCol) {
		return _values.length / nCol;
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

	private double sumRow(int k, int nrColumns) {
		int valOff = k * nrColumns;
		int res = 0;
		for(int i = 0; i < nrColumns; i++) {
			res += _values[valOff + i];
		}
		return res * _scale;
	}

	private double sumRowSq(int k, int nrColumns) {
		int valOff = k * nrColumns;
		double res = 0.0;
		for(int i = 0; i < nrColumns; i++)
			res += (_values[valOff + i] * _values[valOff + i]) * _scale * _scale;
		return res;
	}

	public String getString(int colIndexes) {
		StringBuilder sb = new StringBuilder();
		for(int i = 0; i < size(); i++) {
			sb.append(_values[i]);
			sb.append((i) % (colIndexes) == colIndexes - 1 ? "\n" : " ");
		}
		return sb.toString();
	}

	public IDictionary sliceOutColumnRange(int idxStart, int idxEnd, int previousNumberOfColumns) {
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
		return new QDictionary(newDictValues, _scale, _nCol);
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
	public int[] countNNZZeroColumns(int[] counts) {
		final int nRow = counts.length;
		final int nCol = _values.length / nRow;

		final int[] ret = new int[nCol];
		for(int i = 0; i < nRow; i++) {
			for(int j = 0; j < nCol; j++) {
				final int off = i * nCol + j;
				if(_values[off] != 0)
					ret[j] += counts[i];
			}
		}
		return ret;
	}

	@Override
	public DictType getDictType() {
		return DictType.UInt8;
	}

	@Override
	public double getSparsity() {
		int nnz = 0;
		for(int i = 0; i < _values.length; i++) {
			nnz += _values[i] == 0 ? 0 : 1;
		}
		return (double) nnz / _values.length;
	}

	@Override
	public boolean equals(IDictionary o) {
		return getMBDict().equals(o);
	}

	@Override
	public MatrixBlockDictionary getMBDict() {
		return getMBDict(_nCol);
	}

	@Override
	public MatrixBlockDictionary createMBDict(int nCol) {
		MatrixBlock mb = new MatrixBlock(_values.length / nCol, nCol, false);
		mb.allocateDenseBlock();
		double[] dbv = mb.getDenseBlockValues();
		for(int i = 0; i < _values.length; i++)
			dbv[i] = _values[i] * _scale;
		mb.recomputeNonZeros();
		return new MatrixBlockDictionary(mb);
	}

}
