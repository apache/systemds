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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.utils.BitmapLossy;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.KahanFunction;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.KahanPlusSq;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.utils.MemoryEstimates;

/**
 * This dictionary class aims to encapsulate the storage and operations over unique floating point values of a column
 * group. The primary reason for its introduction was to provide an entry point for specialization such as shared
 * dictionaries, which require additional information.
 */
public class QDictionary extends ADictionary {

	protected static final Log LOG = LogFactory.getLog(QDictionary.class.getName());
	protected double _scale;
	protected byte[] _values;

	public QDictionary(BitmapLossy bm) {
		_values = bm.getValues();
		_scale = bm.getScale();
	}

	private QDictionary(byte[] values, double scale) {
		_values = values;
		_scale = scale;
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
		return (i == _values.length) ? 0.0 : _values[i] * _scale;
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
		return getInMemorySize(_values.length);
	}

	public static long getInMemorySize(int valuesCount) {
		// object + values array + double
		return 16 + MemoryEstimates.byteArrayCost(valuesCount) + 8;
	}

	@Override
	public int hasZeroTuple(int ncol) {
		int len = _values.length / ncol;
		for(int i = 0, off = 0; i < len; i++, off += ncol) {
			boolean allZeros = true;
			for(int j = 0; j < ncol; j++)
				allZeros &= (_values[off + j] == 0);
			if(allZeros)
				return i;
		}
		return -1;
	}

	@Override
	public double aggregate(double init, Builtin fn) {
		// full aggregate can disregard tuple boundaries
		int len = _values.length;
		double ret = init;
		for(int i = 0; i < len; i++)
			ret = fn.execute(ret, getValue(i));
		return ret;
	}

	@Override
	public QDictionary apply(ScalarOperator op) {

		if(op.fn instanceof Multiply) {
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
			double max = op.executeScalar(getValue(0));
			temp[0] = max;
			for(int i = 1; i < _values.length; i++) {
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
	public QDictionary applyScalarOp(ScalarOperator op, double newVal, int numCols) {
		double[] temp = getValues();
		double max = newVal;
		for(int i = 0; i < _values.length; i++) {
			temp[i] = op.executeScalar(temp[i]);
			double absTemp = Math.abs(temp[i]);
			if(absTemp > max) {
				max = absTemp;
			}
		}
		double scale = max / (double) (Byte.MAX_VALUE);
		byte[] res = new byte[_values.length + numCols];
		for(int i = 0; i < _values.length; i++) {
			res[i] = (byte) Math.round(temp[i] / scale);
		}
		Arrays.fill(res, _values.length, _values.length + numCols, (byte) Math.round(newVal / scale));
		return new QDictionary(res, scale);
	}

	@Override
	public int getValuesLength() {
		return _values.length;
	}

	@Override
	public QDictionary clone() {
		return new QDictionary(_values.clone(), _scale);
	}

	public static QDictionary read(DataInput in) throws IOException {
		double scale = in.readDouble();
		int numVals = in.readInt();
		// read distinct values
		byte[] values = new byte[numVals];
		for(int i = 0; i < numVals; i++)
			values[i] = in.readByte();
		return new QDictionary(values, scale);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeDouble(_scale);
		out.writeInt(_values.length);
		for(int i = 0; i < _values.length; i++)
			out.writeByte(_values[i]);
	}

	@Override
	public long getExactSizeOnDisk() {
		return 8 + 4 + _values.length;
	}

	@Override
	public int getNumberOfValues(int nCol) {
		return _values.length / nCol;
	}

	@Override
	protected double[] sumAllRowsToDouble(KahanFunction kplus, KahanObject kbuff, int nrColumns) {
		if(nrColumns == 1 && kplus instanceof KahanPlus)
			return getValues(); // shallow copy of values

		final int numVals = _values.length / nrColumns;
		double[] ret = ColGroupValue.allocDVector(numVals, false);
		for(int k = 0; k < numVals; k++) {
			ret[k] = sumRow(k, kplus, kbuff, nrColumns);
		}

		return ret;
	}

	@Override
	protected double sumRow(int k, KahanFunction kplus, KahanObject kbuff, int nrColumns) {
		int valOff = k * nrColumns;
		if(kplus instanceof KahanPlus) {
			short res = 0;
			for(int i = 0; i < nrColumns; i++) {
				res += _values[valOff + i];
			}
			return res * _scale;
		}
		else {
			kbuff.set(0, 0);
			for(int i = 0; i < nrColumns; i++)
				kplus.execute2(kbuff, _values[valOff + i] * _scale);
			return kbuff._sum;
		}
	}

	@Override
	protected void colSum(double[] c, int[] counts, int[] colIndexes, KahanFunction kplus) {

		if(!(kplus instanceof KahanPlusSq)) {
			int[] sum = new int[colIndexes.length];
			for(int k = 0, valOff = 0; k < _values.length; k++, valOff += colIndexes.length) {
				int cntk = counts[k];
				for(int j = 0; j < colIndexes.length; j++) {
					sum[j] += cntk * getValueByte(valOff + j);
				}
			}
			for(int j = 0; j < colIndexes.length; j++) {
				c[colIndexes[j]] = c[colIndexes[j]] + sum[j] * _scale;
			}
		}
		else {
			KahanObject kbuff = new KahanObject(0, 0);
			for(int k = 0, valOff = 0; k < _values.length; k++, valOff += colIndexes.length) {
				int cntk = counts[k];
				for(int j = 0; j < colIndexes.length; j++) {
					kbuff.set(c[colIndexes[j]], c[colIndexes[j] + colIndexes.length]);
					kplus.execute3(kbuff, getValue(valOff + j), cntk);
					c[colIndexes[j]] = kbuff._sum;
					c[colIndexes[j] + colIndexes.length] = kbuff._correction;
				}
			}
		}
	}

	@Override
	protected double sum(int[] counts, int ncol, KahanFunction kplus) {
		if(!(kplus instanceof KahanPlusSq)) {
			int sum = 0;
			for(int k = 0, valOff = 0; k < _values.length; k++, valOff += ncol) {
				int cntk = counts[k];
				for(int j = 0; j < ncol; j++) {
					sum += cntk * getValueByte(valOff + j);
				}
			}
			return sum * _scale;
		}
		else {
			KahanObject kbuff = new KahanObject(0, 0);
			for(int k = 0, valOff = 0; k < _values.length; k++, valOff += ncol) {
				int cntk = counts[k];
				for(int j = 0; j < ncol; j++) {
					kplus.execute3(kbuff, getValue(valOff + j), cntk);
				}
			}
			return kbuff._sum;
		}
	}
}
