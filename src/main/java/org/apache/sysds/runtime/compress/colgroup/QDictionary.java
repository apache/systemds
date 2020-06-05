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
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.utils.MemoryEstimates;

/**
 * This dictionary class aims to encapsulate the storage and operations over unique floating point values of a column
 * group. The primary reason for its introduction was to provide an entry point for specialization such as shared
 * dictionaries, which require additional information.
 */
public class QDictionary extends IDictionary {

	protected static final Log LOG = LogFactory.getLog(QDictionary.class.getName());
	protected final double _scale;
	protected final byte[] _values;

	public QDictionary(BitmapLossy bm) {
		_values = bm.getValues();
		_scale = bm.getScale();
	}

	public QDictionary(byte[] values, double scale) {
		_values = values;
		_scale = scale;
	}

	public double[] getValues() {
		LOG.warn("Decompressing Quantized Representation");
		double[] res = new double[_values.length];
		for(int i = 0; i < _values.length; i++) {
			res[i] = _values[i] * _scale;
		}
		return res;
	}

	public double getValue(int i) {
		return _values[i] * _scale;
	}

	public byte getValueByte(int i) {
		return _values[i];
	}

	public double getScale() {
		return _scale;
	}

	public long getInMemorySize() {
		// object + values array + double
		return getInMemorySize(_values.length);
	}

	public static long getInMemorySize(int valuesCount) {
		// object + values array + double
		return 16 + MemoryEstimates.byteArrayCost(valuesCount) + 8;
	}

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

	public double aggregate(double init, Builtin fn) {
		// full aggregate can disregard tuple boundaries
		int len = _values.length;
		double ret = init;
		for(int i = 0; i < len; i++)
			ret = fn.execute(ret, getValue(i));
		return ret;
	}

	public QDictionary apply(ScalarOperator op) {

		if(op.fn instanceof Multiply) {
			return new QDictionary(_values, op.executeScalar(_scale));
		}
		double[] temp = new double[_values.length];
		double max = op.executeScalar((double) _values[0] * _scale);
		temp[0] = max;
		for(int i = 1; i < _values.length; i++) {
			temp[i] = op.executeScalar((double) _values[i] * _scale);
			double absTemp = Math.abs(temp[i]);
			if(absTemp > max) {
				max = absTemp;
			}
		}
		byte[] newValues = new byte[_values.length];
		double newScale = max / (double) (Byte.MAX_VALUE);
		for(int i = 0; i < _values.length; i++) {
			newValues[i] = (byte) ((double) temp[i] / newScale);
		}

		return new QDictionary(newValues, newScale);
	}

	@Override
	public int getValuesLength() {
		return _values.length;
	}

	@Override
	public IDictionary clone() {
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
		return 8 + 4 + _values.length + 10000;
	}

	public static QDictionary materializeZeroValueLossy(QDictionary OldDictionary, int numCols) {
		return new QDictionary(Arrays.copyOf(OldDictionary._values, OldDictionary._values.length + numCols),
			OldDictionary._scale);
	}

	public int getNumberOfValues(int nCol) {
		return _values.length / nCol;
	}

	public short[] sumAllRowsToShort(int nCol) {
		short[] res = new short[getNumberOfValues(nCol)];
		for(int i = 0, off = 0; off < _values.length; i++, off += nCol) {
			for(int j = 0; j < nCol; j++) {
				res[i] += _values[off + j];
			}
		}
		return res;
	}

	@Override
	protected double[] sumAllRowsToDouble(KahanFunction kplus, KahanObject kbuff, int nrColumns, boolean allocNew) {
		if(nrColumns == 1 && kplus instanceof KahanPlus)
			return getValues(); // shallow copy of values

		final int numVals = _values.length / nrColumns;
		double[] ret = allocNew ? new double[numVals] : ColGroupValue.allocDVector(numVals, false);
		for(int k = 0; k < numVals; k++) {
			ret[k] = sumRow(k, kplus, kbuff, nrColumns);
		}

		return ret;
	}

	@Override
	protected double sumRow(int k, KahanFunction kplus, KahanObject kbuff, int nrColumns) {
		int valOff = k * nrColumns;
		if(kplus instanceof KahanPlus){
			short res = 0;
			for (int i = 0; i < nrColumns; i++){
				res += _values[valOff + i];
			}
			return res * _scale;
		} else{
			kbuff.set(0, 0);
			for(int i = 0; i < nrColumns; i++)
				kplus.execute2(kbuff, _values[valOff + i] *_scale);
			return kbuff._sum;
		}
	}
}
