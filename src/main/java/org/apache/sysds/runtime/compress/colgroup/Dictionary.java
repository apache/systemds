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

import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.KahanFunction;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.utils.MemoryEstimates;

/**
 * This dictionary class aims to encapsulate the storage and operations over unique floating point values of a column
 * group. The primary reason for its introduction was to provide an entry point for specialization such as shared
 * dictionaries, which require additional information.
 */
public class Dictionary extends ADictionary {

	private final double[] _values;

	public Dictionary(double[] values) {
		_values = values;
	}

	@Override
	public double[] getValues() {
		return _values;
	}

	@Override
	public double getValue(int i) {
		return _values[i];
	}

	@Override
	public long getInMemorySize() {
		// object + values array + double
		return getInMemorySize(_values.length);
	}

	protected static long getInMemorySize(int valuesCount) {
		// object + values array
		return 16 + MemoryEstimates.doubleArrayCost(valuesCount);
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
			ret = fn.execute(ret, _values[i]);
		return ret;
	}

	@Override
	public Dictionary apply(ScalarOperator op) {
		// in-place modification of the dictionary
		int len = _values.length;
		for(int i = 0; i < len; i++)
			_values[i] = op.executeScalar(_values[i]);
		return this;
	}

	@Override
	public Dictionary applyScalarOp(ScalarOperator op, double newVal, int numCols) {
		// allocate new array just once because we need to add the newVal.
		double[] values = Arrays.copyOf(_values, _values.length + numCols);
		for(int i = 0; i < _values.length; i++) {
			values[i] = op.executeScalar(values[i]);
		}
		Arrays.fill(values, _values.length, _values.length + numCols, newVal);
		return new Dictionary(values);
	}

	@Override
	public Dictionary clone() {
		return new Dictionary(_values.clone());
	}

	@Override
	public int getValuesLength() {
		return _values.length;
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
		out.writeInt(_values.length);
		for(int i = 0; i < _values.length; i++)
			out.writeDouble(_values[i]);
	}

	@Override
	public long getExactSizeOnDisk() {
		return 4 + 8 * _values.length;
	}

	@Override
	public int getNumberOfValues(int ncol) {
		return _values.length / ncol;
	}

	@Override
	protected double[] sumAllRowsToDouble(KahanFunction kplus, KahanObject kbuff, int nrColumns) {
		if(nrColumns == 1 && kplus instanceof KahanPlus)
			return getValues(); // shallow copy of values

		// pre-aggregate value tuple
		final int numVals = _values.length / nrColumns;
		double[] ret = ColGroupValue.allocDVector(numVals, false);
		for(int k = 0; k < numVals; k++) {
			ret[k] = sumRow(k, kplus, kbuff, nrColumns);
		}

		return ret;
	}

	@Override
	protected double sumRow(int k, KahanFunction kplus, KahanObject kbuff, int nrColumns) {
		kbuff.set(0, 0);
		int valOff = k * nrColumns;
		for(int i = 0; i < nrColumns; i++)
			kplus.execute2(kbuff, _values[valOff + i]);
		return kbuff._sum;
	}

	@Override
	protected void colSum(double[] c, int[] counts, int[] colIndexes, KahanFunction kplus) {
		KahanObject kbuff = new KahanObject(0, 0);
		for(int k = 0, valOff = 0; k < _values.length; k++, valOff += colIndexes.length) {
			int cntk = counts[k];
			for(int j = 0; j < colIndexes.length; j++) {
				kbuff.set(c[colIndexes[j]], c[colIndexes[j] + colIndexes.length]);
				// int index = getIndex();
				kplus.execute3(kbuff, getValue(valOff + j), cntk);
				c[colIndexes[j]] = kbuff._sum;
				c[colIndexes[j] + colIndexes.length] = kbuff._correction;
			}
		}

	}

	@Override
	protected double sum(int[] counts, int ncol, KahanFunction kplus) {
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
