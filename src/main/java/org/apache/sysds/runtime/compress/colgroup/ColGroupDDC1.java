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
import java.util.HashMap;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.utils.ABitmap;
import org.apache.sysds.runtime.compress.utils.LinearAlgebraUtils;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

/**
 * Class to encapsulate information about a column group that is encoded with dense dictionary encoding (DDC) using 1
 * byte codes.
 */
public class ColGroupDDC1 extends ColGroupDDC {
	private static final long serialVersionUID = 5204955589230760157L;

	private byte[] _data;

	protected ColGroupDDC1() {
		super();
	}

	protected ColGroupDDC1(int[] colIndices, int numRows, ABitmap ubm, CompressionSettings cs) {
		super(colIndices, numRows, ubm, cs);

		int numVals = ubm.getNumValues();
		int numCols = ubm.getNumColumns();

		_data = new byte[numRows];
		// materialize zero values, if necessary
		if(ubm.getNumOffsets() < (long) numRows * numCols) {
			int zeroIx = containsAllZeroValue();
			if(zeroIx < 0) {
				// Utilize the index of the length as a zero index, Makes lookups slower, but removes
				// the need to allocate a new Dictionary
				zeroIx = numVals;
			}
			Arrays.fill(_data, (byte) zeroIx);
			_zeros = true;
		}

		// iterate over values and write dictionary codes
		for(int i = 0; i < numVals; i++) {
			int[] tmpList = ubm.getOffsetsList(i).extractValues();
			int tmpListSize = ubm.getNumOffsets(i);
			for(int k = 0; k < tmpListSize; k++)
				_data[tmpList[k]] = (byte) i;
		}
	}

	protected ColGroupDDC1(int[] colIndices, int numRows, ADictionary dict, byte[] data, boolean zeros) {
		super(colIndices, numRows, dict);
		_data = data;
		_zeros = zeros;
	}

	@Override
	protected ColGroupType getColGroupType() {
		return ColGroupType.DDC1;
	}

	/**
	 * Getter method to get the data, contained in The DDC ColGroup.
	 * 
	 * @return The contained data
	 */
	public byte[] getData() {
		return _data;
	}

	@Override
	public void rightMultByVector(double[] b, double[] c, int rl, int ru, double[] dictVals) {
		final int numVals = getNumValues();
		double[] vals = preaggValues(numVals, b, dictVals);
		LinearAlgebraUtils.vectListAdd(vals, c, _data, rl, ru);
	}

	@Override
	protected int getIndex(int r) {
		return _data[r] & 0xFF;
	}

	@Override
	protected int getIndex(int r, int colIx) {
		return _data[r] & 0xFF * getNumCols() + colIx;
	}

	@Override
	protected double getData(int r, double[] values) {
		int index = (_data[r] & 0xFF);
		return (index == values.length) ? 0.0 : values[index];
	}

	@Override
	protected double getData(int r, int colIx, double[] values) {
		int index = (_data[r] & 0xFF) * getNumCols() + colIx;
		return (index == values.length) ? 0.0 : values[index];
	}

	@Override
	protected void setData(int r, int code) {
		_data[r] = (byte) code;
	}

	public void recodeData(HashMap<Double, Integer> map) {
		// prepare translation table
		final int numVals = getNumValues();
		final double[] values = getValues();
		byte[] lookup = new byte[numVals];
		for(int k = 0; k < numVals; k++)
			lookup[k] = map.get(values[k]).byteValue();

		// recode the data
		for(int i = 0; i < _numRows; i++)
			_data[i] = lookup[_data[i] & 0xFF];
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		// write data
		for(int i = 0; i < _numRows; i++)
			out.writeByte(_data[i]);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		super.readFields(in);
		// read data
		_data = new byte[_numRows];
		for(int i = 0; i < _numRows; i++)
			_data[i] = in.readByte();
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += _data.length;
		return ret;
	}

	@Override
	public long estimateInMemorySize() {
		return ColGroupSizes.estimateInMemorySizeDDC1(getNumCols(), getNumValues(), _data.length, isLossy());
	}

	@Override
	public ColGroup scalarOperation(ScalarOperator op) {
		double val0 = op.executeScalar(0);
		if(op.sparseSafe || val0 == 0 || !_zeros) {
			return new ColGroupDDC1(_colIndexes, _numRows, applyScalarOp(op), _data, _zeros);
		}
		else {
			return new ColGroupDDC1(_colIndexes, _numRows, applyScalarOp(op, val0, _colIndexes.length), _data, false);
		}
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(" DataLength: " + this._data.length);
		return sb.toString();
	}
}
