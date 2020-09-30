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

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.utils.ABitmap;
import org.apache.sysds.runtime.compress.utils.LinearAlgebraUtils;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

/**
 * Class to encapsulate information about a column group that is encoded with dense dictionary encoding (DDC) using 2
 * byte codes.
 */
public class ColGroupDDC2 extends ColGroupDDC {
	private static final long serialVersionUID = -3995768285207071013L;

	private char[] _data;

	protected ColGroupDDC2() {
		super();
	}

	protected ColGroupDDC2(int[] colIndices, int numRows, ABitmap ubm, CompressionSettings cs) {
		super(colIndices, numRows, ubm, cs);
		_data = new char[numRows];

		int numVals = ubm.getNumValues();
		int numCols = ubm.getNumColumns();

		// materialize zero values, if necessary
		if(ubm.getNumOffsets() < (long) numRows * numCols) {
			int zeroIx = containsAllZeroValue();
			if(zeroIx < 0) {
				zeroIx = numVals;
			}
			Arrays.fill(_data, (char) zeroIx);
			_zeros = true;
		}

		// iterate over values and write dictionary codes
		for(int i = 0; i < numVals; i++) {
			int[] tmpList = ubm.getOffsetsList(i).extractValues();
			int tmpListSize = ubm.getNumOffsets(i);
			for(int k = 0; k < tmpListSize; k++)
				_data[tmpList[k]] = (char) i;
		}
	}

	protected ColGroupDDC2(int[] colIndices, int numRows, ADictionary dict, char[] data, boolean zeros) {
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
	public char[] getData() {
		return _data;
	}

	@Override
	protected int getIndex(int r) {
		return _data[r];
	}

	@Override
	protected int getIndex(int r, int colIx) {
		return _data[r] * getNumCols() + colIx;
	}

	@Override
	protected double getData(int r, double[] dictionary) {
		return _dict.getValue(_data[r]);
	}

	@Override
	protected double getData(int r, int colIx, double[] dictionary) {
		return _dict.getValue(_data[r] * getNumCols() + colIx);
	}

	@Override
	protected void setData(int r, int code) {
		_data[r] = (char) code;
	}

	@Override
	public void rightMultByVector(double[] b, double[] c, int rl, int ru, double[] dictVals) {
		final int numVals = getNumValues();
		double[] vals = preaggValues(numVals, b, dictVals);
		LinearAlgebraUtils.vectListAdd(vals, c, _data, rl, ru);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		// write data
		// out.writeInt(_data.length);
		for(int i = 0; i < _numRows; i++)
			out.writeChar(_data[i]);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		super.readFields(in);
		// read data
		_data = new char[_numRows];
		for(int i = 0; i < _numRows; i++)
			_data[i] = in.readChar();
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		// data
		ret += 2 * _data.length;

		return ret;
	}

	@Override
	public long estimateInMemorySize() {
		// LOG.debug(this.toString());
		return ColGroupSizes.estimateInMemorySizeDDC2(getNumCols(), getNumValues(), _data.length, isLossy());
	}

	@Override
	public ColGroup scalarOperation(ScalarOperator op) {
		double val0 = op.executeScalar(0);
		if(op.sparseSafe || val0 == 0 || !_zeros) {
			return new ColGroupDDC2(_colIndexes, _numRows, applyScalarOp(op), _data, _zeros);
		}
		else {
			return new ColGroupDDC2(_colIndexes, _numRows, applyScalarOp(op, val0, _colIndexes.length), _data, false);
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
