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

package org.apache.sysds.runtime.data;

import java.util.Arrays;

import org.apache.sysds.common.Warnings;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.UtilFunctions;

public class DenseBlockInt64 extends DenseBlockDRB
{
	private static final long serialVersionUID = 7166657802668966730L;

	private long[] _data;

	public DenseBlockInt64(int[] dims) {
		super(dims);
		reset(_rlen, _odims, 0);
	}

	@Override
	protected void allocateBlock(int bix, int length) {
		_data = new long[length];
	}

	public DenseBlockInt64(int[] dims, long[] data) {
		super(dims);
		_data = data;
	}

	public long[] getData() {
		return _data;
	}

	@Override
	public boolean isNumeric() {
		return true;
	}
	
	@Override
	public long capacity() {
		return (_data!=null) ? _data.length : -1;
	}

	@Override
	protected long computeNnz(int bix, int start, int length) {
		return UtilFunctions.computeNnz(_data, start, length);
	}

	@Override
	public double[] values(int r) {
		double[] ret = getReuseRow(false);
		int ix = pos(r);
		int ncol = _odims[0];
		for(int j=0; j<ncol; j++)
			ret[j] = _data[ix+j];
		return ret;
	}
	
	@Override
	public double[] valuesAt(int bix) {
		Warnings.warnFullFP64Conversion(_data.length);
		return DataConverter.toDouble(_data);
	}

	@Override
	public int index(int r) {
		return 0;
	}

	@Override
	public void incr(int r, int c) {
		_data[pos(r, c)] ++;
	}
	
	@Override
	public void incr(int r, int c, double delta) {
		_data[pos(r, c)] += delta;
	}

	@Override
	protected void fillBlock(int bix, int fromIndex, int toIndex, double v) {
		Arrays.fill(_data, fromIndex, toIndex, UtilFunctions.toLong(v));
	}

	@Override
	protected void setInternal(int bix, int ix, double v) {
		_data[ix] = UtilFunctions.toLong(v);
	}

	@Override
	public DenseBlock set(int r, int c, double v) {
		_data[pos(r, c)] = UtilFunctions.toLong(v);
		return this;
	}
	
	@Override
	public DenseBlock set(DenseBlock db) {
		double[] data = db.valuesAt(0);
		Arrays.parallelSetAll(_data, (i) -> UtilFunctions.toLong(data[i]));
		return this;
	}
	
	@Override
	public DenseBlock set(int r, double[] v) {
		System.arraycopy(DataConverter.toLong(v), 0, _data, pos(r), _odims[0]);
		return this;
	}

	@Override
	public DenseBlock set(int[] ix, double v) {
		_data[pos(ix)] = UtilFunctions.toLong(v);
		return this;
	}

	@Override
	public DenseBlock set(int[] ix, long v) {
		_data[pos(ix)] = v;
		return this;
	}

	@Override
	public DenseBlock set(int[] ix, String v) {
		_data[pos(ix)] = Long.parseLong(v);
		return this;
	}

	@Override
	public double get(int r, int c) {
		return _data[pos(r, c)];
	}

	@Override
	public double get(int[] ix) {
		return _data[pos(ix)];
	}

	@Override
	public String getString(int[] ix) {
		return String.valueOf(_data[pos(ix)]);
	}

	@Override
	public long getLong(int[] ix) {
		return _data[pos(ix)];
	}
}
