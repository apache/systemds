/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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


package org.tugraz.sysds.runtime.data;

import org.tugraz.sysds.common.Warnings;
import org.tugraz.sysds.runtime.util.DataConverter;
import org.tugraz.sysds.runtime.util.UtilFunctions;

import java.util.Arrays;

public class DenseBlockString extends DenseBlockDRB {
	private static final long serialVersionUID = 7071870563356352352L;

	protected String[] _data;

	public DenseBlockString(int[] dims) {
		super(dims);
		reset(_rlen, _odims, 0);
	}

	@Override
	protected void allocateBlock(int bix, int length) {
		_data = new String[length];
	}

	public DenseBlockString(int[] dims, String[] data) {
		super(dims);
		_data = data;
	}

	public String[] getData() {
		return _data;
	}

	@Override
	public boolean isNumeric() {
		return false;
	}

	@Override
	public long capacity() {
		return (_data != null) ? _data.length : -1;
	}

	@Override
	protected long computeNnz(int bix, int start, int length) {
		return UtilFunctions.computeNnz(_data, start, length);
	}

	@Override
	public double[] values(int r) {
		return DataConverter.toDouble(_data);
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
	public int pos(int r) {
		return r * _odims[0];
	}

	@Override
	public int pos(int r, int c) {
		return r * _odims[0] + c;
	}

	@Override
	public void incr(int r, int c) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void incr(int r, int c, double delta) {
		throw new UnsupportedOperationException();
	}

	@Override
	protected void fillBlock(int bix, int fromIndex, int toIndex, double v) {
		Arrays.fill(_data, fromIndex, toIndex, String.valueOf(v));
	}

	@Override
	protected void setInternal(int bix, int ix, double v) {
		_data[ix] = String.valueOf(v);
	}

	@Override
	public DenseBlock set(String s) {
		Arrays.fill(_data, 0, _data.length, s);
		return this;
	}

	@Override
	public DenseBlock set(int r, int c, double v) {
		_data[pos(r, c)] = String.valueOf(v);
		return this;
	}

	@Override
	public DenseBlock set(DenseBlock db) {
		int[] ix = new int[numDims()];
		for (int r = 0; r < _rlen; r++) {
			ix[0] = r;
			for (int c = 0; c < _odims[0]; c++) {
				ix[ix.length - 1] = c; // for linear scan
				_data[pos(r, c)] = db.getString(ix);
			}
		}
		return this;
	}

	@Override
	public DenseBlock set(int r, double[] v) {
		System.arraycopy(DataConverter.toString(v), 0, _data, pos(r), _odims[0]);
		return this;
	}

	@Override
	public DenseBlock set(int[] ix, double v) {
		_data[pos(ix)] = String.valueOf(v);
		return this;
	}

	@Override
	public DenseBlock set(int[] ix, long v) {
		_data[pos(ix)] = String.valueOf(v);
		return this;
	}

	@Override
	public double get(int r, int c) {
		String s = _data[pos(r, c)];
		return s == null || s.isEmpty() ? 0 : Double.parseDouble(s);
	}

	@Override
	public double get(int[] ix) {
		String s = _data[pos(ix)];
		return s == null || s.isEmpty() ? 0 : Double.parseDouble(s);
	}

	@Override
	public DenseBlock set(int[] ix, String v) {
		_data[pos(ix)] = v;
		return this;
	}

	@Override
	public String getString(int[] ix) {
		return _data[pos(ix)];
	}

	@Override
	public long getLong(int[] ix) {
		String s = _data[pos(ix)];
		return s == null || s.isEmpty() ? 0 : Long.parseLong(s);
	}
}
