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

import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.MemoryEstimates;

import java.util.Arrays;

public class DenseBlockFP64 extends DenseBlockDRB
{
	private static final long serialVersionUID = 8546723684649816489L;

	private double[] _data;

	public DenseBlockFP64(int[] dims) {
		super(dims);
		reset(_rlen, _odims, 0);
	}

	@Override
	protected void allocateBlock(int bix, int length) {
		_data = new double[length];
	}

	public DenseBlockFP64(int[] dims, double[] data) {
		super(dims);
		_data = data;
	}
	
	@Override
	public boolean isNumeric() {
		return true;
	}
	
	@Override
	public void reset(int rlen, int[] odims, double v) {
		int len = rlen * odims[0];
		if( len > capacity() ) {
			_data = new double[len];
			if( v != 0 )
				Arrays.fill(_data, v);
		}
		else {
			Arrays.fill(_data, 0, len, v);
		}
		_rlen = rlen;
		_odims = odims;
	}
	
	public static double estimateMemory(long nrows, long ncols) {
		if( (double)nrows + ncols > Long.MAX_VALUE )
			return Long.MAX_VALUE;
		return DenseBlock.estimateMemory(nrows, ncols)
			+ MemoryEstimates.doubleArrayCost(nrows * ncols);
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
		return _data;
	}
	
	@Override
	public double[] valuesAt(int bix) {
		return _data;
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
		_data[pos(r, c)] ++;
	}
	
	@Override
	public void incr(int r, int c, double delta) {
		_data[pos(r, c)] += delta;
	}

	@Override
	protected void fillBlock(int bix, int fromIndex, int toIndex, double v) {
		Arrays.fill(_data, fromIndex, toIndex, v);
	}

	@Override
	protected void setInternal(int bix, int ix, double v) {
		_data[ix] = v;
	}

	@Override
	public DenseBlock set(int r, int c, double v) {
		_data[pos(r, c)] = v;
		return this;
	}
	
	@Override
	public DenseBlock set(DenseBlock db) {
		System.arraycopy(db.valuesAt(0), 0, _data, 0, _rlen*_odims[0]);
		return this;
	}
	
	@Override
	public DenseBlock set(int rl, int ru, int ol, int ou, DenseBlock db) {
		//TODO generalize to non-double dense blocks
		//TODO should we specify that the length HAS TO BE (ru - rl) * (ou - ol) == db.size()? or allow partial sets
		double[] a = db.valuesAt(0);
		// if we can make use of the linearized array, just do an arraycopy.
		// all columns for our DenseBlock mean we can just have to limit the range by rows
		if( ol == 0 && ou == _odims[0])
			System.arraycopy(a, 0, _data, rl*_odims[0], (ru - rl) * _odims[0]);
		else {
			int len = ou - ol;
			for(int i=rl, ix1=0, ix2=rl*_odims[0]+ol; i<ru; i++, ix1+=len, ix2+=_odims[0])
				System.arraycopy(a, ix1, _data, ix2, len);
		}
		return this;
	}

	@Override
	public DenseBlock set(int r, double[] v) {
		System.arraycopy(v, 0, _data, pos(r), _odims[0]);
		return this;
	}
	
	@Override
	public DenseBlock set(int[] ix, double v) {
		_data[pos(ix)] = v;
		return this;
	}

	@Override
	public DenseBlock set(int[] ix, long v) {
		_data[pos(ix)] = v;
		return this;
	}

	@Override
	public DenseBlock set(int[] ix, String v) {
		_data[pos(ix)] = Double.parseDouble(v);
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
		return UtilFunctions.toLong(_data[pos(ix)]);
	}
}
