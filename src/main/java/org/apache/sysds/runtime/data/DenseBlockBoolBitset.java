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

import java.util.BitSet;

import org.apache.sysds.common.Warnings;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.UtilFunctions;

public class DenseBlockBoolBitset extends DenseBlockDRB
{
	private static final long serialVersionUID = -2228057308997136969L;
	
	private BitSet _data;

	public DenseBlockBoolBitset(int[] dims) {
		super(dims);
		reset(_rlen, _odims, 0);
	}

	@Override
	protected void allocateBlock(int bix, int length) {
		_data = new BitSet(length);
	}

	public DenseBlockBoolBitset(int[] dims, BitSet data) {
		super(dims);
		_data = data;
	}

	public DenseBlockBoolBitset(int[] dims, boolean[] data) {
		super(dims);
		_data = new BitSet(data.length);
		for(int i=0; i<data.length; i++)
			if( data[i] )
				_data.set(i);
	}

	public BitSet getData() {
		return _data;
	}
	
	@Override
	public boolean isNumeric() {
		return true;
	}
	
	@Override
	public boolean isNumeric(ValueType vt) {
		return ValueType.BITSET == vt;
	}
	
	@Override
	public void reset(int rlen, int[] odims, double v) {
		boolean bv = v != 0;
		int len = rlen * odims[0];
		if( len > capacity() ) {
			_data = new BitSet(len);
			if( bv )
				_data.set(0, len);
		}
		else {
			_data.set(0, _data.size(), bv);
		}
		_rlen = rlen;
		_odims = odims;
	}

	@Override
	public void resetNoFill(int rlen, int[] odims){
		int len = rlen * odims[0];
		if( len > capacity() )
			_data = new BitSet(len);
		
		_rlen = rlen;
		_odims = odims;
	}

	@Override
	public long capacity() {
		return (_data!=null) ? _data.size() : -1;
	}

	@Override
	protected long computeNnz(int bix, int start, int length) {
		return (start == 0 && length == _rlen * _odims[0]) ?
			_data.cardinality() : UtilFunctions.computeNnz(_data, start, length);
	}

	@Override
	public double[] values(int r) {
		double[] ret = getReuseRow(false);
		int ix = pos(r);
		int ncol = _odims[0];
		for(int j=0; j<ncol; j++)
			ret[j] = _data.get(ix+j) ? 1 : 0;
		return ret;
	}
	
	@Override
	public double[] valuesAt(int bix) {
		int len = _rlen*_odims[0];
		Warnings.warnFullFP64Conversion(len);
		return DataConverter.toDouble(_data, len);
	}

	@Override
	public int index(int r) {
		return 0;
	}

	@Override
	public void incr(int r, int c) {
		Warnings.warnInvalidBooleanIncrement(1);
		_data.set(pos(r, c));
	}
	
	@Override
	public void incr(int r, int c, double delta) {
		Warnings.warnInvalidBooleanIncrement(delta);
		_data.set(pos(r, c));
	}

	@Override
	protected void fillBlock(int bix, int fromIndex, int toIndex, double v) {
		_data.set(fromIndex, toIndex, v != 0);
	}

	@Override
	protected void setInternal(int bix, int ix, double v) {
		_data.set(ix, v != 0);
	}

	@Override
	public DenseBlock set(String s) {
		_data.set(0, blockSize() * _odims[0], Boolean.parseBoolean(s));
		return this;
	}

	@Override
	public DenseBlock set(int r, int c, double v) {
		_data.set(pos(r, c), v != 0);
		return this;
	}

	public DenseBlock set(int r, int c, boolean v) {
		_data.set(pos(r, c), v);
		return this;
	}
	
	@Override
	public DenseBlock set(DenseBlock db) {
		// ToDo: Performance tests and improvements
		double[] data = db.valuesAt(0);
		for (int i = 0; i < _rlen*_odims[0]; i++) {
			_data.set(i, data[i] != 0);
		}
		return this;
	}
	
	@Override
	public DenseBlock set(int rl, int ru, int cl, int cu, DenseBlock db) {
		//TODO perf computed indexes
		for (int r = rl; r < ru; r++) {
			for (int c = cl; c < cu; c++) {
				int i = r * _odims[0] + c;
				_data.set(i, db.get(r - rl, c - cl) != 0);
			}
		}
		return this;
	}

	@Override
	public DenseBlock set(int r, double[] v) {
		int ri = r * _odims[0];
		for (int i = ri; i < ri + v.length; i++) {
			_data.set(i, v[i - ri] != 0);
		}
		return this;
	}
	
	@Override
	public DenseBlock set(int[] ix, double v) {
		_data.set(pos(ix), v != 0);
		return this;
	}

	@Override
	public DenseBlock set(int[] ix, long v) {
		_data.set(pos(ix), v != 0);
		return this;
	}

	@Override
	public DenseBlock set(int[] ix, String v) {
		_data.set(pos(ix), Boolean.parseBoolean(v));
		return this;
	}

	@Override
	public double get(int r, int c) {
		return _data.get(pos(r, c)) ? 1 : 0;
	}

	@Override
	public double get(int[] ix) {
		return _data.get(pos(ix)) ? 1 : 0;
	}

	@Override
	public String getString(int[] ix) {
		return String.valueOf(_data.get(pos(ix)));
	}

	@Override
	public long getLong(int[] ix) {
		return _data.get(pos(ix)) ? 1 : 0;
	}
}
