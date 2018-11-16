/*
 * Modifications Copyright 2018 Graz University of Technology
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

import java.util.BitSet;

import org.tugraz.sysds.common.Warnings;
import org.tugraz.sysds.runtime.util.DataConverter;
import org.tugraz.sysds.runtime.util.UtilFunctions;

public class DenseBlockBool extends DenseBlockDRB
{
	private static final long serialVersionUID = -2228057308997136969L;
	
	private BitSet _data;

	public DenseBlockBool(int[] dims) {
		super(dims);
		reset(_rlen, _odims, 0);
	}
	
	public DenseBlockBool(int[] dims, boolean[] data) {
		super(dims);
		_data = new BitSet(data.length);
		for(int i=0; i<data.length; i++)
			if( data[i] )
			_data.set(i);
	}
	
	@Override
	public boolean isNumeric() {
		return true;
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
			_data.set(0, len, bv);
		}
		_rlen = rlen;
		_odims = odims;
	}

	@Override
	public long capacity() {
		return (_data!=null) ? _data.size() : -1;
	}

	@Override
	public long countNonZeros() {
		return _data.cardinality();
	}
	
	@Override
	public int countNonZeros(int r) {
		return UtilFunctions.computeNnz(_data, r*_odims[0], _odims[0]);
	}

	@Override
	public long countNonZeros(int rl, int ru, int ol, int ou) {
		long nnz = 0;
		if( ol == 0 && ou == _odims[0] ) { //specific case: all cols
			nnz += UtilFunctions.computeNnz(_data, rl*_odims[0], (ru-rl)*_odims[0]);
		}
		else {
			for( int i=rl, ix=rl*_odims[0]; i<ru; i++, ix+=_odims[0] )
				nnz += UtilFunctions.computeNnz(_data, ix+ol, ou-ol);
		}
		return nnz;
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
		Warnings.warnInvaldBooleanIncrement(1);
		_data.set(pos(r, c));
	}
	
	@Override
	public void incr(int r, int c, double delta) {
		Warnings.warnInvaldBooleanIncrement(delta);
		_data.set(pos(r, c));
	}
	
	@Override
	public DenseBlock set(double v) {
		_data.set(0, _rlen*_odims[0], v != 0);
		return this;
	}
	
	@Override
	public DenseBlock set(int rl, int ru, int ol, int ou, double v) {
		boolean bv = v != 0;
		if( ol==0 && ou == _odims[0] )
			_data.set(rl*_odims[0], ru*_odims[0], bv);
		else
			for(int i=rl, ix=rl*_odims[0]; i<ru; i++, ix+=_odims[0])
				_data.set(ix+ol, ix+ou, bv);
		return this;
	}

	@Override
	public DenseBlock set(int r, int c, double v) {
		_data.set(pos(r, c), v != 0);
		return this;
	}
	
	@Override
	public DenseBlock set(DenseBlock db) {
		System.arraycopy(db.valuesAt(0), 0, _data, 0, _rlen*_odims[0]);
		return this;
	}
	
	@Override
	public DenseBlock set(int rl, int ru, int ol, int ou, DenseBlock db) {
		double[] a = db.valuesAt(0);
		if( ol == 0 && ou == _odims[0])
			System.arraycopy(a, 0, _data, rl*_odims[0]+ol, (int)db.size());
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
	public double get(int r, int c) {
		return _data.get(pos(r, c)) ? 1 : 0;
	}

	@Override
	public double get(int[] ix) {
		return _data.get(pos(ix)) ? 1 : 0;
	}
}
