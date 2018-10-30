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

import java.util.Arrays;

import org.tugraz.sysds.runtime.util.UtilFunctions;

public class DenseBlockFP64 extends DenseBlockDRB
{
	private static final long serialVersionUID = 8546723684649816489L;

	private double[] data;

	@Override
	public void reset() {
		reset(rlen, clen, 0);
	}

	@Override
	public void reset(int rlen, int clen) {
		reset(rlen, clen, 0);
	}
	
	@Override
	public void reset(int rlen, int clen, double v) {
		int len = rlen * clen;
		if( len > capacity() ) {
			data = new double[len];
			if( v != 0 )
				Arrays.fill(data, v);
		}
		else {
			Arrays.fill(data, 0, len, v);
		}
		this.rlen = rlen;
		this.clen = clen;
	}

	@Override
	public long capacity() {
		return (data!=null) ? data.length : -1;
	}

	@Override
	public long countNonZeros() {
		return UtilFunctions.computeNnz(data, 0, rlen*clen);
	}
	
	@Override
	public int countNonZeros(int r) {
		return UtilFunctions.computeNnz(data, r*clen, clen);
	}

	@Override
	public long countNonZeros(int rl, int ru, int cl, int cu) {
		long nnz = 0;
		if( cl == 0 && cu == clen ) { //specific case: all cols
			nnz += UtilFunctions.computeNnz(data, rl*clen, (ru-rl)*clen);
		}
		else {
			for( int i=rl, ix=rl*clen; i<ru; i++, ix+=clen )
				nnz += UtilFunctions.computeNnz(data, ix+cl, cu-cl);
		}
		return nnz;
	}

	@Override
	public double[][] values() {
		return new double[][]{data};
	}

	@Override
	public double[] values(int r) {
		return data;
	}
	
	@Override
	public double[] valuesAt(int bix) {
		return data;
	}

	@Override
	public int index(int r) {
		return 0;
	}

	@Override
	public int pos(int r) {
		return r * clen;
	}

	@Override
	public int pos(int r, int c) {
		return r * clen + c;
	}

	@Override
	public void incr(int r, int c) {
		data[pos(r, c)] ++;
	}
	
	@Override
	public void incr(int r, int c, double delta) {
		data[pos(r, c)] += delta;
	}
	
	@Override
	public DenseBlock set(double v) {
		Arrays.fill(data, 0, rlen*clen, v);
		return this;
	}
	
	@Override
	public DenseBlock set(int rl, int ru, int cl, int cu, double v) {
		if( cl==0 && cu == clen )
			Arrays.fill(data, rl*clen, ru*clen, v);
		else
			for(int i=rl, ix=rl*clen; i<ru; i++, ix+=clen)
				Arrays.fill(data, ix+cl, ix+cu, v);
		return this;
	}

	@Override
	public DenseBlock set(int r, int c, double v) {
		data[pos(r, c)] = v;
		return this;
	}
	
	@Override
	public DenseBlock set(DenseBlock db) {
		System.arraycopy(db.valuesAt(0), 0, data, 0, rlen*clen);
		return this;
	}
	
	@Override
	public DenseBlock set(int rl, int ru, int cl, int cu, DenseBlock db) {
		double[] a = db.valuesAt(0);
		if( cl == 0 && cu == clen)
			System.arraycopy(a, 0, data, rl*clen+cl, (int)db.size());
		else {
			int len = cu - cl;
			for(int i=rl, ix1=0, ix2=rl*clen+cl; i<ru; i++, ix1+=len, ix2+=clen)
				System.arraycopy(a, ix1, data, ix2, len);
		}
		return this;
	}

	@Override
	public DenseBlock set(int r, double[] v) {
		System.arraycopy(v, 0, data, pos(r), clen);
		return this;
	}

	@Override
	public double get(int r, int c) {
		return data[pos(r, c)];
	}
}
