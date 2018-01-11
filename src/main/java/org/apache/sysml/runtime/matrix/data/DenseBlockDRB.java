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


package org.apache.sysml.runtime.matrix.data;

import java.util.Arrays;

import org.apache.sysml.runtime.util.UtilFunctions;

public class DenseBlockDRB extends DenseBlock
{
	private static final long serialVersionUID = 8546723684649816489L;

	private double[] data;
	private int rlen;
	private int clen;

	public DenseBlockDRB(int rlen, int clen) {
		reset(rlen, clen, 0);
	}

	public DenseBlockDRB(double[] data, int rlen, int clen) {
		this.data = data;
		this.rlen = rlen;
		this.clen = clen;
	}

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
	public int numRows() {
		return rlen;
	}

	@Override
	public int numBlocks() {
		return 1;
	}
	
	@Override
	public int blockSize() {
		return rlen;
	}
	
	@Override
	public int blockSize(int bix) {
		return rlen;
	}
	
	@Override
	public boolean isContiguous() {
		return true;
	}
	
	@Override
	public boolean isContiguous(int rl, int ru) {
		return true;
	}

	@Override
	public long size() {
		return rlen * clen;
	}
	
	@Override
	public int size(int bix) {
		return rlen * clen;
	}

	@Override
	public long capacity() {
		return (data!=null) ? data.length : -1;
	}

	@Override
	public long countNonZeros() {
		final int len = rlen * clen;
		double[] a = data;
		int nnz = 0;
		for(int i=0; i<len; i++)
			nnz += (a[i]!=0) ? 1 : 0;
		return nnz;
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
	public void set(double v) {
		Arrays.fill(data, 0, rlen*clen, v);
	}
	
	@Override
	public void set(int rl, int ru, int cl, int cu, double v) {
		if( cl==0 && cu == clen )
			Arrays.fill(data, rl*clen, ru*clen, v);
		else
			for(int i=rl, ix=rl*clen; i<ru; i++, ix+=clen)
				Arrays.fill(data, ix+cl, ix+cu, v);
	}

	@Override
	public void set(int r, int c, double v) {
		data[pos(r, c)] = v;
	}
	
	@Override
	public void set(DenseBlock db) {
		System.arraycopy(db.valuesAt(0), 0, data, 0, rlen*clen);
	}
	
	@Override
	public void set(int rl, int ru, int cl, int cu, DenseBlock db) {
		double[] a = db.valuesAt(0);
		if( cl == 0 && cu == clen)
			System.arraycopy(a, 0, data, rl*clen+cl, (int)db.size());
		else {
			int len = cu - cl;
			for(int i=rl, ix1=0, ix2=rl*clen+cl; i<ru; i++, ix1+=len, ix2+=clen)
				System.arraycopy(a, ix1, data, ix2, len);
		}
	}

	@Override
	public void set(int r, double[] v) {
		System.arraycopy(v, 0, data, pos(r), clen);
	}

	@Override
	public double get(int r, int c) {
		return data[pos(r, c)];
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		for(int i=0, ix=0; i<rlen; i++, ix+=clen) {
			for(int j=0; j<clen; j++) {
				sb.append(data[ix+j]);
				sb.append("\t");
			}
			sb.append("\n");
		}
		return sb.toString();
	}
}
