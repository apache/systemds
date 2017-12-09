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

public class DenseBlockLDRB extends DenseBlock
{
	private static final long serialVersionUID = -7285459683402612969L;

	private double[][] data;
	private int rlen;
	private int clen;
	private int blen;
	
	public DenseBlockLDRB(int rlen, int clen) {
		this(rlen, clen, blocksize(rlen, clen));
	}
	
	public DenseBlockLDRB(int rlen, int clen, int blen) {
		reset(rlen, clen, blen, 0);
	}
	
	@Override
	public void reset() {
		reset(rlen, clen, blen, 0);
	}

	@Override
	public void reset(int rlen, int clen) {
		reset(rlen, clen, blen, 0);
	}
	
	@Override
	public void reset(int rlen, int clen, double v) {
		reset(rlen, clen, blen, v);
	}
	
	private void reset(int rlen, int clen, int blen, double v) {
		long llen = (long) rlen * clen;
		int numPart = (int)Math.ceil((double)rlen / blen);
		if( this.blen == blen && llen < capacity() ) {
			for(int i=0; i<numPart; i++) {
				int len = Math.min((i+1)*blen,rlen)-i*blen;
				Arrays.fill(data[i], 0, len, v);
			}
		}
		else {
			data = new double[numPart][];
			for(int i=0; i<numPart; i++) {
				int len = Math.min((i+1)*blen,rlen)-i*blen;
				data[i] = new double[len];
				if( v != 0 )
					Arrays.fill(data[i], v);
			}
		}
		this.rlen = rlen;
		this.clen = clen;
		this.blen = blen;
	}

	@Override
	public int numRows() {
		return rlen;
	}

	@Override
	public int numBlocks() {
		return data.length;
	}

	@Override
	public long size() {
		return (long)rlen * clen;
	}

	@Override
	public long capacity() {
		int len = 0;
		for(int i=0; i<numBlocks(); i++)
			len += data[i].length;
		return len;
	}

	@Override
	public long countNonZeros() {
		long nnz = 0;
		for(int i=0; i<numBlocks(); i++ ) {
			double[] a = values(i);
			for(int j=0; j<a.length; j++)
				nnz += (a[j]!=0) ? 1 : 0;
		}
		return nnz;
	}
	
	@Override
	public long countNonZeros(int rl, int ru, int cl, int cu) {
		long nnz = 0;
		for(int bi=index(rl); bi<index(ru); bi++) {
			double[] a = data[bi];
			for(int i=pos(bi), ix=pos(bi)*clen; i<blen(bi); i++, ix+=clen)
				for( int j=cl; j<cu; j++ )
					nnz += (a[ix+j]!=0) ? 1 : 0;
		}
		return nnz;
	}

	@Override
	public double[][] values() {
		return data;
	}

	@Override
	public double[] values(int bix) {
		return data[bix];
	}

	@Override
	public int index(int r) {
		return r / blen;
	}

	@Override
	public int pos(int r) {
		return (r % blen) * clen;
	}

	@Override
	public int pos(int r, int c) {
		return (r % blen) * clen + c;
	}

	@Override
	public void set(double v) {
		for(int i=0; i<numBlocks(); i++)
			Arrays.fill(data[i], v);
	}
	
	@Override
	public void set(int rl, int ru, int cl, int cu, double v) {
		for(int bi=index(rl); bi<=index(ru); bi++) {
			int imin = pos(bi);
			int imax = Math.min(ru, (bi+1)*blen)-bi;
			for(int i=imin, ix=imin*clen; i<imax; i++, ix+=clen)
				Arrays.fill(data, ix+cl, ix+cu, v);
		}
	}

	@Override
	public void set(int r, int c, double v) {
		data[index(r)][pos(r, c)] = v;
	}

	@Override
	public void set(int r, double[] v) {
		System.arraycopy(v, 0, data[index(r)], pos(r), v.length);
	}

	@Override
	public double get(int r, int c) {
		return data[index(r)][pos(r, c)];
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		for(int i=0; i<rlen; i++) {
			double[] data = values(index(i));
			int ix = pos(i);
			for(int j=0; j<clen; j++) {
				sb.append(data[ix+j]);
				sb.append("\t");
			}
			sb.append("\n");
		}
		return sb.toString();
	}
	
	private int blen(int bix) {
		return Math.min(blen, rlen-bix*blen);
	}

	private static int blocksize(int rlen, int clen) {
		return Math.min(rlen, Integer.MAX_VALUE / clen);
	}
}
