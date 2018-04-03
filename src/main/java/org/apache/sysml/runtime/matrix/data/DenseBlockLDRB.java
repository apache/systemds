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
import java.util.stream.IntStream;

import org.apache.sysml.runtime.util.UtilFunctions;

public class DenseBlockLDRB extends DenseBlock
{
	private static final long serialVersionUID = -7285459683402612969L;

	private static final boolean PARALLEL_ALLOC = true;
	
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
	
	@SuppressWarnings("resource")
	private void reset(int rlen, int clen, int blen, double v) {
		long llen = (long) rlen * clen;
		int numPart = (int)Math.ceil((double)rlen / blen);
		if( this.blen == blen && llen < capacity() ) {
			for(int i=0; i<numPart; i++) {
				int lrlen = (int)(Math.min((i+1)*blen,rlen)-i*blen);
				Arrays.fill(data[i], 0, lrlen*clen, v);
			}
		}
		else {
			data = new double[numPart][];
			IntStream range = PARALLEL_ALLOC ?
				IntStream.range(0, numPart).parallel() :
				IntStream.range(0, numPart);
			range.forEach(i ->
				data[i] = allocArray(i, rlen, clen, blen, v));
		}
		this.rlen = rlen;
		this.clen = clen;
		this.blen = blen;
	}
	
	private static double[] allocArray(int i, int rlen, int clen, int blen, double v) {
		int lrlen = (int)(Math.min((i+1)*blen,rlen)-i*blen);
		double[] ret = new double[lrlen*clen];
		if( v != 0 )
			Arrays.fill(ret, v);
		return ret;
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
	public int blockSize() {
		return blen;
	}
	
	@Override
	public int blockSize(int bix) {
		return Math.min(blen, rlen-bix*blen);
	}
	
	@Override
	public boolean isContiguous() {
		return rlen <= blen;
	}
	
	@Override
	public boolean isContiguous(int rl, int ru) {
		return isContiguous() || index(rl)==index(ru);
	}

	@Override
	public long size() {
		return (long)rlen * clen;
	}
	
	@Override
	public int size(int bix) {
		return blockSize(bix) * clen;
	}

	@Override
	public long capacity() {
		long len = 0;
		for(int i=0; i<numBlocks(); i++)
			len += data[i].length;
		return len;
	}

	@Override
	public long countNonZeros() {
		long nnz = 0;
		for(int i=0; i<numBlocks(); i++ )
			nnz += UtilFunctions.computeNnz(valuesAt(i), 0, size(i));
		return nnz;
	}
	
	@Override
	public int countNonZeros(int r) {
		return UtilFunctions.computeNnz(values(r), pos(r), clen);
	}
	
	@Override
	public long countNonZeros(int rl, int ru, int cl, int cu) {
		long nnz = 0;
		boolean rowBlock = (cl == 0 && cu == clen);
		final int bil = index(rl);
		final int biu = index(ru-1);
		for(int bi=bil; bi<=biu; bi++) {
			int lpos = (bi==bil) ? pos(rl) : 0;
			int len = (bi==biu) ? pos(ru-1)-lpos+clen : blockSize(bi)*clen;
			if( rowBlock )
				nnz += UtilFunctions.computeNnz(data[bi], lpos, len);
			else
				for(int i=lpos; i<lpos+len; i+=clen)
					nnz += UtilFunctions.computeNnz(data[i], i+cl, cu-cl);
		}
		return nnz;
	}

	@Override
	public double[][] values() {
		return data;
	}
	
	@Override
	public double[] values(int r) {
		return data[r / blen];
	}

	@Override
	public double[] valuesAt(int bix) {
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
	public DenseBlock set(double v) {
		for(int i=0; i<numBlocks(); i++)
			Arrays.fill(data[i], v);
		return this;
	}
	
	@Override
	public DenseBlock set(int rl, int ru, int cl, int cu, double v) {
		boolean rowBlock = (cl == 0 && cu == clen);
		final int bil = index(rl);
		final int biu = index(ru-1);
		for(int bi=bil; bi<=biu; bi++) {
			int lpos = (bi==bil) ? pos(rl) : 0;
			int len = (bi==biu) ? pos(ru-1)-lpos+clen : blockSize(bi)*clen;
			if( rowBlock )
				Arrays.fill(data[bi], lpos, lpos+len, v);
			else
				for(int i=lpos; i<lpos+len; i+=clen)
					Arrays.fill(data[bi], i+cl, i+cu, v);
		}
		return this;
	}

	@Override
	public DenseBlock set(int r, int c, double v) {
		data[index(r)][pos(r, c)] = v;
		return this;
	}

	@Override
	public DenseBlock set(int r, double[] v) {
		System.arraycopy(v, 0, data[index(r)], pos(r), clen);
		return this;
	}
	
	@Override
	public DenseBlock set(DenseBlock db) {
		for(int bi=0; bi<numBlocks(); bi++)
			System.arraycopy(db.valuesAt(bi), 0, data[bi], 0, size(bi));
		return this;
	}
	
	@Override
	public DenseBlock set(int rl, int ru, int cl, int cu, DenseBlock db) {
		for(int i=rl; i<ru; i++) {
			System.arraycopy(db.values(i-rl),
				db.pos(i-rl), values(i), pos(i, cl), cu-cl);
		}
		return this;
	}

	@Override
	public double get(int r, int c) {
		return data[index(r)][pos(r, c)];
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		for(int i=0; i<rlen; i++) {
			double[] data = values(i);
			int ix = pos(i);
			for(int j=0; j<clen; j++) {
				sb.append(data[ix+j]);
				sb.append("\t");
			}
			sb.append("\n");
		}
		return sb.toString();
	}

	private static int blocksize(int rlen, int clen) {
		return Math.min(rlen, Integer.MAX_VALUE / clen);
	}
}
