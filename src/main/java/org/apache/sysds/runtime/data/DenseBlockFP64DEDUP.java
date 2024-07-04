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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.MemoryEstimates;

import java.util.HashMap;

public class DenseBlockFP64DEDUP extends DenseBlockDRB
{
	private static final long serialVersionUID = -4012376952006079198L;
	private double[][] _data;
	//TODO: implement estimator for nr of distinct
	private int _distinct = 0;
	private int _emb_size = 0;
	private int _embPerRow = 0;

	public void setDistinct(int d){
		_distinct = d;
	}

	public void setEmbeddingSize(int s){
		_emb_size = s;
		if (_odims[0] % _emb_size != 0)
			throw new RuntimeException("[Error] DedupDenseBlock: ncols[=" + _odims[0] + "] % emb_size[=" + _emb_size + "] != 0");
		_embPerRow = _odims[0] / _emb_size;
	}
	protected DenseBlockFP64DEDUP(int[] dims) {
		super(dims);
		reset(_rlen, _odims, 0);
	}

	public int getNrDistinctRows(){
		return _distinct;
	}

	public int getNrEmbsPerRow(){
		return _embPerRow;
	}

	public int getEmbSize(){
		return _emb_size;
	}

	@Override
	protected void allocateBlock(int bix, int length) {
		_data = new double[length][];
	}

	@Override
	public void reset(int rlen, int[] odims, double v) {
		if(rlen >  _rlen)
			allocateBlock(0,rlen);
		else{
			if(_data == null)
				allocateBlock(0,rlen);
			if(v == 0.0)
				 for(int i = 0; i < rlen; i++)
					 _data[i] = null;
			else
				throw new NotImplementedException("Reset of DedupBlock with constant value is supported");
		}
		_rlen = rlen;
		_odims = odims;
	}

	@Override
	public void resetNoFill(int rlen, int[] odims) {
		if(_data == null || rlen > _rlen)
			_data = new double[rlen][];
		_rlen = rlen;
		_odims = odims;
	}

	public void resetNoFillDedup(int rlen, int embsPerRow) {
		if(_data == null || rlen > _rlen)
			_data = new double[rlen*embsPerRow][];
		_embPerRow = embsPerRow;
		_rlen = rlen;
	}

	@Override
	public boolean isNumeric() {
		return true;
	}

	@Override
	public boolean isNumeric(Types.ValueType vt) {
		return Types.ValueType.FP64 == vt;
	}

	@Override
	public long capacity() {
		return (_data != null) ? _data.length : -1;
	}

	@Override
	public long countNonZeros() {
		long nnz = 0;
		HashMap<double[], Long> cache = new HashMap<>();
		for (int i = 0; i < _rlen; i++) {
			double[] row = this._data[i];
			if(row == null)
				continue;
			Long count = cache.getOrDefault(row, null);
			if(count == null){
				count = Long.valueOf(countNonZeros(i));
				cache.put(row, count);
			}
			nnz += count;
		}
		this._distinct = cache.size();
		return nnz;
	}

	@Override
	public int countNonZeros(int r) {
		return _data[r] == null ? 0 : UtilFunctions.computeNnz(_data[r], 0, _odims[0]);
	}

	@Override
	public long countNonZeros(int rl, int ru, int ol, int ou) {
		long nnz = 0;
		HashMap<double[], Long> cache = new HashMap<>();
		for (int i = rl; i < ru; i++) {
			double[] row = this._data[i];
			if(row == null)
				continue;
			Long count = cache.getOrDefault(row, null);
			if(count == null){
				count = Long.valueOf(UtilFunctions.computeNnz(_data[i], ol, ou));
				cache.put(row, count);
			}
			nnz += count;
		}
		return nnz;
	}

	@Override
	protected long computeNnz(int bix, int start, int length) {
		int nnz = 0;
		int row_start = (int) Math.floor(((double) start) / _odims[0]);
		int col_start = start % _odims[0];
		for (int i = 0; i < length; i++) {
			if(_data[row_start] == null){
				i += _odims[0] - 1 - col_start;
				col_start = 0;
				row_start += 1;
				continue;
			}
			nnz += _data[row_start][col_start] != 0 ? 1 : 0;
			col_start += 1;
			if(col_start == _odims[0]) {
				col_start = 0;
				row_start += 1;
			}
		}
		return nnz;
	}

	@Override
	public int numBlocks() {
		int blocksize = blockSize();
		if(blocksize < _rlen){
			int numBlocks = _rlen / blocksize;
			if (_rlen % blocksize > 0)
				numBlocks += 1;
			return  numBlocks;
		}
		else
			return 1;
	}

	@Override
	public int blockSize() {
		int blocksize = Integer.MAX_VALUE / _odims[0];
		return Math.min(blocksize, _rlen);
	}

	@Override
	public int blockSize(int bix) {
		int blocksize = blockSize();
		return Math.min(blocksize, _rlen - bix * blocksize);
	}

	@Override
	public boolean isContiguous() {
		return numBlocks() == 1;
	}

	@Override
	public boolean isContiguous(int rl, int ru) {
		return index(rl) == index(ru);
	}

	@Override
	public int pos(int r) {
		return (r % blockSize()) * _odims[0];
	}

	@Override
	public int pos(int r, int c) {
		return (r % blockSize()) * _odims[0] + c;
	}

	@Override
	public int pos(int[] ix){
		int pos = pos(ix[0]);
		pos += ix[ix.length - 1];
		for(int i = 1; i < ix.length - 1; i++)
			pos += ix[i] * _odims[i];
		return pos;
	}

	@Override
	public double[] values(int r) {
		return valuesAt(index(r));
	}

	@Override
	public double[] valuesAt(int bix) {
		int blocksize = blockSize(bix);
		int blocksizeOther = blockSize();
		double[] out = new double[_odims[0]*blocksize];
		if(_data != null) {
			for (int i = 0; i < blocksize; i++) {
				for (int j = 0; j < _embPerRow; j++) {
					int posInDedup = i * _embPerRow + j;
					int posInDense = posInDedup * _emb_size;
					posInDedup += bix*blocksizeOther*_embPerRow;
					if(_data[posInDedup] != null)
						System.arraycopy(_data[posInDedup], 0, out, posInDense, _emb_size);
				}
			}
		}
		return out;
	}

	@Override
	public int index(int r) {
		return r / blockSize();
	}

	@Override
	public int size(int bix) {
		return blockSize(bix) * _odims[0];
	}

	@Override
	public void incr(int r, int c) {
		incr(r,c,1.0);
	}

	public void createDeepCopyOfEmbedding(int pos){
		if(_data[pos] == null)
			_data[pos] = new double[_emb_size];
		else {
			double[] tmp = new double[_emb_size];
			System.arraycopy(_data[pos], 0, tmp, 0, _emb_size);
			_data[pos] = tmp;
		}
	}

	@Override
	public void incr(int r, int c, double delta) {
		int roffset = c / _emb_size;
		int coffset = c % _emb_size;

		//creates a deep copy to avoid unexpected changes in other rows due deduplication
		createDeepCopyOfEmbedding(r*_embPerRow + roffset);
		_data[r*_embPerRow + roffset][coffset] += delta;
	}

	@Override
	public void fillBlock(int bix, int fromIndex, int toIndex, double v) {
		int roffset = fromIndex / _emb_size;
		int coffset = fromIndex % _emb_size;
		int r2offset = fromIndex / _emb_size;
		int c2offset = fromIndex % _emb_size;
		int blockoffset = bix*blockSize();

		int c = coffset;
		int cmax = _emb_size;
		int rmax = r2offset;

		if(c2offset != 0)
			rmax += 1;
		for (int r = roffset; r < rmax; r++) {
			//creates a deep copy to avoid unexpected changes in other rows due deduplication
			createDeepCopyOfEmbedding(blockoffset + roffset);
			if(r == r2offset)
				cmax = c2offset;
			for(; c < cmax; c++){
				_data[blockoffset + r][c] = v;
			}
			c = 0;
		}
	}

	@Override 
	public void fillRow(int r, double v){
		throw new NotImplementedException();
	}


	@Override
	protected void setInternal(int bix, int ix, double v) {
		set(bix, ix, v);
	}

	@Override
	public DenseBlock set(int r, int c, double v) {
		int roffset = c / _emb_size;
		int coffset = c % _emb_size;

		//creates a deep copy to avoid unexpected changes in other rows due deduplication
		createDeepCopyOfEmbedding(r*_embPerRow + roffset);
		_data[r*_embPerRow + roffset][coffset] = v;
		return this;
	}

	@Override
	public DenseBlock set(int r, double[] v) {
		if(_embPerRow == 1)
			_data[r] = v;
		else
			for (int i = 0; i < _embPerRow; i++) {
				//creates a deep copy to avoid unexpected changes in other rows due deduplication
				createDeepCopyOfEmbedding(r*_embPerRow + i);
				System.arraycopy(v, i*_emb_size, _data[r*_embPerRow + i],0, _emb_size);
			}
		return this;
	}

	public DenseBlock setDedupDirectly(int r, double[] v) {
		_data[r] = v;
		return this;
	}

	@Override
	public DenseBlock set(DenseBlock db) {
		throw new NotImplementedException();
	}

	@Override
	//todo
	public DenseBlock set(int rl, int ru, int ol, int ou, DenseBlock db) {
		if( !(db instanceof DenseBlockFP64DEDUP))
			throw new NotImplementedException();
		HashMap<double[], double[]> cache = new HashMap<>();
		int len = ou - ol;
		for(int i=rl, ix1 = 0; i<ru; i++, ix1++){
			double[] row = db.values(ix1);
			double[] newRow = cache.get(row);
			if (newRow == null) {
				 newRow = new double[len];
				 System.arraycopy(row, 0, newRow, 0, len);
				 cache.put(row, newRow);
			}
			set(i, newRow);
		}
		return this;
	}

	@Override
	public DenseBlock set(int[] ix, double v) {
		return set(ix[0], pos(ix), v);
	}

	@Override
	public DenseBlock set(int[] ix, long v) {
		return set(ix[0], pos(ix), v);
	}

	@Override
	public DenseBlock set(int[] ix, String v) {
		return set(ix[0], pos(ix), Double.parseDouble(v));
	}

	public double[] getDedupDirectly(int pos){
		return _data[pos];
	}

	@Override
	public double get(int r, int c) {
		if(_embPerRow == 1)
			return _data[r][c];
		int roffset = c / _emb_size;
		int coffset = c % _emb_size;
		if(_data[r*_embPerRow + roffset] == null)
			return 0.0;
		else
			return _data[r*_embPerRow + roffset][coffset];
	}

	@Override
	public double get(int[] ix) {
		return get(ix[0], pos(ix));
	}

	@Override
	public String getString(int[] ix) {
		return String.valueOf(get(ix[0], pos(ix)));
	}

	@Override
	public long getLong(int[] ix) {
		return UtilFunctions.toLong(get(ix[0], pos(ix)));
	}

	public long estimateMemory(){
		return estimateMemory(_rlen, _odims[0], _distinct);
	}

	public static long estimateMemory(int rows, int cols, int duplicates){
		return estimateMemory((long) rows, (long)  cols, (long) duplicates);
	}

	public static long estimateMemory(long rows, long cols, long duplicates){
		return ((long) (DenseBlock.estimateMemory(rows, cols)))
				+ ((long) MemoryEstimates.doubleArrayCost(cols)*duplicates)
				+ ((long) MemoryEstimates.objectArrayCost(rows));
	}
}