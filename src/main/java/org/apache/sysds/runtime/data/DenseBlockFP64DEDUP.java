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

import java.util.Arrays;
import java.util.HashMap;

public class DenseBlockFP64DEDUP extends DenseBlockDRB{
	private static final long serialVersionUID = 4124905190428752213L;
	private double[][] _data;

	protected DenseBlockFP64DEDUP(int[] dims) {
		super(dims);
		reset(_rlen, _odims, 0);
	}

	@Override
	protected void allocateBlock(int bix, int length) {
		_data[bix] = new double[length];
	}

	@Override
	public void reset(int rlen, int[] odims, double v) {
		if(rlen >  capacity() / _odims[0])
			_data = new double[rlen][];
		else{
			if(v == 0.0){
				for(int i = 0; i < rlen; i++)
					_data[i] = null;
			}
			else {
				for(int i = 0; i < rlen; i++){
					if(odims[0] > _odims[0] ||_data[i] == null )
						allocateBlock(i, odims[0]);
					Arrays.fill(_data[i], 0, odims[0], v);
				}
			}
		}
		_rlen = rlen;
		_odims = odims;
	}

	@Override
	public void resetNoFill(int rlen, int[] odims) {
		if(_data == null || rlen > _rlen){
			_data = new double[rlen][];
		}
		_rlen = rlen;
		_odims = odims;
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
		return (_data != null) ? _data.length*_odims[0] : -1;
	}

	@Override
	public long countNonZeros(){
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
		return nnz;
	}

	@Override
	public int countNonZeros(int r) {
		return _data[r] == null ? 0 : UtilFunctions.computeNnz(_data[r], 0, _odims[0]);
	}

	@Override
	protected long computeNnz(int bix, int start, int length) {
		int nnz = 0;
		int row_start = (int) Math.floor(start / _odims[0]);
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
	public int pos(int r){
		return 0;
	}

	@Override
	public int pos(int[] ix){
		int pos = ix[ix.length - 1];
		for(int i = 1; i < ix.length - 1; i++)
			pos += ix[i] * _odims[i];
		return pos;
	}

	@Override
	public double[] values(int r) {
		return valuesAt(r);
	}

	@Override
	public double[] valuesAt(int bix) {
		return _data[bix] == null ? new double[_odims[0]] : _data[bix];
	}

	@Override
	public int index(int r) {
		return r;
	}

	@Override
	public int numBlocks(){
		return _data.length;
	}

	@Override
	public int size(int bix) {
		return _odims[0];
	}

	@Override
	public void incr(int r, int c) {
		incr(r,c,1.0);
	}

	@Override
	public void incr(int r, int c, double delta) {
		if(_data[r] == null)
			allocateBlock(r, _odims[0]);
		_data[r][c] += delta;
	}

	@Override
	protected void fillBlock(int bix, int fromIndex, int toIndex, double v) {
		if(_data[bix] == null)
			allocateBlock(bix, _odims[0]);
		Arrays.fill(_data[bix], fromIndex, toIndex, v);
	}

	@Override
	protected void setInternal(int bix, int ix, double v) {
		set(bix, ix, v);
	}

	@Override
	public DenseBlock set(int r, int c, double v) {
		if(_data[r] == null)
			_data[r] = new double[_odims[0]];
		_data[r][c] = v;
		return this;
	}

	@Override
	public DenseBlock set(int r, double[] v) {
		if(v.length == _odims[0])
			_data[r] = v;
		else
			throw new RuntimeException("set Denseblock called with an array length [" + v.length +"], array to overwrite is of length [" + _odims[0] + "]");
		return this;
	}

	@Override
	public DenseBlock set(DenseBlock db) {
		throw new NotImplementedException();
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

	@Override
	public double get(int r, int c) {
		if(_data[r] == null)
			return 0.0;
		else
			return _data[r][c];
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
}
