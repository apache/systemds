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

package org.apache.sysds.runtime.compress.colgroup.mapping;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.IMapToDataGroup;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
import org.apache.sysds.utils.MemoryEstimates;

public class MapToInt extends AMapToData {

	private static final long serialVersionUID = -5557070920888782274L;

	private final int[] _data;

	protected MapToInt(int size) {
		this(Integer.MAX_VALUE, size);
	}

	public MapToInt(int unique, int size) {
		super(unique);
		_data = new int[size];
	}

	private MapToInt(int unique, int[] data) {
		super(unique);
		_data = data;
		verify();
	}

	protected int[] getData() {
		return _data;
	}

	@Override
	public MAP_TYPE getType() {
		return MapToFactory.MAP_TYPE.INT;
	}

	@Override
	public int getIndex(int n) {
		return _data[n];
	}

	@Override
	public void fill(int v) {
		Arrays.fill(_data, v);
	}

	@Override
	public long getInMemorySize() {
		return getInMemorySize(_data.length);
	}

	protected static long getInMemorySize(int dataLength) {
		long size = 16 + 8; // object header + object reference
		size += MemoryEstimates.intArrayCost(dataLength);
		return size;
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 4 + 4 + _data.length * 4;
	}

	@Override
	public void set(int n, int v) {
		_data[n] = v;
	}

	@Override
	public void set(int l, int u, int off, AMapToData tm) {
		for(int i = l; i < u; i++, off++) {
			set(i, tm.getIndex(off));
		}
	}

	@Override
	public int setAndGet(int n, int v) {
		return _data[n] = v;
	}

	@Override
	public int size() {
		return _data.length;
	}

	@Override
	public void replace(int v, int r) {
		for(int i = 0; i < size(); i++)
			if(_data[i] == v)
				_data[i] = r;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(MAP_TYPE.INT.ordinal());
		out.writeInt(getUnique());
		out.writeInt(_data.length);
		for(int i = 0; i < _data.length; i++)
			out.writeInt(_data[i]);
	}

	protected static MapToInt readFields(DataInput in) throws IOException {
		int unique = in.readInt();
		final int length = in.readInt();
		final int[] data = new int[length];
		for(int i = 0; i < length; i++)
			data[i] = in.readInt();
		return new MapToInt(unique, data);
	}

	@Override
	protected void preAggregateDenseToRowBy8(double[] mV, double[] preAV, int cl, int cu, int off) {
		final int h = (cu - cl) % 8;
		off += cl;
		for(int rc = cl; rc < cl + h; rc++, off++)
			preAV[getIndex(rc)] += mV[off];
		for(int rc = cl + h; rc < cu; rc += 8, off += 8)
			preAggregateDenseToRowVec8(mV, preAV, rc, off);
	}

	@Override
	protected void preAggregateDenseToRowVec8(double[] mV, double[] preAV, int rc, int off) {
		preAV[getIndex(rc)] += mV[off];
		preAV[getIndex(rc + 1)] += mV[off + 1];
		preAV[getIndex(rc + 2)] += mV[off + 2];
		preAV[getIndex(rc + 3)] += mV[off + 3];
		preAV[getIndex(rc + 4)] += mV[off + 4];
		preAV[getIndex(rc + 5)] += mV[off + 5];
		preAV[getIndex(rc + 6)] += mV[off + 6];
		preAV[getIndex(rc + 7)] += mV[off + 7];
	}

	@Override
	protected void preAggregateDenseMultiRowContiguousBy8(double[] mV, int nCol, int nVal, double[] preAV, int rl,
		int ru, int cl, int cu) {
		final int h = (cu - cl) % 8;
		preAggregateDenseMultiRowContiguousBy1(mV, nCol, nVal, preAV, rl, ru, cl, cl + h);
		final int offR = nCol * rl;
		final int offE = nCol * ru;
		for(int c = cl + h; c < cu; c += 8) {
			final int id1 = _data[c], id2 = _data[c + 1], id3 = _data[c + 2], id4 = _data[c + 3], id5 = _data[c + 4],
				id6 = _data[c + 5], id7 = _data[c + 6], id8 = _data[c + 7];

			final int start = c + offR;
			final int end = c + offE;
			int nValOff = 0;
			for(int off = start; off < end; off += nCol) {
				preAV[id1 + nValOff] += mV[off];
				preAV[id2 + nValOff] += mV[off + 1];
				preAV[id3 + nValOff] += mV[off + 2];
				preAV[id4 + nValOff] += mV[off + 3];
				preAV[id5 + nValOff] += mV[off + 4];
				preAV[id6 + nValOff] += mV[off + 5];
				preAV[id7 + nValOff] += mV[off + 6];
				preAV[id8 + nValOff] += mV[off + 7];
				nValOff += nVal;
			}
		}
	}

	@Override
	public int getUpperBoundValue() {
		return Integer.MAX_VALUE;
	}

	@Override
	public void copyInt(int[] d, int start, int end) {
		for(int i = start; i < end; i++)
			_data[i] = d[i];
	}

	@Override
	public int[] getCounts(int[] ret) {
		final int h = (_data.length) % 8;
		for(int i = 0; i < h; i++)
			ret[_data[i]]++;
		getCountsBy8P(ret, h, _data.length);
		return ret;
	}

	private void getCountsBy8P(int[] ret, int s, int e) {
		for(int i = s; i < e; i += 8) {
			ret[_data[i]]++;
			ret[_data[i + 1]]++;
			ret[_data[i + 2]]++;
			ret[_data[i + 3]]++;
			ret[_data[i + 4]]++;
			ret[_data[i + 5]]++;
			ret[_data[i + 6]]++;
			ret[_data[i + 7]]++;
		}
	}


	@Override
	public int countRuns() {
		int c = 1;
		int prev = _data[0];
		for(int i = 1; i < _data.length; i++) {
			c += prev == _data[i] ? 0 : 1;
			prev = _data[i];
		}
		return c;
	}

	@Override
	public AMapToData resize(int unique) {
		final int size = _data.length;
		AMapToData ret;
		if(unique <= 1)
			return new MapToZero(size);
		else if(unique == 2 && size > 32)
			ret = new MapToBit(unique, size);
		else if(unique < 128)
			ret = new MapToUByte(unique, size);
		else if(unique < 256)
			ret = new MapToByte(unique, size);
		else if(unique < Character.MAX_VALUE)
			ret = new MapToChar(unique, size);
		else if(unique < MapToCharPByte.max)
			ret = new MapToCharPByte(unique, size);
		else {
			setUnique(unique);
			return this;
		}
		ret.copyInt(_data);
		return ret;
	}

	@Override
	public AMapToData slice(int l, int u) {
		return new MapToInt(getUnique(), Arrays.copyOfRange(_data, l, u));
	}

	@Override
	public AMapToData append(AMapToData t) {
		if(t instanceof MapToInt) {
			MapToInt tb = (MapToInt) t;
			int[] tbb = tb._data;
			final int newSize = _data.length + t.size();
			final int newDistinct = Math.max(getUnique(), t.getUnique());

			// copy
			int[] ret = Arrays.copyOf(_data, newSize);
			System.arraycopy(tbb, 0, ret, _data.length, t.size());

			return new MapToInt(newDistinct, ret);
		}
		else {
			throw new NotImplementedException("Not implemented append on Bit map different type");
		}
	}

	@Override
	public AMapToData appendN(IMapToDataGroup[] d) {
		int p = 0; // pointer
		for(IMapToDataGroup gd : d)
			p += gd.getMapToData().size();
		final int[] ret = new int[p];

		p = 0;
		for(int i = 0; i < d.length; i++) {
			if(d[i].getMapToData().size() > 0) {
				final MapToInt mm = (MapToInt) d[i].getMapToData();
				final int ms = mm.size();
				System.arraycopy(mm._data, 0, ret, p, ms);
				p += ms;
			}
		}

		return new MapToInt(getUnique(), ret);
	}

	@Override
	public boolean equals(AMapToData e) {
		return e instanceof MapToInt && //
			e.getUnique() == getUnique() && //
			Arrays.equals(((MapToInt) e)._data, _data);
	}

	@Override
	public void decompressToRange(double[] c, int rl, int ru, int offR, double[] values) {
		// OVERWRITTEN FOR JIT COMPILE!
		if(offR == 0)
			decompressToRangeNoOff(c, rl, ru, values);
		else
			decompressToRangeOff(c, rl, ru, offR, values);
	}

	@Override
	public void decompressToRangeOff(double[] c, int rl, int ru, int offR, double[] values) {
		for(int i = rl, offT = rl + offR; i < ru; i++, offT++)
			c[offT] += values[getIndex(i)];
	}

	@Override
	public void decompressToRangeNoOff(double[] c, int rl, int ru, double[] values) {
		// OVERWRITTEN FOR JIT COMPILE!
		final int h = (ru - rl) % 8;
		for(int rc = rl; rc < rl + h; rc++)
			c[rc] += values[getIndex(rc)];
		for(int rc = rl + h; rc < ru; rc += 8)
			decompressToRangeNoOffBy8(c, rc, values);
	}
}
