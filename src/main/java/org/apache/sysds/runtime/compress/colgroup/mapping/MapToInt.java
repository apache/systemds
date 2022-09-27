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
import java.util.BitSet;

import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
import org.apache.sysds.utils.MemoryEstimates;

public class MapToInt extends AMapToData {

	private static final long serialVersionUID = -5557070920888782274L;

	private final int[] _data;

	protected MapToInt(int size) {
		this(Character.MAX_VALUE + 1, size);
	}

	public MapToInt(int unique, int size) {
		super(unique);
		_data = new int[size];
	}

	private MapToInt(int unique, int[] data) {
		super(unique);
		_data = data;
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
			preAV[_data[rc]] += mV[off];
		for(int rc = cl + h; rc < cu; rc += 8, off += 8) {
			preAV[_data[rc]] += mV[off];
			preAV[_data[rc + 1]] += mV[off + 1];
			preAV[_data[rc + 2]] += mV[off + 2];
			preAV[_data[rc + 3]] += mV[off + 3];
			preAV[_data[rc + 4]] += mV[off + 4];
			preAV[_data[rc + 5]] += mV[off + 5];
			preAV[_data[rc + 6]] += mV[off + 6];
			preAV[_data[rc + 7]] += mV[off + 7];
		}
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
	public void copyInt(int[] d) {
		for(int i = 0; i < _data.length; i++)
			_data[i] = d[i];
	}

	@Override
	public void copyBit(BitSet d) {
		for(int i = d.nextSetBit(0); i >= 0; i = d.nextSetBit(i + 1))
			_data[i] = 1;
	}

	@Override
	public int[] getCounts(int[] ret) {
		for(int i = 0; i < _data.length; i++)
			ret[_data[i]]++;
		return ret;
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
		else if(unique <= 127)
			ret = new MapToUByte(unique, size);
		else if(unique < 256)
			ret = new MapToByte(unique, size);
		else if(unique < Character.MAX_VALUE - 1)
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
}
