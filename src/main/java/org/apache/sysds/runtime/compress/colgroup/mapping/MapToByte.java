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

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.AMapToDataGroup;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
import org.apache.sysds.utils.MemoryEstimates;

public class MapToByte extends AMapToData {

	private static final long serialVersionUID = -2498505439667351828L;

	protected final byte[] _data;

	protected MapToByte(int size) {
		this(256, size);
	}

	public MapToByte(int unique, int size) {
		super(Math.min(unique, 256));
		_data = new byte[size];
	}

	protected MapToByte(int unique, byte[] data) {
		super(unique);
		_data = data;
	}

	protected MapToUByte toUByte() {
		return new MapToUByte(getUnique(), _data);
	}

	@Override
	public MAP_TYPE getType() {
		return MapToFactory.MAP_TYPE.BYTE;
	}

	@Override
	public int getIndex(int n) {
		return _data[n] & 0xFF;
	}

	@Override
	public void fill(int v) {
		Arrays.fill(_data, (byte) v);
	}

	@Override
	public long getInMemorySize() {
		return getInMemorySize(_data.length);
	}

	public static long getInMemorySize(int dataLength) {
		long size = 16 + 8; // object header + object reference
		size += MemoryEstimates.byteArrayCost(dataLength);
		return size;
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 4 + 4 + _data.length;
	}

	@Override
	public void set(int n, int v) {
		_data[n] = (byte) v;
	}

	@Override
	public int setAndGet(int n, int v) {
		_data[n] = (byte) v;
		return _data[n] & 0xFF;
	}

	@Override
	public int size() {
		return _data.length;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(MAP_TYPE.BYTE.ordinal());
		writeBytes(out);
	}

	protected void writeBytes(DataOutput out) throws IOException {
		out.writeInt(getUnique());
		out.writeInt(_data.length);
		for(int i = 0; i < _data.length; i++)
			out.writeByte(_data[i]);
	}

	protected static MapToByte readFields(DataInput in) throws IOException {
		final int unique = in.readInt();
		final int length = in.readInt();
		final byte[] data = new byte[length];
		for(int i = 0; i < length; i++)
			data[i] = in.readByte();
		return new MapToByte(unique, data);
	}

	@Override
	public void replace(int v, int r) {
		byte cv = (byte) v;
		byte rv = (byte) r;
		for(int i = 0; i < size(); i++)
			if(_data[i] == cv)
				_data[i] = rv;
	}

	@Override
	public void copyInt(int[] d) {
		for(int i = 0; i < _data.length; i++)
			_data[i] = (byte) d[i];
	}

	@Override
	public void copyBit(BitSet d) {
		for(int i = d.nextSetBit(0); i >= 0; i = d.nextSetBit(i + 1)) {
			_data[i] = 1;
		}
	}

	@Override
	public int[] getCounts(int[] ret) {
		for(int i = 0; i < _data.length; i++)
			ret[_data[i] & 0xFF]++;
		return ret;
	}

	@Override
	protected void preAggregateDenseToRowBy8(double[] mV, double[] preAV, int cl, int cu, int off) {
		final int h = (cu - cl) % 8;
		off += cl;
		for(int rc = cl; rc < cl + h; rc++, off++)
			preAV[_data[rc] & 0xFF] += mV[off];
		for(int rc = cl + h; rc < cu; rc += 8, off += 8) {
			preAV[_data[rc] & 0xFF] += mV[off];
			preAV[_data[rc + 1] & 0xFF] += mV[off + 1];
			preAV[_data[rc + 2] & 0xFF] += mV[off + 2];
			preAV[_data[rc + 3] & 0xFF] += mV[off + 3];
			preAV[_data[rc + 4] & 0xFF] += mV[off + 4];
			preAV[_data[rc + 5] & 0xFF] += mV[off + 5];
			preAV[_data[rc + 6] & 0xFF] += mV[off + 6];
			preAV[_data[rc + 7] & 0xFF] += mV[off + 7];
		}
	}

	@Override
	public int getUpperBoundValue() {
		return 255;
	}

	@Override
	public int countRuns() {
		int c = 1;
		byte prev = _data[0];
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
		else if(unique <= 127) {
			ret = toUByte();
			ret.setUnique(unique);
			return ret;
		}
		else {
			setUnique(unique);
			return this;
		}
		ret.copy(this);
		return ret;
	}

	@Override
	public AMapToData slice(int l, int u) {
		return new MapToByte(getUnique(), Arrays.copyOfRange(_data, l, u));
	}

	@Override
	public AMapToData append(AMapToData t) {
		if(t instanceof MapToByte) {
			final MapToByte tb = (MapToByte) t;
			final byte[] tbb = tb._data;
			final int newSize = _data.length + t.size();
			final int newDistinct = Math.max(getUnique(), t.getUnique());

			// copy
			final byte[] ret = Arrays.copyOf(_data, newSize);
			System.arraycopy(tbb, 0, ret, _data.length, t.size());

			// return
			if(newDistinct < 127)
				return new MapToUByte(newDistinct, ret);
			else
				return new MapToByte(newDistinct, ret);
		}
		else {
			throw new NotImplementedException("Not implemented append on Bit map different type");
		}
	}

	@Override
	public AMapToData appendN(AMapToDataGroup[] d) {
		int p = 0; // pointer
		for(AMapToDataGroup gd : d)
			p += gd.getMapToData().size();
		final byte[] ret = Arrays.copyOf(_data, p);

		p = size();
		for(int i = 1; i < d.length; i++) {
			final MapToByte mm = (MapToByte) d[i].getMapToData();
			final int ms = mm.size();
			System.arraycopy(mm._data, 0, ret, p, ms);
			p += ms;
		}

		if(getUnique() < 127)
			return new MapToUByte(getUnique(), ret);
		else
			return new MapToByte(getUnique(), ret);
	}
}
