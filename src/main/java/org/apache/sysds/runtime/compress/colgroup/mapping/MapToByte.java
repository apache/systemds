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
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
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
		verify();
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
	public void set(int l, int u, int off, AMapToData tm) {
		if(tm instanceof MapToByte) {
			MapToByte tbm = (MapToByte) tm;
			byte[] tbv = tbm._data;
			for(int i = l; i < u; i++, off++) {
				_data[i] = tbv[off];
			}
		}
		else {
			for(int i = l; i < u; i++, off++) {
				_data[i] = (byte) tm.getIndex(off);
			}
		}
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
		out.write(_data);
	}

	protected static MapToByte readFields(DataInput in) throws IOException {
		final int unique = in.readInt();
		final int length = in.readInt();
		final byte[] data = new byte[length];
		in.readFully(data);
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
	public void copyInt(int[] d, int start, int end) {
		for(int i = start; i < end; i++)
			_data[i] = (byte) d[i];
	}

	@Override
	public void copyBit(MapToBit d) {
		fill(0);
		for(int i = d.nextSetBit(0); i >= 0; i = d.nextSetBit(i + 1)) {
			_data[i] = 1;
		}
	}

	@Override
	public int[] getCounts(int[] ret) {
		final int h = (_data.length) % 8;
		for(int i = 0; i < h; i++)
			ret[_data[i] & 0xFF]++;
		getCountsBy8P(ret, h, _data.length);
		return ret;
	}

	private void getCountsBy8P(int[] ret, int s, int e) {
		for(int i = s; i < e; i += 8) {
			ret[_data[i] & 0xFF]++;
			ret[_data[i + 1] & 0xFF]++;
			ret[_data[i + 2] & 0xFF]++;
			ret[_data[i + 3] & 0xFF]++;
			ret[_data[i + 4] & 0xFF]++;
			ret[_data[i + 5] & 0xFF]++;
			ret[_data[i + 6] & 0xFF]++;
			ret[_data[i + 7] & 0xFF]++;
		}
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
		else if(unique <= 128) {
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
			if(newDistinct < 128)
				return new MapToUByte(newDistinct, ret);
			else
				return new MapToByte(newDistinct, ret);
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
		final byte[] ret = new byte[p];

		p = 0;
		for(int i = 0; i < d.length; i++) {
			if(d[i].getMapToData().size() > 0) {
				final MapToByte mm = (MapToByte) d[i].getMapToData();
				final int ms = mm.size();
				System.arraycopy(mm._data, 0, ret, p, ms);
				p += ms;
			}
		}

		if(getUnique() < 128)
			return new MapToUByte(getUnique(), ret);
		else
			return new MapToByte(getUnique(), ret);
	}

	@Override
	public void lmSparseMatrixRow(SparseBlock sb, final int r, DenseBlock db, final IColIndex colIndexes,
		final IDictionary dict) {

		if(sb.isEmpty(r))
			return;
		// dense output blocks locations
		final int pos = db.pos(r);
		final double[] retV = db.values(r);

		// sparse left block locations
		final int apos = sb.pos(r);
		final int alen = sb.size(r) + apos;
		final int[] aix = sb.indexes(r);
		final double[] aval = sb.values(r);

		for(int i = apos; i < alen; i++)
			dict.multiplyScalar(aval[i], retV, pos, getIndex(aix[i]), colIndexes);
	}

	@Override
	public boolean equals(AMapToData e) {
		return e instanceof MapToByte && //
			e.getUnique() == getUnique() && //
			Arrays.equals(((MapToByte) e)._data, _data);
	}

	@Override
	protected void decompressToRangeNoOffBy8(double[] c, int r, double[] values) {
		c[r] += values[getIndex(r)];
		c[r + 1] += values[getIndex(r + 1)];
		c[r + 2] += values[getIndex(r + 2)];
		c[r + 3] += values[getIndex(r + 3)];
		c[r + 4] += values[getIndex(r + 4)];
		c[r + 5] += values[getIndex(r + 5)];
		c[r + 6] += values[getIndex(r + 6)];
		c[r + 7] += values[getIndex(r + 7)];
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
	protected void decompressToRangeOff(double[] c, int rl, int ru, int offR, double[] values) {
		for(int i = rl, offT = rl + offR; i < ru; i++, offT++)
			c[offT] += values[getIndex(i)];
	}

	@Override
	protected void decompressToRangeNoOff(double[] c, int rl, int ru, double[] values) {
		// OVERWRITTEN FOR JIT COMPILE!
		final int h = (ru - rl) % 8;
		for(int rc = rl; rc < rl + h; rc++)
			c[rc] += values[getIndex(rc)];
		for(int rc = rl + h; rc < ru; rc += 8)
			decompressToRangeNoOffBy8(c, rc, values);
	}

}
