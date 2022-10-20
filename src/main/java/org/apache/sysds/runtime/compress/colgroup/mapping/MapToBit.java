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
import java.util.BitSet;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.AMapToDataGroup;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
import org.apache.sysds.utils.MemoryEstimates;

public class MapToBit extends AMapToData {

	private static final long serialVersionUID = -8065234231282619923L;

	private final BitSet _data;
	private final int _size;

	protected MapToBit(int size) {
		this(2, size);
	}

	public MapToBit(int unique, int size) {
		super(Math.min(unique, 2));
		_data = new BitSet(size);
		_size = size;
	}

	private MapToBit(int unique, BitSet d, int size) {
		super(unique);
		_data = d;
		_size = size;
		if(_data.isEmpty()) {
			unique = 1;
			LOG.warn("Empty bit set should not happen");
		}
	}

	protected BitSet getData() {
		return _data;
	}

	@Override
	public MAP_TYPE getType() {
		return MapToFactory.MAP_TYPE.BIT;
	}

	@Override
	public int getIndex(int n) {
		return _data.get(n) ? 1 : 0;
	}

	@Override
	public void fill(int v) {
		_data.set(0, _size, true);
	}

	@Override
	public long getInMemorySize() {
		return getInMemorySize(_data.size());
	}

	public static long getInMemorySize(int dataLength) {
		long size = 16 + 8 + 4; // object header + object reference + int size
		size += MemoryEstimates.bitSetCost(dataLength);
		return size;
	}

	@Override
	public void set(int n, int v) {
		_data.set(n, v == 1);
	}

	@Override
	public int setAndGet(int n, int v) {
		_data.set(n, v == 1);
		return 1;
	}

	@Override
	public int size() {
		return _size;
	}

	@Override
	public void replace(int v, int r) {
		// Note that this method assume that replace is called correctly.
		if(v == 0) // set all to 1
			_data.set(0, size(), true);
		else // set all to 0
			_data.clear();
	}

	@Override
	public long getExactSizeOnDisk() {
		long size = 1 + 4 + 4; // base variables
		size += _data.toLongArray().length * 8;
		return size;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		long[] internals = _data.toLongArray();
		out.writeByte(MAP_TYPE.BIT.ordinal());
		out.writeInt(_size);
		out.writeInt(internals.length);
		for(int i = 0; i < internals.length; i++)
			out.writeLong(internals[i]);
	}

	protected static MapToBit readFields(DataInput in) throws IOException {
		int size = in.readInt();
		long[] internalLong = new long[in.readInt()];
		for(int i = 0; i < internalLong.length; i++)
			internalLong[i] = in.readLong();
		BitSet ret = BitSet.valueOf(internalLong);
		return new MapToBit(2, ret, size);
	}

	@Override
	public int getUpperBoundValue() {
		return 1;
	}

	@Override
	public int[] getCounts(int[] ret) {
		final int sz = size();
		ret[1] = _data.cardinality();
		ret[0] = sz - ret[1];
		return ret;
	}

	@Override
	public void preAggregateDDC_DDCSingleCol(AMapToData tm, double[] td, double[] v) {
		if(tm instanceof MapToBit)
			preAggregateDDCSingleColBitBit((MapToBit) tm, td, v);
		else // fallback
			super.preAggregateDDC_DDCSingleCol(tm, td, v);
	}

	private void preAggregateDDCSingleColBitBit(MapToBit tmb, double[] td, double[] v) {

		JoinBitSets j = new JoinBitSets(tmb._data, _data, _size);

		// multiply and scale with actual values
		v[1] += td[1] * j.tt;
		v[0] += td[1] * j.ft;
		v[1] += td[0] * j.tf;
		v[0] += td[0] * j.ff;
	}

	@Override
	public void preAggregateDDC_DDCMultiCol(AMapToData tm, ADictionary td, double[] v, int nCol) {
		if(tm instanceof MapToBit)
			preAggregateDDCMultiColBitBit((MapToBit) tm, td, v, nCol);
		else // fallback
			super.preAggregateDDC_DDCMultiCol(tm, td, v, nCol);
	}

	private void preAggregateDDCMultiColBitBit(MapToBit tmb, ADictionary td, double[] v, int nCol) {

		JoinBitSets j = new JoinBitSets(tmb._data, _data, _size);

		final double[] tv = td.getValues();

		// multiply and scale with actual values
		for(int i = 0; i < nCol; i++) {
			final int off = nCol + i;
			v[i] += tv[i] * j.ff;
			v[off] += tv[i] * j.tf;
			v[off] += tv[off] * j.tt;
			v[i] += tv[off] * j.ft;
		}
	}

	public boolean isEmpty() {
		return _data.isEmpty();
	}

	@Override
	public void copy(AMapToData d) {
		if(d instanceof MapToBit)
			copyBit((MapToBit) d);
		else if(d instanceof MapToInt)
			copyInt((MapToInt) d);
		else {
			final int sz = size();
			for(int i = 0; i < sz; i++)
				if(d.getIndex(i) != 0)
					_data.set(i);
		}
	}

	@Override
	public void copyInt(int[] d) {
		// start from end because bitset is allocating based on last bit set.
		for(int i = d.length - 1; i > -1; i--)
			if(d[i] != 0)
				_data.set(i);
	}

	@Override
	public void copyBit(BitSet d) {
		_data.clear();
		_data.or(d);
	}

	private static class JoinBitSets {
		int tt = 0;
		int ft = 0;
		int tf = 0;
		int ff = 0;

		protected JoinBitSets(BitSet t_data, BitSet o_data, int size) {

			// This naively rely on JDK implementation using long arrays to encode bit Arrays.
			final long[] t_longs = t_data.toLongArray();
			final long[] _longs = o_data.toLongArray();

			final int common = Math.min(t_longs.length, _longs.length);

			for(int i = 0; i < common; i++) {
				long t = t_longs[i];
				long v = _longs[i];
				tt += Long.bitCount(t & v);
				ft += Long.bitCount(t & ~v);
				tf += Long.bitCount(~t & v);
				ff += Long.bitCount(~t & ~v);
			}

			if(t_longs.length > common) {
				for(int i = common; i < t_longs.length; i++) {
					int v = Long.bitCount(t_longs[i]);
					ft += v;
					ff += 64 - v;
				}
			}
			else if(_longs.length > common) {
				for(int i = common; i < _longs.length; i++) {
					int v = Long.bitCount(_longs[i]);
					tf += v;
					ff += 64 - v;
				}
			}

			final int longest = Math.max(t_longs.length, _longs.length);
			ff += size - (longest * 64); // remainder
		}
	}

	@Override
	public AMapToData resize(int unique) {
		if(unique <= 1)
			return new MapToZero(size());
		else
			return this;
	}

	@Override
	public int countRuns() {
		if(_size <= 64) {
			long l = _data.toLongArray()[0];
			if(_size != 64 && _data.get(_size - 1)) {
				// make last run go over into the last elements.
				long mask = ~((-1L) ^ ((-1L) << (_size - 64)));
				l |= mask;
			}
			long shift1 = (l << 1) | (l & 1L);
			long j = l ^ shift1;
			return 1 + Long.bitCount(j);
		}
		else {
			// at least two locations
			final long[] _longs = _data.toLongArray();
			final long lastMask = (-1L) << (63);

			// first
			long l = _longs[0];
			// set least significant bit to same as previous.
			long shift1 = (l << 1) | (l & 1L);
			long j = l ^ shift1;
			int c = 1 + Long.bitCount(j);

			// middle ones
			for(int i = 1; i < _longs.length - 1; i++) {
				// we move over the most significant bit from last value
				shift1 = (_longs[i] << 1) | ((_longs[i - 1] & lastMask) >>> 63);
				c += Long.bitCount(_longs[i] ^ shift1);
			}

			// last
			final int idx = _longs.length - 1;
			// handle if we are not aligned with 64.
			l = (_size % 64 != 0 && _data.get(_size - 1)) ? _longs[idx] |
				(~((-1L) ^ ((-1L) << (_size - 64)))) : _longs[idx];
			shift1 = (l << 1) | ((_longs[idx - 1] & lastMask) >>> 63);

			c += Long.bitCount(l ^ shift1);

			return c;
		}
	}

	@Override
	public AMapToData slice(int l, int u) {
		return new MapToBit(getUnique(), _data.get(l, u), u - l);
	}

	@Override
	public AMapToData append(AMapToData t) {
		if(t instanceof MapToBit) {
			MapToBit tb = (MapToBit) t;
			BitSet tbb = tb._data;
			final int newSize = _size + t.size();
			BitSet ret = new BitSet(newSize);
			ret.xor(_data);

			tbb.stream().forEach(x -> ret.set(x + _size, true));
			return new MapToBit(2, ret, newSize);
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
		final long[] ret = new long[(p - 1) / 64 + 1];
		long[] or = _data.toLongArray();
		System.arraycopy(or, 0, ret, 0, or.length);

		p = size();
		for(int i = 1; i < d.length; i++) {
			final MapToBit mm = (MapToBit) d[i].getMapToData();
			final int ms = mm.size();
			or = mm._data.toLongArray();
			final int remainder = p % 64;
			int retLp = p / 64;
			if(remainder == 0)// Easy lining up
				System.arraycopy(or, 0, ret, retLp, or.length);
			else { // Not Lining up
				// all but last
				for(int j = 0; j < or.length - 1; j++) {
					long v = or[j];
					ret[retLp] = ret[retLp] ^ (v << remainder);
					retLp++;
					ret[retLp] = v >>> (64 - remainder);
				}
				// last
				long v = or[or.length - 1];
				ret[retLp] = ret[retLp] ^ (v << remainder);
				retLp++;
				if(retLp < ret.length)
					ret[retLp] = v >>> (64 - remainder);
			}
			p += ms;
		}

		BitSet retBS = BitSet.valueOf(ret);
		return new MapToBit(getUnique(), retBS, p);

	}
}
