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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.IMapToDataGroup;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
import org.apache.sysds.runtime.frame.data.columns.BitSetArray;
import org.apache.sysds.utils.MemoryEstimates;

/**
 * A Bit map containing the values of the map inside a long array.
 * 
 * By convention inside the indexes are from the right most bit. so for instance index 0 is the left most bit in the
 * first long and index 63 is the leftmost bit in the first long. Similarly 64 is the right most bit in the second long.
 * and so on.
 * 
 */
public class MapToBit extends AMapToData {

	private static final long serialVersionUID = -8065234231282619903L;

	private final long[] _data;
	private final int _size;

	/**
	 * A Bit map containing the values of the map inside a long array.
	 * 
	 * By convention inside the indexes are from the right most bit. so for instance index 0 is the left most bit in the
	 * first long and index 63 is the leftmost bit in the first long. Similarly 64 is the right most bit in the second
	 * long. and so on.
	 * 
	 * @param size The size to allocate, as in the number of bits to enable encoding not the number of longs to allocate
	 */
	protected MapToBit(int size) {
		this(2, size);
	}

	/**
	 * A Bit map containing the values of the map inside a long array.
	 * 
	 * By convention inside the indexes are from the right most bit. so for instance index 0 is the left most bit in the
	 * first long and index 63 is the leftmost bit in the first long. Similarly 64 is the right most bit in the second
	 * long. and so on.
	 * 
	 * @param unique the number of unique values to encode ... is basically always 2 in this case
	 * @param size   The size to allocate, as in number of bits to enable encoding not the number of longs to allocate.
	 */
	public MapToBit(int unique, int size) {
		super(Math.min(unique, 2));
		_data = new long[longSize(size)];
		_size = size;
	}

	private MapToBit(int unique, BitSet d, int size) {
		this(unique, d.toLongArray(), size);
	}

	private MapToBit(int unique, long[] bsd, int size) {
		super(unique);
		if(bsd.length == longSize(size))
			_data = bsd;
		else {
			_data = new long[longSize(size)];
			System.arraycopy(bsd, 0, _data, 0, bsd.length);
		}
		_size = size;
	}

	@Override
	public MAP_TYPE getType() {
		return MapToFactory.MAP_TYPE.BIT;
	}

	@Override
	public int getIndex(int n) {
		int wIdx = n >> 6; // same as divide by 64 but faster
		return (_data[wIdx] & (1L << n)) != 0L ? 1 : 0;
	}

	@Override
	public void fill(int v) {
		final long re = (_data.length * 64) - _size;
		final boolean fillZero = v == 0;
		final long fillValue = fillZero ? 0L : -1L;
		if(re == 0 || fillZero)
			Arrays.fill(_data, fillValue);
		else {
			Arrays.fill(_data, 0, _data.length - 1, fillValue);
			_data[_data.length - 1] = -1L >>> re;
		}
	}

	@Override
	public long getInMemorySize() {
		return getInMemorySize(_size);
	}

	public static long getInMemorySize(int dataLength) {
		long size = 16 + 4; // object header + object reference + int size
		size += MemoryEstimates.longArrayCost(dataLength >> 6 + 1);
		return size;
	}

	@Override
	public void set(int n, int v) {
		int wIdx = n >> 6; // same as divide by 64 bit faster
		if(v == 1)
			_data[wIdx] |= (1L << n);
		else
			_data[wIdx] &= ~(1L << n);
	}

	@Override
	public void set(int l, int u, int off, AMapToData tm) {
		for(int i = l; i < u; i++, off++) {
			set(i, tm.getIndex(off));
		}
	}

	@Override
	public int setAndGet(int n, int v) {
		set(n, v);
		return v == 1 ? 1 : 0;
	}

	@Override
	public int size() {
		return _size;
	}

	@Override
	public void replace(int v, int r) {
		// Note that this method assume that replace is called correctly.
		if(v == 0) // set all to 1
			fill(1);
		else // set all to 0
			fill(0);
	}

	@Override
	public long getExactSizeOnDisk() {
		long size = 1 + 4 + 4; // base variables
		size += _data.length * 8;
		return size;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(MAP_TYPE.BIT.ordinal());
		out.writeInt(_size);
		out.writeInt(_data.length);
		for(int i = 0; i < _data.length; i++)
			out.writeLong(_data[i]);
	}

	protected static MapToBit readFields(DataInput in) throws IOException {
		final int size = in.readInt();
		final long[] data = new long[in.readInt()];
		for(int i = 0; i < data.length; i++)
			data[i] = in.readLong();
		return new MapToBit(2, data, size);
	}

	@Override
	public int getUpperBoundValue() {
		return 1;
	}

	@Override
	public int[] getCounts(int[] ret) {
		final int sz = size();
		for(int i = 0; i < _data.length; i++)
			ret[1] += Long.bitCount(_data[i]);
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

		JoinBitSets j = new JoinBitSets(tmb, this, _size);

		// multiply and scale with actual values
		v[1] += td[1] * j.tt;
		v[0] += td[1] * j.ft;
		v[1] += td[0] * j.tf;
		v[0] += td[0] * j.ff;
	}

	@Override
	public void preAggregateDDC_DDCMultiCol(AMapToData tm, IDictionary td, double[] v, int nCol) {
		if(tm instanceof MapToBit)
			preAggregateDDCMultiColBitBit((MapToBit) tm, td, v, nCol);
		else // fallback
			super.preAggregateDDC_DDCMultiCol(tm, td, v, nCol);
	}

	private void preAggregateDDCMultiColBitBit(MapToBit tmb, IDictionary td, double[] v, int nCol) {

		JoinBitSets j = new JoinBitSets(tmb, this, _size);

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
		for(int i = 0; i < _data.length; i++)
			if(_data[i] != 0)
				return false;
		return true;
	}

	@Override
	public void copyInt(int[] d, int start, int end) {
		for(int i = start; i < end; i++)
			set(i, d[i]);
	}

	@Override
	public void copyBit(MapToBit d) {
		long[] vals = d._data;
		System.arraycopy(vals, 0, _data, 0, Math.min(vals.length, _data.length));
	}

	/**
	 * Return the index of the next bit set to one. If no more bits are set to one return -1. The method behaves
	 * similarly to and is inspired from java's BitSet. If a negative value is given as input it fails.
	 * 
	 * @param fromIndex The index to start from (inclusive)
	 * @return The next valid index.
	 */
	public int nextSetBit(int fromIndex) {
		if(fromIndex >= _size)
			return -1;
		int u = fromIndex >> 6; // long trick instead of division by 64.
		final int s = _data.length;
		// mask out previous set bits in this word.
		long word = _data[u] & (0xffffffffffffffffL << fromIndex);

		while(true) {
			if(word != 0)
				return (u * 64) + Long.numberOfTrailingZeros(word);
			if(++u == s)
				return -1;
			word = _data[u];
		}
	}

	private static class JoinBitSets {
		int tt = 0;
		int ft = 0;
		int tf = 0;
		int ff = 0;

		protected JoinBitSets(MapToBit t_data, MapToBit o_data, int size) {

			// This naively rely on JDK implementation using long arrays to encode bit Arrays.
			final long[] t_longs = t_data._data;
			final long[] _longs = o_data._data;

			if(t_longs.length != _longs.length)
				throw new RuntimeException("Invalid to join bit sets not same length");

			for(int i = 0; i < _longs.length; i++) {
				long t = t_longs[i];
				long v = _longs[i];
				tt += Long.bitCount(t & v);
				ft += Long.bitCount(t & ~v);
				tf += Long.bitCount(~t & v);
				ff += Long.bitCount(~t & ~v);
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
			long l = _data[0];
			if(_size != 64 && getIndex(_size - 1) == 1) {
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
			final long[] _longs = _data;
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
			l = (_size % 64 != 0 && getIndex(_size - 1) == 1) ? _longs[idx] |
				(~((-1L) ^ ((-1L) << (_size - 64)))) : _longs[idx];
			shift1 = (l << 1) | ((_longs[idx - 1] & lastMask) >>> 63);

			c += Long.bitCount(l ^ shift1);

			return c;
		}
	}

	@Override
	public AMapToData slice(int l, int u) {
		long[] s = BitSetArray.sliceVectorized(_data, l, u);
		MapToBit m = new MapToBit(getUnique(), s, u - l);

		if(m.isEmpty())
			return new MapToZero(u - l);
		else
			return m;
	}

	@Override
	public AMapToData append(AMapToData t) {
		if(t instanceof MapToBit) {
			MapToBit tb = (MapToBit) t;
			final int newSize = _size + t.size();
			long[] ret = new long[longSize(newSize)];
			System.arraycopy(_data, 0, ret, 0, _data.length);
			BitSetArray.setVectorizedLongs(_size, newSize, ret, tb._data);
			return new MapToBit(2, ret, newSize);
		}
		else {
			throw new NotImplementedException("Not implemented append on Bit map different type");
		}
	}

	@Override
	public boolean equals(AMapToData e) {
		return e instanceof MapToBit && //
			e.getUnique() == getUnique() && //
			((MapToBit) e)._size == _size && //
			Arrays.equals(((MapToBit) e)._data, _data);
	}

	@Override
	public AMapToData appendN(IMapToDataGroup[] d) {
		int p = 0; // pointer
		for(IMapToDataGroup gd : d)
			p += gd.getMapToData().size();
		final long[] ret = new long[longSize(p)];

		long[] or = null;

		p = 0;
		for(int i = 0; i < d.length; i++) {
			if(d[i].getMapToData().size() > 0) {
				final MapToBit mm = (MapToBit) d[i].getMapToData();
				final int ms = mm.size();
				or = mm._data;
				BitSetArray.setVectorizedLongs(p, p + ms, ret, or);
				p += ms;
			}
		}

		BitSet retBS = BitSet.valueOf(ret);
		return new MapToBit(getUnique(), retBS, p);

	}

	private static int longSize(int size) {
		return Math.max(size >> 6, 0) + 1;
	}

	@Override
	public void decompressToRangeOff(double[] c, int rl, int ru, int offR, double[] values) {
		for(int i = rl, offT = rl + offR; i < ru; i++, offT++)
			c[offT] += values[getIndex(i)];
	}

	@Override
	public void decompressToRangeNoOff(double[] c, int rl, int ru, double[] values) {
		for(int i = rl; i < ru; i++)
			c[i] += values[getIndex(i)];
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(" size: " + _size);
		sb.append(" longLength:[");
		sb.append(_data.length);
		sb.append("]");
		return sb.toString();

	}
}
