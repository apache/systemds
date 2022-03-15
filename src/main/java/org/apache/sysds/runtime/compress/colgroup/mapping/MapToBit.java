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

import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.utils.MemoryEstimates;

public class MapToBit extends AMapToData {

	private static final long serialVersionUID = -8065234231282619923L;

	private final BitSet _data;
	private final int _size;

	protected MapToBit(int size){
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
	public long getExactSizeOnDisk() {
		final int dSize = _data.size();
		long size = 1 + 4 + 4 + 4; // base variables
		size += (dSize / 64) * 8; // all longs except last
		// size += (dSize % 64 == 0 ? 0 : 8); // last long
		return size;
	}

	@Override
	public void set(int n, int v) {
		_data.set(n, v == 1);
	}

	@Override
	public int setAndGet(int n, int v){
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
	public void write(DataOutput out) throws IOException {
		long[] internals = _data.toLongArray();
		out.writeByte(MAP_TYPE.BIT.ordinal());
		out.writeInt(getUnique());
		out.writeInt(_size);
		out.writeInt(internals.length);
		for(int i = 0; i < internals.length; i++)
			out.writeLong(internals[i]);
	}

	protected static MapToBit readFields(DataInput in) throws IOException {
		int unique = in.readInt();
		int size = in.readInt();
		long[] internalLong = new long[in.readInt()];
		for(int i = 0; i < internalLong.length; i++)
			internalLong[i] = in.readLong();

		return new MapToBit(unique, BitSet.valueOf(internalLong), size);
	}

	@Override
	public void preAggregateDense(MatrixBlock m, double[] preAV, int rl, int ru, int cl, int cu, AOffset indexes) {
		indexes.preAggregateDenseMap(m, preAV, rl, ru, cl, cu, getUnique(), _data);
	}

	@Override
	public void preAggregateSparse(SparseBlock sb, double[] preAV, int rl, int ru, AOffset indexes) {
		indexes.preAggregateSparseMap(sb, preAV, rl, ru, getUnique(), _data);
	}

	@Override
	public int getUpperBoundValue() {
		return 1;
	}

	@Override
	public int[] getCounts(int[] counts) {
		final int sz = size();

		if(counts.length == 1)
			counts[0] = sz;
		else {
			counts[1] = _data.cardinality();
			counts[0] = sz - counts[1];
		}

		return counts;
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

	public boolean isEmpty(){
		return _data.isEmpty();
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
}
