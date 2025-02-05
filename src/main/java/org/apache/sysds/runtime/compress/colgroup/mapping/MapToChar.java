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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.IMapToDataGroup;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.utils.MemoryEstimates;

public class MapToChar extends AMapToData {

	private static final long serialVersionUID = 6315708056775476541L;

	private final char[] _data;

	protected MapToChar(int size) {
		this(Character.MAX_VALUE + 1, size);
	}

	public MapToChar(int unique, int size) {
		super(Math.min(unique, Character.MAX_VALUE + 1));
		_data = new char[size];
	}

	public MapToChar(int unique, char[] data) {
		super(unique);
		_data = data;
		verify();
	}

	@Override
	public MAP_TYPE getType() {
		return MapToFactory.MAP_TYPE.CHAR;
	}

	@Override
	public int getIndex(int n) {
		return _data[n];
	}

	@Override
	public void fill(int v) {
		Arrays.fill(_data, (char) v);
	}

	@Override
	public long getInMemorySize() {
		return getInMemorySize(_data.length);
	}

	public static long getInMemorySize(int dataLength) {
		long size = 16 + 8; // object header + object reference
		size += MemoryEstimates.charArrayCost(dataLength);
		return size;
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 4 + 4 + _data.length * 2;
	}

	@Override
	public void set(int n, int v) {
		_data[n] = (char) v;
	}

	@Override
	public void set(int l, int u, int off, AMapToData tm) {
		if(tm instanceof MapToChar) {
			MapToChar tbm = (MapToChar) tm;
			char[] tbv = tbm._data;
			for(int i = l; i < u; i++, off++) {
				_data[i] = tbv[off];
			}
		}
		else {
			for(int i = l; i < u; i++, off++) {
				set(i, tm.getIndex(off));
			}
		}
	}

	@Override
	public int setAndGet(int n, int v) {
		return _data[n] = (char) v;
	}

	@Override
	public int size() {
		return _data.length;
	}

	@Override
	public void replace(int v, int r) {
		char cv = (char) v;
		char rv = (char) r;
		for(int i = 0; i < size(); i++)
			if(_data[i] == cv)
				_data[i] = rv;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(MAP_TYPE.CHAR.ordinal());
		out.writeInt(getUnique());
		out.writeInt(_data.length);
		writeChars(out, _data);

	}

	protected static void writeChars(DataOutput out, char[] _data_c) throws IOException {
		final int BS = 100;
		if(_data_c.length > BS) {
			final byte[] buff = new byte[BS * 2];
			for(int i = 0; i < _data_c.length;) {
				i = writeCharsBlock(out, _data_c, BS, buff, i);
			}
		}
		else {
			for(int i = 0; i < _data_c.length; i++)
				out.writeChar(_data_c[i]);
		}
	}

	private static int writeCharsBlock(DataOutput out, char[] _data_c, final int BS, final byte[] buff, int i)
		throws IOException {
		if(i + BS <= _data_c.length) {
			for(int o = 0; o < BS; o++) {
				IOUtilFunctions.shortToBa(_data_c[i++], buff, o * 2);
			}
			out.write(buff);
		}
		else {// remaining.
			for(; i < _data_c.length; i++)
				out.writeChar(_data_c[i]);
		}
		return i;
	}

	protected static MapToChar readFields(DataInput in) throws IOException {
		int unique = in.readInt();
		final int length = in.readInt();
		final char[] data = new char[length];
		for(int i = 0; i < length; i++)
			data[i] = (char) in.readUnsignedShort();
		return new MapToChar(unique, data);
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
		return Character.MAX_VALUE;
	}

	@Override
	public void copyInt(int[] d, int start, int end) {
		for(int i = start; i < end; i++)
			_data[i] = (char) d[i];
	}

	@Override
	public int[] getCounts(int[] ret) {
		final int h = (_data.length) % 8;
		for(int i = 0; i < h; i++)
			ret[getIndex(i)]++;
		getCountsBy8P(ret, h, _data.length);
		return ret;
	}

	private void getCountsBy8P(int[] ret, int s, int e) {
		for(int i = s; i < e; i += 8) {
			ret[getIndex(i)]++;
			ret[getIndex(i + 1)]++;
			ret[getIndex(i + 2)]++;
			ret[getIndex(i + 3)]++;
			ret[getIndex(i + 4)]++;
			ret[getIndex(i + 5)]++;
			ret[getIndex(i + 6)]++;
			ret[getIndex(i + 7)]++;
		}
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
		else {
			setUnique(unique);
			return this;
		}
		ret.copy(this);
		return ret;
	}

	@Override
	public int countRuns() {
		int c = 1;
		char prev = _data[0];
		for(int i = 1; i < _data.length; i++) {
			c += prev == _data[i] ? 0 : 1;
			prev = _data[i];
		}
		return c;
	}

	@Override
	public AMapToData slice(int l, int u) {
		return new MapToChar(getUnique(), Arrays.copyOfRange(_data, l, u));
	}

	@Override
	public AMapToData append(AMapToData t) {
		if(t instanceof MapToChar) {
			MapToChar tb = (MapToChar) t;
			char[] tbb = tb._data;
			final int newSize = _data.length + t.size();
			final int newDistinct = Math.max(getUnique(), t.getUnique());

			// copy
			char[] ret = Arrays.copyOf(_data, newSize);
			System.arraycopy(tbb, 0, ret, _data.length, t.size());

			return new MapToChar(newDistinct, ret);
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
		final char[] ret = new char[p];

		p = 0;
		for(int i = 0; i < d.length; i++) {
			if(d[i].getMapToData().size() > 0) {
				final MapToChar mm = (MapToChar) d[i].getMapToData();
				final int ms = mm.size();
				System.arraycopy(mm._data, 0, ret, p, ms);
				p += ms;
			}
		}

		return new MapToChar(getUnique(), ret);
	}

	@Override
	public boolean equals(AMapToData e) {
		return e instanceof MapToChar && //
			e.getUnique() == getUnique() && //
			Arrays.equals(((MapToChar) e)._data, _data);
	}

	@Override
	protected void preAggregateDDC_DDCSingleCol_vec(AMapToData tm, double[] td, double[] v, int r) {
		if(tm instanceof MapToChar)
			preAggregateDDC_DDCSingleCol_vecChar((MapToChar) tm, td, v, r);
		else
			super.preAggregateDDC_DDCSingleCol_vec(tm, td, v, r);
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

	protected final void preAggregateDDC_DDCSingleCol_vecChar(MapToChar tm, double[] td, double[] v, int r) {
		final int r2 = r + 1, r3 = r + 2, r4 = r + 3, r5 = r + 4, r6 = r + 5, r7 = r + 6, r8 = r + 7;
		v[getIndex(r)] += td[tm.getIndex(r)];
		v[getIndex(r2)] += td[tm.getIndex(r2)];
		v[getIndex(r3)] += td[tm.getIndex(r3)];
		v[getIndex(r4)] += td[tm.getIndex(r4)];
		v[getIndex(r5)] += td[tm.getIndex(r5)];
		v[getIndex(r6)] += td[tm.getIndex(r6)];
		v[getIndex(r7)] += td[tm.getIndex(r7)];
		v[getIndex(r8)] += td[tm.getIndex(r8)];
	}

	@Override
	public AMapToData[] splitReshapeDDCPushDown(final int multiplier, final ExecutorService pool) throws Exception {
		final int s = size();
		final MapToChar[] ret = new MapToChar[multiplier];
		final int eachSize = s / multiplier;
		for(int i = 0; i < multiplier; i++)
			ret[i] = new MapToChar(getUnique(), eachSize);

		final int blkz = Math.max(eachSize / 8, 2048) * multiplier;
		List<Future<?>> tasks = new ArrayList<>();
		for(int i = 0; i < s; i += blkz) {
			final int start = i;
			final int end = Math.min(i + blkz, s);
			tasks.add(pool.submit(() -> splitReshapeDDCBlock(ret, multiplier, start, end)));
		}

		for(Future<?> t : tasks)
			t.get();

		return ret;
	}

	private void splitReshapeDDCBlock(final MapToChar[] ret, final int multiplier, final int start, final int end) {
		for(int i = start; i < end; i += multiplier)
			splitReshapeDDCRow(ret, multiplier, i);
	}

	private void splitReshapeDDCRow(final MapToChar[] ret, final int multiplier, final int i) {
		final int off = i / multiplier;
		final int end = i + multiplier;
		for(int j = i; j < end; j++)
			ret[j % multiplier]._data[off] = _data[j];
	}

}
