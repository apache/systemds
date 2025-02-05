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

import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;

public class MapToUByte extends MapToByte {

	private static final long serialVersionUID = -2498505439667351828L;

	protected MapToUByte(int size) {
		this(128, size);
	}

	public MapToUByte(int unique, int size) {
		super(Math.min(unique, 128), new byte[size]);
	}

	protected MapToUByte(int unique, byte[] data) {
		super(unique, data);
	}

	@Override
	public MAP_TYPE getType() {
		return MapToFactory.MAP_TYPE.UBYTE;
	}

	@Override
	public int getIndex(int n) {
		return _data[n];
	}

	@Override
	public int setAndGet(int n, int v) {
		return _data[n] = (byte) v;
	}

	@Override
	public void fill(int v) {
		Arrays.fill(_data, (byte) (v % 128));
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(MAP_TYPE.UBYTE.ordinal());
		super.writeBytes(out);
	}

	protected static MapToUByte readFields(DataInput in) throws IOException {
		int unique = in.readInt();
		final int length = in.readInt();
		final byte[] data = new byte[length];
		for(int i = 0; i < length; i++)
			data[i] = in.readByte();
		return new MapToUByte(unique, data);
	}

	@Override
	public void replace(int v, int r) {
		byte cv = (byte) v;
		byte rv = (byte) (r % 128);
		for(int i = 0; i < size(); i++)
			if(_data[i] == cv)
				_data[i] = rv;
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
		return 127;
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
	protected void decompressToRangeNoOffBy8(double[] c, int r, double[] values) {
		c[r] += values[_data[r]];
		c[r + 1] += values[_data[r + 1]];
		c[r + 2] += values[_data[r + 2]];
		c[r + 3] += values[_data[r + 3]];
		c[r + 4] += values[_data[r + 4]];
		c[r + 5] += values[_data[r + 5]];
		c[r + 6] += values[_data[r + 6]];
		c[r + 7] += values[_data[r + 7]];
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
	public AMapToData resize(int unique) {
		final int size = _data.length;
		if(unique <= 1)
			return new MapToZero(size);
		else if(unique == 2 && size > 32) {
			AMapToData ret = new MapToBit(unique, size);
			ret.copy(this);
			return ret;
		}
		else {
			setUnique(unique);
			return this;
		}
	}
}
