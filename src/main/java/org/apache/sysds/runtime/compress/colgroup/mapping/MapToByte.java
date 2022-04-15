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
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
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
		int unique = in.readInt();
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
	public void copy(AMapToData d) {
		if(d instanceof MapToChar) {
			char[] dd = ((MapToChar) d).getChars();
			for(int i = 0; i < size(); i++)
				_data[i] = (byte) dd[i];
		}
		else
			super.copy(d);
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
	public final void preAggregateDense(MatrixBlock m, double[] preAV, int rl, int ru, int cl, int cu, AOffset indexes) {
		indexes.preAggregateDenseMap(m, preAV, rl, ru, cl, cu, getUnique(), _data);
	}

	@Override
	public void preAggregateSparse(SparseBlock sb, double[] preAV, int rl, int ru, AOffset indexes) {
		indexes.preAggregateSparseMap(sb, preAV, rl, ru, getUnique(), _data);
	}

	@Override
	public int getUpperBoundValue() {
		return 255;
	}
}
