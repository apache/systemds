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
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.utils.MemoryEstimates;

public class MapToByte extends AMapToData {

	private static final long serialVersionUID = -2498505439667351828L;

	private final byte[] _data;

	public MapToByte(int unique, int size) {
		super(unique);
		_data = new byte[size];
	}

	private MapToByte(int unique, byte[] data) {
		super(unique);
		_data = data;
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
	public int size() {
		return _data.length;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(MAP_TYPE.BYTE.ordinal());
		out.writeInt(getUnique());
		out.writeInt(_data.length);
		for(int i = 0; i < _data.length; i++)
			out.writeByte(_data[i]);
	}

	public static MapToByte readFields(DataInput in) throws IOException {
		int unique = in.readInt();
		final int length = in.readInt();
		final byte[] data = new byte[length];
		for(int i = 0; i < length; i++)
			data[i] = in.readByte();
		return new MapToByte(unique, data);
	}

	public byte[] getBytes() {
		return _data;
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
		else {
			for(int i = 0; i < size(); i++)
				set(i, d.getIndex(i));
		}
	}

	@Override
	public void preAggregateDense(MatrixBlock m, MatrixBlock pre, int rl, int ru, int cl, int cu) {
		final int nRow = m.getNumColumns();
		final int nVal = pre.getNumColumns();
		final double[] preAV = pre.getDenseBlockValues();
		final double[] mV = m.getDenseBlockValues();
		final int blockSize = 4000;
		for(int block = cl; block < cu; block += blockSize) {
			final int blockEnd = Math.min(block + blockSize, nRow);
			for(int rowLeft = rl, offOut = 0; rowLeft < ru; rowLeft++, offOut += nVal) {
				final int offLeft = rowLeft * nRow;
				for(int rc = block; rc < blockEnd; rc++) {
					final int idx = _data[rc] & 0xFF;
					preAV[offOut + idx] += mV[offLeft + rc];
				}
			}
		}
	}

	@Override
	public int getUpperBoundValue() {
		return 255;
	}
}
