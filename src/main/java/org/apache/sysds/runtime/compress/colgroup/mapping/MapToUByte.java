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

	public MapToUByte(int unique, int size) {
		super(Math.min(unique, 127), new byte[size]);
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

	public static long getInMemorySize(int dataLength) {
		return MapToByte.getInMemorySize(dataLength);
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

	// @Override
	// public void copy(AMapToData d) {
	// if(d instanceof MapToChar) {
	// char[] dd = ((MapToChar) d).getChars();
	// for(int i = 0; i < size(); i++)
	// _data[i] = (byte) (dd[i] % 128);
	// }
	// else
	// for(int i = 0; i < size(); i++)
	// set(i, d.getIndex(i) % 128);
	// }

	@Override
	protected void preAggregateDenseToRowBy8(double[] mV, int off, double[] preAV, int cl, int cu) {
		final int h = (cu - cl) % 8;
		off += cl;
		for(int rc = cl; rc < cl + h; rc++, off++)
			preAV[_data[rc]] += mV[off];
		for(int rc = cl + h; rc < cu; rc += 8, off += 8) {
			int id1 = _data[rc], id2 = _data[rc + 1], id3 = _data[rc + 2], id4 = _data[rc + 3], id5 = _data[rc + 4],
				id6 = _data[rc + 5], id7 = _data[rc + 6], id8 = _data[rc + 7];
			preAV[id1] += mV[off];
			preAV[id2] += mV[off + 1];
			preAV[id3] += mV[off + 2];
			preAV[id4] += mV[off + 3];
			preAV[id5] += mV[off + 4];
			preAV[id6] += mV[off + 5];
			preAV[id7] += mV[off + 6];
			preAV[id8] += mV[off + 7];
		}
	}

	@Override
	public int getUpperBoundValue() {
		return 127;
	}
}
