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

import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
import org.apache.sysds.utils.MemoryEstimates;

public class MapToCharPByte extends AMapToData {

	private static final long serialVersionUID = 6315708056775476541L;

	// 8323073
	public static final int max = 0xFFFF * 127;
	private final char[] _data_c;
	private final byte[] _data_b; // next byte after the char

	protected MapToCharPByte(int size) {
		this(Character.MAX_VALUE, size);
	}

	public MapToCharPByte(int unique, int size) {
		super(Math.min(unique, max));
		_data_c = new char[size];
		_data_b = new byte[size];
	}

	public MapToCharPByte(int unique, char[] data_c, byte[] data_b) {
		super(unique);
		_data_c = data_c;
		_data_b = data_b;
	}

	@Override
	public MAP_TYPE getType() {
		return MapToFactory.MAP_TYPE.CHAR_BYTE;
	}

	@Override
	public int getIndex(int n) {
		return _data_c[n] + ((int) _data_b[n] << 16);
	}

	@Override
	public void fill(int v) {
		int m = v & 0xffffff;
		Arrays.fill(_data_c, (char) m);
		Arrays.fill(_data_b, (byte) (m >> 16));
	}

	@Override
	public long getInMemorySize() {
		return getInMemorySize(_data_c.length);
	}

	public static long getInMemorySize(int dataLength) {
		long size = 16 + 8 + 8; // object header + object reference
		size += MemoryEstimates.charArrayCost(dataLength);
		size += MemoryEstimates.byteArrayCost(dataLength);
		return size;
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 4 + 4 + _data_c.length * 3;
	}

	@Override
	public void set(int n, int v) {
		int m = v & 0xffffff;
		_data_c[n] = (char) m;
		_data_b[n] = (byte) (m >> 16);
	}

	@Override
	public int setAndGet(int n, int v) {
		int m = v & 0xffffff;
		_data_c[n] = (char) m;
		_data_b[n] = (byte) (m >> 16);
		return m;
	}

	@Override
	public int size() {
		return _data_c.length;
	}

	@Override
	public void replace(int v, int r) {
		int m = v & 0xffffff;
		int mr = r & 0xffffff;
		char c = (char) m;
		char cr = (char) mr;
		byte b = (byte) (m >> 16);
		byte br = (byte) (mr >> 16);

		for(int i = 0; i < _data_c.length; i++)
			if(_data_b[i] == b && _data_c[i] == c) {
				_data_b[i] = br;
				_data_c[i] = cr;
			}
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(MAP_TYPE.CHAR_BYTE.ordinal());
		out.writeInt(getUnique());
		out.writeInt(_data_c.length);
		for(int i = 0; i < _data_c.length; i++)
			out.writeChar(_data_c[i]);
		for(int i = 0; i < _data_c.length; i++)
			out.writeByte(_data_b[i]);
	}

	protected static MapToCharPByte readFields(DataInput in) throws IOException {
		int unique = in.readInt();
		final int length = in.readInt();
		final char[] data_c = new char[length];
		for(int i = 0; i < length; i++)
			data_c[i] = in.readChar();
		final byte[] data_b = new byte[length];
		for(int i = 0; i < length; i++)
			data_b[i] = in.readByte();
		return new MapToCharPByte(unique, data_c, data_b);
	}

	protected char[] getChars() {
		return _data_c;
	}

	protected byte[] getBytes() {
		return _data_b;
	}

	@Override
	public int getUpperBoundValue() {
		return max;
	}

	@Override
	public void copyInt(int[] d) {
		for(int i = 0; i < d.length; i++)
			set(i, d[i]);
	}

	@Override
	public void copyBit(BitSet d) {
		for(int i = d.nextSetBit(0); i >= 0; i = d.nextSetBit(i + 1))
			_data_c[i] = 1;
	}

	@Override
	public AMapToData resize(int unique) {
		final int size = _data_c.length;
		AMapToData ret;
		if(unique <= 1)
			return new MapToZero(size);
		else if(unique == 2 && size > 32)
			ret = new MapToBit(unique, size);
		else if(unique <= 127)
			ret = new MapToUByte(unique, size);
		else if(unique < 256)
			ret = new MapToByte(unique, size);
		else if(unique < Character.MAX_VALUE - 1)
			ret = new MapToChar(unique, size);
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
		char prev = _data_c[0];
		byte prev_b = _data_b[0];
		for(int i = 1; i <_data_c.length; i++){
			c += prev == _data_c[i] && prev_b == _data_b[i] ? 0: 1;
			prev = _data_c[i];
			prev_b = _data_b[i];
		}
		return c;
	}
}
