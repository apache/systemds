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

package org.apache.sysds.runtime.frame.data.columns;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.MemoryEstimates;

public class HashLongArray extends Array<Object> implements IHashArray {
	private long[] _data;

	private HashLongArray(int nRow) {
		this(new long[nRow]);
	}

	public HashLongArray(long[] data) {
		super(data.length);
		_data = data;
	}

	public HashLongArray(String[] data) {
		super(data.length);
		_data = new long[data.length];
		for(int i = 0; i < data.length; i++) {
			_data[i] = parseHashLong(data[i]);
		}
	}

	@Override
	public Object get() {
		throw new NotImplementedException("Invalid to get underlying array in Hash");
	}

	@Override
	public Object get(int index) {
		return Long.toHexString(_data[index]);
	}

	@Override
	public Object getInternal(int index) {
		return Long.valueOf(_data[index]);
	}

	@Override
	public long getLong(int index) {
		return _data[index];
	}

	@Override
	public int getInt(int index) {
		return (int) _data[index];
	}

	protected long[] getLongs() {
		return _data;
	}

	@Override
	public void set(int index, Object value) {
		if(value instanceof String)
			_data[index] = parseHashLong((String) value);
		else if(value instanceof Long)
			_data[index] = (Long) value;
		else if(value == null)
			_data[index] = 0L;
		else
			throw new NotImplementedException("not supported : " + value);
	}

	@Override
	public void set(int index, String value) {
		_data[index] = parseHashLong(value);
	}

	@Override
	public void set(int index, double value) {
		_data[index] = (long) value;
	}

	@Override
	public void setFromOtherType(int rl, int ru, Array<?> value) {
		for(int i = rl; i <= ru; i++)
			_data[i] = parseHashLong(value.get(i));
	}

	@Override
	public void setNz(int rl, int ru, Array<Object> value) {
		if(value instanceof HashLongArray) {
			long[] thatVals = ((HashLongArray) value)._data;
			for(int i = rl; i <= ru; i++)
				if(thatVals[i] != 0)
					_data[i] = thatVals[i];
		}
		else {
			throw new NotImplementedException("Not supported type of array: " + value.getClass().getSimpleName());
		}
	}

	@Override
	public void setFromOtherTypeNz(int rl, int ru, Array<?> value) {
		if(value instanceof HashLongArray)
			setNz(rl, ru, (HashLongArray) value);
		else if(value instanceof StringArray) {
			StringArray st = ((StringArray) value);
			for(int i = rl; i <= ru; i++)
				if(st.get(i) != null)
					_data[i] = parseHashLong(st.get(i));
		}
		else {
			ValueType vt = value.getValueType();
			for(int i = rl; i <= ru; i++) {
				Object v = value.get(i);
				if(v != null)
					_data[i] = UtilFunctions.objectToLong(vt, v);
			}
		}
	}

	@Override
	public void append(Object value) {
		append(parseHashLong(value));
	}

	@Override
	public void append(String value) {
		append(parseHashLong(value));
	}

	public void append(long value) {
		if(_data.length <= _size)
			_data = Arrays.copyOf(_data, newSize());
		_data[_size++] = value;
	}

	@Override
	public Array<Object> append(Array<Object> other) {
		if(other instanceof HashLongArray) {

			final int endSize = this._size + other.size();
			final long[] ret = new long[endSize];
			System.arraycopy(_data, 0, ret, 0, this._size);
			System.arraycopy(((HashLongArray) other)._data, 0, ret, this._size, other.size());
			return new HashLongArray(ret);
		}
		else if(other instanceof OptionalArray) {
			OptionalArray<Object> ot = (OptionalArray<Object>) other;
			if(ot._a instanceof HashLongArray) {
				Array<Object> a = this.append(ot._a);
				return OptionalArray.appendOther(ot, a);
			}
		}
		throw new NotImplementedException(other.getClass().getSimpleName() + " not append supported in hashColumn");
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(FrameArrayType.HASH64.ordinal());
		for(int i = 0; i < _size; i++)
			out.writeLong(_data[i]);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		_size = _data.length;
		for(int i = 0; i < _size; i++)
			_data[i] = in.readLong();
	}

	protected static HashLongArray read(DataInput in, int nRow) throws IOException {
		final HashLongArray arr = new HashLongArray(nRow);
		arr.readFields(in);
		return arr;
	}

	@Override
	public Array<Object> clone() {
		return new HashLongArray(Arrays.copyOf(_data, _size));
	}

	@Override
	public Array<Object> slice(int rl, int ru) {
		return new HashLongArray(Arrays.copyOfRange(_data, rl, ru));
	}

	@Override
	public void reset(int size) {
		if(_data.length < size || _data.length > 2 * size)
			_data = new long[size];
		else
			for(int i = 0; i < size; i++)
				_data[i] = 0;
		_size = size;
	}

	@Override
	public byte[] getAsByteArray() {
		throw new NotImplementedException("Unclear how this byte array should look like for Hash");
	}

	@Override
	public ValueType getValueType() {
		return ValueType.HASH64;
	}

	@Override
	public Pair<ValueType, Boolean> analyzeValueType(int maxCells) {
		return new Pair<>(ValueType.HASH64, false);
	}

	@Override
	public FrameArrayType getFrameArrayType() {
		return FrameArrayType.HASH64;
	}

	@Override
	public long getInMemorySize() {
		long size = super.getInMemorySize(); // object header + object reference
		size += MemoryEstimates.longArrayCost(_data.length);
		return size;
	}

	@Override
	public long getExactSerializedSize() {
		return 1 + 8 * _size;
	}

	@Override
	protected Array<Boolean> changeTypeBitSet(Array<Boolean> ret, int l, int u) {
		for(int i = l; i < u; i++) {
			if(_data[i] != 0 && _data[i] != 1)
				throw new DMLRuntimeException("Unable to change to Boolean from Hash array because of value:" + _data[i]);
			ret.set(i, _data[i] == 0 ? false : true);
		}
		return ret;
	}

	@Override
	protected Array<Boolean> changeTypeBoolean(Array<Boolean> retA, int l, int u) {
		boolean[] ret = (boolean[]) retA.get();
		for(int i = l; i < u; i++) {
			if(_data[i] != 0 && _data[i] != 1)
				throw new DMLRuntimeException("Unable to change to Boolean from Hash array because of value:" + _data[i]);
			ret[i] = _data[i] == 0 ? false : true;
		}
		return retA;
	}

	@Override
	protected Array<Double> changeTypeDouble(Array<Double> retA, int l, int u) {
		double[] ret = (double[]) retA.get();
		for(int i = l; i < u; i++)
			ret[i] = _data[i];
		return retA;
	}

	@Override
	protected Array<Float> changeTypeFloat(Array<Float> retA, int l, int u) {
		float[] ret = (float[]) retA.get();
		for(int i = l; i < u; i++)
			ret[i] = _data[i];
		return retA;
	}

	@Override
	protected Array<Integer> changeTypeInteger(Array<Integer> retA, int l, int u) {
		int[] ret = (int[]) retA.get();
		for(int i = l; i < u; i++)
			ret[i] = (int) _data[i];
		return retA;
	}

	@Override
	protected Array<Long> changeTypeLong(Array<Long> retA, int l, int u) {
		long[] ret = (long[]) retA.get();
		for(int i = l; i < u; i++)
			ret[i] = _data[i];
		return retA;
	}

	@Override
	protected Array<Object> changeTypeHash64(Array<Object> retA, int l, int u) {
		long[] ret = ((HashLongArray) retA).getLongs();
		for(int i = l; i < u; i++)
			ret[i] = _data[i];
		return retA;
	}

	@Override
	protected Array<Object> changeTypeHash32(Array<Object> retA, int l, int u) {
		int[] ret = ((HashIntegerArray) retA).getInts();
		for(int i = l; i < u; i++)
			ret[i] = (int) _data[i];
		return retA;
	}

	@Override
	protected Array<String> changeTypeString(Array<String> retA, int l, int u) {
		String[] ret = (String[]) retA.get();
		for(int i = l; i < u; i++)
			ret[i] = get(i).toString();
		return retA;
	}

	@Override
	public Array<Character> changeTypeCharacter(Array<Character> retA, int l, int u) {
		char[] ret = (char[]) retA.get();
		for(int i = l; i < u; i++)
			ret[i] = get(i).toString().charAt(0);
		return retA;
	}

	@Override
	public void fill(String value) {
		fill(parseHashLong(value));
	}

	@Override
	public void fill(Object value) {
		fill(parseHashLong(value));
	}

	private void fill(long value) {
		Arrays.fill(_data, value);
	}

	@Override
	public double getAsDouble(int i) {
		return _data[i];
	}

	public static long parseHashLong(Object s) {
		if(s == null)
			return 0L;
		else if(s instanceof String)
			return parseHashLong((String) s);
		else if(s instanceof Long)
			return (Long) s;
		else if(s instanceof Integer)
			return (Integer) s;
		else
			throw new NotImplementedException("not supported" + s);
	}

	public static long parseHashLong(String s) {
		if(s == null || s.isEmpty())
			return 0L;
		return Long.parseUnsignedLong(s, 16);
	}

	@Override
	public boolean isShallowSerialize() {
		return true;
	}

	@Override
	public boolean isEmpty() {
		for(int i = 0; i < _size; i++)
			if(_data[i] != 0L)
				return false;
		return true;
	}

	@Override
	public Array<Object> select(int[] indices) {
		final long[] ret = new long[indices.length];
		for(int i = 0; i < indices.length; i++)
			ret[i] = _data[indices[i]];
		return new HashLongArray(ret);
	}

	@Override
	public Array<Object> select(boolean[] select, int nTrue) {
		final long[] ret = new long[nTrue];
		int k = 0;
		for(int i = 0; i < select.length; i++)
			if(select[i])
				ret[k++] = _data[i];
		return new HashLongArray(ret);
	}

	@Override
	public final boolean isNotEmpty(int i) {
		return _data[i] != 0;
	}

	@Override
	public double hashDouble(int idx) {
		return Long.hashCode(_data[idx]);
	}

	@Override
	public boolean equals(Array<Object> other) {
		if(other instanceof HashLongArray)
			return Arrays.equals(_data, ((HashLongArray) other)._data);
		else
			return false;
	}

	@Override
	public boolean possiblyContainsNaN() {
		return false;
	}

	@Override
	public void setM(HashMapToInt<Object> map, AMapToData m, int i) {
		m.set(i, map.get(Long.valueOf(getLong(i))) - 1);
	}

	@Override
	public void setM(HashMapToInt<Object> map, int si, AMapToData m, int i) {
		m.set(i, map.get(Long.valueOf(getLong(i))) - 1);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder(_size * 5 + 2);
		sb.append(super.toString() + ":[");
		for(int i = 0; i < _size - 1; i++)
			sb.append(_data[i] + ",");
		sb.append(_data[_size - 1]);
		sb.append("]");
		return sb.toString();
	}
}
