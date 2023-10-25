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
import java.util.BitSet;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.MemoryEstimates;

public class HashLongArray extends Array<Object> {
	private long[] _data;

	public HashLongArray(long[] data) {
		super(data.length);
		_data = data;
	}

	public long[] get() {
		return _data;
	}

	@Override
	public Object get(int index) {
		return Long.toHexString(_data[index]);
	}

	public long getLong(int index){
		return _data[index];
	}

	@Override
	public void set(int index, Object value) {
		if(value instanceof String)
			_data[index] = parseLong((String)value);
		else if(value instanceof Long)
			_data[index] = (Long)value;
		else 
			throw new NotImplementedException("not supported : " + value);
	}


	@Override
	public void set(int index, String value) {
		_data[index] = parseLong((String)value);
	}

	@Override
	public void set(int index, double value) {
		_data[index] = (long) value;
	}

	@Override
	public void set(int rl, int ru, Array<Object> value) {
		set(rl, ru, value, 0);
	}

	@Override
	public void setFromOtherType(int rl, int ru, Array<?> value) {
		final ValueType vt = value.getValueType();
		for(int i = rl; i <= ru; i++)
			_data[i] = UtilFunctions.objectToLong(vt, value.get(i));
	}

	@Override
	public void setNz(int rl, int ru, Array<Object> value) {
	
		throw new NotImplementedException();
		
		// long[] data2 = ((LongArray) value)._data;
		// for(int i = rl; i <= ru; i++)
		// 	if(data2[i] != 0)
		// 		_data[i] = data2[i];
	}

	@Override
	public void setFromOtherTypeNz(int rl, int ru, Array<?> value) {
		throw new NotImplementedException();
	}

	@Override
	public void append(Object value) {
		append(parseLong(value));
	}


	@Override
	public void append(String value) {
		append(parseLong(value));
	}

	public void append(long value) {
		if(_data.length <= _size)
			_data = Arrays.copyOf(_data, newSize());
		_data[_size++] = value;
	}

	@Override
	public Array<Object> append(Array<Object> other) {
		throw new NotImplementedException();
		// final int endSize = this._size + other.size();
		// final long[] ret = new long[endSize];
		// System.arraycopy(_data, 0, ret, 0, this._size);
		// System.arraycopy(other.get(), 0, ret, this._size, other.size());
		// if(other instanceof OptionalArray)
		// 	return OptionalArray.appendOther((OptionalArray<Long>) other, new LongArray(ret));
		// else
		// 	return new LongArray(ret);
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
		throw new NotImplementedException();
		// ByteBuffer longBuffer = ByteBuffer.allocate(8 * _size);
		// longBuffer.order(ByteOrder.LITTLE_ENDIAN);
		// for(int i = 0; i < _size; i++)
		// 	longBuffer.putLong(_data[i]);
		// return longBuffer.array();
	}

	@Override
	public ValueType getValueType() {
		return ValueType.HASH64;
	}

	@Override
	public Pair<ValueType, Boolean> analyzeValueType() {
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
		return 1 + 8 * _data.length;
	}

	@Override
	protected Array<Boolean> changeTypeBitSet() {
		BitSet ret = new BitSet(size());
		for(int i = 0; i < size(); i++) {
			if(_data[i] != 0 && _data[i] != 1)
				throw new DMLRuntimeException(
					"Unable to change to Boolean from Integer array because of value:" + _data[i]);
			ret.set(i, _data[i] == 0 ? false : true);
		}
		return new BitSetArray(ret, size());
	}

	@Override
	protected Array<Boolean> changeTypeBoolean() {
		boolean[] ret = new boolean[size()];
		for(int i = 0; i < size(); i++) {
			if(_data[i] < 0 || _data[i] > 1)
				throw new DMLRuntimeException(
					"Unable to change to Boolean from Integer array because of value:" + _data[i]);
			ret[i] = _data[i] == 0 ? false : true;
		}
		return new BooleanArray(ret);
	}

	@Override
	protected Array<Double> changeTypeDouble() {
		double[] ret = new double[size()];
		for(int i = 0; i < size(); i++)
			ret[i] = _data[i];
		return new DoubleArray(ret);
	}

	@Override
	protected Array<Float> changeTypeFloat() {
		float[] ret = new float[size()];
		for(int i = 0; i < size(); i++)
			ret[i] = _data[i];
		return new FloatArray(ret);
	}

	@Override
	protected Array<Integer> changeTypeInteger() {
		int[] ret = new int[size()];
		for(int i = 0; i < size(); i++) {
			if(Math.abs(_data[i]) > Integer.MAX_VALUE )
				throw new DMLRuntimeException("Unable to change to integer from long array because of value:" + _data[i]);
			ret[i] = (int) _data[i];
		}
		return new IntegerArray(ret);
	}

	@Override
	protected Array<Long> changeTypeLong() {
		return new LongArray(_data);
	}

	@Override
	protected Array<Object> changeTypeHash64() {
		return this;
	}

	@Override
	protected Array<String> changeTypeString() {
		String[] ret = new String[size()];
		for(int i = 0; i < size(); i++)
			ret[i] = get(i).toString();
		return new StringArray(ret);
	}

	@Override
	public void fill(String value) {
		fill(parseLong(value));
	}

	@Override
	public void fill(Object value) {
		fill(parseLong(value));
	}

	public void fill(Long value) {
		value = value != null ? value : 0L;
		Arrays.fill(_data, value);
	}

	@Override
	public double getAsDouble(int i) {
		return _data[i];
	}

	public static long parseLong(Object s) {
		if(s instanceof String)
			return parseLong((String) s);
		else if (s instanceof Long)
			return (Long)s;
		else 
			throw new NotImplementedException("not supported" + s); 
	}


	public static long parseLong(String s) {
		if(s == null || s.isEmpty())
			return 0L;
		return Long.parseUnsignedLong(s, 16);
	}

	@Override
	public Array<Character> changeTypeCharacter() {
		char[] ret = new char[size()];
		for(int i = 0; i < size(); i++)
			ret[i] = get(i).toString().charAt(0);
		return new CharArray(ret);
	}

	@Override
	public boolean isShallowSerialize() {
		return true;
	}

	@Override
	public boolean isEmpty() {
		for(int i = 0; i < _data.length; i++)
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
	public boolean possiblyContainsNaN(){
		return false;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder(_data.length * 5 + 2);
		sb.append(super.toString() + ":[");
		for(int i = 0; i < _size - 1; i++)
			sb.append(_data[i] + ",");
		sb.append(_data[_size - 1]);
		sb.append("]");
		return sb.toString();
	}
}
