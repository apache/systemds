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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.BitSet;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.MemoryEstimates;

public class LongArray extends Array<Long> {
	private long[] _data;

	public LongArray(long[] data) {
		super(data.length);
		_data = data;
	}

	public long[] get() {
		return _data;
	}

	@Override
	public Long get(int index) {
		return _data[index];
	}

	@Override
	public void set(int index, Long value) {
		_data[index] = (value != null) ? value : 0L;
	}

	@Override
	public void set(int index, double value) {
		_data[index] = (long) value;
	}

	@Override
	public void set(int index, String value) {
		set(index, parseLong(value));
	}

	@Override
	public void set(int rl, int ru, Array<Long> value) {
		set(rl, ru, value, 0);
	}

	@Override
	public void setFromOtherType(int rl, int ru, Array<?> value) {
		final ValueType vt = value.getValueType();
		for(int i = rl; i <= ru; i++)
			_data[i] = UtilFunctions.objectToLong(vt, value.get(i));
	}

	@Override
	public void set(int rl, int ru, Array<Long> value, int rlSrc) {
		try {
			// try system array copy.
			// but if it does not work, default to get.
			System.arraycopy(value.get(), rlSrc, _data, rl, ru - rl + 1);
		}
		catch(Exception e) {
			super.set(rl, ru, value, rlSrc);
		}
	}

	@Override
	public void setNz(int rl, int ru, Array<Long> value) {
		long[] data2 = ((LongArray) value)._data;
		for(int i = rl; i <= ru; i++)
			if(data2[i] != 0)
				_data[i] = data2[i];
	}

	@Override
	public void setFromOtherTypeNz(int rl, int ru, Array<?> value) {
		final ValueType vt = value.getValueType();
		for(int i = rl; i <= ru; i++) {
			long v = UtilFunctions.objectToLong(vt, value.get(i));
			if(v != 0)
				_data[i] = v;
		}
	}

	@Override
	public void append(String value) {
		append(parseLong(value));
	}

	@Override
	public void append(Long value) {
		if(_data.length <= _size)
			_data = Arrays.copyOf(_data, newSize());
		_data[_size++] = (value != null) ? value : 0L;
	}

	@Override
	public Array<Long> append(Array<Long> other) {
		final int endSize = this._size + other.size();
		final long[] ret = new long[endSize];
		System.arraycopy(_data, 0, ret, 0, this._size);
		System.arraycopy(other.get(), 0, ret, this._size, other.size());
		if(other instanceof OptionalArray)
			return OptionalArray.appendOther((OptionalArray<Long>) other, new LongArray(ret));
		else
			return new LongArray(ret);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(FrameArrayType.INT64.ordinal());
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
	public Array<Long> clone() {
		return new LongArray(Arrays.copyOf(_data, _size));
	}

	@Override
	public Array<Long> slice(int rl, int ru) {
		return new LongArray(Arrays.copyOfRange(_data, rl, ru));
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
		ByteBuffer longBuffer = ByteBuffer.allocate(8 * _size);
		longBuffer.order(ByteOrder.LITTLE_ENDIAN);
		for(int i = 0; i < _size; i++)
			longBuffer.putLong(_data[i]);
		return longBuffer.array();
	}

	@Override
	public ValueType getValueType() {
		return ValueType.INT64;
	}

	@Override
	public Pair<ValueType, Boolean> analyzeValueType() {
		return new Pair<>(ValueType.INT64, false);
	}

	@Override
	public FrameArrayType getFrameArrayType() {
		return FrameArrayType.INT64;
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
		return this;
	}

	@Override
	protected Array<Object> changeTypeHash64() {
		return new HashLongArray(_data);
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
	public void fill(Long value) {
		value = value != null ? value : 0L;
		Arrays.fill(_data, value);
	}

	@Override
	public double getAsDouble(int i) {
		return _data[i];
	}

	public static long parseLong(String s) {
		if(s == null || s.isEmpty())
			return 0;
		try {
			return Long.parseLong(s);
		}
		catch(NumberFormatException e) {
			if(s.contains("."))
				return (long) Double.parseDouble(s);
			else
				throw e;
		}
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
	public Array<Long> select(int[] indices) {
		final long[] ret = new long[indices.length];
		for(int i = 0; i < indices.length; i++)
			ret[i] = _data[indices[i]];
		return new LongArray(ret);
	}

	@Override
	public Array<Long> select(boolean[] select, int nTrue) {
		final long[] ret = new long[nTrue];
		int k = 0;
		for(int i = 0; i < select.length; i++)
			if(select[i])
				ret[k++] = _data[i];
		return new LongArray(ret);
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
	public boolean equals(Array<Long> other) {
		if(other instanceof LongArray)
			return Arrays.equals(_data, ((LongArray) other)._data);
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
