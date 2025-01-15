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

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.MemoryEstimates;

public class IntegerArray extends Array<Integer> {
	private int[] _data;

	private IntegerArray(int nRow) {
		this(new int[nRow]);
	}

	public IntegerArray(int[] data) {
		super(data.length);
		_data = data;
	}

	public int[] get() {
		return _data;
	}

	@Override
	public Integer get(int index) {
		return _data[index];
	}

	@Override
	public void set(int index, Integer value) {
		_data[index] = (value != null) ? value : 0;
	}

	@Override
	public void set(int index, double value) {
		_data[index] = (int) value;
	}

	@Override
	public void set(int index, String value) {
		set(index, parseInt(value));
	}
	
	@Override
	public void setFromOtherType(int rl, int ru, Array<?> value) {
		final ValueType vt = value.getValueType();
		for(int i = rl; i <= ru; i++)
			_data[i] = UtilFunctions.objectToInteger(vt, value.get(i));

	}

	@Override
	public void set(int rl, int ru, Array<Integer> value, int rlSrc) {
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
	public void setNz(int rl, int ru, Array<Integer> value) {
		int[] data2 = ((IntegerArray) value)._data;
		for(int i = rl; i <= ru; i++)
			if(data2[i] != 0)
				_data[i] = data2[i];
	}

	@Override
	public void setFromOtherTypeNz(int rl, int ru, Array<?> value) {
		final ValueType vt = value.getValueType();
		for(int i = rl; i <= ru; i++) {
			int v = UtilFunctions.objectToInteger(vt, value.get(i));
			if(v != 0)
				_data[i] = v;
		}
	}

	@Override
	public void append(String value) {
		append(parseInt(value));
	}

	@Override
	public void append(Integer value) {
		if(_data.length <= _size)
			_data = Arrays.copyOf(_data, newSize());
		_data[_size++] = (value != null) ? value : 0;
	}

	@Override
	public Array<Integer> append(Array<Integer> other) {
		final int endSize = this._size + other.size();
		final int[] ret = new int[endSize];
		System.arraycopy(_data, 0, ret, 0, this._size);
		System.arraycopy(other.get(), 0, ret, this._size, other.size());
		if(other instanceof OptionalArray)
			return OptionalArray.appendOther((OptionalArray<Integer>) other, new IntegerArray(ret));
		else
			return new IntegerArray(ret);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(FrameArrayType.INT32.ordinal());
		for(int i = 0; i < _size; i++)
			out.writeInt(_data[i]);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		_size = _data.length;
		for(int i = 0; i < _size; i++)
			_data[i] = in.readInt();
	}

	protected static IntegerArray read(DataInput in, int nRow) throws IOException {
		final IntegerArray arr = new IntegerArray(nRow);
		arr.readFields(in);
		return arr;
	}

	@Override
	public Array<Integer> clone() {
		return new IntegerArray(Arrays.copyOf(_data, _size));
	}

	@Override
	public Array<Integer> slice(int rl, int ru) {
		return new IntegerArray(Arrays.copyOfRange(_data, rl, ru));
	}

	public void reset(int size) {
		if(_data.length < size || _data.length > 2 * size)
			_data = new int[size];
		else
			for(int i = 0; i < size; i++)
				_data[i] = 0;
		_size = size;
	}

	@Override
	public byte[] getAsByteArray() {
		ByteBuffer intBuffer = ByteBuffer.allocate(4 * _size);
		intBuffer.order(ByteOrder.LITTLE_ENDIAN);
		for(int i = 0; i < _size; i++)
			intBuffer.putInt(_data[i]);
		return intBuffer.array();
	}

	@Override
	public ValueType getValueType() {
		return ValueType.INT32;
	}

	@Override
	public Pair<ValueType, Boolean> analyzeValueType(int maxCells) {
		return new Pair<>(ValueType.INT32, false);
	}

	@Override
	public FrameArrayType getFrameArrayType() {
		return FrameArrayType.INT32;
	}

	@Override
	public long getInMemorySize() {
		long size = super.getInMemorySize(); // object header + object reference
		size += MemoryEstimates.intArrayCost(_data.length);
		return size;
	}

	@Override
	public long getExactSerializedSize() {
		return 1 + 4 * _size;
	}

	@Override
	protected Array<Boolean> changeTypeBitSet(Array<Boolean> ret, int l, int u) {
		for(int i = l; i < u; i++) {
			if(_data[i] != 0 && _data[i] != 1)
				throw new DMLRuntimeException(
					"Unable to change to Boolean from Integer array because of value:" + _data[i]);
			ret.set(i, _data[i] == 0 ? false : true);
		}
		return ret;
	}

	@Override
	protected Array<Boolean> changeTypeBoolean(Array<Boolean> retA, int l, int u) {
		boolean[] ret = (boolean[]) retA.get();
		for(int i = l; i < u; i++) {
			if(_data[i] < 0 || _data[i] > 1)
				throw new DMLRuntimeException(
					"Unable to change to Boolean from Integer array because of value:" + _data[i]);
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
			ret[i] = _data[i];
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
			ret[i] = _data[i];
		return retA;
	}

	@Override
	protected Array<String> changeTypeString(Array<String> retA, int l, int u) {
		String[] ret = (String[]) retA.get();
		for(int i = l; i < u; i++)
			ret[i] = Integer.toString(_data[i]);
		return retA;
	}

	@Override
	public Array<Character> changeTypeCharacter(Array<Character> retA, int l, int u) {
		char[] ret = (char[]) retA.get();
		for(int i = l; i < u; i++)
			ret[i] = Integer.toString(_data[i]).charAt(0);
		return retA;
	}

	@Override
	public void fill(String value) {
		fill(parseInt(value));
	}

	@Override
	public void fill(Integer value) {
		value = value != null ? value : 0;
		Arrays.fill(_data, value);
	}

	@Override
	public double getAsDouble(int i) {
		return _data[i];
	}

	public static int parseInt(String s) {
		if(s == null || s.isEmpty())
			return 0;
		try {
			return Integer.parseInt(s);
		}
		catch(NumberFormatException e) {
			// we use exceptions as normal behavior here since we want faster default parsing.
			if(s.contains("."))
				return (int) Double.parseDouble(s);
			else
				throw e;
		}
	}

	@Override
	public boolean isShallowSerialize() {
		return true;
	}

	@Override
	public boolean isEmpty() {
		for(int i = 0; i < _size; i++)
			if(_data[i] != 0)
				return false;
		return true;
	}

	@Override
	public Array<Integer> select(int[] indices) {
		final int[] ret = new int[indices.length];
		for(int i = 0; i < indices.length; i++)
			ret[i] = _data[indices[i]];
		return new IntegerArray(ret);
	}

	@Override
	public Array<Integer> select(boolean[] select, int nTrue) {
		final int[] ret = new int[nTrue];
		int k = 0;
		for(int i = 0; i < select.length; i++)
			if(select[i])
				ret[k++] = _data[i];
		return new IntegerArray(ret);
	}

	@Override
	public final boolean isNotEmpty(int i) {
		return _data[i] != 0;
	}

	@Override
	public double hashDouble(int idx) {
		return Integer.hashCode(_data[idx]);
	}

	@Override
	public boolean equals(Array<Integer> other) {
		if(other instanceof IntegerArray)
			return Arrays.equals(_data, ((IntegerArray) other)._data);
		else
			return false;
	}

	@Override
	public boolean possiblyContainsNaN() {
		return false;
	}

	@Override
	protected int addValRecodeMap(HashMapToInt<Integer> map, int id, int i) {
		if( map.putIfAbsentI(_data[i], id) == -1) 
			id++;
		return id;
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
