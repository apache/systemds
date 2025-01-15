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
import org.apache.sysds.runtime.frame.data.lib.FrameUtil;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.MemoryEstimates;

public class CharArray extends Array<Character> {

	private char[] _data;

	private CharArray(int nRow) {
		this(new char[nRow]);
	}

	public CharArray(char[] data) {
		super(data.length);
		_data = data;
	}

	public char[] get() {
		return _data;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(FrameArrayType.CHARACTER.ordinal());
		for(int i = 0; i < _size; i++)
			out.writeChar(_data[i]);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		_size = _data.length;
		for(int i = 0; i < _size; i++)
			_data[i] = in.readChar();
	}

	protected static CharArray read(DataInput in, int nRow) throws IOException {
		final CharArray arr = new CharArray(nRow);
		arr.readFields(in);
		return arr;
	}

	@Override
	public Character get(int index) {
		return _data[index];
	}

	@Override
	public double getAsDouble(int i) {
		return _data[i];
	}

	@Override
	public void set(int index, Character value) {
		_data[index] = value != null ? value : 0;
	}

	@Override
	public void set(int index, double value) {
		_data[index] = (char) (int) value;
	}

	@Override
	public void set(int index, String value) {
		_data[index] = parseChar(value);
	}

	@Override
	public void setFromOtherType(int rl, int ru, Array<?> value) {
		for(int i = rl; i <= ru; i++)
			_data[i] = value.get(i).toString().charAt(0);
	}

	@Override
	public void set(int rl, int ru, Array<Character> value, int rlSrc) {
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
	public void setNz(int rl, int ru, Array<Character> value) {
		char[] data2 = ((CharArray) value)._data;
		for(int i = rl; i <= ru; i++)
			if(data2[i] != 0)
				_data[i] = data2[i];
	}

	@Override
	public void setFromOtherTypeNz(int rl, int ru, Array<?> value) {
		final ValueType vt = value.getValueType();
		for(int i = rl; i <= ru; i++) {
			Object vv = value.get(i);
			if(vv != null) {
				char v = UtilFunctions.objectToCharacter(vt, vv);
				if(v != 0)
					_data[i] = v;
			}
		}
	}

	@Override
	public void append(String value) {
		append(parseChar(value));
	}

	@Override
	public void append(Character value) {
		if(_data.length <= _size)
			_data = Arrays.copyOf(_data, newSize());
		_data[_size++] = (value != null) ? value : 0;
	}

	@Override
	public Array<Character> append(Array<Character> other) {
		final int endSize = this._size + other.size();
		final char[] ret = new char[endSize];
		System.arraycopy(_data, 0, ret, 0, this._size);
		System.arraycopy(other.get(), 0, ret, this._size, other.size());
		if(other instanceof OptionalArray)
			return OptionalArray.appendOther((OptionalArray<Character>) other, new CharArray(ret));
		else
			return new CharArray(ret);
	}

	@Override
	public Array<Character> slice(int rl, int ru) {
		return new CharArray(Arrays.copyOfRange(_data, rl, ru));
	}

	@Override
	public void reset(int size) {
		if(_data.length < size || _data.length > 2 * size)
			_data = new char[size];
		else
			for(int i = 0; i < size; i++)
				_data[i] = 0;
		_size = size;
	}

	@Override
	public byte[] getAsByteArray() {
		ByteBuffer charBuffer = ByteBuffer.allocate(2 * _size);
		charBuffer.order(ByteOrder.nativeOrder());
		for(int i = 0; i < _size; i++)
			charBuffer.putChar(_data[i]);
		return charBuffer.array();

	}

	@Override
	public ValueType getValueType() {
		return ValueType.CHARACTER;
	}

	@Override
	public Pair<ValueType, Boolean> analyzeValueType(int maxCells) {
		return new Pair<>(ValueType.CHARACTER, false);
	}

	@Override
	public FrameArrayType getFrameArrayType() {
		return FrameArrayType.CHARACTER;
	}

	@Override
	public long getExactSerializedSize() {
		return 1L + 2L * _data.length;
	}

	@Override
	protected Array<Boolean> changeTypeBitSet(Array<Boolean> ret, int l, int u) {
		for(int i = l; i < u; i++) {
			final int di = _data[i];
			if(di != 0 && di != 1)
				throw new DMLRuntimeException("Unable to change to boolean from char array because of value:" //
					+ _data[i] + " (as int: " + di + ")");
			ret.set(i, di != 0);
		}
		return ret;
	}

	@Override
	protected Array<Boolean> changeTypeBoolean(Array<Boolean> retA, int l, int u) {
		boolean[] ret = (boolean[]) retA.get();
		for(int i = l; i < u; i++) {
			final int di = _data[i];
			if(di != 0 && di != 1)
				throw new DMLRuntimeException("Unable to change to boolean from char array because of value:" //
					+ _data[i] + " (as int: " + di + ")");
			ret[i] = di != 0;
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
			ret[i] = "" + _data[i];
		return retA;
	}

	@Override
	public Array<Character> changeTypeCharacter(Array<Character> retA, int l, int u) {
		char[] ret = (char[]) retA.get();
		for(int i = l; i < u; i++)
			ret[i] = _data[i];
		return retA;
	}

	@Override
	public void fill(String val) {
		fill(parseChar(val));
	}

	@Override
	public void fill(Character value) {
		value = value != null ? value : 0;
		Arrays.fill(_data, value);
	}

	@Override
	public boolean isShallowSerialize() {
		return true;
	}

	@Override
	public Array<Character> clone() {
		return new CharArray(Arrays.copyOf(_data, _size));
	}

	public static char parseChar(String value) {
		if(value == null)
			return 0;
		else if(value.length() == 1)
			return value.charAt(0);
		else if(FrameUtil.isIntType(value, value.length()) != null)
			return (char) IntegerArray.parseInt(value);
		else
			throw new NumberFormatException("Invalid parsing of Character: " + value);
	}

	@Override
	public boolean isEmpty() {
		for(int i = 0; i < _size; i++)
			if(_data[i] != 0)
				return false;
		return true;
	}

	@Override
	public Array<Character> select(int[] indices) {
		final char[] ret = new char[indices.length];
		for(int i = 0; i < indices.length; i++)
			ret[i] = _data[indices[i]];
		return new CharArray(ret);
	}

	@Override
	public Array<Character> select(boolean[] select, int nTrue) {
		final char[] ret = new char[nTrue];
		int k = 0;
		for(int i = 0; i < select.length; i++)
			if(select[i])
				ret[k++] = _data[i];
		return new CharArray(ret);
	}

	@Override
	public final boolean isNotEmpty(int i) {
		return _data[i] != 0;
	}

	@Override
	public double hashDouble(int idx) {
		return Character.hashCode(_data[idx]);
	}

	@Override
	public boolean equals(Array<Character> other) {
		if(other instanceof CharArray)
			return Arrays.equals(_data, ((CharArray) other)._data);
		else
			return false;
	}

	@Override
	public boolean possiblyContainsNaN() {
		return false;
	}

	@Override
	public long getInMemorySize() {
		return estimateInMemorySize(_size);
	}

	public static long estimateInMemorySize(int nRow) {
		long size = baseMemoryCost(); // object header + object reference
		size += MemoryEstimates.charArrayCost(nRow);
		return size;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder(_size * 2 + 15);
		sb.append(super.toString());
		sb.append(":[");
		for(int i = 0; i < _size - 1; i++) {
			sb.append(_data[i]);
			sb.append(',');
		}
		sb.append(_data[_size - 1]);
		sb.append("]");
		return sb.toString();
	}
}
