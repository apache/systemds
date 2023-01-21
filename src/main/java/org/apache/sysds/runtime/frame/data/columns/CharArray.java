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
import org.apache.sysds.runtime.frame.data.lib.FrameUtil;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.UtilFunctions;

public class CharArray extends Array<Character> {

	protected char[] _data;

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

	@Override
	public Character get(int index) {
		return _data[index];
	}

	@Override
	public double getAsDouble(int i) {
		return (int) _data[i];
	}

	@Override
	public void set(int index, Character value) {
		_data[index] = value != null ? value : 0;
	}

	@Override
	public void set(int index, double value) {
		_data[index] = parseChar(Double.toString(value));
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
	public void set(int rl, int ru, Array<Character> value) {
		set(rl, ru, value, 0);
	}

	@Override
	public void set(int rl, int ru, Array<Character> value, int rlSrc) {
		System.arraycopy(((CharArray) value)._data, rlSrc, _data, rl, ru - rl + 1);
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
			char v = UtilFunctions.objectToCharacter(vt, value.get(i));
			if(v != 0)
				_data[i] = v;
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
		System.arraycopy((char[]) other.get(), 0, ret, this._size, other.size());
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
	public Pair<ValueType, Boolean> analyzeValueType() {
		return new Pair<ValueType, Boolean>(ValueType.CHARACTER, false);
	}

	@Override
	public FrameArrayType getFrameArrayType() {
		return FrameArrayType.CHARACTER;
	}

	@Override
	public long getExactSerializedSize() {
		return 1 + 2 * _data.length;
	}

	@Override
	protected Array<Boolean> changeTypeBitSet() {
		BitSet ret = new BitSet(size());
		for(int i = 0; i < size(); i++) {
			if(_data[i] != 0 && _data[i] != 1)
				throw new DMLRuntimeException("Unable to change to boolean from char array because of value:" + _data[i]);
			ret.set(i, _data[i] != 0);
		}
		return new BitSetArray(ret, size());
	}

	@Override
	protected Array<Boolean> changeTypeBoolean() {
		boolean[] ret = new boolean[size()];
		for(int i = 0; i < size(); i++) {
			if(_data[i] != 0 && _data[i] != 1)
				throw new DMLRuntimeException("Unable to change to boolean from char array because of value:" + _data[i]);
			ret[i] = _data[i] != 0;
		}
		return new BooleanArray(ret);
	}

	@Override
	protected Array<Double> changeTypeDouble() {
		try {
			double[] ret = new double[size()];
			for(int i = 0; i < size(); i++)
				ret[i] = (int) _data[i];
			return new DoubleArray(ret);
		}
		catch(NumberFormatException e) {
			throw new DMLRuntimeException("Invalid parsing of char to double", e);
		}
	}

	@Override
	protected Array<Float> changeTypeFloat() {
		try {
			float[] ret = new float[size()];
			for(int i = 0; i < size(); i++)
				ret[i] = (int) _data[i];
			return new FloatArray(ret);
		}
		catch(NumberFormatException e) {
			throw new DMLRuntimeException("Invalid parsing of char to float", e);
		}
	}

	@Override
	protected Array<Integer> changeTypeInteger() {
		int[] ret = new int[size()];
		for(int i = 0; i < size(); i++)
			ret[i] = (int) _data[i];

		return new IntegerArray(ret);
	}

	@Override
	protected Array<Long> changeTypeLong() {
		long[] ret = new long[size()];
		for(int i = 0; i < size(); i++)
			ret[i] = (int) _data[i];
		return new LongArray(ret);
	}

	@Override
	protected Array<String> changeTypeString() {
		String[] ret = new String[size()];
		for(int i = 0; i < size(); i++)
			ret[i] = _data[i] + "";
		return new StringArray(ret);
	}

	@Override
	public Array<Character> changeTypeCharacter() {
		return this;
	}

	@Override
	public void fill(String val) {
		fill(parseChar(val));
	}

	@Override
	public void fill(Character val) {
		Arrays.fill(_data, (char) val);
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
			return (char) Double.parseDouble(value);
		else
			throw new DMLRuntimeException("Invalid parsing of Character");
	}

	@Override
	public boolean isEmpty() {
		for(int i = 0; i < _data.length; i++)
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
	public String toString() {
		StringBuilder sb = new StringBuilder(_data.length * 2 + 15);
		sb.append(super.toString());
		sb.append(":[");
		if(_size > 0) {
			for(int i = 0; i < _size - 1; i++) {
				sb.append(_data[i]);
				sb.append(',');
			}
			sb.append(_data[_size - 1]);
		}
		sb.append("]");
		return sb.toString();
	}
}
