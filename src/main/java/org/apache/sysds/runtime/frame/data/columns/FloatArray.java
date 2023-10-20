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

public class FloatArray extends Array<Float> {
	private float[] _data;

	public FloatArray(float[] data) {
		super(data.length);
		_data = data;
	}

	public float[] get() {
		return _data;
	}

	@Override
	public Float get(int index) {
		return _data[index];
	}

	@Override
	public void set(int index, Float value) {
		_data[index] = (value != null) ? value : 0f;
	}

	@Override
	public void set(int index, double value) {
		_data[index] = (float) value;
	}

	@Override
	public void set(int index, String value) {
		set(index, parseFloat(value));
	}

	@Override
	public void set(int rl, int ru, Array<Float> value) {
		set(rl, ru, value, 0);
	}

	@Override
	public void setFromOtherType(int rl, int ru, Array<?> value) {
		final ValueType vt = value.getValueType();
		for(int i = rl; i <= ru; i++)
			_data[i] = UtilFunctions.objectToFloat(vt, value.get(i));
	}

	@Override
	public void set(int rl, int ru, Array<Float> value, int rlSrc) {
		System.arraycopy(((FloatArray) value)._data, rlSrc, _data, rl, ru - rl + 1);
	}

	@Override
	public void setNz(int rl, int ru, Array<Float> value) {
		float[] data2 = ((FloatArray) value)._data;
		for(int i = rl; i <= ru; i++)
			if(data2[i] != 0)
				_data[i] = data2[i];
	}

	@Override
	public void setFromOtherTypeNz(int rl, int ru, Array<?> value) {
		final ValueType vt = value.getValueType();
		for(int i = rl; i <= ru; i++) {
			float v = UtilFunctions.objectToFloat(vt, value.get(i));
			if(v != 0)
				_data[i] = v;
		}
	}

	@Override
	public void append(String value) {
		append(parseFloat(value));
	}

	@Override
	public void append(Float value) {
		if(_data.length <= _size)
			_data = Arrays.copyOf(_data, newSize());
		_data[_size++] = (value != null) ? value : 0f;
	}

	@Override
	public Array<Float> append(Array<Float> other) {
		final int endSize = this._size + other.size();
		final float[] ret = new float[endSize];
		System.arraycopy(_data, 0, ret, 0, this._size);
		System.arraycopy(other.get(), 0, ret, this._size, other.size());
		if(other instanceof OptionalArray)
			return OptionalArray.appendOther((OptionalArray<Float>) other, new FloatArray(ret));
		else
			return new FloatArray(ret);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(FrameArrayType.FP32.ordinal());
		for(int i = 0; i < _size; i++)
			out.writeFloat(_data[i]);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		_size = _data.length;
		for(int i = 0; i < _size; i++)
			_data[i] = in.readFloat();
	}

	@Override
	public Array<Float> clone() {
		return new FloatArray(Arrays.copyOf(_data, _size));
	}

	@Override
	public Array<Float> slice(int rl, int ru) {
		return new FloatArray(Arrays.copyOfRange(_data, rl, ru));
	}

	@Override
	public void reset(int size) {
		if(_data.length < size || _data.length > 2 * size)
			_data = new float[size];
		else
			for(int i = 0; i < size; i++)
				_data[i] = 0;
		_size = size;
	}

	@Override
	public byte[] getAsByteArray() {
		ByteBuffer floatBuffer = ByteBuffer.allocate(8 * _size);
		floatBuffer.order(ByteOrder.nativeOrder());
		for(int i = 0; i < _size; i++)
			floatBuffer.putFloat(_data[i]);
		return floatBuffer.array();
	}

	@Override
	public ValueType getValueType() {
		return ValueType.FP32;
	}

	@Override
	public Pair<ValueType, Boolean> analyzeValueType() {
		return new Pair<>(ValueType.FP32, false);
	}

	@Override
	public FrameArrayType getFrameArrayType() {
		return FrameArrayType.FP32;
	}

	@Override
	public long getInMemorySize() {
		long size = super.getInMemorySize(); // object header + object reference
		size += MemoryEstimates.floatArrayCost(_data.length);
		return size;
	}

	@Override
	public long getExactSerializedSize() {
		return 1 + 4 * _data.length;
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
			if(_data[i] != 0 && _data[i] != 1)
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
	protected Array<Integer> changeTypeInteger() {
		int[] ret = new int[size()];
		for(int i = 0; i < size(); i++) {
			if(_data[i] != (int) _data[i])
				throw new DMLRuntimeException("Unable to change to integer from float array because of value:" + _data[i]);
			ret[i] = (int) _data[i];
		}
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
	protected Array<Float> changeTypeFloat() {
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
	public Array<Character> changeTypeCharacter() {
		char[] ret = new char[size()];
		for(int i = 0; i < size(); i++)
			ret[i] = CharArray.parseChar(get(i).toString());
		return new CharArray(ret);
	}

	@Override
	public void fill(String value) {
		fill(parseFloat(value));
	}

	@Override
	public void fill(Float value) {
		value = value != null ? value : 0.0f;
		Arrays.fill(_data, value);
	}

	@Override
	public double getAsDouble(int i) {
		return _data[i];
	}

	public static float parseFloat(String value) {
		try {
			if(value == null || value.isEmpty())
				return 0.0f;
			return Float.parseFloat(value);
		}
		catch(NumberFormatException e) {
			final int len = value.length();
			// check for common extra cases.
			if(len == 3 && value.compareToIgnoreCase("Inf") == 0)
				return Float.POSITIVE_INFINITY;
			else if(len == 4 && value.compareToIgnoreCase("-Inf") == 0)
				return Float.NEGATIVE_INFINITY;
			throw new DMLRuntimeException(e);
		}
	}

	@Override
	public boolean isShallowSerialize() {
		return true;
	}

	@Override
	public boolean isEmpty() {
		for(int i = 0; i < _data.length; i++)
			if(isNotEmpty(i))
				return false;
		return true;
	}

	@Override
	public Array<Float> select(int[] indices) {
		final float[] ret = new float[indices.length];
		for(int i = 0; i < indices.length; i++)
			ret[i] = _data[indices[i]];
		return new FloatArray(ret);
	}

	@Override
	public Array<Float> select(boolean[] select, int nTrue) {
		final float[] ret = new float[nTrue];
		int k = 0;
		for(int i = 0; i < select.length; i++)
			if(select[i])
				ret[k++] = _data[i];
		return new FloatArray(ret);
	}

	@Override
	public final boolean isNotEmpty(int i) {
		return _data[i] != 0.0f;
	}

	@Override
	public double hashDouble(int idx) {
		return Float.hashCode(_data[idx]);
	}

	@Override
	public boolean equals(Array<Float> other) {
		if(other instanceof FloatArray)
			return Arrays.equals(_data, ((FloatArray) other)._data);
		else
			return false;
	}

	@Override
	public boolean possiblyContainsNaN() {
		return true;
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
