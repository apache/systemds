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
import org.apache.sysds.utils.MemoryEstimates;

public class DoubleArray extends Array<Double> {
	private double[] _data;

	public DoubleArray(double[] data) {
		super(data.length);
		_data = data;
	}

	public double[] get() {
		return _data;
	}

	@Override
	public Double get(int index) {
		return _data[index];
	}

	@Override
	public void set(int index, Double value) {
		_data[index] = (value != null) ? value : 0d;
	}

	@Override
	public void set(int index, double value) {
		_data[index] = value;
	}

	@Override
	public void set(int index, String value) {
		set(index, parseDouble(value));
	}

	@Override
	public void set(int rl, int ru, Array<Double> value) {
		set(rl, ru, value, 0);
	}

	@Override
	public void setFromOtherType(int rl, int ru, Array<?> value) {
		final ValueType vt = value.getValueType();
		for(int i = rl; i <= ru; i++)
			_data[i] = UtilFunctions.objectToDouble(vt, value.get(i));
	}

	@Override
	public void set(int rl, int ru, Array<Double> value, int rlSrc) {
		System.arraycopy(((DoubleArray) value)._data, rlSrc, _data, rl, ru - rl + 1);
	}

	@Override
	public void setNz(int rl, int ru, Array<Double> value) {
		double[] data2 = ((DoubleArray) value)._data;
		for(int i = rl; i <= ru; i++)
			if(data2[i] != 0)
				_data[i] = data2[i];
	}

	@Override
	public void setFromOtherTypeNz(int rl, int ru, Array<?> value) {
		final ValueType vt = value.getValueType();
		for(int i = rl; i <= ru; i++) {
			double v = UtilFunctions.objectToDouble(vt, value.get(i));
			if(v != 0)
				_data[i] = v;
		}
	}

	@Override
	public void append(String value) {
		append(parseDouble(value));
	}

	@Override
	public void append(Double value) {
		if(_data.length <= _size)
			_data = Arrays.copyOf(_data, newSize());
		_data[_size++] = (value != null) ? value : 0d;
	}

	@Override
	public Array<Double> append(Array<Double> other) {
		final int endSize = this._size + other.size();
		final double[] ret = new double[endSize];
		System.arraycopy(_data, 0, ret, 0, this._size);
		System.arraycopy((double[]) other.get(), 0, ret, this._size, other.size());
		if(other instanceof OptionalArray)
			return OptionalArray.appendOther((OptionalArray<Double>) other, new DoubleArray(ret));
		else
			return new DoubleArray(ret);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(FrameArrayType.FP64.ordinal());
		for(int i = 0; i < _size; i++)
			out.writeDouble(_data[i]);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		_size = _data.length;
		for(int i = 0; i < _size; i++)
			_data[i] = in.readDouble();
	}

	@Override
	public Array<Double> clone() {
		return new DoubleArray(Arrays.copyOf(_data, _size));
	}

	@Override
	public Array<Double> slice(int rl, int ru) {
		return new DoubleArray(Arrays.copyOfRange(_data, rl, ru));
	}

	@Override
	public void reset(int size) {
		if(_data.length < size || _data.length > 2 * size)
			_data = new double[size];
		else
			for(int i = 0; i < size; i++)
				_data[i] = 0;
		_size = size;
	}

	@Override
	public byte[] getAsByteArray() {
		ByteBuffer doubleBuffer = ByteBuffer.allocate(8 * _size);
		doubleBuffer.order(ByteOrder.nativeOrder());
		for(int i = 0; i < _size; i++)
			doubleBuffer.putDouble(_data[i]);
		return doubleBuffer.array();
	}

	@Override
	public ValueType getValueType() {
		return ValueType.FP64;
	}

	@Override
	public Pair<ValueType, Boolean> analyzeValueType() {
		ValueType state = FrameUtil.isType(_data[0]);
		for(int i = 0; i < _size; i++) {
			ValueType c = FrameUtil.isType(_data[i], state);
			if(state == ValueType.FP64)
				return new Pair<ValueType, Boolean>(ValueType.FP64, false);

			switch(state) {
				case FP32:
					switch(c) {
						case FP64:
							state = c;
						default:
					}
					break;
				case INT64:
					switch(c) {
						case FP64:
						case FP32:
							state = c;
						default:
					}
					break;
				case INT32:
					switch(c) {
						case FP64:
						case FP32:
						case INT64:
							state = c;
						default:
					}
					break;
				default:
				case BITSET:
					switch(c) {
						case FP64:
						case FP32:
						case INT64:
						case INT32:
							state = c;
						default:
					}
					break;
			}
		}
		return new Pair<ValueType, Boolean>(state, false);
	}

	@Override
	public FrameArrayType getFrameArrayType() {
		return FrameArrayType.FP64;
	}

	@Override
	public long getInMemorySize() {
		long size = super.getInMemorySize(); // object header + object reference
		size += MemoryEstimates.doubleArrayCost(_data.length);
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
			if(_data[i] != 0 && _data[i] != 1)
				throw new DMLRuntimeException(
					"Unable to change to Boolean from Integer array because of value:" + _data[i]);
			ret[i] = _data[i] == 0 ? false : true;
		}
		return new BooleanArray(ret);
	}

	@Override
	protected Array<Double> changeTypeDouble() {
		return this;
	}

	@Override
	protected Array<Float> changeTypeFloat() {
		float[] ret = new float[size()];
		for(int i = 0; i < size(); i++)
			ret[i] = (float) _data[i];
		return new FloatArray(ret);
	}

	@Override
	protected Array<Integer> changeTypeInteger() {
		int[] ret = new int[size()];
		for(int i = 0; i < size(); i++) {
			if(_data[i] != (int) _data[i])
				throw new DMLRuntimeException("Unable to change to Integer from Double array because of value:" + _data[i]);
			ret[i] = (int) _data[i];
		}
		return new IntegerArray(ret);
	}

	@Override
	protected Array<Long> changeTypeLong() {
		long[] ret = new long[size()];
		for(int i = 0; i < size(); i++) {
			if(_data[i] != (long) _data[i])
				throw new DMLRuntimeException("Unable to change to Long from Double array because of value:" + _data[i]);
			ret[i] = (long) _data[i];
		}
		return new LongArray(ret);
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
		fill(parseDouble(value));
	}

	@Override
	public void fill(Double value) {
		value = value != null ? value : 0.0d;
		Arrays.fill(_data, value);
	}

	@Override
	public double getAsDouble(int i) {
		return _data[i];
	}

	public static double parseDouble(String value) {
		if(value == null || value.isEmpty())
			return 0.0;
		else
			return Double.parseDouble(value);
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
	public Array<Double> select(int[] indices) {
		final double[] ret = new double[indices.length];
		for(int i = 0; i < indices.length; i++)
			ret[i] = _data[indices[i]];
		return new DoubleArray(ret);
	}

	@Override
	public Array<Double> select(boolean[] select, int nTrue) {
		final double[] ret = new double[nTrue];
		int k = 0;
		for(int i = 0; i < select.length; i++)
			if(select[i])
				ret[k++] = _data[i];
		return new DoubleArray(ret);
	}

	@Override
	public final boolean isNotEmpty(int i) {
		return _data[i] != 0.0d;
	}


	@Override
	public double hashDouble(int idx){
		return Double.hashCode(_data[idx]);
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
