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

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameUtil;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;
import org.apache.sysds.utils.MemoryEstimates;

public class DoubleArray extends Array<Double> {
	private double[] _data;

	public DoubleArray(double[] data) {
		_data = data;
		_size = _data.length;
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
	public void set(int rl, int ru, Array<Double> value) {
		set(rl, ru, value, 0);
	}

	@Override
	public void setFromOtherType(int rl, int ru, Array<?> value) {
		throw new NotImplementedException();
	}

	@Override
	public void set(int rl, int ru, Array<Double> value, int rlSrc) {
		System.arraycopy(((DoubleArray) value)._data, rlSrc, _data, rl, ru - rl + 1);
	}

	@Override
	public void setNz(int rl, int ru, Array<Double> value) {
		double[] data2 = ((DoubleArray) value)._data;
		for(int i = rl; i < ru + 1; i++)
			if(data2[i] != 0)
				_data[i] = data2[i];
	}

	@Override
	public void append(String value) {
		append((value != null) ? Double.parseDouble(value) : null);
	}

	@Override
	public void append(Double value) {
		if(_data.length <= _size)
			_data = Arrays.copyOf(_data, newSize());
		_data[_size++] = (value != null) ? value : 0d;
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
	public Array<Double> sliceTransform(int rl, int ru, ValueType vt) {
		return slice(rl, ru);
	}

	@Override
	public void reset(int size) {
		if(_data.length < size)
			_data = new double[size];
		_size = size;
	}

	@Override
	public byte[] getAsByteArray(int nRow) {
		ByteBuffer doubleBuffer = ByteBuffer.allocate(8 * nRow);
		doubleBuffer.order(ByteOrder.nativeOrder());
		for(int i = 0; i < nRow; i++)
			doubleBuffer.putDouble(_data[i]);
		return doubleBuffer.array();
	}

	@Override
	public ValueType getValueType() {
		return ValueType.FP64;
	}

	@Override
	public ValueType analyzeValueType() {
		ValueType state = FrameUtil.isType(_data[0]);
		for(int i = 0; i < _size; i++) {
			ValueType c = FrameUtil.isType(_data[i]);
			if(state == ValueType.FP64)
				return ValueType.FP64;
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
				case BOOLEAN:
					switch(c) {
						case FP64:
						case FP32:
						case INT64:
						case INT32:
							state = c;
						default:
					}
					break;
				default:
			}
		}
		return state;
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
	protected Array<?> changeTypeBitSet() {
		BitSet ret = new BitSet(size());
		for(int i = 0; i < size(); i++) {
			if(_data[i] != 0 && _data[i] != 1)
				throw new DMLRuntimeException(
					"Unable to change to Boolean from Integer array because of value:" + _data[i]);
			ret.set(i,  _data[i] == 0 ? false : true);
		}
		return new BitSetArray(ret, size());
	}

	@Override
	protected Array<?> changeTypeBoolean() {
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
	protected Array<?> changeTypeDouble() {
		return clone();
	}

	@Override
	protected Array<?> changeTypeFloat() {
		float[] ret = new float[size()];
		for(int i = 0; i < size(); i++)
			ret[i] = (float) _data[i];
		return new FloatArray(ret);
	}

	@Override
	protected Array<?> changeTypeInteger() {
		int[] ret = new int[size()];
		for(int i = 0; i < size(); i++) {
			if(_data[i] != (int) _data[i])
				throw new DMLRuntimeException("Unable to change to Integer from Double array because of value:" + _data[i]);
			ret[i] = (int) _data[i];
		}
		return new IntegerArray(ret);
	}

	@Override
	protected Array<?> changeTypeLong() {
		long[] ret = new long[size()];
		for(int i = 0; i < size(); i++) {
			if(_data[i] != (long) _data[i])
				throw new DMLRuntimeException("Unable to change to Long from Double array because of value:" + _data[i]);
			ret[i] = (long) _data[i];
		}
		return new LongArray(ret);
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
