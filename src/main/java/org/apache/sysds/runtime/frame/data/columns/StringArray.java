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
import org.apache.sysds.runtime.frame.data.FrameUtil;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.utils.MemoryEstimates;

public class StringArray extends Array<String> {
	private String[] _data;

	public StringArray(String[] data) {
		super(data.length);
		_data = data;
	}

	public String[] get() {
		return _data;
	}

	@Override
	public String get(int index) {
		return _data[index];
	}

	@Override
	public void set(int index, String value) {
		_data[index] = value;
	}

	@Override
	public void set(int index, double value) {
		_data[index] = Double.toString(value);
	}

	@Override
	public void set(int rl, int ru, Array<String> value) {
		set(rl, ru, value, 0);
	}

	@Override
	public void setFromOtherType(int rl, int ru, Array<?> value) {
		for(int i = rl; i <= ru; i++) {
			final Object v = value.get(i);
			if(v != null)
				_data[i] = v.toString();
			else
				_data[i] = null;
		}
	}

	@Override
	public void set(int rl, int ru, Array<String> value, int rlSrc) {
		System.arraycopy(((StringArray) value)._data, rlSrc, _data, rl, ru - rl + 1);
	}

	@Override
	public void setNz(int rl, int ru, Array<String> value) {
		String[] data2 = ((StringArray) value)._data;
		for(int i = rl; i <= ru; i++)
			if(data2[i] != null)
				_data[i] = data2[i];
	}

	@Override
	public void setFromOtherTypeNz(int rl, int ru, Array<?> value) {
		for(int i = rl; i <= ru; i++) {
			Object v = value.get(i);
			if(v != null)
				_data[i] = v.toString();
		}
	}

	@Override
	public void append(String value) {
		if(_data.length <= _size)
			_data = Arrays.copyOf(_data, newSize());
		_data[_size++] = value;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(FrameArrayType.STRING.ordinal());
		for(int i = 0; i < _size; i++)
			out.writeUTF((_data[i] != null) ? _data[i] : "");
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		_size = _data.length;
		for(int i = 0; i < _size; i++) {
			String tmp = in.readUTF();
			_data[i] = (!tmp.isEmpty()) ? tmp : null;
		}
	}

	@Override
	public Array<String> clone() {
		return new StringArray(Arrays.copyOf(_data, _size));
	}

	@Override
	public Array<String> slice(int rl, int ru) {
		return new StringArray(Arrays.copyOfRange(_data, rl, ru));
	}

	@Override
	public void reset(int size) {
		if(_data.length < size)
			_data = new String[size];
		_size = size;
	}

	@Override
	public byte[] getAsByteArray() {
		throw new NotImplementedException("Not Implemented getAsByte for string");
	}

	/**
	 * Python interface to extract strings from systemds.
	 * 
	 * @param r the index to extract
	 * @return The value in bytes for py4j
	 */
	public byte[] getIndexAsBytes(int r) {
		if(_data[r] != null)
			return _data[r].getBytes();
		else
			return null;
	}

	@Override
	public ValueType getValueType() {
		return ValueType.STRING;
	}

	@Override
	public ValueType analyzeValueType() {
		ValueType state = FrameUtil.isType(_data[0]);
		for(int i = 1; i < _size; i++) {
			ValueType c = FrameUtil.isType(_data[i]);
			if(c == ValueType.STRING || c == ValueType.UNKNOWN)
				return ValueType.STRING;
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
		return FrameArrayType.STRING;
	}

	@Override
	public long getInMemorySize() {
		long size = super.getInMemorySize(); // object header + object reference
		size += MemoryEstimates.stringArrayCost(_data);
		return size;
	}

	@Override
	public long getExactSerializedSize() {
		long si = 1; // byte identifier
		for(String s : _data)
			si += IOUtilFunctions.getUTFSize(s);
		return si;
	}

	@Override
	protected Array<Boolean> changeTypeBitSet() {
		return changeTypeBoolean();
	}

	@Override
	protected Array<Boolean> changeTypeBoolean() {
		// detect type of transform.
		if(_data[0].toLowerCase().equals("true") || _data[0].toLowerCase().equals("false"))
			return changeTypeBooleanStandard();
		if(_data[0].equals("0") || _data[0].equals("1"))
			return changeTypeBooleanNumeric();
		else
			throw new DMLRuntimeException("Not supported type of Strings to change to Booleans value: " + _data[0]);
	}

	protected Array<Boolean> changeTypeBooleanStandard() {
		if(size() > ArrayFactory.bitSetSwitchPoint)
			return changeTypeBooleanStandardBitSet();
		else
			return changeTypeBooleanStandardArray();
	}

	protected Array<Boolean> changeTypeBooleanStandardBitSet() {
		BitSet ret = new BitSet(size());
		for(int i = 0; i < size(); i++)
			ret.set(i, Boolean.parseBoolean(_data[i]));

		return new BitSetArray(ret, size());
	}

	protected Array<Boolean> changeTypeBooleanStandardArray() {
		boolean[] ret = new boolean[size()];
		for(int i = 0; i < size(); i++)
			ret[i] = Boolean.parseBoolean(_data[i]);
		return new BooleanArray(ret);
	}

	protected Array<Boolean> changeTypeBooleanNumeric() {
		if(size() > ArrayFactory.bitSetSwitchPoint)
			return changeTypeBooleanNumericBitSet();
		else
			return changeTypeBooleanNumericArray();
	}

	protected Array<Boolean> changeTypeBooleanNumericBitSet() {
		BitSet ret = new BitSet(size());
		for(int i = 0; i < size(); i++) {
			final boolean zero = _data[i].equals("0");
			final boolean one = _data[i].equals("1");
			if(zero | one)
				ret.set(i, one);
			else
				throw new DMLRuntimeException("Unable to change to Boolean from String array, value:" + _data[i]);
		}
		return new BitSetArray(ret, size());
	}

	protected Array<Boolean> changeTypeBooleanNumericArray() {
		boolean[] ret = new boolean[size()];
		for(int i = 0; i < size(); i++) {
			final boolean zero = _data[i].equals("0");
			final boolean one = _data[i].equals("1");
			if(zero | one)
				ret[i] = one;
			else
				throw new DMLRuntimeException("Unable to change to Boolean from String array, value:" + _data[i]);
		}
		return new BooleanArray(ret);
	}

	@Override
	protected Array<Double> changeTypeDouble() {
		try {
			double[] ret = new double[size()];
			for(int i = 0; i < size(); i++)
				ret[i] = Double.parseDouble(_data[i]);
			return new DoubleArray(ret);
		}
		catch(NumberFormatException e) {
			throw new DMLRuntimeException("Unable to change to Double from String array", e);
		}
	}

	@Override
	protected Array<Float> changeTypeFloat() {
		try {
			float[] ret = new float[size()];
			for(int i = 0; i < size(); i++)
				ret[i] = Float.parseFloat(_data[i]);
			return new FloatArray(ret);
		}
		catch(NumberFormatException e) {
			throw new DMLRuntimeException("Unable to change to Float from String array", e);
		}
	}

	@Override
	protected Array<Integer> changeTypeInteger() {
		try {
			int[] ret = new int[size()];
			for(int i = 0; i < size(); i++)
				ret[i] = Integer.parseInt(_data[i]);
			return new IntegerArray(ret);
		}
		catch(NumberFormatException e) {
			throw new DMLRuntimeException("Unable to change to Integer from String array", e);
		}
	}

	@Override
	protected Array<Long> changeTypeLong() {
		try {
			long[] ret = new long[size()];
			for(int i = 0; i < size(); i++)
				ret[i] = Long.parseLong(_data[i]);
			return new LongArray(ret);
		}
		catch(NumberFormatException e) {
			throw new DMLRuntimeException("Unable to change to Long from String array", e);
		}
	}

	@Override
	public Array<String> changeTypeString() {
		return clone();
	}

	@Override
	public Pair<Integer, Integer> getMinMaxLength() {
		int minLength = Integer.MAX_VALUE;
		int maxLength = 0;
		for(int i = 0; i < _size; i++) {
			if(_data[i] == null)
				continue;
			int l = _data[i].length();
			minLength = minLength < l ? minLength : l;
			maxLength = maxLength > l ? maxLength : l;

		}
		return new Pair<>(minLength, maxLength);
	}

	@Override
	public void fill(String value) {
		for(int i = 0; i < _size; i++)
			_data[i] = value;
	}

	@Override
	public double getAsDouble(int i){
		return Double.parseDouble(_data[i]);
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
