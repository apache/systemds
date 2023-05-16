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
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.MemoryEstimates;

public class BooleanArray extends ABooleanArray {
	protected boolean[] _data;

	public BooleanArray(boolean[] data) {
		super(data.length);
		_data = data;
	}

	public boolean[] get() {
		return _data;
	}

	@Override
	public Boolean get(int index) {
		return _data[index];
	}

	@Override
	public void set(int index, Boolean value) {
		_data[index] = value != null && value;
	}

	@Override
	public void set(int index, double value) {
		_data[index] = value == 1.0;
	}

	@Override
	public void set(int index, String value) {
		set(index, BooleanArray.parseBoolean(value));
	}

	@Override
	public void set(int rl, int ru, Array<Boolean> value) {
		set(rl, ru, value, 0);
	}

	@Override
	public void setFromOtherType(int rl, int ru, Array<?> value) {
		final ValueType vt = value.getValueType();
		for(int i = rl; i <= ru; i++)
			set(i, UtilFunctions.objectToBoolean(vt, value.get(i)));
	}

	@Override
	public void set(int rl, int ru, Array<Boolean> value, int rlSrc) {
		if(value instanceof BooleanArray)
			System.arraycopy((boolean[]) value.get(), rlSrc, _data, rl, ru - rl + 1);
		else
			for(int i = rl, off = rlSrc; i <= ru; i++, off++)
				_data[i] = value.get(off);
	}

	@Override
	public void setNz(int rl, int ru, Array<Boolean> value) {
		if(value instanceof BooleanArray) {
			boolean[] data2 = ((BooleanArray) value)._data;
			for(int i = rl; i <= ru; i++)
				if(data2[i])
					_data[i] = data2[i];
		}
		else {
			for(int i = rl; i <= ru; i++) {
				final boolean v = value.get(i);
				if(v)
					_data[i] = v;
			}
		}
	}

	@Override
	public void setFromOtherTypeNz(int rl, int ru, Array<?> value) {
		final ValueType vt = value.getValueType();
		for(int i = rl; i <= ru; i++) {
			boolean v = UtilFunctions.objectToBoolean(vt, value.get(i));
			if(v)
				_data[i] = v;
		}
	}

	@Override
	public void append(String value) {
		append(parseBoolean(value));
	}

	@Override
	public void append(Boolean value) {
		if(_data.length <= _size)
			_data = Arrays.copyOf(_data, newSize());
		_data[_size++] = (value != null) ? value : false;
	}

	@Override
	public Array<Boolean> append(Array<Boolean> other) {
		final int endSize = this._size + other.size();
		final ABooleanArray retBS = ArrayFactory.allocateBoolean(endSize);
		retBS.set(0, this._size - 1, this);
		if(other instanceof OptionalArray) {
			retBS.set(this._size, endSize - 1, ((OptionalArray<Boolean>) other)._a);
			return OptionalArray.appendOther((OptionalArray<Boolean>) other, retBS);
		}
		else {
			retBS.set(this._size, endSize - 1, other);
			return retBS;
		}
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(FrameArrayType.BOOLEAN.ordinal());
		for(int i = 0; i < _size; i++)
			out.writeBoolean(_data[i]);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		_size = _data.length;
		for(int i = 0; i < _size; i++)
			_data[i] = in.readBoolean();
	}

	@Override
	public ABooleanArray clone() {
		return new BooleanArray(Arrays.copyOf(_data, _size));
	}

	@Override
	public ABooleanArray slice(int rl, int ru) {
		return new BooleanArray(Arrays.copyOfRange(_data, rl, ru));
	}

	@Override
	public void reset(int size) {
		if(_data.length < size || _data.length > 2 * size)
			_data = new boolean[size];
		else
			for(int i = 0; i < size; i++)
				_data[i] = false;
		_size = size;
	}

	@Override
	public byte[] getAsByteArray() {
		// over allocating here.. we could maybe bit pack?
		ByteBuffer booleanBuffer = ByteBuffer.allocate(_size);
		booleanBuffer.order(ByteOrder.nativeOrder());
		for(int i = 0; i < _size; i++)
			booleanBuffer.put((byte) (_data[i] ? 1 : 0));
		return booleanBuffer.array();
	}

	@Override
	public ValueType getValueType() {
		return ValueType.BOOLEAN;
	}

	@Override
	public Pair<ValueType, Boolean> analyzeValueType() {
		return new Pair<ValueType, Boolean>(ValueType.BOOLEAN, false);
	}

	@Override
	public FrameArrayType getFrameArrayType() {
		return FrameArrayType.BOOLEAN;
	}

	@Override
	public long getInMemorySize() {
		return estimateInMemorySize(_size);
	}

	public static long estimateInMemorySize(int nRow) {
		long size = baseMemoryCost(); // object header + object reference
		size += MemoryEstimates.booleanArrayCost(nRow);
		return size;
	}

	@Override
	public long getExactSerializedSize() {
		return 1 + _data.length;
	}

	@Override
	protected ABooleanArray changeTypeBitSet() {
		return new BitSetArray(_data);
	}

	@Override
	protected ABooleanArray changeTypeBoolean() {
		return this;
	}

	@Override
	protected Array<Double> changeTypeDouble() {
		double[] ret = new double[size()];
		for(int i = 0; i < size(); i++)
			ret[i] = _data[i] ? 1.0 : 0.0;
		return new DoubleArray(ret);
	}

	@Override
	protected Array<Float> changeTypeFloat() {
		float[] ret = new float[size()];
		for(int i = 0; i < size(); i++)
			ret[i] = _data[i] ? 1.0f : 0.0f;
		return new FloatArray(ret);
	}

	@Override
	protected Array<Integer> changeTypeInteger() {
		int[] ret = new int[size()];
		for(int i = 0; i < size(); i++)
			ret[i] = _data[i] ? 1 : 0;
		return new IntegerArray(ret);
	}

	@Override
	protected Array<Long> changeTypeLong() {
		long[] ret = new long[size()];
		for(int i = 0; i < size(); i++)
			ret[i] = _data[i] ? 1L : 0L;
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
			ret[i] = (char) (_data[i] ? 1 : 0);
		return new CharArray(ret);
	}

	@Override
	public void fill(String value) {
		fill(parseBoolean(value));
	}

	@Override
	public void fill(Boolean value) {
		value = value != null ? value : false;
		Arrays.fill(_data, value);
	}

	@Override
	public boolean isShallowSerialize() {
		return true;
	}

	@Override
	public double getAsDouble(int i) {
		return _data[i] ? 1.0 : 0.0;
	}

	@Override
	public boolean isEmpty() {
		for(int i = 0; i < _data.length; i++)
			if(_data[i])
				return false;
		return true;
	}

	@Override
	public boolean isAllTrue() {
		for(int i = 0; i < _data.length; i++)
			if(!_data[i])
				return false;
		return true;
	}

	@Override
	public ABooleanArray select(int[] indices) {
		final boolean[] ret = new boolean[indices.length];
		for(int i = 0; i < indices.length; i++)
			ret[i] = _data[indices[i]];
		return new BooleanArray(ret);
	}

	@Override
	public ABooleanArray select(boolean[] select, int nTrue) {
		final boolean[] ret = new boolean[nTrue];
		int k = 0;
		for(int i = 0; i < select.length; i++)
			if(select[i])
				ret[k++] = _data[i];
		return new BooleanArray(ret);
	}

	@Override
	public final boolean isNotEmpty(int i) {
		return _data[i];
	}

	public static boolean parseBoolean(String value) {
		return value != null && //
			!value.isEmpty() && //
			(Boolean.parseBoolean(value) || value.equals("1") || value.equals("1.0") || value.equals("t"));
	}

	@Override
	public double hashDouble(int idx){
		return get(idx) ? 1.0 : 0.0;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder(_data.length * 2 + 10);
		sb.append(super.toString() + ":[");
		for(int i = 0; i < _size - 1; i++)
			sb.append((_data[i] ? 1 : 0) + ",");
		sb.append(_data[_size - 1] ? 1 : 0);
		sb.append("]");
		return sb.toString();
	}
}
