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

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.UtilFunctions;

public class OptionalArray<T> extends Array<T> {

	/** Underlying values not able to contain null values */
	protected Array<T> _a;
	/** A Bitset specifying where there are null, in it false means null */
	protected ABooleanArray _n;

	@SuppressWarnings("unchecked")
	public OptionalArray(T[] a) {
		super(a.length);

		if(a instanceof Boolean[])
			_a = (Array<T>) ArrayFactory.allocate(ValueType.BOOLEAN, a.length);
		else if(a instanceof Integer[])
			_a = (Array<T>) ArrayFactory.allocate(ValueType.INT32, a.length);
		else if(a instanceof Double[])
			_a = (Array<T>) ArrayFactory.allocate(ValueType.FP64, a.length);
		else if(a instanceof Float[])
			_a = (Array<T>) ArrayFactory.allocate(ValueType.FP32, a.length);
		else if(a instanceof Long[])
			_a = (Array<T>) ArrayFactory.allocate(ValueType.INT64, a.length);
		else if(a instanceof Character[]) // Character
			_a = (Array<T>) ArrayFactory.allocate(ValueType.CHARACTER, a.length);
		else
			throw new DMLRuntimeException("Invalid type for Optional Array: " + a.getClass().getSimpleName());

		_n = ArrayFactory.allocateBoolean(a.length);
		for(int i = 0; i < a.length; i++) {
			_a.set(i, a[i]);
			_n.set(i, a[i] != null);
		}
	}

	public OptionalArray(Array<T> a, boolean empty) {
		super(a.size());
		if(a instanceof OptionalArray)
			throw new DMLRuntimeException("Not allowed optional optional array");
		else if(a instanceof StringArray)
			throw new DMLRuntimeException("Not allowed StringArray in OptionalArray");
		_a = a;
		_n = ArrayFactory.allocateBoolean(a.size());
		if(!empty)
			_n.fill(true);
	}

	public OptionalArray(Array<T> a, ABooleanArray n) {
		super(a.size());
		if(a instanceof OptionalArray)
			throw new DMLRuntimeException("Not allowed optional optional array");
		else if(a instanceof StringArray)
			throw new DMLRuntimeException("Not allowed StringArray in OptionalArray");
		if(n.size() != a.size())
			throw new DMLRuntimeException("Incompatible sizes of arrays for optional array");
		_a = a;
		_n = n;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(FrameArrayType.OPTIONAL.ordinal());
		_a.write(out);
		_n.write(out);
	}

	@Override
	public long getExactSerializedSize() {
		return 1L + _a.getExactSerializedSize() + _n.getExactSerializedSize();
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		throw new DMLRuntimeException("Should not be called");
	}

	@SuppressWarnings("unchecked")
	protected static OptionalArray<?> readOpt(DataInput in, int nRow) throws IOException {
		final Array<?> a = ArrayFactory.read(in, nRow);
		final ABooleanArray n = (ABooleanArray) ArrayFactory.read(in, nRow);
		switch(a.getValueType()) {
			case BOOLEAN:
				return new OptionalArray<Boolean>((Array<Boolean>) a, n);
			case FP32:
				return new OptionalArray<Float>((Array<Float>) a, n);
			case FP64:
				return new OptionalArray<Double>((Array<Double>) a, n);
			case UINT8:
			case INT32:
				return new OptionalArray<Integer>((Array<Integer>) a, n);
			case INT64:
				return new OptionalArray<Long>((Array<Long>) a, n);
			case CHARACTER:
			default:
				return new OptionalArray<Character>((Array<Character>) a, n);
		}
	}

	@Override
	public T get(int index) {
		return _n.get(index) ? _a.get(index) : null;
	}

	@Override
	public Object get() {
		return _a.get();
	}

	@Override
	public double getAsDouble(int i) {
		return _n.get(i) ? _a.getAsDouble(i) : 0.0;
	}

	@Override
	public double getAsNaNDouble(int i) {
		return _n.get(i) ? _a.getAsDouble(i) : Double.NaN;
	}

	@Override
	public void set(int index, T value) {
		if(value == null)
			_n.set(index, false);
		else {
			_n.set(index, true);
			_a.set(index, value);
		}
	}

	@Override
	public void set(int index, double value) {
		_a.set(index, value);
		_n.set(index, true);
	}

	@Override
	public void set(int index, String value) {
		if(value == null)
			_n.set(index, false);
		else {
			_a.set(index, value);
			_n.set(index, true);
		}
	}

	@Override
	public void setFromOtherType(int rl, int ru, Array<?> value) {
		for(int i = rl; i <= ru; i++) {
			final Object o = value.get(i);
			if(o == null)
				_n.set(i, false);
			else {
				_a.set(i, UtilFunctions.objectToString(value.get(i)));
				_n.set(i, true);
			}
		}
	}

	@Override
	public void set(int rl, int ru, Array<T> value) {
		set(rl, ru, value, 0);
	}

	@Override
	public void set(int rl, int ru, Array<T> value, int rlSrc) {
		if(value instanceof OptionalArray)
			_a.set(rl, ru, getBasic(value), rlSrc);
		else
			_a.set(rl, ru, value, rlSrc);

		Array<Boolean> nulls = value.getNulls();
		if(nulls != null)
			_n.set(rl, ru, nulls, rlSrc);
		else{
			for(int i = rl; i <= ru; i++)
				_n.set(i, true);
		}
	}

	private static <T> Array<T> getBasic(Array<T> value) {
		while(value instanceof OptionalArray)
			value = ((OptionalArray<T>) value)._a;
		return value;
	}

	@Override
	public void setNz(int rl, int ru, Array<T> value) {
		for(int i = rl; i <= ru; i++) {
			T v = value.get(i);
			if(v != null)
				set(i, v);

		}
	}

	@Override
	public void setFromOtherTypeNz(int rl, int ru, Array<?> value) {
		for(int i = rl; i <= ru; i++) {
			String v = UtilFunctions.objectToString(value.get(i));
			if(v != null)
				set(i, v);

		}
	}

	@Override
	public void append(String value) {
		_n.append(value != null);
		_a.append(value);
		_size = _a.size();
	}

	@Override
	public void append(T value) {
		_n.append(value != null);
		_a.append(value);
		_size = _a.size();
	}

	@Override
	public Array<T> append(Array<T> other) {
		OptionalArray<T> otherOpt = (other instanceof OptionalArray) ? //
			(OptionalArray<T>) other : new OptionalArray<T>(other, false);
		ABooleanArray n = (ABooleanArray) _n.append(otherOpt._n);
		Array<T> a = _a.append(otherOpt._a);
		return new OptionalArray<T>(a, n);
	}

	public static <T> OptionalArray<T> appendOther(OptionalArray<T> that, Array<T> appended) {
		final int endSize = appended.size();
		ABooleanArray nullsThat = that._n;
		ABooleanArray optsEnd = ArrayFactory.allocateBoolean(endSize);
		optsEnd.fill(true);
		optsEnd.set(endSize - that.size(), endSize - 1, nullsThat);
		return new OptionalArray<T>(appended, optsEnd);
	}

	@Override
	public Array<T> slice(int rl, int ru) {
		return new OptionalArray<T>(_a.slice(rl, ru), _n.slice(rl, ru));
	}

	@Override
	public void reset(int size) {
		_size = size;
		_a.reset(size);
		_n.reset(size);
	}

	@Override
	public byte[] getAsByteArray() {
		// technically not correct.
		return _a.getAsByteArray();
	}

	@Override
	public ValueType getValueType() {
		return _a.getValueType();
	}

	@Override
	public Pair<ValueType, Boolean> analyzeValueType() {
		return new Pair<ValueType, Boolean>(getValueType(), true);
	}

	@Override
	public FrameArrayType getFrameArrayType() {
		return FrameArrayType.OPTIONAL;
	}

	@Override
	protected Array<Boolean> changeTypeBitSet() {
		Array<Boolean> a = _a.changeTypeBitSet();
		return new OptionalArray<Boolean>(a, _n);
	}

	@Override
	protected Array<Boolean> changeTypeBoolean() {
		Array<Boolean> a = _a.changeTypeBoolean();
		return new OptionalArray<Boolean>(a, _n);
	}

	@Override
	protected Array<Double> changeTypeDouble() {
		Array<Double> a = _a.changeTypeDouble();
		return new OptionalArray<Double>(a, _n);
	}

	@Override
	protected Array<Float> changeTypeFloat() {
		Array<Float> a = _a.changeTypeFloat();
		return new OptionalArray<Float>(a, _n);
	}

	@Override
	protected Array<Integer> changeTypeInteger() {
		Array<Integer> a = _a.changeTypeInteger();
		return new OptionalArray<Integer>(a, _n);
	}

	@Override
	protected Array<Long> changeTypeLong() {
		Array<Long> a = _a.changeTypeLong();
		return new OptionalArray<Long>(a, _n);
	}

	@Override
	protected Array<Character> changeTypeCharacter() {
		Array<Character> a = _a.changeTypeCharacter();
		return new OptionalArray<Character>(a, _n);
	}

	@Override
	protected Array<String> changeTypeString() {
		StringArray a = (StringArray) _a.changeTypeString();
		String[] d = a.get();
		for(int i = 0; i < _size; i++)
			if(!_n.get(i))
				d[i] = null;
		return a;
	}

	@Override
	public void fill(String val) {
		_n.fill(val != null);
		_a.fill(val);
	}

	@Override
	public void fill(T val) {
		_n.fill(val != null);
		_a.fill(val);
	}

	@Override
	public boolean isShallowSerialize() {
		return _a.isShallowSerialize();
	}

	@Override
	public ABooleanArray getNulls() {
		return _n;
	}

	@Override
	public Array<T> clone() {
		return new OptionalArray<T>(_a.clone(), _n.clone());
	}

	@Override
	public boolean isEmpty() {
		return !_n.isAllTrue();
	}

	@Override
	public Array<T> select(int[] indices) {
		return new OptionalArray<T>(_a.select(indices), _n.select(indices));
	}

	@Override
	public Array<T> select(boolean[] select, int nTrue) {
		return new OptionalArray<T>(_a.select(select, nTrue), _n.select(select, nTrue));
	}

	@Override
	public final boolean isNotEmpty(int i) {
		return _n.isNotEmpty(i) && _a.isNotEmpty(i);
	}

	@Override
	public Array<?> changeTypeWithNulls(ValueType t) {

		switch(t) {
			case BOOLEAN:
				if(size() > ArrayFactory.bitSetSwitchPoint)
					return changeTypeBitSet();
				else
					return changeTypeBoolean();
			case FP32:
				return changeTypeFloat();
			case FP64:
				return changeTypeDouble();
			case UINT8:
				throw new NotImplementedException();
			case INT32:
				return changeTypeInteger();
			case INT64:
				return changeTypeLong();
			case CHARACTER:
				return changeTypeCharacter();
			case STRING:
			case UNKNOWN:
			default:
				return changeTypeString(); // String can contain null
		}

	}

	@Override
	public boolean containsNull(){
		return !_n.isAllTrue();
	}


	@Override
	public double hashDouble(int idx){
		if(_n.get(idx))
			return _a.hashDouble(idx);
		else
			return Double.NaN;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder(_size + 2);
		sb.append(super.toString() + "<" + _a.getClass().getSimpleName() + ">:[");
		for(int i = 0; i < _size - 1; i++)
			sb.append(get(i) + ",");
		sb.append(get(_size - 1));
		sb.append("]");
		return sb.toString();
	}
}
