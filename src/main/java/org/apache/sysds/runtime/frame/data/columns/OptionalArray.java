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

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.UtilFunctions;

public class OptionalArray<T> extends Array<T> {

	/** Underlying values not able to contain null values */
	protected final Array<T> _a;
	/** A Bitset specifying where there are null, in it false means null */
	protected final ABooleanArray _n;

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

	@SuppressWarnings("unchecked")
	public OptionalArray(T[] a, ValueType vt) {
		super(a.length);
		_a = (Array<T>) ArrayFactory.allocate(vt, a.length);
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
	public long getInMemorySize() {
		long size = super.getInMemorySize(); // object header + object reference
		size += 16; // object pointers.
		size += _a.getInMemorySize();
		size += _n.getInMemorySize();
		return size;
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		throw new DMLRuntimeException("Should not be called");
	}

	@SuppressWarnings("unchecked")
	protected static OptionalArray<?> read(DataInput in, int nRow) throws IOException {
		final Array<?> a = ArrayFactory.read(in, nRow);
		final ABooleanArray n = (ABooleanArray) ArrayFactory.read(in, nRow);
		switch(a.getValueType()) {
			case BOOLEAN:
				return new OptionalArray<>((Array<Boolean>) a, n);
			case FP32:
				return new OptionalArray<>((Array<Float>) a, n);
			case FP64:
				return new OptionalArray<>((Array<Double>) a, n);
			case UINT8:
			case INT32:
				return new OptionalArray<>((Array<Integer>) a, n);
			case INT64:
				return new OptionalArray<>((Array<Long>) a, n);
			case CHARACTER:
			default:
				return new OptionalArray<>((Array<Character>) a, n);
		}
	}

	@Override
	public T get(int index) {
		return _n.get(index) ? _a.get(index) : null;
	}

	@Override
	public T getInternal(int index) {
		return _n.get(index) ? _a.getInternal(index) : null;
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
	public void set(int rl, int ru, Array<T> value, int rlSrc) {
		if(value instanceof OptionalArray)
			_a.set(rl, ru, getBasic(value), rlSrc);
		else
			_a.set(rl, ru, value, rlSrc);

		Array<Boolean> nulls = value.getNulls();
		if(nulls != null)
			_n.set(rl, ru, nulls, rlSrc);
		else {
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
		if(_a instanceof HashIntegerArray || _a instanceof HashLongArray) {
			Array<?> noOpt = value instanceof OptionalArray //
				? ((OptionalArray<?>) value)._a : value;
			_a.setFromOtherTypeNz(rl, ru, noOpt);
			for(int i = rl; i <= ru; i++) {
				Object v = value.get(i);
				if(v != null)
					_n.set(i, true);
			}
		}
		else {
			for(int i = rl; i <= ru; i++) {
				Object v = value.get(i);
				if(v != null)
					set(i, UtilFunctions.objectToString(v));
			}
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
			(OptionalArray<T>) other : new OptionalArray<>(other, false);
		ABooleanArray n = (ABooleanArray) _n.append(otherOpt._n);
		Array<T> a = _a.append(otherOpt._a);
		return new OptionalArray<>(a, n);
	}

	public static <T> OptionalArray<T> appendOther(OptionalArray<T> that, Array<T> appended) {
		final int endSize = appended.size();
		ABooleanArray nullsThat = that._n;
		ABooleanArray optsEnd = ArrayFactory.allocateBoolean(endSize);
		optsEnd.fill(true);
		optsEnd.set(endSize - that.size(), endSize - 1, nullsThat);
		return new OptionalArray<>(appended, optsEnd);
	}

	@Override
	public Array<T> slice(int rl, int ru) {
		return new OptionalArray<>(_a.slice(rl, ru), _n.slice(rl, ru));
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
	public Array<?> changeType(ValueType t) {
		if(t == ValueType.STRING) // String can contain null.
			return changeType(ArrayFactory.allocate(t, size()));
		return changeTypeWithNulls(t);
	}

	@Override
	public Pair<ValueType, Boolean> analyzeValueType(int maxCells) {
		return new Pair<>(getValueType(), true);
	}

	@Override
	public FrameArrayType getFrameArrayType() {
		return FrameArrayType.OPTIONAL;
	}

	@Override
	protected Array<Boolean> changeTypeBitSet(Array<Boolean> ret, int l, int u) {
		return _a.changeTypeBitSet(ret, l, u);
	}

	@Override
	protected Array<Boolean> changeTypeBoolean(Array<Boolean> retA, int l, int u) {
		return _a.changeTypeBoolean(retA, l, u);
	}

	@Override
	protected Array<Double> changeTypeDouble(Array<Double> retA, int l, int u) {
		return _a.changeTypeDouble(retA, l, u);
	}

	@Override
	protected Array<Float> changeTypeFloat(Array<Float> retA, int l, int u) {
		return _a.changeTypeFloat(retA, l, u);
	}

	@Override
	protected Array<Integer> changeTypeInteger(Array<Integer> retA, int l, int u) {
		return _a.changeTypeInteger(retA, l, u);
	}

	@Override
	protected Array<Long> changeTypeLong(Array<Long> retA, int l, int u) {

		return _a.changeTypeLong(retA, l, u);
	}

	@Override
	protected Array<Object> changeTypeHash64(Array<Object> retA, int l, int u) {
		return _a.changeTypeHash64(retA, l, u);
	}

	@Override
	protected Array<Object> changeTypeHash32(Array<Object> retA, int l, int u) {
		return _a.changeTypeHash32(retA, l, u);
	}

	@Override
	protected Array<Character> changeTypeCharacter(Array<Character> retA, int l, int u) {
		return _a.changeTypeCharacter(retA, l, u);
	}

	@Override
	protected Array<String> changeTypeString(Array<String> retA, int l, int u) {
		String[] d = (String[]) retA.get();
		for(int i = 0; i < _size; i++)
			if(_n.get(i))
				d[i] = _a.get(i).toString();
			else
				d[i] = null;
		return retA;
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
		return new OptionalArray<>(_a.clone(), _n.clone());
	}

	@Override
	public boolean isEmpty() {
		return !_n.isAllTrue();
	}

	@Override
	public Array<T> select(int[] indices) {
		return new OptionalArray<>(_a.select(indices), _n.select(indices));
	}

	@Override
	public Array<T> select(boolean[] select, int nTrue) {
		return new OptionalArray<>(_a.select(select, nTrue), _n.select(select, nTrue));
	}

	@Override
	public final boolean isNotEmpty(int i) {
		return _n.isNotEmpty(i) && _a.isNotEmpty(i);
	}

	@Override
	public boolean containsNull() {
		return !_n.isAllTrue();
	}

	@Override
	public double hashDouble(int idx) {
		if(_n.get(idx))
			return _a.hashDouble(idx);
		else
			return Double.NaN;
	}

	@Override
	public boolean equals(Array<T> other) {
		if(other instanceof OptionalArray) {
			OptionalArray<T> ot = (OptionalArray<T>) other;
			return _n.equals(ot._n) && ot._a.equals(_a);
		}
		else
			return false;
	}

	@Override
	public boolean possiblyContainsNaN() {
		return true;
	}

	@Override
	public void setM(HashMapToInt<T> map, AMapToData m, int i) {
		_a.setM(map, m, i);
	}

	@Override
	public void setM(HashMapToInt<T> map, int si, AMapToData m, int i) {
		if(_n.get(i))
			_a.setM(map, si, m, i);
		else
			m.set(i, si);
	}

	@Override
	protected int addValRecodeMap(HashMapToInt<T> map, int id, int i) {
		if(_n.get(i))
			id = _a.addValRecodeMap(map, id, i);
		return id;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder(_size + 2);
		sb.append(super.toString()).append(Opcodes.LESS.toString()).append(_a.getClass().getSimpleName()).append(">:[");
		for(int i = 0; i < _size - 1; i++)
			sb.append(get(i)).append(",");
		sb.append(get(_size - 1));
		sb.append("]");
		return sb.toString();
	}
}
