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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;
import org.apache.sysds.runtime.matrix.data.Pair;

/**
 * A Ragged array for a single column contains a smaller array, only containing the values of the top most part of the
 * 
 * This makes the allocation much better in cases where only the top n rows of a m row frame are used for the specific
 * column. It is typically used for instances of transform encode, where the transform encode return a metadata frame to
 * enable encoding and decoding the matrix.
 */
public class RaggedArray<T> extends Array<T> {

	/** Underlying values in general shorter than expected. */
	protected Array<T> _a;

	/**
	 * The allocation where, a's length is shorter than m, and we handle all accesses above len(a) as null.
	 * 
	 * @param a The underlying array that is shorter than length m
	 * @param m The overall supported length m
	 */
	public RaggedArray(T[] a, int m) {
		super(m);
		this._a = ArrayFactory.create(a);
	}

	/**
	 * The allocation where, a's length is shorter than m, and we handle all accesses above len(a) as null.
	 * 
	 * @param a The underlying array that is shorter than length m
	 * @param m The overall supported length m
	 */
	public RaggedArray(Array<T> a, int m) {
		super(m);
		this._a = a;
	}

	protected Array<T> getInnerArray() {
		return _a;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(FrameArrayType.RAGGED.ordinal());
		out.writeInt(_size);
		out.writeInt(_a.size());
		_a.write(out);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		throw new DMLRuntimeException("Should not be called");
	}

	protected static RaggedArray<?> read(DataInput in, int nRow) throws IOException {
		int m = in.readInt();
		final Array<?> a = ArrayFactory.read(in, in.readInt());
		return new RaggedArray<>(a, m);
	}

	@Override
	public T get(int index) {
		if(index > _size || index < 0)
			throw new ArrayIndexOutOfBoundsException("Index " + index + " out of bounds " + _size);
		return index < _a._size ? _a.get(index) : null;
	}

	@Override
	public Object get() {
		throw new NotImplementedException("Should not be called");
	}

	@Override
	public double getAsDouble(int i) {
		return i < _a._size ? _a.getAsDouble(i) : 0;
	}

	@Override
	public double getAsNaNDouble(int i) {
		return i < _a._size ? _a.getAsNaNDouble(i) : Double.NaN;
	}

	@Override
	public void set(int index, T value) {
		if(index < _a._size)
			_a.set(index, value);
		else if(index < super.size()) {
			_a.reset(index + 1);
			_a.set(index, value);
			LOG.warn("Reallocated ragged array");
		}
	}

	@Override
	public void set(int index, double value) {
		if(index < _a._size)
			_a.set(index, value);
		else if(index < super.size()) {
			_a.reset(index + 1);
			_a.set(index, value);
			LOG.warn("Reallocated ragged array");
		}
	}

	@Override
	public void set(int index, String value) {
		if(index < _a._size)
			_a.set(index, value);
		else if(index < super.size()) {
			_a.reset(index + 1);
			_a.set(index, value);
			LOG.warn("Reallocated ragged array");
		}
	}

	@Override
	public void setFromOtherType(int rl, int ru, Array<?> value) {
		if(rl >= 0 && rl < _a._size && ru < _a._size)
			_a.setFromOtherType(rl, ru, value);
		else
			throw new NotImplementedException("Unimplemented method 'setFromOtherType'");
	}


	@Override
	public void set(int rl, int ru, Array<T> value, int rlSrc) {
		if(rl >= 0 && rl < _a._size && ru < _a._size)
			if(value instanceof RaggedArray)
				_a.set(rl, ru, ((RaggedArray<T>) value).getInnerArray(), rlSrc);
			else if(_a.getClass() == value.getClass())
				_a.set(rl, ru, value, rlSrc);
			else
				throw new DMLRuntimeException(
					"RaggedArray set: value type should be same to RaggedArray type " + _a.getClass());
	}

	@Override
	public void setNz(int rl, int ru, Array<T> value) {
		if(rl >= 0 && rl < _a._size && ru < _a._size)
			_a.setNz(rl, ru, value);
		else
			throw new NotImplementedException();
	}

	@Override
	public void setFromOtherTypeNz(int rl, int ru, Array<?> value) {
		if(rl >= 0 && rl < _a._size && ru < _a._size)
			_a.setFromOtherTypeNz(rl, ru, value);
		else
			throw new NotImplementedException();
	}

	@Override
	public void append(String value) {
		Array<T> oldVals = _a.clone();
		_a.reset(super.size() + 1);
		_a.set(0, oldVals.size() - 1, oldVals);
		_a.set(super.size(), value);
		super._size += 1;

		LOG.warn("Fully allocated ragged array");
	}

	@Override
	public void append(T value) {
		Array<T> oldVals = _a.clone();
		_a.reset(super.size() + 1);
		_a.set(0, oldVals.size() - 1, oldVals);
		_a.set(super.size(), value);
		super._size += 1;

		LOG.warn("Fully allocated ragged array");
	}

	@Override
	public Array<T> append(Array<T> other) {
		Array<T> oldVals = _a.clone();
		_a.reset(super.size() + other._size + 1);
		_a.set(0, oldVals.size() - 1, oldVals);
		_a.set(super.size(), super.size() + other.size() - 1, other);
		super._size += other.size();

		LOG.warn("Fully allocated ragged array");

		return this;
	}

	@Override
	public Array<T> slice(int rl, int ru) {
		if(rl >= 0 && rl < _a._size && ru < _a._size)
			return _a.slice(rl, ru);
		else if(rl >= 0 && ru >= _a._size)
			return _a.slice(rl, _a._size - 1);
		return null;
	}

	@Override
	public void reset(int size) {
		_a.reset(size);
		super._size = size;
	}

	@Override
	public byte[] getAsByteArray() {
		throw new NotImplementedException("Unimplemented method 'getAsByteArray'");
	}

	@Override
	public ValueType getValueType() {
		return _a.getValueType();
	}

	@Override
	public Pair<ValueType, Boolean> analyzeValueType(int maxCells) {
		return _a.analyzeValueType(maxCells);
	}

	@Override
	public FrameArrayType getFrameArrayType() {
		return FrameArrayType.RAGGED;
	}

	@Override
	public long getExactSerializedSize() {
		return _a.getExactSerializedSize() + 8 + 1;
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
	protected Array<String> changeTypeString(Array<String> retA, int l, int u) {
		return _a.changeTypeString(retA, l, u);
	}

	@Override
	protected Array<Character> changeTypeCharacter(Array<Character> retA, int l, int u) {
		return _a.changeTypeCharacter(retA, l, u);
	}

	@Override
	public Array<?> changeTypeWithNulls(ValueType t) {
		throw new NotImplementedException("Not Implemented ragged array with nulls");
	}

	@Override
	public void fill(String val) {
		_a.reset(super.size());
		_a.fill(val);
	}

	@Override
	public void fill(T val) {
		_a.reset(super.size());
		_a.fill(val);
	}

	@Override
	public boolean isShallowSerialize() {
		return _a.isShallowSerialize();
	}

	@Override
	public boolean isEmpty() {
		return _a.isEmpty();
	}

	@Override
	@SuppressWarnings("unchecked")
	public Array<T> select(int[] indices) {
		Array<T> ret = _a.getFrameArrayType() == FrameArrayType.OPTIONAL ? //
			(Array<T>) ArrayFactory.allocateOptional(_a.getValueType(), indices.length) : //
			(Array<T>) ArrayFactory.allocate(_a.getValueType(), indices.length);
		for(int i = 0; i < indices.length; i++)
			ret.set(i, get(indices[i]));
		return ret;
	}

	@Override
	@SuppressWarnings("unchecked")
	public Array<T> select(boolean[] select, int nTrue) {
		Array<T> ret = _a.getFrameArrayType() == FrameArrayType.OPTIONAL ? //
			(Array<T>) ArrayFactory.allocateOptional(_a.getValueType(), nTrue) : //
			(Array<T>) ArrayFactory.allocate(_a.getValueType(), nTrue);
		int k = 0;
		for(int i = 0; i < _a.size(); i++) {
			if(select[i])
				ret.set(k++, _a.get(i));
		}

		for(int i = _a.size(); i < select.length; i++) {
			if(select[i])
				ret.set(k++, get(i));
		}

		return ret;
	}

	@Override
	public boolean isNotEmpty(int i) {
		return i < _a.size() && _a.isNotEmpty(i);
	}

	@Override
	public Array<T> clone() {
		return new RaggedArray<>(_a.clone(), super._size);
	}

	@Override
	public double hashDouble(int idx) {
		return idx < _a.size() ? _a.hashDouble(idx) : Double.NaN;
	}

	@Override
	public boolean equals(Array<T> other) {
		if(other._size == this._size && //
			other.getValueType() == this.getValueType() && //
			other instanceof RaggedArray) {
			if(other == this) {// same pointer
				return true;
			}
			RaggedArray<T> ot = (RaggedArray<T>) other;
			return ot._a.equals(this._a);
		}
		return false;
	}

	@Override
	public long getInMemorySize() {
		return baseMemoryCost() + _a.getInMemorySize() + 8;
	}

	@Override
	public boolean containsNull() {
		return (_a.size() < super._size) || _a.containsNull();
	}

	@Override
	public boolean possiblyContainsNaN() {
		return true;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder(_size + 2);
		sb.append(super.toString()).append(Opcodes.LESS.toString());
		sb.append(_a.getClass().getSimpleName()).append(">:[");
		for(int i = 0; i < _size - 1; i++)
			sb.append(get(i)).append(",");
		sb.append(get(_size - 1));
		sb.append("]");
		return sb.toString();
	}

}
