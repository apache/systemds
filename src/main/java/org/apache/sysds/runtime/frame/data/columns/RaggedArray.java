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

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;
import org.apache.sysds.runtime.matrix.data.Pair;

/**
 * A Ragged array for a single column contains a smaller array, only containing the values of the top most part of the
 * column.
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

	@Override
	public void write(DataOutput out) throws IOException {
		_a.write(out);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		_a.readFields(in);
	}

	@Override
	public T get(int index) {
		return index < _a._size ? _a.get(index) : null;
	}

	@Override
	public Object get() {
		throw new NotImplementedException("Unimplemented method Object 'get'");
	}

	@Override
	public double getAsDouble(int i) {
		return i < _a._size ? _a.getAsDouble(i) : Double.NaN;
	}

	@Override
	public void set(int index, T value) {
		if (index < _a._size)
			_a.set(index, value);
	}

	@Override
	public void set(int index, double value) {
		if (index < _a._size)
			_a.set(index, value);
	}

	@Override
	public void set(int index, String value) {
		if (index < _a._size)
			_a.set(index, value);
	}

	@Override
	public void setFromOtherType(int rl, int ru, Array<?> value) {
		if(rl >= 0 && rl < _a._size && ru < _a._size)
			_a.setFromOtherType(rl, ru, value);
	}

	@Override
	public void set(int rl, int ru, Array<T> value) {
		if(rl >= 0 && rl < _a._size && ru < _a._size)
			_a.set(rl, ru, value);
	}

	@Override
	public void set(int rl, int ru, Array<T> value, int rlSrc) {
		if(rl >= 0 && rlSrc >= 0 && rl < _a._size && ru < _a._size)
			_a.set(rl, ru, value);
	}

	@Override
	public void setNz(int rl, int ru, Array<T> value) {
		if(rl >= 0 && rl < _a._size && ru < _a._size)
			_a.setNz(rl, ru, value);
	}

	@Override
	public void setFromOtherTypeNz(int rl, int ru, Array<?> value) {
		if(rl >= 0 && rl < _a._size && ru < _a._size)
			_a.setFromOtherTypeNz(rl, ru, value);
	}

	@Override
	public void append(String value) {
		_a.append(value);
	}

	@Override
	public void append(T value) {
		_a.append(value);
	}

	@Override
	public Array<T> append(Array<T> other) {
		return _a.append(other);
	}

	@Override
	public Array<T> slice(int rl, int ru) {
		if(rl >= 0 && rl < _a._size && ru < _a._size)
			return _a.slice(rl, ru);
		else if(rl >= 0 && ru >= _a._size )
			return _a.slice(rl, _a._size - 1);
		return null;
	}

	@Override
	public void reset(int size) {
		_a.reset(size);
	}

	@Override
	public byte[] getAsByteArray() {
		return _a.getAsByteArray();
	}

	@Override
	public ValueType getValueType() {
		return _a.getValueType();
	}

	@Override
	public Pair<ValueType, Boolean> analyzeValueType() {
		return _a.analyzeValueType();
	}

	@Override
	public FrameArrayType getFrameArrayType() {
		return _a.getFrameArrayType();
	}

	@Override
	public long getExactSerializedSize() {
		return _a.getExactSerializedSize();
	}

	@Override
	protected Array<Boolean> changeTypeBitSet() {
		return _a.changeTypeBitSet();
	}

	@Override
	protected Array<Boolean> changeTypeBoolean() {
		return _a.changeTypeBoolean();
	}

	@Override
	protected Array<Double> changeTypeDouble() {
		return _a.changeTypeDouble();
	}

	@Override
	protected Array<Float> changeTypeFloat() {
		return _a.changeTypeFloat();
	}

	@Override
	protected Array<Integer> changeTypeInteger() {
		return _a.changeTypeInteger();
	}

	@Override
	protected Array<Long> changeTypeLong() {
		return _a.changeTypeLong();
	}

	@Override
	protected Array<String> changeTypeString() {
		return _a.changeTypeString();
	}

	@Override
	protected Array<Character> changeTypeCharacter() {
		return _a.changeTypeCharacter();
	}

	@Override
	public void fill(String val) {
		_a.fill(val);
	}

	@Override
	public void fill(T val) {
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
	public Array<T> select(int[] indices) {
		return _a.select(Arrays.stream(indices).filter(x -> x < _size && x >= 0).toArray());
	}

	@Override
	public Array<T> select(boolean[] select, int nTrue) {
		T[] ret = (T[]) new Object[nTrue];
		int k = 0;
		for(int i = 0; i < Math.max(select.length, _a.size()); i++) {
			ret[k++] = _a.get(i);
		}
		return ArrayFactory.create(ret);
	}

	@Override
	public boolean isNotEmpty(int i) {
		return i < _a.size() && _a.isNotEmpty(i);
	}

	@Override
	public Array<T> clone() {
		return _a.clone();
	}

	@Override
	public double hashDouble(int idx) {
		return idx < _a.size() ? _a.hashDouble(idx): Double.NaN;
	}

}
