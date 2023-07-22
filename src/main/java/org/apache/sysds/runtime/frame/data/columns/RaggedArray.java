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
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;
import org.apache.sysds.runtime.matrix.data.Pair;

/**
 * A Ragged array for the columns contains a smaller array, only containing the values of the top most part of the
 * column.
 * 
 * This makes the allocation much better in cases where only the top n rows of a m row frame is used for the specific
 * column. It is typically used for instances of transform encode, where the transform encode return a metadata frame to
 * enable encoding and decoding the matrix
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
		throw new NotImplementedException();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		throw new NotImplementedException("Unimplemented method 'write'");
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		throw new NotImplementedException("Unimplemented method 'readFields'");
	}

	@Override
	public T get(int index) {
		throw new NotImplementedException("Unimplemented method 'get'");
	}

	@Override
	public Object get() {
		throw new NotImplementedException("Unimplemented method 'get'");
	}

	@Override
	public double getAsDouble(int i) {
		throw new NotImplementedException("Unimplemented method 'getAsDouble'");
	}

	@Override
	public void set(int index, T value) {
		throw new NotImplementedException("Unimplemented method 'set'");
	}

	@Override
	public void set(int index, double value) {
		throw new NotImplementedException("Unimplemented method 'set'");
	}

	@Override
	public void set(int index, String value) {
		throw new NotImplementedException("Unimplemented method 'set'");
	}

	@Override
	public void setFromOtherType(int rl, int ru, Array<?> value) {
		throw new NotImplementedException("Unimplemented method 'setFromOtherType'");
	}

	@Override
	public void set(int rl, int ru, Array<T> value) {
		throw new NotImplementedException("Unimplemented method 'set'");
	}

	@Override
	public void set(int rl, int ru, Array<T> value, int rlSrc) {
		throw new NotImplementedException("Unimplemented method 'set'");
	}

	@Override
	public void setNz(int rl, int ru, Array<T> value) {
		throw new NotImplementedException("Unimplemented method 'setNz'");
	}

	@Override
	public void setFromOtherTypeNz(int rl, int ru, Array<?> value) {
		throw new NotImplementedException("Unimplemented method 'setFromOtherTypeNz'");
	}

	@Override
	public void append(String value) {
		throw new NotImplementedException("Unimplemented method 'append'");
	}

	@Override
	public void append(T value) {
		throw new NotImplementedException("Unimplemented method 'append'");
	}

	@Override
	public Array<T> append(Array<T> other) {
		throw new NotImplementedException("Unimplemented method 'append'");
	}

	@Override
	public Array<T> slice(int rl, int ru) {
		throw new NotImplementedException("Unimplemented method 'slice'");
	}

	@Override
	public void reset(int size) {
		throw new NotImplementedException("Unimplemented method 'reset'");
	}

	@Override
	public byte[] getAsByteArray() {
		throw new NotImplementedException("Unimplemented method 'getAsByteArray'");
	}

	@Override
	public ValueType getValueType() {
		throw new NotImplementedException("Unimplemented method 'getValueType'");
	}

	@Override
	public Pair<ValueType, Boolean> analyzeValueType() {
		throw new NotImplementedException("Unimplemented method 'analyzeValueType'");
	}

	@Override
	public FrameArrayType getFrameArrayType() {
		throw new NotImplementedException("Unimplemented method 'getFrameArrayType'");
	}

	@Override
	public long getExactSerializedSize() {
		throw new NotImplementedException("Unimplemented method 'getExactSerializedSize'");
	}

	@Override
	protected Array<Boolean> changeTypeBitSet() {
		throw new NotImplementedException("Unimplemented method 'changeTypeBitSet'");
	}

	@Override
	protected Array<Boolean> changeTypeBoolean() {
		throw new NotImplementedException("Unimplemented method 'changeTypeBoolean'");
	}

	@Override
	protected Array<Double> changeTypeDouble() {
		throw new NotImplementedException("Unimplemented method 'changeTypeDouble'");
	}

	@Override
	protected Array<Float> changeTypeFloat() {
		throw new NotImplementedException("Unimplemented method 'changeTypeFloat'");
	}

	@Override
	protected Array<Integer> changeTypeInteger() {
		throw new NotImplementedException("Unimplemented method 'changeTypeInteger'");
	}

	@Override
	protected Array<Long> changeTypeLong() {
		throw new NotImplementedException("Unimplemented method 'changeTypeLong'");
	}

	@Override
	protected Array<String> changeTypeString() {
		throw new NotImplementedException("Unimplemented method 'changeTypeString'");
	}

	@Override
	protected Array<Character> changeTypeCharacter() {
		throw new NotImplementedException("Unimplemented method 'changeTypeCharacter'");
	}

	@Override
	public void fill(String val) {
		throw new NotImplementedException("Unimplemented method 'fill'");
	}

	@Override
	public void fill(T val) {
		throw new NotImplementedException("Unimplemented method 'fill'");
	}

	@Override
	public boolean isShallowSerialize() {
		throw new NotImplementedException("Unimplemented method 'isShallowSerialize'");
	}

	@Override
	public boolean isEmpty() {
		throw new NotImplementedException("Unimplemented method 'isEmpty'");
	}

	@Override
	public Array<T> select(int[] indices) {
		throw new NotImplementedException("Unimplemented method 'select'");
	}

	@Override
	public Array<T> select(boolean[] select, int nTrue) {
		throw new NotImplementedException("Unimplemented method 'select'");
	}

	@Override
	public boolean isNotEmpty(int i) {
		throw new NotImplementedException("Unimplemented method 'isNotEmpty'");
	}

	@Override
	public Array<T> clone() {
		throw new NotImplementedException("Unimplemented method 'clone'");
	}

	@Override
	public double hashDouble(int idx) {
		throw new NotImplementedException("Unimplemented method 'hashDouble'");
	}

	@Override
	public boolean equals(Array<T> other) {
		throw new NotImplementedException("Unimplemented method 'equals'");
	}

}
