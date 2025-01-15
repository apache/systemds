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

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.frame.data.compress.ArrayCompressionStatistics;

/**
 * A Compressed Array, in general does not allow us to set or modify the array.
 * 
 * In all cases of modification it throws an DMLCompressionException.
 */
public abstract class ACompressedArray<T> extends Array<T> {

	protected ACompressedArray(int size) {
		super(size);
	}

	@Override
	public Object get() {
		throw new DMLCompressionException("Invalid to call 'get' to access primitive array on CompressedArray");
	}

	@Override
	public void set(int index, T value) {
		throw new DMLCompressionException("Invalid to set value in CompressedArray");
	}

	@Override
	public void set(int index, double value) {
		throw new DMLCompressionException("Invalid to set value in CompressedArray");
	}

	@Override
	public void set(int index, String value) {
		throw new DMLCompressionException("Invalid to set value in CompressedArray");
	}

	@Override
	public void setFromOtherType(int rl, int ru, Array<?> value) {
		throw new DMLCompressionException("Invalid to set value in CompressedArray");
	}

	@Override
	public void setNz(int rl, int ru, Array<T> value) {
		throw new DMLCompressionException("Invalid to set value in CompressedArray");
	}

	@Override
	public void setFromOtherTypeNz(int rl, int ru, Array<?> value) {
		throw new DMLCompressionException("Invalid to set value in CompressedArray");
	}

	@Override
	public void append(String value) {
		throw new DMLCompressionException("Invalid to 'append' single values in CompressedArray");
	}

	@Override
	public void append(T value) {
		throw new DMLCompressionException("Invalid to 'append' single values in CompressedArray");
	}

	@Override
	public void fill(String val) {
		throw new DMLCompressionException("Unimplemented method 'fill'");
	}

	@Override
	public void fill(T val) {
		throw new DMLCompressionException("Unimplemented method 'fill'");
	}

	@Override
	public void reset(int size) {
		throw new DMLCompressionException("Invalid to reset compressed array");
	}

	@Override
	public abstract ArrayCompressionStatistics statistics(int nSamples);

	@Override
	public abstract Array<?> changeType(ValueType t);

	@Override
	protected Array<Boolean> changeTypeBitSet(Array<Boolean> ret, int l, int u) {
		throw new DMLCompressionException("Invalid to change sub compressed array");
	}

	@Override
	protected Array<Boolean> changeTypeBoolean(Array<Boolean> retA, int l, int u) {
		throw new DMLCompressionException("Invalid to change sub compressed array");
	}

	@Override
	protected Array<Double> changeTypeDouble(Array<Double> retA, int l, int u) {
		throw new DMLCompressionException("Invalid to change sub compressed array");
	}

	@Override
	protected Array<Float> changeTypeFloat(Array<Float> retA, int l, int u) {
		throw new DMLCompressionException("Invalid to change sub compressed array");
	}

	@Override
	protected Array<Integer> changeTypeInteger(Array<Integer> retA, int l, int u) {
		throw new DMLCompressionException("Invalid to change sub compressed array");
	}

	@Override
	protected Array<Long> changeTypeLong(Array<Long> retA, int l, int u) {
		throw new DMLCompressionException("Invalid to change sub compressed array");
	}

	@Override
	protected Array<String> changeTypeString(Array<String> retA, int l, int u) {
		throw new DMLCompressionException("Invalid to change sub compressed array");
	}

	@Override
	protected Array<Character> changeTypeCharacter(Array<Character> retA, int l, int u) {
		throw new DMLCompressionException("Invalid to change sub compressed array");
	}

	@Override
	protected Array<Object> changeTypeHash64(Array<Object> retA, int l, int u) {
		throw new DMLCompressionException("Invalid to change sub compressed array");
	}

	@Override
	protected Array<Object> changeTypeHash32(Array<Object> ret, int l, int u) {
		throw new DMLCompressionException("Invalid to change sub compressed array");
	}
}
