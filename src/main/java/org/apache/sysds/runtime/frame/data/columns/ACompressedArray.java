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

import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.frame.data.compress.ArrayCompressionStatistics;

/**
 * A Compressed Array, in general does not allow us to set or modify the array.
 * 
 * In all cases of modification it throws an DMLCompressionException.
 */
public abstract class ACompressedArray<T> extends Array<T> {

	public ACompressedArray(int size) {
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
	public void set(int rl, int ru, Array<T> value) {
		throw new DMLCompressionException("Invalid to set value in CompressedArray");
	}

	@Override
	public void set(int rl, int ru, Array<T> value, int rlSrc) {
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
	public ArrayCompressionStatistics statistics(int nSamples) {
		// already compressed
		return null;
	}

}
