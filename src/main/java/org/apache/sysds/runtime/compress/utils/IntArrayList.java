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

package org.apache.sysds.runtime.compress.utils;

import java.util.Arrays;

import org.apache.sysds.runtime.compress.DMLCompressionException;

public class IntArrayList {
	private static final int INIT_CAPACITY = 4;
	private static final int RESIZE_FACTOR = 2;

	private int[] _data;
	private int _size;

	public IntArrayList() {
		this(INIT_CAPACITY);
	}

	public IntArrayList(int initialSize) {
		_data = new int[initialSize];
		_size = 0;
	}

	public IntArrayList(int[] values) {
		if(values == null)
			throw new DMLCompressionException("Invalid initialization of IntArrayList");
		_data = values;
		_size = values.length;
	}

	public int size() {
		return _size;
	}

	public void appendValue(int value) {
		// allocate or resize array if necessary
		if(_size + 1 > _data.length)
			resize();

		// append value
		_data[_size] = value;
		_size++;
	}

	public void appendValue(IntArrayList value) {
		// allocate or resize array if necessary
		if(_size + value._size >= _data.length)
			_data = Arrays.copyOf(_data, _size + value._size);
		System.arraycopy(value._data, 0, _data, _size, value._size);
		_size = _size + value._size;
	}

	/**
	 * Returns the underlying array of offsets. Note that this array might be physically larger than the actual length of
	 * the offset lists. Use size() to obtain the actual length.
	 * 
	 * @return integer array of offsets, the physical array length may be larger than the length of the offset list
	 */
	public int[] extractValues() {
		return _data;
	}

	public int get(int index) {
		return _data[index];
	}

	public int[] extractValues(boolean trim) {
		if(trim ){
			if(_data.length == _size)
				return _data;
			return Arrays.copyOfRange(_data, 0, _size);
		}
		else
			return _data;
	}

	private void resize() {

		// resize data array and copy existing contents
		_data = Arrays.copyOf(_data, _data.length * RESIZE_FACTOR);
	}

	public void reset() {
		_size = 0;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		if(_size == 0)
			return "[]";
		sb.append("[");
		int i = 0;
		for(; i < _size - 1; i++)
			sb.append(_data[i]).append(", ");
		sb.append(_data[i]);
		sb.append("]");

		return sb.toString();
	}
}
