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

public class IntArrayList {
	private static final int INIT_CAPACITY = 4;
	private static final int RESIZE_FACTOR = 2;

	private int[] _data = null;
	private int _size;

	public IntArrayList() {
		_data = null;
		_size = 0;
	}

	public IntArrayList(int initialSize) {
		_data = new int[initialSize];
		_size = 0;
	}

	public IntArrayList(int[] values) {
		_data = values;
		_size = values.length;
	}

	public int size() {
		return _size;
	}

	public void appendValue(int value) {
		// allocate or resize array if necessary
		if(_data == null) {
			_data = new int[INIT_CAPACITY];
		}
		else if(_size + 1 >= _data.length)
			resize();

		// append value
		_data[_size] = value;
		_size++;
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
		if(_data != null)
			return _data[index];
		else
			throw new RuntimeException("invalid index to get");
	}

	public int[] extractValues(boolean trim) {
		int[] ret = extractValues();
		return (trim && _size < ret.length) ? Arrays.copyOfRange(ret, 0, _size) : ret;
	}

	private void resize() {
		// check for integer overflow on resize
		if(_data.length > Integer.MAX_VALUE / RESIZE_FACTOR)
			throw new RuntimeException("IntArrayList resize leads to integer overflow: size=" + _size);

		// resize data array and copy existing contents
		_data = Arrays.copyOf(_data, _data.length * RESIZE_FACTOR);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();

		sb.append("[");
		int i = 0;
		for(; i < _size - 1; i++)
			sb.append(_data[i] + ",");

		sb.append(_data[i] + "]");

		return sb.toString();
	}
}
