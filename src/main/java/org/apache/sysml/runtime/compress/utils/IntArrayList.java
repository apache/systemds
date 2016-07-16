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

package org.apache.sysml.runtime.compress.utils;

import java.util.Arrays;

/**
 * This class provides a memory-efficient replacement for ArrayList<Integer> for
 * restricted use cases.
 * 
 */
public class IntArrayList 
{
	private static final int INIT_CAPACITY = 4;
	private static final int RESIZE_FACTOR = 2;

	private int[] _data = null;
	private int _size = -1;
	private int _val0 = -1;

	public IntArrayList() {
		_data = null;
		_size = 0;
	}

	/**
	 * 
	 * @return
	 */
	public int size() {
		return _size;
	}

	/**
	 * 
	 * @param value
	 */
	public void appendValue(int value) {
		// embedded value (no array allocation)
		if( _size == 0 ) {
			_val0 = value;
			_size = 1;
			return;
		}

		// allocate or resize array if necessary
		if( _data == null ) {
			_data = new int[INIT_CAPACITY];
			_data[0] = _val0;
		} 
		else if( _size + 1 >= _data.length ) {
			resize();
		}

		// append value
		_data[_size] = value;
		_size++;
	}

	/**
	 * 
	 * @return
	 */
	public int[] extractValues() {
		if( _size == 1 )
			return new int[] { _val0 };
		else
			return Arrays.copyOfRange(_data, 0, _size);
	}

	/**
	 * 
	 */
	private void resize() {
		// check for integer overflow on resize
		if( _data.length > Integer.MAX_VALUE / RESIZE_FACTOR )
			throw new RuntimeException(
					"IntArrayList resize leads to integer overflow: size=" + _size);

		// resize data array and copy existing contents
		int[] newdata = new int[_data.length * RESIZE_FACTOR];
		System.arraycopy(_data, 0, newdata, 0, _size);
		_data = newdata;
	}
}
