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
 * Helper class used for bitmap extraction.
 *
 */
public class DblArray 
{
	private double[] _arr = null;
	private boolean _zero = false;
	
	public DblArray() {
		this(null, false);
	}
	
	public DblArray(double[] arr) {
		this(arr, false);
	}
	
	public DblArray(DblArray that) {
		this(Arrays.copyOf(that._arr, that._arr.length), that._zero);
	}

	public DblArray(double[] arr, boolean allZeros) {
		_arr = arr;
		_zero = allZeros;
	}
	
	public double[] getData() {
		return _arr;
	}
	
	@Override
	public int hashCode() {
		return _zero ? 0 : Arrays.hashCode(_arr);
	}

	@Override
	public boolean equals(Object o) {
		return ( o instanceof DblArray
			&& _zero == ((DblArray) o)._zero
			&& Arrays.equals(_arr, ((DblArray) o)._arr) );
	}

	@Override
	public String toString() {
		return Arrays.toString(_arr);
	}

	/**
	 * 
	 * @param ds
	 * @return
	 */
	public static boolean isZero(double[] ds) {
		for (int i = 0; i < ds.length; i++)
			if (ds[i] != 0.0)
				return false;
		return true;
	}

	/**
	 * 
	 * @param val
	 * @return
	 */
	public static boolean isZero(DblArray val) {
		return val._zero || isZero(val._arr);
	}
}
