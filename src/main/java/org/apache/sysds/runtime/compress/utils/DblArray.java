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

/**
 * Helper class used for bitmap extraction.
 */
public class DblArray {
	private final double[] _arr;
	private int _hash;

	public DblArray(double[] arr) {
		_arr = arr;
	}

	private DblArray(double[] arr, int hash) {
		_arr = arr;
		_hash = hash;
	}

	public DblArray(DblArray that) {
		this(Arrays.copyOf(that._arr, that._arr.length), that.hashCode());
	}

	public double[] getData() {
		return _arr;
	}

	public void resetHash() {
		_hash = 0;
	}

	public boolean isEmpty() {
		if(_arr == null)
			return true;

		for(int i = 0; i < _arr.length; i++) {
			if(_arr[i] != 0)
				return false;
		}
		return true;
	}

	@Override
	public final int hashCode() {
		if(_hash != 0)
			return _hash;
		_hash = hashCode(_arr);
		return _hash;
	}

	private final int hashCode(final double[] arr) {
		int h = 1;
		for(double element : _arr) {
			long bits = Double.doubleToLongBits(element);
			h = 857 * h + (int) (bits ^ (bits >>> 32));
		}
		h ^= (h >>> 20) ^ (h >>> 12);
		h = h ^ (h >>> 7) ^ (h >>> 4);
		return h;
	}

	public final boolean equals(DblArray that) {
		if(hashCode() == that.hashCode()) {
			final double[] t = _arr;
			final double[] o = that._arr;
			for(int i = 0; i < t.length; i++)
				if(!Util.eq(t[i], o[i]))
					return false;
			return true;
		}
		return false;
	}

	@Override
	public boolean equals(Object o) {
		return o instanceof DblArray && this.equals((DblArray) o);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder(2 + _arr.length * 4);
		sb.append("[");
		sb.append(doubleToString(_arr[0]));
		for(int i = 1; i < _arr.length; i++) {
			sb.append(", ");
			sb.append(doubleToString(_arr[i]));
		}
		sb.append("]");
		return sb.toString();
	}

	private static String doubleToString(double v) {
		if(v == (long) v)
			return Long.toString(((long) v));
		else
			return Double.toString(v);
	}
}
