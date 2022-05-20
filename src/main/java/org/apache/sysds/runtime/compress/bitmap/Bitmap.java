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

package org.apache.sysds.runtime.compress.bitmap;

import java.util.Arrays;

import org.apache.sysds.runtime.compress.utils.IntArrayList;

/**
 * Uncompressed representation of one or more columns in bitmap format.
 */
public final class Bitmap extends ABitmap {

	/** Distinct values contained in the bitmap */
	private double[] _values;

	/**
	 * Single column version of a bitmap.
	 * 
	 * it should be guaranteed that the offsetLists are not null.
	 * 
	 * @param offsetsLists The offsets for the values
	 * @param values       The values matched with the offsets
	 * @param rows         The number of rows encoded
	 */
	protected Bitmap(IntArrayList[] offsetsLists, double[] values, int rows) {
		super(offsetsLists, rows);
		_values = values;
	}

	/**
	 * Get all values without unnecessary allocations and copies.
	 * 
	 * @return Dictionary of distinct values
	 */
	public final double[] getValues() {
		return _values;
	}

	@Override
	public final int getNumNonZerosInOffset(int idx) {
		// all values are non zero therefore since this bitmap type contains one column
		// always return 1.
		return 1;
	}

	@Override
	public int getNumColumns() {
		return 1;
	}

	@Override
	protected void addToString(StringBuilder sb) {
		sb.append("\nValues: " + Arrays.toString(_values));
	}
}
