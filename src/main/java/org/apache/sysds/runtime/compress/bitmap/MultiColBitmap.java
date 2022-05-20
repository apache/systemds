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
public final class MultiColBitmap extends ABitmap {

	/** Distinct tuples that appear in the columnGroup */
	private double[][] _values;

	/**
	 * Multi column version of a Bitmap.
	 * 
	 * it should be guaranteed that the offsetLists are not null.
	 * 
	 * @param offsetsLists The offsets for the values
	 * @param values       The tuples matched with the offsets
	 * @param rows         The number of rows encoded
	 */
	protected MultiColBitmap(IntArrayList[] offsetsLists, double[][] values, int rows) {
		super(offsetsLists, rows);
		_values = values;
	}

	/**
	 * Obtain tuple of column values associated with index.
	 * 
	 * @param ix index of a particular distinct value
	 * @return the tuple of column values associated with the specified index
	 */
	public double[] getValues(int ix) {
		return _values[ix];
	}

	@Override
	public int getNumNonZerosInOffset(int idx) {
		int nz = 0;
		for(double v : getValues(idx))
			nz += v == 0 ? 0 : 1;
		return nz;
	}

	@Override
	public int getNumColumns() {
		// values are always guaranteed to be allocated
		return _values[0].length;
	}

	@Override
	protected void addToString(StringBuilder sb) {
		sb.append("\nValues:");
		for(double[] vv : _values)
			sb.append("\n" + Arrays.toString(vv));
	}
}
