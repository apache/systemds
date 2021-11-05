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

import org.apache.commons.lang.ArrayUtils;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.util.SortUtils;

/**
 * Uncompressed representation of one or more columns in bitmap format.
 */
public final class MultiColBitmap extends ABitmap {

	/** Distinct tuples that appear in the columnGroup */
	private double[][] _values;

	protected MultiColBitmap(IntArrayList[] offsetsLists, double[][] values, int rows) {
		super(offsetsLists, rows);
		_values = values;
	}

	/**
	 * Get all values without unnecessary allocations and copies.
	 * 
	 * @return dictionary of value tuples
	 */
	public double[][] getValues() {
		return _values;
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

	public int getNumNonZerosInOffset(int idx) {
		int nz = 0;
		for(double v : getValues(idx))
			nz += v == 0 ? 0 : 1;

		return nz;
	}

	public int getNumValues() {
		return (_values == null) ? 0 : _values.length;
	}

	public void sortValuesByFrequency() {
		final int numVals = getNumValues();

		final double[] freq = new double[numVals];
		final int[] pos = new int[numVals];

		// populate the temporary arrays
		for(int i = 0; i < numVals; i++) {
			freq[i] = getNumOffsets(i);
			pos[i] = i;
		}

		// sort ascending and reverse (descending)
		SortUtils.sortByValue(0, numVals, freq, pos);
		ArrayUtils.reverse(pos);

		// create new value and offset list arrays
		double[][] lvalues = new double[numVals][];
		IntArrayList[] loffsets = new IntArrayList[numVals];
		for(int i = 0; i < numVals; i++) {
			lvalues[i] = _values[pos[i]];
			loffsets[i] = _offsetsLists[pos[i]];
		}
		_values = lvalues;
		_offsetsLists = loffsets;
	}

	@Override
	public int getNumColumns() {
		return _values[0].length;
	}

	@Override
	protected void addToString(StringBuilder sb) {
		sb.append("\nValues:");
		for(double[] vv : _values)
			sb.append("\n" + Arrays.toString(vv));
	}
}
