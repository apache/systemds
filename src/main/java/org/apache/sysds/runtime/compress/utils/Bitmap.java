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

import org.apache.commons.lang.ArrayUtils;
import org.apache.sysds.runtime.util.SortUtils;

/**
 * Uncompressed representation of one or more columns in bitmap format.
 */
public final class Bitmap extends ABitmap {

	/**
	 * Distinct values that appear in the column. Linearized as value groups <v11 v12> <v21 v22>.
	 */
	private double[] _values;

	public Bitmap(int numCols, IntArrayList[] offsetsLists, int numZeroGroups, double[] values) {
		super(numCols, offsetsLists, numZeroGroups);
		_values = values;
	}

	/**
	 * Get all values without unnecessary allocations and copies.
	 * 
	 * @return dictionary of value tuples
	 */
	public double[] getValues() {
		return _values;
	}

	/**
	 * Obtain tuple of column values associated with index.
	 * 
	 * @param ix index of a particular distinct value
	 * @return the tuple of column values associated with the specified index
	 */
	public double[] getValues(int ix) {
		return Arrays.copyOfRange(_values, ix * _numCols, (ix + 1) * _numCols);
	}

	public int getNumValues() {
		return _values.length / _numCols;
	}

	public void sortValuesByFrequency() {
		int numVals = getNumValues();
		int numCols = getNumColumns();

		double[] freq = new double[numVals];
		int[] pos = new int[numVals];

		// populate the temporary arrays
		for(int i = 0; i < numVals; i++) {
			freq[i] = getNumOffsets(i);
			pos[i] = i;
		}

		// sort ascending and reverse (descending)
		SortUtils.sortByValue(0, numVals, freq, pos);
		ArrayUtils.reverse(pos);

		// create new value and offset list arrays
		double[] lvalues = new double[numVals * numCols];
		IntArrayList[] loffsets = new IntArrayList[numVals];
		for(int i = 0; i < numVals; i++) {
			System.arraycopy(_values, pos[i] * numCols, lvalues, i * numCols, numCols);
			loffsets[i] = _offsetsLists[pos[i]];
		}
		_values = lvalues;
		_offsetsLists = loffsets;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append("\nValues: " + Arrays.toString(_values));
		return sb.toString();
	}

	@Override
	public BitmapType getType() {
		return BitmapType.Full;
	}
}
