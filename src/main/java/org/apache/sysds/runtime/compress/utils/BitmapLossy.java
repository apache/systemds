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

import org.apache.commons.lang.NotImplementedException;

/**
 * Uncompressed but Quantized representation of contained data.
 */
public final class BitmapLossy extends ABitmap {

	/**
	 * Distinct values that appear in the column. Linearized as value groups <v11 v12> <v21 v22>.
	 */
	private final byte[] _values;
	private final double _scale;

	public BitmapLossy(int numCols, IntArrayList[] offsetsLists, int numZeroGroups, byte[] values, double scale) {
		super(numCols, offsetsLists, numZeroGroups);
		_values = values;
		_scale = scale;
	}


	/**
	 * Get all values without unnecessary allocations and copies.
	 * 
	 * @return dictionary of value tuples
	 */
	public byte[] getValues() {
		return _values;
	}

	/**
	 * Obtain tuple of column values associated with index.
	 * 
	 * @param ix index of a particular distinct value
	 * @return the tuple of column values associated with the specified index
	 */
	public byte[] getValues(int ix) {
		return Arrays.copyOfRange(_values, ix * _numCols, (ix + 1) * _numCols);
	}

	public double getScale() {
		return _scale;
	}

	/**
	 * Obtain number of distinct values in the column.
	 * 
	 * @return number of distinct values in the column; this number is also the number of bitmaps, since there is one
	 *         bitmap per value
	 */
	public int getNumValues() {
		return _values.length / _numCols;
	}

	public IntArrayList getOffsetsList(int ix) {
		return _offsetsLists[ix];
	}

	public long getNumOffsets() {
		long ret = 0;
		for(IntArrayList offlist : _offsetsLists)
			ret += offlist.size();
		return ret;
	}

	public int getNumOffsets(int ix) {
		return _offsetsLists[ix].size();
	}

	@Override
	public void sortValuesByFrequency() {
		throw new NotImplementedException("Not Implemented Sorting of Lossy Bit Map");
	}

	@Override
	public BitmapType getType() {
		return BitmapType.Lossy;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append("\nValues: " + Arrays.toString(_values));
		sb.append("\ncolumns:" + _numCols);
		sb.append("\nScale:  " + _scale);
		sb.append("\nOffsets:" + Arrays.toString(_offsetsLists));
		return sb.toString();
	}

}
