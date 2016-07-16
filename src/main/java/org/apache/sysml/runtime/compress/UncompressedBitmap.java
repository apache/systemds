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

package org.apache.sysml.runtime.compress;

import java.util.Arrays;

import org.apache.sysml.runtime.compress.utils.DblArrayIntListHashMap;
import org.apache.sysml.runtime.compress.utils.DoubleIntListHashMap;
import org.apache.sysml.runtime.compress.utils.DblArrayIntListHashMap.DArrayIListEntry;
import org.apache.sysml.runtime.compress.utils.DoubleIntListHashMap.DIListEntry;

/** 
 * Uncompressed representation of one or more columns in bitmap format. 
 * 
 */
public final class UncompressedBitmap 
{
	private int _numCols;

	/** Distinct values that appear in the column. Linearized as value groups <v11 v12> <v21 v22>.*/
	private double[] _values;

	/** Bitmaps (as lists of offsets) for each of the values. */
	private int[][] _offsetsLists;

	public UncompressedBitmap( DblArrayIntListHashMap distinctVals, int numColumns ) 
	{
		// added for one pass bitmap construction
		// Convert inputs to arrays
		int numVals = distinctVals.size();
		_values = new double[numVals*numColumns];
		_offsetsLists = new int[numVals][];
		int bitmapIx = 0;
		for( DArrayIListEntry val : distinctVals.extractValues()) {
			System.arraycopy(val.key.getData(), 0, _values, bitmapIx*numColumns, numColumns);
			_offsetsLists[bitmapIx++] = val.value.extractValues();
		}
		_numCols = numColumns;
	}

	public UncompressedBitmap( DoubleIntListHashMap distinctVals ) 
	{
		// added for one pass bitmap construction
		// Convert inputs to arrays
		int numVals = distinctVals.size();
		_values = new double[numVals];
		_offsetsLists = new int[numVals][];
		int bitmapIx = 0;
		for(DIListEntry val : distinctVals.extractValues()) {
			_values[bitmapIx] = val.key;
			_offsetsLists[bitmapIx++] = val.value.extractValues();
		}
		_numCols = 1;
	}
	
	public int getNumColumns() {
		return _numCols;
	}

	/**
	 * @param ix   index of a particular distinct value
	 * @return the tuple of column values associated with the specified index
	 */
	public double[] getValues(int ix) {
		return Arrays.copyOfRange(_values, ix*_numCols, (ix+1)*_numCols);
	}

	/**
	 * @return number of distinct values in the column; this number is also the
	 *         number of bitmaps, since there is one bitmap per value
	 */
	public int getNumValues() {
		return _values.length / _numCols;
	}

	/**
	 * @param ix   index of a particular distinct value
	 * @return IMMUTABLE array of the offsets of the rows containing the value
	 *         with the indicated index
	 */
	public int[] getOffsetsList(int ix) {
		return _offsetsLists[ix];
	}
}
