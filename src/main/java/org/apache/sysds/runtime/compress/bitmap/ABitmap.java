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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.utils.IntArrayList;

public abstract class ABitmap {
	protected static final Log LOG = LogFactory.getLog(ABitmap.class.getName());

	/** Bitmaps (as lists of offsets) for each of the values. */
	protected IntArrayList[] _offsetsLists;

	/** int specifying the number of zero value tuples contained in the rows. */
	protected final int _numZeros;

	/**
	 * Main constructor of bitMap, it should be guaranteed that the offsetLists are not null.
	 * 
	 * @param offsetsLists The offsets to the values
	 * @param rows         The number of rows encoded
	 */
	protected ABitmap(IntArrayList[] offsetsLists, int rows) {
		int offsetsTotal = 0;
		for(IntArrayList a : offsetsLists)
			offsetsTotal += a.size();
		_numZeros = rows - offsetsTotal;

		_offsetsLists = offsetsLists;
	}

	/**
	 * Get all the offset lists.
	 * 
	 * @return the contained offset lists
	 */
	public final IntArrayList[] getOffsetList() {
		return _offsetsLists;
	}

	/**
	 * Get a specific offset list.
	 * 
	 * @param idx The index to look at inside the contained array
	 * @return the Offset list at the index
	 */
	public final IntArrayList getOffsetsList(int idx) {
		return _offsetsLists[idx];
	}

	/**
	 * Get the sum of offsets contained.
	 * 
	 * @return The sum of offsets
	 */
	public final long getNumOffsets() {
		long ret = 0;
		for(IntArrayList off : _offsetsLists)
			ret += off.size();
		return ret;
	}

	/**
	 * Get the number of offsets for a specific unique offset.
	 * 
	 * @param ix The offset index.
	 * @return The number of offsets for this unique value.
	 */
	public final int getNumOffsets(int ix) {
		return _offsetsLists[ix].size();
	}

	/**
	 * Get the number of zero tuples contained in this bitmap.
	 * 
	 * @return The number of zero tuples
	 */
	public final int getNumZeros() {
		return _numZeros;
	}

	/**
	 * Find out if the map contains zeros.
	 * 
	 * @return A boolean specifying if the bitmap contains zero offsets
	 */
	public final boolean containsZero() {
		return _numZeros > 0;
	}

	/**
	 * Obtain number of distinct value groups in the column. this number is also the number of bitmaps, since there is
	 * one bitmap per value
	 * 
	 * @return number of distinct value groups in the column;
	 */
	public final int getNumValues(){
		return _offsetsLists.length;
	}

	/**
	 * Get the number of non zeros in a specific offset's tuple value.
	 * 
	 * @param idx The offset index to look at.
	 * @return The nnz in the tuple.
	 */
	public abstract int getNumNonZerosInOffset(int idx);

	/**
	 * Get the number of columns encoded in this bitmap
	 * 
	 * @return The column count
	 */
	public abstract int getNumColumns();

	/**
	 * Internal function to construct a string representation of the bitmap.
	 * 
	 * @param sb The string builder to append subclasses information to.
	 */
	protected abstract void addToString(StringBuilder sb);

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append("  zeros:  " + _numZeros);
		sb.append("\nOffsets:" + Arrays.toString(_offsetsLists));
		addToString(sb);
		return sb.toString();
	}
}
