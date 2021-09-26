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

	protected ABitmap(IntArrayList[] offsetsLists, int rows) {
		int offsetsTotal = 0;
		if(offsetsLists != null) {
			for(IntArrayList a : offsetsLists)
				offsetsTotal += a.size();
			_numZeros = rows - offsetsTotal;
		}
		else
			_numZeros = rows;

		_offsetsLists = offsetsLists;
	}

	public final boolean isEmpty() {
		return _offsetsLists == null;
	}

	public final IntArrayList[] getOffsetList() {
		return _offsetsLists;
	}

	public final IntArrayList getOffsetsList(int idx) {
		return _offsetsLists[idx];
	}

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

	public final int getNumZeros(){
		return _numZeros;
	}

	public final boolean containsZero(){
		return _numZeros > 0;
	}

	/**
	 * Obtain number of distinct value groups in the column. this number is also the number of bitmaps, since there is
	 * one bitmap per value
	 * 
	 * @return number of distinct value groups in the column;
	 */
	public abstract int getNumValues();

	public abstract void sortValuesByFrequency();

	public abstract int getNumNonZerosInOffset(int idx);

	public abstract int getNumColumns();

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
