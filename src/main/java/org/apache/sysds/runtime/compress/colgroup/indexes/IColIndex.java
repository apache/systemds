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

package org.apache.sysds.runtime.compress.colgroup.indexes;

import java.io.DataOutput;
import java.io.IOException;

/**
 * Class to contain column indexes for the compression column groups.
 */
public interface IColIndex {

	public static enum ColIndexType {
		SINGLE, TWO, ARRAY, RANGE, UNKNOWN;
	}

	/**
	 * Get the size of the index aka, how many columns is contained
	 * 
	 * @return The size of the array
	 */
	public int size();

	/**
	 * Get the index at a specific location, Note that many of the underlying implementations does not throw exceptions
	 * on indexes that are completely wrong, so all implementations that use this index should always be well behaved.
	 * 
	 * @param i The index to get
	 * @return the column index at the index.
	 */
	public int get(int i);

	/**
	 * Return a new column index where the values are shifted by the specified amount.
	 * 
	 * It is returning a new instance of the index.
	 * 
	 * @param i The amount to shift
	 * @return the new instance of an index.
	 */
	public IColIndex shift(int i);

	/**
	 * Write out the IO representation of this column index
	 * 
	 * @param out The Output to write into
	 * @throws IOException IO exceptions in case of for instance not enough disk space
	 */
	public void write(DataOutput out) throws IOException;

	/**
	 * Get the exact size on disk to enable preallocation of the disk output buffer sizes
	 * 
	 * @return The exact disk representation size
	 */
	public long getExactSizeOnDisk();

	/**
	 * Get the in memory size of this object.
	 * 
	 * @return The memory size of this object
	 */
	public long estimateInMemorySize();

	/**
	 * A Iterator of the indexes see the iterator interface for details.
	 * 
	 * @return A iterator for the indexes contained.
	 */
	public IIterate iterator();

	/**
	 * Find the index of the value given return negative if non existing.
	 * 
	 * @param i the value to find inside the allocation
	 * @return The index of the value.
	 */
	public int findIndex(int i);

	public SliceResult slice(int l, int u);

	@Override
	public boolean equals(Object other);

	public boolean equals(IColIndex other);

	@Override
	public int hashCode();

	/**
	 * This contains method is not strict since it only verifies one element is contained from each a and b.
	 * 
	 * @param a one array to contain at least one value from
	 * @param b another array to contain at least one value from
	 * @return if the other arrays contain values from this array
	 */
	public boolean contains(IColIndex a, IColIndex b);

	/**
	 * This contains both a and b ... it is strict because it verifies all cells.
	 * 
	 * Note it returns false if there are more elements in this than the sum of a and b.
	 * 
	 * @param a one other array to contain
	 * @param b another array to contain
	 * @return if this array contains both a and b
	 */
	public boolean containsStrict(IColIndex a, IColIndex b);

	/**
	 * combine the indexes of this colIndex with another, it is expected that all calls to this contains unique indexes,
	 * and no copies of values.
	 * 
	 * @param other The other array
	 * @return The combined array
	 */
	public IColIndex combine(IColIndex other);

	/**
	 * Get if these columns are contiguous, meaning all indexes are integers at increments of 1.
	 * 
	 * ex:
	 * 
	 * 1,2,3,4,5,6 is contiguous
	 * 
	 * 1,3,4 is not.
	 * 
	 * 
	 * @return If the Columns are contiguous.
	 */
	public boolean isContiguous();

	/** A Class for slice results containing indexes for the slicing of dictionaries, and the resulting column index */
	public static class SliceResult {
		/** Start index to slice inside the dictionary */
		public final int idStart;
		/** End index (not inclusive) to slice inside the dictionary */
		public final int idEnd;
		/** The already modified column index to return on slices */
		public final IColIndex ret;

		protected SliceResult(int idStart, int idEnd, IColIndex ret) {
			this.idStart = idStart;
			this.idEnd = idEnd;
			this.ret = ret;
		}

		@Override
		public String toString() {
			StringBuilder sb = new StringBuilder(50);
			sb.append("SliceResult:[");
			sb.append(idStart);
			sb.append("-");
			sb.append(idEnd);
			sb.append(" ");
			sb.append(ret);
			sb.append("]");
			return sb.toString();
		}
	}
}
