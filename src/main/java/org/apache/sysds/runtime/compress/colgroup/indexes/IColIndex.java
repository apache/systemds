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

	public boolean contains(IColIndex a, IColIndex b);


	public IColIndex combine(IColIndex other);

	// public boolean contains(ColIndexes a, ColIndexes b) {
	// 	if(a == null || b == null)
	// 		return false;
	// 	int id = _indexes.findIndex(a._indexes.get(0));
	// 	if(id >= 0)
	// 		return true;
	// 	id = _indexes.findIndex(b._indexes.get(0));
	// 	return id >= 0;
	// }

	public static class SliceResult {
		public final int idStart;
		public final int idEnd;
		public final IColIndex ret;

		protected SliceResult(int idStart, int idEnd, IColIndex ret) {
			this.idStart = idStart;
			this.idEnd = idEnd;
			this.ret = ret;
		}
	}
}
