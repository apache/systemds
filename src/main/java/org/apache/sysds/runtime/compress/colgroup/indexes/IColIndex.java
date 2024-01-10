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

import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.Pair;

/**
 * Class to contain column indexes for the compression column groups.
 */
public interface IColIndex {

	public static enum ColIndexType {
		SINGLE, TWO, ARRAY, RANGE, TWORANGE, COMBINED, UNKNOWN;
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

	/**
	 * Slice the range given.
	 * 
	 * The slice result is an object containing the indexes in the original array to slice out and a new index for the
	 * sliced columns offset by l.
	 * 
	 * Example:
	 * 
	 * ArrayIndex(1,3,5).slice(2,6)
	 * 
	 * returns
	 * 
	 * SliceResult(1,3,ArrayIndex(1,3))
	 * 
	 * 
	 * @param l inclusive lower bound
	 * @param u exclusive upper bound
	 * @return A slice result
	 */
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
	 * @return If the Columns are contiguous.
	 */
	public boolean isContiguous();

	/**
	 * If the columns are not in sorted incrementing order this method can be called to get the sorting index for this
	 * set of column indexes.
	 * 
	 * The returned list should be the mapping of elements for each column to where it should be after sorting.
	 * 
	 * @return A Reordered index.
	 */
	public int[] getReorderingIndex();

	/**
	 * Get if the Index is sorted.
	 * 
	 * @return If the index is sorted
	 */
	public boolean isSorted();

	/**
	 * Sort the index and return a new object if there are modifications otherwise return this.
	 * 
	 * @return The sorted instance of this column index.
	 */
	public IColIndex sort();

	/**
	 * Analyze if this column group contain the given column id
	 * 
	 * @param i id to search for
	 * @return if it is contained
	 */
	public boolean contains(int i);

	/**
	 * Analyze if this column group contain any of the given column Ids.
	 * 
	 * @param idx A List of indexes
	 * @return If it is contained
	 */
	public boolean containsAny(IColIndex idx);

	/**
	 * Get the average of this index. We use this to sort the priority que when combining equivalent costly groups
	 * 
	 * @return The average of the indexes.
	 */
	public double avgOfIndex();

	/**
	 * Decompress this
	 */
	/**
	 * Decompress this column index into the dense c array.
	 * 
	 * @param sb  A sparse block to extract values out of and insert into c
	 * @param vr  The row to extract from the sparse block
	 * @param off The offset that the row starts at in c.
	 * @param c   The dense output to decompress into
	 */
	public void decompressToDenseFromSparse(SparseBlock sb, int vr, int off, double[] c);

	/**
	 * Decompress into c using the values provided. The offset to start into c is off and then row index is similarly the
	 * offset of values. nCol specify the number of values to add over.
	 * 
	 * @param nCol   The number of columns to copy.
	 * @param c      The output to add into
	 * @param off    The offset to start in c
	 * @param values the values to copy from
	 * @param rowIdx The offset to start in values
	 */
	public void decompressVec(int nCol, double[] c, int off, double[] values, int rowIdx);

	/**
	 * Indicate if the two given column indexes are in order such that the first set of indexes all are of lower value
	 * than the second.
	 * 
	 * @param a the first column index
	 * @param b the second column index
	 * @return If the first all is lower than the second.
	 */
	public static boolean inOrder(IColIndex a, IColIndex b) {
		return a.get(a.size() - 1) < b.get(0);
	}

	public static Pair<int[], int[]> reorderingIndexes(IColIndex a, IColIndex b){
		final int[] ar = new int[a.size()];
		final int[] br = new int[b.size()];
		final IIterate ai = a.iterator();
		final IIterate bi = b.iterator();

		int ia = 0;
		int ib = 0;
		int i = 0;
		while(ai.hasNext() && bi.hasNext()){
			if(ai.v()< bi.v()){
				ar[ia++] = i++;
				ai.next();
			}
			else{
				br[ib++] = i++;
				bi.next();
			}
		}

		while(ai.hasNext()){
				ar[ia++] = i++;
			ai.next();
		}

		while(bi.hasNext()){
				br[ib++] = i++;
			bi.next();
		}

		return new Pair<int[],int[]>(ar, br);
	}

	/** A Class for slice results containing indexes for the slicing of dictionaries, and the resulting column index */
	public static class SliceResult {
		/** Start index to slice inside the dictionary */
		public final int idStart;
		/** End index (not inclusive) to slice inside the dictionary */
		public final int idEnd;
		/** The already modified column index to return on slices */
		public final IColIndex ret;

		/**
		 * The slice result
		 * 
		 * @param idStart The starting index
		 * @param idEnd   The ending index (not inclusive)
		 * @param ret     The resulting IColIndex
		 */
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
