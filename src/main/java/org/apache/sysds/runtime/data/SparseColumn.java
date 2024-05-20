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

package org.apache.sysds.runtime.data;

import java.io.Serializable;

public abstract class SparseColumn implements Serializable {

	private static final long serialVersionUID = -2421613741597245419L;

	/**
	 * Get the number of non-zero values of the sparse column.
	 *
	 * @return number of non-zeros
	 */
	public abstract int size();

	/**
	 * Indicates if the sparse column is empty, i.e., if is has
	 * size zero.
	 *
	 * @return true if empty
	 */
	public abstract boolean isEmpty();

	/**
	 * Get the value array of non-zero entries, co-aligned
	 * with the array of indexes.
	 *
	 * @return array of values
	 */
	public abstract double[] values();

	/**
	 * Get the row-index array of non-zero entries, co-aligned
	 * with the array of values.
	 *
	 * @return array of indexes
	 */
	public abstract int[] indexes();

	/**
	 * Resets the sparse column to empty, after this call size and
	 * isEmpty are guaranteed to return 0 and true, respectively.
	 *
	 * @param estnns estimated number of non-zeros
	 * @param maxnns maximum number of non-zeros, e.g., number of rows
	 */
	public abstract void reset(int estnns, int maxnns);

	/**
	 * Sets the value of a specified row with awareness of
	 * potential overwrites or deletes (set to value zero).
	 *
	 * @param row row index, zero-based
	 * @param v value
	 * @return true if the size of the sparse column changed
	 */
	public abstract boolean set(int row, double v);

	/**
	 * Add a value to a specified row with awareness of
	 * potential insertions.
	 *
	 * @param row row index, zero-based
	 * @param v value
	 * @return true if the size of the sparse column changed
	 */
	public abstract boolean add(int row, double v);

	/**
	 * Appends a value to the end of the sparse column.
	 *
	 * @param row row index, zero-based
	 * @param v value
	 * @return the row with an appended element
	 */
	public abstract SparseRow append(int row, double v);

	/**
	 * Gets the value of a specified row. If the row
	 * index does not exist in the sparse column, this call
	 * returns zero.
	 *
	 * @param row row index, zero-based
	 * @return value
	 */
	public abstract double get(int row);

	/**
	 * In-place sort of row-index value pairs in order to allow binary search
	 * after constant-time append was used for reading unordered sparse columns. We
	 * first check if already sorted and subsequently sort if necessary in order
	 * to get O(n) best case.
	 *
	 * Note: In-place sort is necessary in order to guarantee the memory estimate
	 * for operations that implicitly read that data set.
	 */
	public abstract void sort();

	/**
	 * In-place compaction of non-zero-entries; removes zero entries
	 * and shifts non-zero entries to the left if necessary.
	 */
	public abstract void compact();

	/**
	 * In-place compaction of values over eps away from zero;
	 * and shifts non-zero entries to the left if necessary.
	 * @param eps epsilon value
	 */
	public abstract void compact(double eps);

	/**
	 * Make a copy of this column.
	 *
	 * @param deep if the copy should be deep
	 * @return A copy
	 */
	public abstract SparseColumn copy(boolean deep);

	/**
	 * Get first index greater than or equal row index.
	 * @param row row to be greater than
	 * @return index
	 */
	public abstract int searchIndexesFirstGTE(int row);

	/**
	 * Get first index greater than row index.
	 * @param row row to be greater than
	 * @return index
	 */
	public abstract int searchIndexesFirstGT(int row);

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();

		// TODO: Adapt to column layout

		return sb.toString();
	}




}
