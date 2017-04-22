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


package org.apache.sysml.runtime.matrix.data;

import java.io.Serializable;

/**
 * Base class for sparse row implementations such as sparse 
 * row vectors and sparse scalars (single value per row).
 * 
 */
public abstract class SparseRow implements Serializable 
{
	private static final long serialVersionUID = 5806895317005796456L;

	/**
	 * Get the number of non-zero values of the sparse row.
	 * 
	 * @return number of non-zeros
	 */
	public abstract int size();
	
	/**
	 * Indicates if the sparse row is empty, i.e., if is has 
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
	 * Get the index array of non-zero entries, co-aligned
	 * with the array of values.
	 * 
	 * @return array of indexes
	 */
	public abstract int[] indexes();
	
	/**
	 * Resets the sparse row to empty, after this call size and
	 * isEmpty are guaranteed to return 0 and true, respectively.
	 * 
	 * @param estnns estimated number of non-zeros
	 * @param maxnns maximum number of non-zeros, e.g., number of columns
	 */
	public abstract void reset(int estnns, int maxnns);
	
	/**
	 * Sets the value of a specified column with awareness of
	 * potential overwrites or deletes (set to value zero).
	 * 
	 * @param col column index, zero-based
	 * @param v value 
	 * @return true if the size of the sparse row changed
	 */
	public abstract boolean set(int col, double v);
	
	/**
	 * Appends a value to the end of the sparse row.
	 * 
	 * @param col column index, zero-based
	 * @param v value
	 */
	public abstract void append(int col, double v);
	
	/**
	 * Gets the value of a specified column. If the column
	 * index does not exist in the sparse row, this call
	 * returns zero.
	 * 
	 * @param col column index, zero-based
	 * @return value 
	 */
	public abstract double get(int col);
	
	/**
	 * In-place sort of column-index value pairs in order to allow binary search
	 * after constant-time append was used for reading unordered sparse rows. We
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
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		for(int i=0; i<size(); i++) {
			sb.append(indexes()[i]);
			sb.append(": ");
			sb.append(values()[i]);
			sb.append("\t");
		}
		return sb.toString();
	}
}
