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
import java.util.Arrays;

import org.apache.sysml.runtime.util.SortUtils;

public class SparseRow implements Serializable 
{
	private static final long serialVersionUID = 5806895317005796456L;

	//initial capacity of any created sparse row
	//WARNING: be aware that this affects the core memory estimates (incl. implicit assumptions)! 
	public static final int initialCapacity = 4;
	
	private int estimatedNzs = initialCapacity;
	private int maxNzs = Integer.MAX_VALUE;
	private int size = 0;
	private double[] values = null;
	private int[] indexes = null;
	
	public SparseRow() {
		this(initialCapacity);
	}
	
	public SparseRow(int capacity)
	{
		estimatedNzs = capacity;
		values = new double[capacity];
		indexes = new int[capacity];
	}
	
	public SparseRow(int estnnz, int maxnnz)
	{
		if( estnnz > initialCapacity )
			estimatedNzs = estnnz;
		maxNzs = maxnnz;
		int capacity = ((estnnz<initialCapacity && estnnz>0) ? 
				         estnnz : initialCapacity);
		values = new double[capacity];
		indexes = new int[capacity];
	}
	
	public SparseRow(SparseRow that)
	{
		size = that.size;
		int cap = Math.max(initialCapacity, that.size);
		
		//allocate arrays and copy new values
		values = Arrays.copyOf(that.values, cap);
		indexes = Arrays.copyOf(that.indexes, cap);
	}
	
	public void truncate(int newsize)
	{
		if( newsize>size || newsize<0 )
			throw new RuntimeException("truncate size: "+newsize+" should <= size: "+size+" and >=0");
		size = newsize;
	}
	
	public int size()
	{
		return size;
	}
	
	public void setSize(int newsize)
	{
		size = newsize;
	}
	
	public boolean isEmpty()
	{
		return (size == 0);
	}
	
	public double[] values()
	{
		return getValueContainer();
	}
	
	public double[] getValueContainer()
	{
		return values;
	}
	
	public int[] indexes()
	{
		return getIndexContainer();
	}
	
	public int[] getIndexContainer()
	{
		return indexes;
	}
	
	public void setValueContainer(double[] d) {
		values = d;
	}
	
	public void setIndexContainer(int[] i) {
		indexes = i;
	}
	
	public int capacity()
	{
		return values.length;
	}
	
	/**
	 * 
	 * @param that
	 */
	public void copy(SparseRow that)
	{
		//note: no recap (if required) + copy, in order to prevent unnecessary copy of old values
		//in case we have to reallocate the arrays
		
		if( values.length < that.size ) {
			//reallocate arrays and copy new values
			values = Arrays.copyOf(that.values, that.size);
			indexes = Arrays.copyOf(that.indexes, that.size);
		}
		else {
			//copy new values
			System.arraycopy(that.values, 0, values, 0, that.size);
			System.arraycopy(that.indexes, 0, indexes, 0, that.size);	
		}
		size = that.size;
	}

	/**
	 * 
	 * @param estnns
	 * @param maxnns
	 */
	public void reset(int estnns, int maxnns)
	{
		estimatedNzs = estnns;
		maxNzs = maxnns;
		size = 0;
	}
	
	/**
	 * 
	 * @param newCap
	 */
	public void recap(int newCap) 
	{
		if( newCap<=values.length )
			return;
		
		//reallocate arrays and copy old values
		values = Arrays.copyOf(values, newCap);
		indexes = Arrays.copyOf(indexes, newCap);
	}
	
	/**
	 * Heuristic for resizing:
	 *   doubling before capacity reaches estimated nonzeros, then 1.1x after that for default behavior: always 1.1x
	 *   (both with exponential size increase for log N steps of reallocation and shifting)
	 * 
	 * @return
	 */
	private int newCapacity()
	{
		if( values.length < estimatedNzs )
			return Math.min(estimatedNzs, values.length*2);
		else
			return (int) Math.min(maxNzs, Math.ceil((double)(values.length)*1.1));
	}

	/**
	 * In-place compaction of non-zero-entries; removes zero entries and
	 * shifts non-zero entries to the left if necessary.
	 */
	public void compact() 
	{
		int nnz = 0;
		for( int i=0; i<size; i++ ) 
			if( values[i] != 0 ){
				values[nnz] = values[i];
				indexes[nnz] = indexes[i];
				nnz++;
			}
		size = nnz; //adjust row size
	}
	
	/**
	 * 
	 * @param col
	 * @param v
	 * @return
	 */
	public boolean set(int col, double v)
	{
		//search for existing col index
		int index = Arrays.binarySearch(indexes, 0, size, col);
		if( index >= 0 ) {
			//delete/overwrite existing value (on value delete, we shift 
			//left for (1) correct nnz maintenance, and (2) smaller size)
			if( v == 0 ) {
				shiftLeftAndDelete(index);
				return true; // nnz--
			}
			else { 	
				values[index] = v;
				return false;
			} 
		}

		//early abort on zero (if no overwrite)
		if( v==0.0 ) {
			return false;
		}
		
		//insert new index-value pair
		index = Math.abs( index+1 );
		if( size==values.length )
			resizeAndInsert(index, col, v);
		else
			shiftRightAndInsert(index, col, v);
		return true; // nnz++
	}
	
	/**
	 * 
	 * @param col
	 * @param v
	 */
	public void append(int col, double v)
	{
		//early abort on zero 
		if( v==0.0 ) {
			return;
		}
		
		//resize if required
		if( size==values.length )
			recap(newCapacity());
		
		//append value at end
		values[size] = v;
		indexes[size] = col;
		size++;
	}
	
	/**
	 * 
	 * @param col
	 * @return
	 */
	public double get(int col)
	{
		//search for existing col index
		int index = Arrays.binarySearch(indexes, 0, size, col);		
		if( index >= 0 )
			return values[index];
		else
			return 0;
	}
	
	/**
	 * 
	 * @param col
	 * @return
	 */
	public int searchIndexesFirstLTE(int col)
	{
		//search for existing col index
		int index = Arrays.binarySearch(indexes, 0, size, col);
		if( index >= 0  ) {
			if( index < size )
				return index;
			else 
				return -1;
		}
		
		//search lt col index (see binary search)
		index = Math.abs( index+1 );
		if( index-1 < size )
			return index-1;
		else 
			return -1;
	}
	

	/**
	 * 
	 * @param col
	 * @return
	 */
	public int searchIndexesFirstGTE(int col)
	{
		//search for existing col index
		int index = Arrays.binarySearch(indexes, 0, size, col);
		if( index >= 0  ) {
			if( index < size )
				return index;
			else 
				return -1;
		}
		
		//search gt col index (see binary search)
		index = Math.abs( index+1 );
		if( index < size )
			return index;
		else 
			return -1;
	}
	
	/**
	 * 
	 * @param col
	 * @return
	 */
	public int searchIndexesFirstGT(int col)
	{
		//search for existing col index
		int index = Arrays.binarySearch(indexes, 0, size, col);
		if( index >= 0  ) {
			if( index+1 < size )
				return index+1;
			else 
				return -1;
		}
		
		//search gt col index (see binary search)
		index = Math.abs( index+1 );
		if( index < size )
			return index;
		else 
			return -1;
	}

	/**
	 * 
	 * @param col
	 */
	public void delete(int col)
	{
		//search for existing col index
		int index = Arrays.binarySearch(indexes, 0, size, col);
		if( index >= 0 ) {
			//shift following entries left by 1
			shiftLeftAndDelete(index);
		}
	}
	
	/**
	 * 
	 * @param lowerCol
	 * @param upperCol
	 */
	public void deleteIndexRange(int lowerCol, int upperCol)
	{
		int start = searchIndexesFirstGTE(lowerCol);
		if( start < 0 ) //nothing to delete 
			return;
		
		int end = searchIndexesFirstGT(upperCol);
		if( end < 0 ) //delete all remaining
			end = size;
		
		//overlapping array copy (shift rhs values left)
		System.arraycopy(values, end, values, start, size-end);
		System.arraycopy(indexes, end, indexes, start, size-end);
		size -= (end-start);
	}
	
	/**
	 * Inserts a dense vector into a column range; calling this methods helps to
	 * avoid repeated shifting of remaining values/indexes for every set value. 
	 * 
	 * @param lowerCol
	 * @param upperCol
	 * @param v
	 * @param vix
	 * @param len
	 */
	public void setIndexRange(int lowerCol, int upperCol, double[] v, int vix, int len)
	{
		int start = searchIndexesFirstGTE(lowerCol);
		if( start < 0 ) { //nothing to delete/shift
			for( int i=vix; i<vix+len; i++ )
				append(lowerCol+i-vix, v[i]);
			return;
		}
		
		int end = searchIndexesFirstGT(upperCol);
		if( end < 0 ) { //delete all remaining
			size = start;
			for( int i=vix; i<vix+len; i++ )
				append(lowerCol+i-vix, v[i]);
			return;
		}
		
		//determine input nnz
		int lnnz = 0;
		for( int i=vix; i<vix+len; i++ )
			lnnz += ( v[i] != 0 ) ? 1 : 0;
		
		//prepare free space (allocate and shift)
		int lsize = size+lnnz-(end-start);
		if( values.length < lsize )
			recap(lsize);
		shiftRightByN(end, lnnz-(end-start));
		
		//insert values
		for( int i=vix, pos=start; i<vix+len; i++ )
			if( v[i] != 0 ) {
				values[ pos ] = v[i];
				indexes[ pos ] = lowerCol+i-vix;
				pos++;
			}
	}

	/**
	 * 
	 * @param index
	 * @param col
	 * @param v
	 */
	private void resizeAndInsert(int index, int col, double v) 
	{
		//allocate new arrays
		int newCap = newCapacity();
		double[] oldvalues = values;
		int[] oldindexes = indexes;
		values = new double[newCap];
		indexes = new int[newCap];
		
		//copy lhs values to new array
		System.arraycopy(oldvalues, 0, values, 0, index);
		System.arraycopy(oldindexes, 0, indexes, 0, index);
		
		//insert new value
		indexes[index] = col;
		values[index] = v;
		
		//copy rhs values to new array
		System.arraycopy(oldvalues, index, values, index+1, size-index);
		System.arraycopy(oldindexes, index, indexes, index+1, size-index);
		size++;
	}
	
	/**
	 * 
	 * @param index
	 * @param col
	 * @param v
	 */
	private void shiftRightAndInsert(int index, int col, double v) 
	{		
		//overlapping array copy (shift rhs values right by 1)
		System.arraycopy(values, index, values, index+1, size-index);
		System.arraycopy(indexes, index, indexes, index+1, size-index);

		//insert new value
		values[index] = v;
		indexes[index] = col;
		size++;
	}
	
	/**
	 * 
	 * @param index
	 * @param n
	 */
	private void shiftRightByN(int index, int n) 
	{		
		//overlapping array copy (shift rhs values right by 1)
		System.arraycopy(values, index, values, index+n, size-index);
		System.arraycopy(indexes, index, indexes, index+n, size-index);
		size += n;
	}
	
	/**
	 * 
	 * @param index
	 */
	private void shiftLeftAndDelete(int index)
	{
		//overlapping array copy (shift rhs values left by 1)
		System.arraycopy(values, index+1, values, index, size-index-1);
		System.arraycopy(indexes, index+1, indexes, index, size-index-1);
		size--;
	}
	
	/**
	 * In-place sort of column-index value pairs in order to allow binary search
	 * after constant-time append was used for reading unordered sparse rows. We
	 * first check if already sorted and subsequently sort if necessary in order
	 * to get O(n) bestcase.
	 * 
	 * Note: In-place sort necessary in order to guarantee the memory estimate
	 * for operations that implicitly read that data set.
	 */
	public void sort()
	{
		if( size<=100 || !SortUtils.isSorted(0, size, indexes) )
			SortUtils.sortByIndex(0, size, indexes, values);
	}
	
	/**
	 * 
	 */
	@Override
	public String toString()
	{
		StringBuilder sb = new StringBuilder();
		for(int i=0; i<size; i++) {
			sb.append(indexes[i]);
			sb.append(": ");
			sb.append(values[i]);
			sb.append("\t");
		}
		return sb.toString();
	}
}
