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

import java.util.Arrays;

import org.apache.sysds.runtime.util.SortUtils;
import org.apache.sysds.runtime.util.UtilFunctions;

/**
 * A sparse row vector that is able to grow dynamically as values are appended to it.
 */
public final class SparseRowVector extends SparseRow {
	private static final long serialVersionUID = 2971077474424464992L;

	/**
	 * <p>Initial capacity of any created sparse row</p>
	 * WARNING: be aware that this affects the core memory estimates (incl. implicit assumptions)! 
	 */
	public static final int initialCapacity = 4;
	
	/**
	 * An estimate of the number of non zero values in this row.
	 * The estimate is used to set a threshold on how much the array should grow at certain
	 * lengths to not double the size at all times.
	 */
	private int estimatedNzs = initialCapacity;
	/** The current size of the row vector */
	private int size;
	/** The values contained in the vector, can be allocated larger than needed */
	private double[] values;
	/** The column indexes of the values contained, can be allocated larger than needed */
	private int[] indexes;
	
	public SparseRowVector() {
		this(initialCapacity);
	}
	
	public SparseRowVector(int capacity) {
		capacity = Math.max(initialCapacity, capacity);
		estimatedNzs = capacity;
		values = new double[capacity];
		indexes = new int[capacity];
		size = 0;
	}
	
	public SparseRowVector(int nnz, double[] v, int vlen) {
		values = new double[nnz];
		indexes = new int[nnz];
		for(int i=0, pos=0; i<vlen; i++)
			if( v[i] != 0 ) {
				values[pos] = v[i];
				indexes[pos] = i;
				pos++;
			}
		size = nnz;
	}

	public SparseRowVector(double[] v, int[] i){
		values = v;
		indexes = i;
		size = v.length;
	}

	public SparseRowVector(double[] v, int[] i, int nnz){
		values = v;
		indexes = i;
		size = nnz;
	}

	/**
	 * Sparse row vector constructor that take a dense array, and allocate sparsely by ignoring zero values
	 * @param v The dense row
	 */
	public SparseRowVector(double[] v){
		int nnz = 0;
		for(double vv: v)
			if(vv != 0)
				nnz++;

		values = new double[nnz];
		indexes = new int[nnz];
		for(int i=0, pos=0; i<v.length; i++)
			if( v[i] != 0 ) {
				values[pos] = v[i];
				indexes[pos] = i;
				pos++;
			}
		size = nnz;
	}
	
	public SparseRowVector(int estnnz, int maxnnz) {
		if( estnnz > initialCapacity )
			estimatedNzs = estnnz;
		int capacity = initialCapacity;
		values = new double[capacity];
		indexes = new int[capacity];
		size = 0;
	}
	
	public SparseRowVector(SparseRow that) {
		size = that.size();
		int cap = Math.max(initialCapacity, size);
		
		//allocate arrays and copy new values
		values = Arrays.copyOf(that.values(), cap);
		indexes = Arrays.copyOf(that.indexes(), cap);
	}

	@Override
	public int size() {
		return size;
	}
	
	public void setSize(int newsize) {
		size = newsize;
	}
	
	@Override
	public final boolean isEmpty() {
		return size == 0;
	}
	
	@Override
	public double[] values() {
		return values;
	}
	
	@Override
	public int[] indexes() {
		return indexes;
	}
	
	public void setValues(double[] d) {
		values = d;
	}
	
	public void setIndexes(int[] i) {
		indexes = i;
	}
	
	public int capacity() {
		return values.length;
	}

	public void copy(SparseRow that)
	{
		//note: no recap (if required) + copy, in order to prevent unnecessary copy of old values
		//in case we have to reallocate the arrays
		
		int thatSize = that.size();
		if( values.length < thatSize ) {
			//reallocate arrays and copy new values
			values = Arrays.copyOf(that.values(), thatSize);
			indexes = Arrays.copyOf(that.indexes(), thatSize);
		}
		else {
			//copy new values
			System.arraycopy(that.values(), 0, values, 0, thatSize);
			System.arraycopy(that.indexes(), 0, indexes, 0, thatSize);
		}
		size = thatSize;
	}
	
	@Override
	public void reset(int estnns, int maxnns) {
		estimatedNzs = estnns;
		// maxNzs = maxnns;
		size = 0;
	}

	public void setEstimatedNzs(int estnnz){
		estimatedNzs = estnnz;
	}

	public int getEstimatedNzs(){
		return estimatedNzs;
	}

	private void recap(int newCap) {
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
	 * @return new capacity for resizing
	 */
	private int newCapacity() {
		final double currLen = values.length;
		final boolean lessThanEstimate = currLen < estimatedNzs;
		final double factor = lessThanEstimate ? 
			SparseBlock.RESIZE_FACTOR1 : SparseBlock.RESIZE_FACTOR2;
		//scale length exponentially based on estimated number of non-zeros
		final int nextLen = (int)Math.ceil(currLen * factor);
		//cap at max number of non-zeros with robustness of initial zero
		return Math.max(2, nextLen);
	}

	@Override
	public boolean set(int col, double v) {
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
		if( v==0.0 )
			return false;
		
		//insert new index-value pair
		index = Math.abs( index+1 );
		if( size==values.length )
			resizeAndInsert(index, col, v);
		else
			shiftRightAndInsert(index, col, v);
		return true; // nnz++
	}
	
	public void setAtPos(int pos, int col, double v) {
		indexes[pos] = col;
		values[pos] = v;
	}
	
	@Override
	public boolean add(int col, double v) {
		//early abort on zero (if no overwrite)
		if( v==0.0 ) return false;
		
		//search for existing col index
		int index = Arrays.binarySearch(indexes, 0, size, col);
		if( index >= 0 ) {
			//add to existing values
			values[index] += v;
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

	@Override
	public SparseRowVector append(int col, double v) {
		// early abort on zero 
		if(v == 0.0)
			return this;
		
		// resize if required
		if( size == values.length )
			recap(newCapacity());
		
		// append value at end
		values[size] = v;
		indexes[size] = col;
		size++;
		return this;
	}

	@Override
	public double get(int col) {
		//search for existing col index
		int index = Arrays.binarySearch(indexes, 0, size, col);
		return (index >= 0) ? values[index] : 0;
	}

	public int getIndex(int col) {
		//search for existing col index
		int index = Arrays.binarySearch(indexes, 0, size, col);
		return (index >= 0) ? index : -1;
	}

	public int searchIndexesFirstLTE(int col) {
		if( size == 0 ) return -1;
		
		//search for existing col index
		int index = Arrays.binarySearch(indexes, 0, size, col);
		if( index >= 0 )
			return (index < size) ? index : -1;
		
		//search lt col index (see binary search)
		index = Math.abs( index+1 );
		return (index-1 >= 0) ? index-1 : -1;
	}

	@Override
	public int searchIndexesFirstGTE(int col) {
		// assumed that empty check is made.
		
		//search for existing col index
		int index = Arrays.binarySearch(indexes, 0, size, col);
		if( index >= 0 )
			return (index < size) ? index : -1;
		
		//search gt col index (see binary search)
		index = Math.abs( index+1 );
		return (index < size) ? index : -1;
	}

	@Override
	public int searchIndexesFirstGT(int col) {
		// assumed that empty check is made.

		//search for existing col index
		int index = Arrays.binarySearch(indexes, 0, size, col);
		if( index >= 0 )
			return (index+1 < size) ? index+1 : -1;
		
		//search gt col index (see binary search)
		index = Math.abs( index+1 );
		return (index < size) ? index : -1;
	}

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
	
	public void setIndexRange(int cl, int cu, double[] v, int vix, int vlen) {
		//handle special cases
		int start = searchIndexesFirstGTE(cl);
		if( start < 0 ) { //nothing to delete/shift
			for( int i=vix; i<vix+vlen; i++ )
				append(cl+i-vix, v[i]);
			return;
		}
		int end = searchIndexesFirstGT(cu);
		if( end < 0 ) { //delete all remaining
			size = start;
			for( int i=vix; i<vix+vlen; i++ )
				append(cl+i-vix, v[i]);
			return;
		}
		
		//determine input nnz
		int lnnz = UtilFunctions.computeNnz(v, vix, vlen);
		
		//prepare free space (allocate and shift)
		int lsize = size+lnnz-(end-start);
		if( values.length < lsize )
			recap(lsize);
		shiftRightByN(end, lnnz-(end-start));
		
		//insert values
		for( int i=vix, pos=start; i<vix+vlen; i++ )
			if( v[i] != 0 ) {
				values[ pos ] = v[i];
				indexes[ pos ] = cl+i-vix;
				pos++;
			}
	}

	public void setIndexRange(int cl, int cu, double[] v, int[] vix, int vpos, int vlen) {
		//handle special cases
		int start = searchIndexesFirstGTE(cl);
		if( start < 0 ) { //nothing to delete/shift
			for( int i=vpos; i<vpos+vlen; i++ )
				append(cl+vix[i], v[i]);
			return;
		}
		int end = searchIndexesFirstGT(cu);
		if( end < 0 ) { //delete all remaining
			size = start;
			for( int i=vpos; i<vpos+vlen; i++ )
				append(cl+vix[i], v[i]);
			return;
		}
		
		//prepare free space (allocate and shift)
		int lsize = size+vlen-(end-start);
		if( values.length < lsize )
			recap(lsize);
		shiftRightByN(end, vlen-(end-start));
		
		//insert values
		for( int i=vpos, pos=start; i<vpos+vlen; i++ ) {
			values[ pos ] = v[i];
			indexes[ pos ] = cl+vix[i];
			pos++;
		}
	}
	
	private final void resizeAndInsert(int index, int col, double v) {
		final int newCap = newCapacity();
		resizeVals(newCap, index, v);
		resizeIndex(newCap, index, col);
		size++;
	}

	private final void resizeVals(int newCap, int index, double v){
		double[] old = values;
		values = new double[newCap];
		System.arraycopy(old, 0, values, 0, index);
		values[index] = v;
		System.arraycopy(old, index, values, index+1, size-index);
	}

	private final void resizeIndex(int newCap, int index, int col){
		int[] old = indexes;
		indexes = new int[newCap];
		System.arraycopy(old, 0, indexes, 0, index);
		indexes[index] = col;
		System.arraycopy(old, index, indexes, index+1, size-index);
	}

	private void shiftRightAndInsert(int index, int col, double v) {
		//overlapping array copy (shift rhs values right by 1)
		System.arraycopy(values, index, values, index+1, size-index);
		System.arraycopy(indexes, index, indexes, index+1, size-index);

		//insert new value
		values[index] = v;
		indexes[index] = col;
		size++;
	}

	private void shiftRightByN(int index, int n) {
		//overlapping array copy (shift rhs values right by 1)
		System.arraycopy(values, index, values, index+n, size-index);
		System.arraycopy(indexes, index, indexes, index+n, size-index);
		size += n;
	}

	private void shiftLeftAndDelete(int index) {
		//overlapping array copy (shift rhs values left by 1)
		System.arraycopy(values, index+1, values, index, size-index-1);
		System.arraycopy(indexes, index+1, indexes, index, size-index-1);
		size--;
	}
	
	@Override
	public void sort() {
		if( size<=100 || !SortUtils.isSorted(0, size, indexes) )
			SortUtils.sortByIndex(0, size, indexes, values);
	}
	
	@Override
	public void compact() {
		int nnz = 0;
		for( int i=0; i<size; i++ ) 
			if( values[i] != 0.0 ){
				values[nnz] = values[i];
				indexes[nnz] = indexes[i];
				nnz++;
			}
		size = nnz; //adjust row size
	}

	@Override
	public void compact(double eps) {
		int nnz = 0;
		for( int i=0; i<size; i++ ) 
			if( Math.abs(values[i]) > eps ){
				values[nnz] = values[i];
				indexes[nnz] = indexes[i];
				nnz++;
			}
		size = nnz; //adjust row size
	}

	@Override
	public SparseRow copy(boolean deep){
		return new SparseRowVector(this);
	}
}
