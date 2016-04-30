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

import java.util.Arrays;

import org.apache.sysml.runtime.util.SortUtils;

/**
 * SparseBlock implementation that realizes a traditional 'compressed sparse row'
 * representation, where the entire sparse block is stored as three arrays: ptr
 * of length rlen+1 to store offsets per row, and indexes/values of length nnz
 * to store column indexes and values of non-zero entries. This format is very
 * memory efficient for sparse (but not ultra-sparse) matrices and provides very 
 * good performance for common operations, partially due to lower memory bandwidth 
 * requirements. However, this format is slow on incremental construction (because 
 * it does not allow append/sort per row) without reshifting. Finally, the total 
 * nnz is limited to INTEGER_MAX, whereas for SparseBlockMCSR only the nnz per 
 * row are limited to INTEGER_MAX.  
 * 
 * TODO: extensions for faster incremental construction (e.g., max row)
 * TODO more efficient fused setIndexRange impl to avoid repeated copies and updates
 * 	
 */
public class SparseBlockCSR extends SparseBlock 
{
	private static final long serialVersionUID = 1922673868466164244L;

	private int[] _ptr = null;       //row pointer array (size: rlen+1)
	private int[] _indexes = null;   //column index array (size: >=nnz)
	private double[] _values = null; //value array (size: >=nnz)
	private int _size = 0;           //actual number of nnz
	
	public SparseBlockCSR(int rlen) {
		this(rlen, INIT_CAPACITY);
	}
	
	public SparseBlockCSR(int rlen, int capacity) {
		_ptr = new int[rlen+1]; //ix0=0
		_indexes = new int[capacity];
		_values = new double[capacity];
		_size = 0;
	}
	
	/**
	 * Copy constructor sparse block abstraction. 
	 */
	public SparseBlockCSR(SparseBlock sblock)
	{
		long size = sblock.size();
		if( size > Integer.MAX_VALUE )
			throw new RuntimeException("SparseBlockCSR supports nnz<=Integer.MAX_VALUE but got "+size);
		
		//special case SparseBlockCSR
		if( sblock instanceof SparseBlockCSR ) { 
			SparseBlockCSR ocsr = (SparseBlockCSR)sblock;
			_ptr = Arrays.copyOf(ocsr._ptr, ocsr.numRows()+1);
			_indexes = Arrays.copyOf(ocsr._indexes, ocsr._size);
			_values = Arrays.copyOf(ocsr._values, ocsr._size);
			_size = ocsr._size;
		}
		//general case SparseBlock
		else {
			int rlen = sblock.numRows();
			
			_ptr = new int[rlen+1];
			_indexes = new int[(int)size];
			_values = new double[(int)size];
			_size = (int)size;

			for( int i=0, pos=0; i<rlen; i++ ) {
				if( !sblock.isEmpty(i) ) {
					int apos = sblock.pos(i);
					int alen = sblock.size(i);
					int[] aix = sblock.indexes(i);
					double[] avals = sblock.values(i);
					for( int j=apos; j<apos+alen; j++ ) {
						_indexes[pos] = aix[j];
						_values[pos] = avals[j];
						pos++;
					}
				}
				_ptr[i+1]=pos;
			}			
		}
	}
	
	/**
	 * Copy constructor old sparse row representation. 
	 */
	public SparseBlockCSR(SparseRow[] rows, int nnz)
	{
		int rlen = rows.length;
		
		_ptr = new int[rlen+1]; //ix0=0
		_indexes = new int[nnz];
		_values = new double[nnz];
		_size = nnz;
		
		for( int i=0, pos=0; i<rlen; i++ ) {
			int alen = rows[i].size();
			int[] aix = rows[i].indexes();
			double[] avals = rows[i].values();
			for( int j=0; j<alen; j++ ) {
				_indexes[pos] = aix[j];
				_values[pos] = avals[j];
				pos++;
			}
			_ptr[i+1]=pos;	
		}
	}
	
	/**
	 * Get the estimated in-memory size of the sparse block in CSR 
	 * with the given dimensions w/o accounting for overallocation. 
	 * 
	 * @param nrows
	 * @param ncols
	 * @param sparsity
	 * @return
	 */
	public static long estimateMemory(long nrows, long ncols, double sparsity) {
		double lnnz = Math.max(INIT_CAPACITY, Math.ceil(sparsity*nrows*ncols));
		
		//32B overhead per array, int arr in nrows, int/double arr in nnz 
		double size = 16 + 4;        //object + int field
		size += 32 + (nrows+1) * 4d; //ptr array (row pointers)
		size += 32 + lnnz * 4d;      //indexes array (column indexes)
		size += 32 + lnnz * 8d;      //values array (non-zero values)
		
		//robustness for long overflows
		return (long) Math.min(size, Long.MAX_VALUE);
	}
	
	///////////////////
	//SparseBlock implementation

	@Override
	public void allocate(int r) {
		//do nothing everything preallocated
	}
	
	@Override
	public void allocate(int r, int nnz) {
		//do nothing everything preallocated
	}
	
	@Override
	public void allocate(int r, int ennz, int maxnnz) {
		//do nothing everything preallocated
	}

	@Override
	public int numRows() {
		return _ptr.length-1;
	}

	@Override
	public boolean isThreadSafe() {
		return false;
	}
	
	@Override
	public boolean isContiguous() {
		return true;
	}
	
	@Override 
	public void reset() {
		_size = 0;
		Arrays.fill(_ptr, 0);
	}

	@Override 
	public void reset(int ennz, int maxnnz) {
		_size = 0;
		Arrays.fill(_ptr, 0);
	}
	
	@Override 
	public void reset(int r, int ennz, int maxnnz) {
		int pos = pos(r);
		int len = size(r);
		
		if( len > 0 ) {
			//overlapping array copy (shift rhs values left)
			System.arraycopy(_indexes, pos+len, _indexes, pos, _size-(pos+len));
			System.arraycopy(_values, pos+len, _values, pos, _size-(pos+len));
			_size -= len;	
			decrPtr(r+1, len);
		}
	}
	
	@Override
	public long size() {
		return _size;
	}

	@Override
	public int size(int r) {
		return _ptr[r+1] - _ptr[r];
	}
	
	@Override
	public long size(int rl, int ru) {
		return _ptr[ru] - _ptr[rl];
	}

	@Override
	public long size(int rl, int ru, int cl, int cu) {
		long nnz = 0;
		for(int i=rl; i<ru; i++)
			if( !isEmpty(i) ) {
				int start = posFIndexGTE(i, cl);
				int end = posFIndexGTE(i, cu);
				nnz += (start!=-1) ? (end-start) : 0;
			}
		return nnz;
	}
	
	@Override
	public boolean isEmpty(int r) {
		return (_ptr[r+1] - _ptr[r] == 0);
	}
	
	@Override
	public int[] indexes(int r) {
		return _indexes;
	}

	@Override
	public double[] values(int r) {
		return _values;
	}

	@Override
	public int pos(int r) {
		return _ptr[r];
	}

	@Override
	public boolean set(int r, int c, double v) {
		int pos = pos(r);
		int len = size(r);
		
		//search for existing col index
		int index = Arrays.binarySearch(_indexes, pos, pos+len, c);
		if( index >= 0 ) {
			//delete/overwrite existing value (on value delete, we shift 
			//left for (1) correct nnz maintenance, and (2) smaller size)
			if( v == 0 ) {
				shiftLeftAndDelete(index);
				decrPtr(r+1);
				return true; // nnz--
			}
			else { 	
				_values[index] = v;
				return false;
			} 
		}

		//early abort on zero (if no overwrite)
		if( v==0 ) return false;
		
		//insert new index-value pair
		index = Math.abs( index+1 );
		if( _size==_values.length )
			resizeAndInsert(index, c, v);
		else
			shiftRightAndInsert(index, c, v);
		incrPtr(r+1);
		return true; // nnz++
	}

	@Override
	public void set(int r, SparseRow row, boolean deep) {
		int pos = pos(r);
		int len = size(r);		
		int alen = row.size();
		int[] aix = row.indexes();
		double[] avals = row.values();
		
		//delete existing values if necessary
		if( len > 0 )
			deleteIndexRange(r, aix[0], aix[alen-1]+1);
		
		//prepare free space (allocate and shift)
		int lsize = _size+alen;
		if( _values.length < lsize )
			resize(lsize);				
		shiftRightByN(pos, alen);
		
		//copy input row into internal representation
		System.arraycopy(aix, 0, _indexes, pos, alen);
		System.arraycopy(avals, 0, _values, pos, alen);
		_size+=alen;
	}
	
	@Override
	public void append(int r, int c, double v) {
		//early abort on zero 
		if( v==0 ) return;
	
		int pos = pos(r);
		int len = size(r);
		if( pos+len == _size ) {
			//resize and append
			if( _size==_values.length )
				resize();
			insert(_size, c, v);		
		}		
		else {
			//resize, shift and insert
			if( _size==_values.length )
				resizeAndInsert(pos+len, c, v);
			else
				shiftRightAndInsert(pos+len, c, v);
		}			
		incrPtr(r+1);
	}

	@Override
	public void setIndexRange(int r, int cl, int cu, double[] v, int vix, int vlen) {
		//delete existing values in range if necessary 
		deleteIndexRange(r, cl, cu);
		
		//determine input nnz
		int lnnz = 0;
		for( int i=vix; i<vix+vlen; i++ )
			lnnz += ( v[i] != 0 ) ? 1 : 0;

		//prepare free space (allocate and shift)
		int lsize = _size+lnnz;
		if( _values.length < lsize )
			resize(lsize);
		int index = posFIndexGT(r, cl);
		shiftRightByN((index>0)?index:pos(r+1), lnnz);
		
		//insert values
		for( int i=vix; i<vix+vlen; i++ )
			if( v[i] != 0 ) {
				_indexes[ index ] = cl+i-vix;
				_values[ index ] = v[i];
				index++;
			}
		incrPtr(r+1, lnnz);
	}

	@Override
	public void deleteIndexRange(int r, int cl, int cu) {
		int start = posFIndexGTE(r,cl);
		if( start < 0 ) //nothing to delete 
			return;		

		int len = size(r);
		int end = posFIndexGTE(r, cu);
		if( end < 0 ) //delete all remaining
			end = start+len;
		
		//overlapping array copy (shift rhs values left)
		System.arraycopy(_indexes, end, _indexes, start, _size-end);
		System.arraycopy(_values, end, _values, start, _size-end);
		_size -= (end-start);		
		
		decrPtr(r+1, end-start);
	}

	@Override
	public void sort() {
		int rlen = numRows();
		for( int i=0; i<rlen && pos(i)<_size; i++ )
			sort(i);
	}

	@Override
	public void sort(int r) {
		int pos = pos(r);
		int len = size(r);
				
		if( len<=100 || !SortUtils.isSorted(pos, pos+len, _indexes) )
			SortUtils.sortByIndex(pos, pos+len, _indexes, _values);
	}

	@Override
	public double get(int r, int c) {
		int pos = pos(r);
		int len = size(r);
		
		//search for existing col index in [pos,pos+len)
		int index = Arrays.binarySearch(_indexes, pos, pos+len, c);		
		return (index >= 0) ? _values[index] : 0;
	}
	
	@Override 
	public SparseRow get(int r) {
		int pos = pos(r);
		int len = size(r);
		
		SparseRow row = new SparseRow(len);
		System.arraycopy(_indexes, pos, row.indexes(), 0, len);
		System.arraycopy(_values, pos, row.values(), 0, len);
		row.setSize(len);
		
		return row;
	}
	
	@Override
	public int posFIndexLTE(int r, int c) {
		int pos = pos(r);
		int len = size(r);
		
		//search for existing col index in [pos,pos+len)
		int index = Arrays.binarySearch(_indexes, pos, pos+len, c);
		if( index >= 0  )
			return (index < pos+len) ? index : -1;
		
		//search lt col index (see binary search)
		index = Math.abs( index+1 );
		return (index-1 >= pos) ? index-1 : -1;
	}

	@Override
	public int posFIndexGTE(int r, int c) {
		int pos = pos(r);
		int len = size(r);
		
		//search for existing col index
		int index = Arrays.binarySearch(_indexes, pos, pos+len, c);
		if( index >= 0  )
			return (index < pos+len) ? index : -1;
		
		//search gt col index (see binary search)
		index = Math.abs( index+1 );
		return (index < pos+len) ? index : -1;
	}

	@Override
	public int posFIndexGT(int r, int c) {
		int pos = pos(r);
		int len = size(r);
		
		//search for existing col index
		int index = Arrays.binarySearch(_indexes, pos, pos+len, c);
		if( index >= 0  )
			return (index+1 < pos+len) ? index+1 : -1;
		
		//search gt col index (see binary search)
		index = Math.abs( index+1 );
		return (index < pos+len) ? index : -1;
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("SparseBlockCSR: rlen=");
		sb.append(numRows());
		sb.append(", nnz=");
		sb.append(size());
		sb.append("\n");
		for( int i=0; i<numRows(); i++ ) {
			sb.append("row +");
			sb.append(i);
			sb.append(": ");
			//append row
			int pos = pos(i);
			int len = size(i);
			for(int j=pos; j<pos+len; j++) {
				sb.append(_indexes[j]);
				sb.append(": ");
				sb.append(_values[j]);
				sb.append("\t");
			}
			sb.append("\n");
		}		
		
		return sb.toString();
	}
	
	///////////////////////////
	// private helper methods
	
	/**
	 * 
	 */
	private void resize() {
		//compute new size
		double tmpCap = _values.length * RESIZE_FACTOR1;
		int newCap = (int)Math.min(tmpCap, Integer.MAX_VALUE);
		
		resize(newCap);
	}
	
	private void resize(int capacity) {
		//reallocate arrays and copy old values
		_indexes = Arrays.copyOf(_indexes, capacity);
		_values = Arrays.copyOf(_values, capacity);
	}
	
	/**
	 * 
	 * @param ix
	 * @param c
	 * @param v
	 */
	private void resizeAndInsert(int ix, int c, double v) {
		//compute new size
		double tmpCap = _values.length * RESIZE_FACTOR1;
		int newCap = (int)Math.min(tmpCap, Integer.MAX_VALUE);
		
		int[] oldindexes = _indexes;
		double[] oldvalues = _values;
		_indexes = new int[newCap];
		_values = new double[newCap];
		
		//copy lhs values to new array
		System.arraycopy(oldindexes, 0, _indexes, 0, ix);
		System.arraycopy(oldvalues, 0, _values, 0, ix);
		
		//copy rhs values to new array
		System.arraycopy(oldindexes, ix, _indexes, ix+1, _size-ix);
		System.arraycopy(oldvalues, ix, _values, ix+1, _size-ix);
		
		//insert new value
		insert(ix, c, v);
	}
	
	/**
	 * 
	 * @param ix
	 * @param c
	 * @param v
	 */
	private void shiftRightAndInsert(int ix, int c, double v)  {		
		//overlapping array copy (shift rhs values right by 1)
		System.arraycopy(_indexes, ix, _indexes, ix+1, _size-ix);
		System.arraycopy(_values, ix, _values, ix+1, _size-ix);
		
		//insert new value
		insert(ix, c, v);
	}
	
	/**
	 * 
	 * @param index
	 */
	private void shiftLeftAndDelete(int ix)
	{
		//overlapping array copy (shift rhs values left by 1)
		System.arraycopy(_indexes, ix+1, _indexes, ix, _size-ix-1);
		System.arraycopy(_values, ix+1, _values, ix, _size-ix-1);
		_size--;
	}

	private void shiftRightByN(int ix, int n) 
	{		
		//overlapping array copy (shift rhs values right by 1)
		System.arraycopy(_indexes, ix, _indexes, ix+n, _size-ix);
		System.arraycopy(_values, ix, _values, ix+n, _size-ix);
		_size += n;
	}
	
	/**
	 * 
	 * @param ix
	 * @param c
	 * @param v
	 */
	private void insert(int ix, int c, double v) {
		_indexes[ix] = c;
		_values[ix] = v;
		_size++;	
	}
	
	/**
	 * 
	 * @param rl
	 */
	private void incrPtr(int rl) {
		incrPtr(rl, 1);
	}
	
	/**
	 * 
	 * @param rl
	 * @param cnt
	 */
	private void incrPtr(int rl, int cnt) {
		int rlen = numRows();
		for( int i=rl; i<rlen+1; i++ )
			_ptr[i]+=cnt;
	}
	
	/**
	 * 
	 * @param rl
	 */
	private void decrPtr(int rl) {
		decrPtr(rl, 1);
	}
	
	/**
	 * 
	 * @param rl
	 * @param cnt
	 */
	private void decrPtr(int rl, int cnt) {
		int rlen = numRows();
		for( int i=rl; i<rlen+1; i++ )
			_ptr[i]-=cnt;
	}
}
