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
import java.util.Iterator;

import org.apache.sysml.runtime.util.SortUtils;

/**
 * SparseBlock implementation that realizes a traditional 'coordinate matrix'
 * representation, where the entire sparse block is stored as triples in three arrays: 
 * row indexes, column indexes, and values, where row indexes and colunm indexes are
 * sorted in order to allow binary search. This format is very memory efficient for
 * ultra-sparse matrices, allows fast incremental construction but has performance
 * drawbacks for row-major access through our sparse block abstraction since there
 * is no constant-time random access to individual rows. Similar to CSR, the nnz
 * is limited to Integer.MAX_VALUE.
 * 
 */
public class SparseBlockCOO extends SparseBlock 
{
	private static final long serialVersionUID = 7223478015917668745L;

	private int _rlen = -1;
	private int[] _rindexes = null;  //row index array (size: >=nnz)
	private int[] _cindexes = null;  //column index array (size: >=nnz)
	private double[] _values = null; //value array (size: >=nnz)
	private int _size = 0;           //actual number of nnz
	
	public SparseBlockCOO(int rlen) {
		this(rlen, INIT_CAPACITY);
	}
	
	public SparseBlockCOO(int rlen, int capacity) {
		_rlen = rlen;
		_rindexes = new int[capacity];
		_cindexes = new int[capacity];
		_values = new double[capacity];
		_size = 0;
	}
	
	/**
	 * Copy constructor old sparse row representation. 
	 */
	public SparseBlockCOO(SparseRow[] rows, int nnz)
	{
		_rlen = rows.length;
		
		_rindexes = new int[nnz];
		_cindexes = new int[nnz];
		_values = new double[nnz];
		_size = nnz;
		
		for( int i=0, pos=0; i<_rlen; i++ ) {
			int alen = rows[i].size();
			int[] aix = rows[i].getIndexContainer();
			double[] avals = rows[i].getValueContainer();
			for( int j=0; j<alen; j++ ) {
				_rindexes[pos] = i;
				_cindexes[pos] = aix[j];
				_values[pos] = avals[j];
				pos++;
			}
		}
	}
	
	@Override
	public void allocate(int r) {
		//do nothing everything preallocated
	}

	@Override
	public int numRows() {
		return _rlen;
	}

	@Override
	public boolean isThreadSafe() {
		return false;
	}
	
	@Override 
	public void reset() {
		_size = 0;
	}
	
	@Override
	public long size() {
		return _size;
	}

	@Override
	public int size(int r) {
		int pos = pos(r);
		if( _rindexes[pos]!=r )
			return 0;
		
		//count number of equal row indexes
		double rix0 = _rindexes[pos];
		int cnt = 0;
		while( pos<_size && rix0 == _rindexes[pos++] )
			cnt ++;		
		return cnt;
	}
	
	@Override
	public long size(int rl, int ru) {
		return pos(ru) - pos(rl);
	}
	
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
		int pos = pos(r);
		return (pos>=_size||_rindexes[pos]!=r);
	}
	
	@Override
	public int[] indexes(int r) {
		return _cindexes;
	}

	@Override
	public double[] values(int r) {
		return _values;
	}

	@Override
	public int pos(int r) {
		//find row index partition
		int index = Arrays.binarySearch(_rindexes, 0, _size, r);
		if( index < 0 )
			return Math.abs(index+1);
			
		//scan to begin of row index partition
		while( index>0 && _rindexes[index-1]==r )
			index--;
		return index;	
	}

	@Override
	public boolean set(int r, int c, double v) {
		int pos = pos(r);
		int len = size(r);
		
		//search for existing col index
		int index = Arrays.binarySearch(_cindexes, pos, pos+len, c);
			
		if( index >= 0 ) {
			//delete/overwrite existing value (on value delete, we shift 
			//left for (1) correct nnz maintenance, and (2) smaller size)
			if( v == 0 ) {
				shiftLeftAndDelete(index);
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
			resizeAndInsert(index, r, c, v);
		else
			shiftRightAndInsert(index, r, c, v);
		return true; // nnz++
	}

	@Override
	public void append(int r, int c, double v) {
		//early abort on zero 
		if( v==0 ) return;
	
		if( _size==_values.length ) 
			resize();
		insert(_size, r, c, v);	
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
				_rindexes[ index ] = r;
				_cindexes[ index ] = cl+i-vix;
				_values[ index ] = v[i];
				index++;
			}
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
		System.arraycopy(_rindexes, end, _rindexes, start, _size-end);
		System.arraycopy(_cindexes, end, _cindexes, start, _size-end);
		System.arraycopy(_values, end, _values, start, _size-end);
		_size -= (end-start);		
	}

	@Override
	public void sort() {
		//sort all three indexes by _rindexes
		SortUtils.sortByIndex(0, _size, _rindexes, _cindexes, _values);
		
		//sort _cindexes/_values by _cindexes per row partition
		int index = 0;
		while( index < _size ){
			int r = _rindexes[index];		
			int len = 0;
			while( r == _rindexes[index] ) {
				len ++;	index ++;	
			}
			SortUtils.sortByIndex(index-len, index, _cindexes, _values);
		}
	}

	@Override
	public void sort(int r) {
		int pos = pos(r);
		int len = size(r);
				
		if( len<=100 || !SortUtils.isSorted(pos, pos+len, _cindexes) )
			SortUtils.sortByIndex(pos, pos+len, _cindexes, _values);
	}

	@Override
	public double get(int r, int c) {
		int pos = pos(r);
		int len = size(r);
		
		//search for existing col index in [pos,pos+len)
		int index = Arrays.binarySearch(_cindexes, pos, pos+len, c);		
		return (index >= 0) ? _values[index] : 0;
	}

	@Override
	public int posFIndexLTE(int r, int c) {
		int pos = pos(r);
		int len = size(r);
		
		//search for existing col index in [pos,pos+len)
		int index = Arrays.binarySearch(_cindexes, pos, pos+len, c);
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
		int index = Arrays.binarySearch(_cindexes, pos, pos+len, c);
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
		int index = Arrays.binarySearch(_cindexes, pos, pos+len, c);
		if( index >= 0  )
			return (index+1 < pos+len) ? index+1 : -1;
		
		//search gt col index (see binary search)
		index = Math.abs( index+1 );
		return (index < pos+len) ? index : -1;
	}

	@Override
	public Iterator<IJV> getIterator() {
		return new SparseBlockCOOIterator(0, _size);
	}

	@Override
	public Iterator<IJV> getIterator(int rl, int ru) {
		return new SparseBlockCOOIterator(pos(rl), pos(ru));
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("SparseBlockCOO: rlen=");
		sb.append(_rlen);
		sb.append(", nnz=");
		sb.append(_size);
		sb.append("\n");
		for( int i=0; i<_size; i++ ) {
			sb.append(_rindexes[i]);
			sb.append(",");
			sb.append(_cindexes[i]);
			sb.append(":");
			sb.append(_values[i]);
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
	
	/**
	 * 
	 * @param capacity
	 */
	private void resize(int capacity) {
		//reallocate arrays and copy old values
		_rindexes = Arrays.copyOf(_rindexes, capacity);
		_cindexes = Arrays.copyOf(_cindexes, capacity);
		_values = Arrays.copyOf(_values, capacity);
	}
	
	/**
	 * 
	 * @param ix
	 * @param r
	 * @param c
	 * @param v
	 */
	private void resizeAndInsert(int ix, int r, int c, double v) {
		//compute new size
		double tmpCap = _values.length * RESIZE_FACTOR1;
		int newCap = (int)Math.min(tmpCap, Integer.MAX_VALUE);
		
		int[] oldrindexes = _rindexes;
		int[] oldcindexes = _cindexes;		
		double[] oldvalues = _values;
		_rindexes = new int[newCap];
		_cindexes = new int[newCap];
		_values = new double[newCap];
		
		//copy lhs values to new array
		System.arraycopy(oldrindexes, 0, _rindexes, 0, ix);
		System.arraycopy(oldcindexes, 0, _cindexes, 0, ix);
		System.arraycopy(oldvalues, 0, _values, 0, ix);
		
		//copy rhs values to new array
		System.arraycopy(oldrindexes, ix, _rindexes, ix+1, _size-ix);
		System.arraycopy(oldcindexes, ix, _cindexes, ix+1, _size-ix);
		System.arraycopy(oldvalues, ix, _values, ix+1, _size-ix);
		
		//insert new value
		insert(ix, r, c, v);
	}
	
	/**
	 * 
	 * @param ix
	 * @param r
	 * @param c
	 * @param v
	 */
	private void shiftRightAndInsert(int ix, int r, int c, double v)  {		
		//overlapping array copy (shift rhs values right by 1)
		System.arraycopy(_rindexes, ix, _rindexes, ix+1, _size-ix);
		System.arraycopy(_cindexes, ix, _cindexes, ix+1, _size-ix);
		System.arraycopy(_values, ix, _values, ix+1, _size-ix);
		
		//insert new value
		insert(ix, r, c, v);
	}
	
	/**
	 * 
	 * @param index
	 */
	private void shiftLeftAndDelete(int ix)
	{
		//overlapping array copy (shift rhs values left by 1)
		System.arraycopy(_rindexes, ix+1, _rindexes, ix, _size-ix-1);
		System.arraycopy(_cindexes, ix+1, _cindexes, ix, _size-ix-1);
		System.arraycopy(_values, ix+1, _values, ix, _size-ix-1);
		_size--;
	}
	
	/**
	 * 
	 * @param ix
	 * @param n
	 */
	private void shiftRightByN(int ix, int n) 
	{		
		//overlapping array copy (shift rhs values right by 1)
		System.arraycopy(_rindexes, ix, _rindexes, ix+n, _size-ix);
		System.arraycopy(_cindexes, ix, _cindexes, ix+n, _size-ix);
		System.arraycopy(_values, ix, _values, ix+n, _size-ix);
		_size += n;
	}
	
	/**
	 * 
	 * @param ix
	 * @param r
	 * @param c
	 * @param v
	 */
	private void insert(int ix, int r, int c, double v) {
		_rindexes[ix] = r;
		_cindexes[ix] = c;
		_values[ix] = v;
		_size++;	
	}
	
	/**
	 * Custom sparse block COO iterator implemented against the 
	 * SparseBlockCOO data structure in order to avoid unnecessary
	 * binary search for row locations and lengths.
	 * 
	 */
	private class SparseBlockCOOIterator implements Iterator<IJV>
	{
		private int _pos = 0; //current nnz position
		private int _len = 0; //upper nnz position (exclusive)
		private IJV retijv = new IJV(); //reuse output tuple

		protected SparseBlockCOOIterator(int posrl, int posru) {
			_pos = posrl;
			_len = posru;
		}
		
		@Override
		public boolean hasNext() {
			return _pos<_len;
		}

		@Override
		public IJV next( ) {
			retijv.set(_rindexes[_pos], _cindexes[_pos], _values[_pos++]);			
			return retijv;
		}

		@Override
		public void remove() {
			throw new RuntimeException("SparseBlockCOOIterator is unsupported!");			
		}		
	}
}
