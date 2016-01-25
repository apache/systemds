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
import java.util.Iterator;

/**
 * This SparseBlock is an abstraction for different sparse matrix formats.
 * Since the design is a tradeoff between performance and generality, we 
 * restrict this abstraction to row-major sparse representations for now. 
 * All sparse matrix block operations are supposed to be implemented 
 * against this abstraction in order to enable variability/extensibility.
 * 
 * Example sparse format that can be implemented efficiently include
 * CSR, MCSR, and - with performance drawbacks - COO.
 * 
 */
public abstract class SparseBlock implements Serializable
{
	private static final long serialVersionUID = -5008747088111141395L;
	
	//internal configuration parameters for all sparse blocks
	protected static final int INIT_CAPACITY = 4;       //initial array capacity
	protected static final double RESIZE_FACTOR1 = 2;   //factor until reaching est nnz
	protected static final double RESIZE_FACTOR2 = 1.1; //factor after reaching est nnz
	
	public enum Type {
		MCSR,
		CSR,
		COO,
	}
	
	
	////////////////////////
	//basic allocation

	/**
	 * Allocate the underlying data structure holding non-zero values
	 * of row r if necessary. 
	 * 
	 * @param r
	 */
	public abstract void allocate(int r);
	
	/**
	 * Allocate the underlying data structure holding non-zero values
	 * of row r if necessary, w/ given size. 
	 * 
	 * @param r
	 */
	public abstract void allocate(int r, int nnz);
	
	/**
	 * Allocate the underlying data structure holding non-zero values
	 * of row r w/ the specified estimated nnz and max nnz.
	 * 
	 * @param r
	 * @param ennz
	 * @param maxnnz
	 */
	public abstract void allocate(int r, int ennz, int maxnnz);
	
	
	////////////////////////
	//obtain basic meta data
	
	/**
	 * Get the number of rows in the sparse block.
	 * 
	 * @return
	 */
	public abstract int numRows();
	
	/**
	 * Indicates if the underlying implementation allows thread-safe row
	 * updates if concurrent threads update disjoint rows. 
	 * 
	 * @return
	 */
	public abstract boolean isThreadSafe();

	/**
	 * Indicates if the underlying data structures returned by values 
	 * and indexes are contiguous arrays, which can be exploited for 
	 * more efficient operations.
	 * 
	 * @return
	 */
	public abstract boolean isContiguous();
	

	/**
	 * Indicates if all non-zero values are aligned with the given
	 * second sparse block instance, which can be exploited for 
	 * more efficient operations. Two non-zeros are aligned if they 
	 * have the same column index and reside in the same array position.
	 * 
	 * @param that
	 * @return
	 */
	public boolean isAligned(SparseBlock that)
	{
		//step 1: cheap meta data comparisons
		if( numRows() != that.numRows() ) //num rows check
			return false;
		
		//step 2: check column indexes per row
		int rlen = numRows();
		for( int i=0; i<rlen; i++ )
			if( !isAligned(i, that) )
				return false;
		
		return true;
	}
	
	/**
	 * Indicates if all non-zero values of row r are aligned with 
	 * the same row of the given second sparse block instance, which 
	 * can be exploited for more efficient operations. Two non-zeros
	 * are aligned if they have the same column index and reside in
	 * the same array position.
	 * 
	 * @param r  row index starting at 0
	 * @param that
	 * @return
	 */
	public boolean isAligned(int r, SparseBlock that)
	{
		//step 1: cheap meta data comparisons
		if( size(r) != that.size(r) || pos(r) != that.pos(r) ) 
			return false;
		
		//step 2: check column indexes per row
		if( !isEmpty(r) ) {
			int alen = size(r);
			int apos = pos(r);
			int[] aix = indexes(r);
			int[] bix = that.indexes(r);
			for( int j=apos; j<apos+alen; j++ )
				if( aix[j] != bix[j] )
					return false;
		}
		
		return true;
	}
	
	/**
	 * Clears the sparse block by deleting non-zero values. After this call
	 * all size() calls are guaranteed to return 0.
	 */
	public abstract void reset();
	
	/**
	 * Clears the sparse block by deleting non-zero values. After this call
	 * all size() calls are guaranteed to return 0.
	 */
	public abstract void reset(int ennz, int maxnnz);
	
	/**
	 * Clears row r of the sparse block by deleting non-zero values. 
	 * After this call size(r) is guaranteed to return 0.
	 */
	public abstract void reset(int r, int ennz, int maxnnz);
	
	
	/**
	 * Get the number of non-zero values in the sparse block.
	 * 
	 * @return
	 */
	public abstract long size();
	
	/**
	 * Get the number of non-zero values in row r.
	 * 
	 * @param r  row index starting at 0
	 * @return
	 */
	public abstract int size(int r);
	
	/**
	 * Get the number of non-zeros values in the row range
	 * of [rl, ru). 
	 * 
	 * @param rl  row index starting at 0
	 * @param ru  column index starting at 0
	 * @return
	 */
	public abstract long size(int rl, int ru);
	
	/**
	 * Get the number of non-zeros values in the row and column
	 * range of [rl/cl, ru/cu);
	 * 
	 * @param rl
	 * @param ru
	 * @param cl
	 * @param cu
	 * @return
	 */
	public abstract long size(int rl, int ru, int cl, int cu);
	
	/**
	 * Get information if row r is empty, i.e., does not contain non-zero 
	 * values. Equivalent to size(r)==0. Users should do this check if 
	 * it is unknown if the underlying row data structure is allocated. 
	 * 
	 * @param r  row index starting at 0
	 * @return
	 */
	public abstract boolean isEmpty(int r); 
	
	
	////////////////////////
	//obtain indexes/values/positions
	
	/**
	 * Get the sorted array of column indexes of all non-zero entries in 
	 * row r. Note that - for flexibility of the implementing format - the 
	 * returned array may be larger, where the range for row r is given by 
	 * [pos(r),pos(r)+size(r)).
	 * 
	 * @param r  row index starting at 0
	 * @return
	 */
	public abstract int[] indexes(int r);
	
	/**
	 * Get the array of all non-zero entries in row r, sorted by their column
	 * indexes. Note that - for flexibility of the implementing format - the 
	 * returned array may be larger, where the range for row r is given by 
	 * [pos(r),pos(r)+size(r)).
	 * 
	 * @param r  row index starting at 0
	 * @return
	 */
	public abstract double[] values(int r);
	
	/**
	 * Get the starting position of row r in the indexes/values arrays returned
	 * by indexes(r) and values(r). 
	 * 
	 * @param r  row index starting at 0
	 * @return
	 */
	public abstract int pos(int r);
	
	
	////////////////////////
	//update operations
	
	/**
	 * Set the value of a matrix cell (r,c). This might update an existing 
	 * non-zero value, insert a new non-zero value, or delete a non-zero value.
	 * 
	 * @param r  row index starting at 0
	 * @param c  column index starting at 0
	 * @param v  zero or non-zero value 
	 * @return
	 */
	public abstract boolean set(int r, int c, double v);
	
	/**
	 * Set the values of row r to the given sparse row. This might update 
	 * existing non-zero values, insert a new row, or delete a row.
	 * 
	 * NOTE: This method exists for incremental runtime integration and might
	 * be deleted in the future.
	 * 
	 * @param r  row index starting at 0
	 * @param row
	 * @return
	 */
	public abstract void set(int r, SparseRow row);
	
	/**
	 * Append a value to the end of the physical representation. This should 
	 * only be used for operations with sequential write pattern or if followed
	 * by a sort() operation. Note that this operation does not perform any 
	 * matrix cell updates.  
	 * 
	 * @param r  row index starting at 0
	 * @param c  column index starting at 0
	 * @param v  zero or non-zero value
	 */
	public abstract void append(int r, int c, double v);
	
	/**
	 * Sets a sorted array of non-zeros values into the column range [cl,cu) 
	 * in row r. The passed value array may be larger and the relevant range 
	 * is given by [vix,vix+len).
	 * 
	 * @param r    row index starting at 0
	 * @param cl   lower column index starting at 0
	 * @param cu   upper column index starting at 0
	 * @param v    value array
	 * @param vix  start index in value array
	 * @param vlen number of relevant values 
	 */
	public abstract void setIndexRange(int r, int cl, int cu, double[] v, int vix, int vlen);
	
	/**
	 * Deletes all non-zero values of the given column range [cl,cu) in row r.
	 * 
	 * @param r   row index starting at 0
	 * @param cl  lower column index starting at 0
	 * @param cu  upper column index starting at 0
	 */
	public abstract void deleteIndexRange(int r, int cl, int cu);
	
	/**
	 * Sort all non-zero value/index pairs of the sparse block by row 
	 * and column index. 
	 */
	public abstract void sort();
	
	/**
	 * Sort all non-zero value/index pairs of row r column index.
	 * 
	 * @param r  row index starting at 0
	 */
	public abstract void sort(int r);
	
	
	////////////////////////
	//search operations
	
	/**
	 * Get value of matrix cell (r,c). In case of non existing values
	 * this call returns 0.
	 * 
	 * @param r  row index starting at 0
	 * @param c  column index starting at 0
	 * @return
	 */
	public abstract double get(int r, int c);
	
	/**
	 * Get values of row r in the format of a sparse row. 
	 * 
	 * NOTE: This method exists for incremental runtime integration and might
	 * be deleted in the future.
	 * 
	 * @param r  row index starting at 0
	 * @return
	 */
	public abstract SparseRow get(int r);
	
	/**
	 * Get position of first column index lower than or equal column c 
	 * in row r. The position is relative to the indexes/values arrays 
	 * returned by indexes(r) and values(r). If no such value exists, 
	 * this call returns -1.
	 * 
	 * @param r  row index starting at 0
	 * @param c  column index starting at 0
	 * @return
	 */
	public abstract int posFIndexLTE(int r, int c);
	
	/**
	 * Get position of first column index greater than or equal column c
	 * in row r. The position is relative to the indexes/values arrays 
	 * returned by indexes(r) and values(r). If no such value exists, 
	 * this call returns -1.
	 * 
	 * @param r
	 * @param c
	 * @return
	 */
	public abstract int posFIndexGTE(int r, int c);
	
	/**
	 * Get position of first column index greater than column c in row r. 
	 * The position is relative to the indexes/values arrays returned by 
	 * indexes(r) and values(r). If no such value exists, this call 
	 * returns -1.
	 * 
	 * @param r
	 * @param c
	 * @return
	 */
	public abstract int posFIndexGT(int r, int c);
	
	
	////////////////////////
	//iterators
	
	/**
	 * Get a non-zero iterator over the entire sparse block. Note that
	 * the returned IJV object is reused across next calls and should 
	 * be directly consumed or deep copied. 
	 * 
	 * @return
	 */
	public Iterator<IJV> getIterator() {
		//default generic iterator, override if necessary
		return new SparseBlockIterator(numRows());
	}
	
	/**
	 * Get a non-zero iterator over the partial sparse block [0,ru). Note 
	 * that the returned IJV object is reused across next calls and should 
	 * be directly consumed or deep copied. 
	 * 
	 * @param ru   exclusive upper row index starting at 0
	 * @return
	 */
	public Iterator<IJV> getIterator(int ru) {
		//default generic iterator, override if necessary
		return new SparseBlockIterator(ru);
	}
	
	/**
	 * Get a non-zero iterator over the subblock [rl, ru). Note that
	 * the returned IJV object is reused across next calls and should 
	 * be directly consumed or deep copied. 
	 * 
	 * @param rl   inclusive lower row index starting at 0
	 * @param ru   exclusive upper row index starting at 0
	 * @return
	 */
	public Iterator<IJV> getIterator(int rl, int ru) {
		//default generic iterator, override if necessary
		return new SparseBlockIterator(rl, Math.min(ru,numRows()));
	}
	
	@Override 
	public abstract String toString();
	
	/**
	 * Default sparse block iterator implemented against the sparse block
	 * api in an implementation-agnostic manner.
	 * 
	 */
	private class SparseBlockIterator implements Iterator<IJV>
	{
		private int _rlen = 0; //row upper
		private int _curRow = -1; //current row
		private int _curColIx = -1; //current col index pos
		private int[] _curIndexes = null; //current col indexes
		private double[] _curValues = null; //current col values
 		private boolean _noNext = false; //end indicator		
		private IJV retijv = new IJV(); //reuse output tuple

		protected SparseBlockIterator(int ru) {
			_rlen = ru;
			_curRow = 0;
			findNextNonZeroRow();
		}
		
		protected SparseBlockIterator(int rl, int ru) {
			_rlen = ru;
			_curRow = rl;
			findNextNonZeroRow();
		}
		
		@Override
		public boolean hasNext() {
			return !_noNext;
		}

		@Override
		public IJV next( ) {
			retijv.set(_curRow, _curIndexes[_curColIx], _curValues[_curColIx]);
			if( ++_curColIx >= pos(_curRow)+size(_curRow) ) {
				_curRow++;
				findNextNonZeroRow();
			}
			
			return retijv;
		}

		@Override
		public void remove() {
			throw new RuntimeException("SparseBlockIterator is unsupported!");			
		}		
		
		/**
		 * Moves cursor to next non-zero value or indicates that no more 
		 * values are available.
		 */
		private void findNextNonZeroRow() {
			while( _curRow<_rlen && isEmpty(_curRow))
				_curRow++;
			if(_curRow >= _rlen)
				_noNext = true;
			else {
				_curColIx = pos(_curRow);
				_curIndexes = indexes(_curRow); 
				_curValues = values(_curRow);
			}
		}		
	}
}
