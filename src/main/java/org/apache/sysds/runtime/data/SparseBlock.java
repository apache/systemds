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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.sysds.runtime.matrix.data.IJV;
import org.apache.sysds.runtime.util.UtilFunctions;

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
public abstract class SparseBlock implements Serializable, Block
{
	private static final long serialVersionUID = -5008747088111141395L;
	
	//internal configuration parameters for all sparse blocks
	protected static final int INIT_CAPACITY = 4;       //initial array capacity
	protected static final double RESIZE_FACTOR1 = 2;   //factor until reaching est nnz
	protected static final double RESIZE_FACTOR2 = 1.1; //factor after reaching est nnz
	
	public enum Type {
		COO,  // coordinate
		CSR,  // compressed sparse rows
		CSC,  // compressed sparse column
		DCSR, // double compressed sparse rows
		MCSR, // modified compressed sparse rows (update-friendly)
		MCSC, // modified compressed sparse column
	}
	
	
	////////////////////////
	//basic allocation

	/**
	 * Allocate the underlying data structure holding non-zero values
	 * of row r if necessary. 
	 * 
	 * @param r row index
	 */
	public abstract void allocate(int r);
	
	/**
	 * Allocate the underlying data structure holding non-zero values
	 * of row r if necessary, w/ given size. 
	 * 
	 * @param r row index
	 * @param nnz number of non-zeros
	 */
	public abstract void allocate(int r, int nnz);
	
	/**
	 * Allocate the underlying data structure holding non-zero values
	 * of row r w/ the specified estimated nnz and max nnz.
	 * 
	 * @param r row index
	 * @param ennz estimated non-zeros
	 * @param maxnnz max non-zeros
	 */
	public abstract void allocate(int r, int ennz, int maxnnz);
	
	/**
	 * Re-allocate physical row if physical size exceeds
	 * logical size plus resize factor.
	 * 
	 * @param r row index
	 */
	public abstract void compact(int r);

	/**
	 * In-place compaction of non-zero-entries; removes zero entries
	 * and shifts non-zero entries to the left if necessary.
	 */
	public abstract void compact();
	
	////////////////////////
	//obtain basic meta data
	
	/**
	 * Get the number of rows in the sparse block.
	 * 
	 * @return number of rows
	 */
	public abstract int numRows();
	
	/**
	 * Indicates if the underlying implementation allows thread-safe row
	 * updates if concurrent threads update disjoint rows. 
	 * 
	 * @return true if thread-safe row updates
	 */
	public abstract boolean isThreadSafe();

	/**
	 * Indicates if the underlying data structures returned by values 
	 * and indexes are contiguous arrays, which can be exploited for 
	 * more efficient operations.
	 * 
	 * @return true if underlying data structures are contiguous arrays
	 */
	public abstract boolean isContiguous();
	

	/**
	 * Indicates if all non-zero values are aligned with the given
	 * second sparse block instance, which can be exploited for 
	 * more efficient operations. Two non-zeros are aligned if they 
	 * have the same column index and reside in the same array position.
	 * 
	 * @param that sparse block
	 * @return true if all non-zero values are aligned with the given second sparse block
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
	 * @param r row index starting at 0
	 * @param that sparse block
	 * @return true if all non-zero values of row r are aligned with the same row
	 * of the given second sparse block instance
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
	 * Indicates if the underlying data structure for a given row
	 * is already allocated.
	 * 
	 * @param r row index
	 * @return true if already allocated
	 */
	public abstract boolean isAllocated(int r);
	
	/**
	 * Clears the sparse block by deleting non-zero values. After this call
	 * all size() calls are guaranteed to return 0.
	 */
	public abstract void reset();
	
	/**
	 * Clears the sparse block by deleting non-zero values. After this call
	 * all size() calls are guaranteed to return 0.
	 * 
	 * @param ennz estimated non-zeros
	 * @param maxnnz max non-zeros
	 */
	public abstract void reset(int ennz, int maxnnz);
	
	/**
	 * Clears row r of the sparse block by deleting non-zero values. 
	 * After this call size(r) is guaranteed to return 0.
	 * 
	 * @param r row index
	 * @param ennz estimated non-zeros
	 * @param maxnnz max non-zeros
	 */
	public abstract void reset(int r, int ennz, int maxnnz);
	
	
	/**
	 * Get the number of non-zero values in the sparse block.
	 * 
	 * @return number of non-zero values in sparse block
	 */
	public abstract long size();
	
	/**
	 * Get the number of non-zero values in row r.
	 * 
	 * @param r row index starting at 0
	 * @return number of non-zero values in row r
	 */
	public abstract int size(int r);
	
	/**
	 * Get the number of non-zeros values in the row range
	 * of [rl, ru). 
	 * 
	 * @param rl  row lower index
	 * @param ru  row upper index
	 * @return number of non-zero values in the row range
	 */
	public abstract long size(int rl, int ru);
	
	/**
	 * Get the number of non-zeros values in the row and column
	 * range of [rl/cl, ru/cu);
	 * 
	 * @param rl row lower index
	 * @param ru row upper index
	 * @param cl column lower index
	 * @param cu column upper index
	 * @return number of non-zero values in the row and column range
	 */
	public abstract long size(int rl, int ru, int cl, int cu);
	
	/**
	 * Get information if row r is empty, i.e., does not contain non-zero 
	 * values. Equivalent to size(r)==0. Users should do this check if 
	 * it is unknown if the underlying row data structure is allocated. 
	 * 
	 * @param r  row index starting at 0
	 * @return true if row does not contain non-zero values
	 */
	public abstract boolean isEmpty(int r);

	/**
	 * Validate the correctness of the internal data structures of the different
	 * sparse block implementations.
	 *
	 * @param rlen number of rows
	 * @param clen number of columns
	 * @param nnz number of non zeros
	 * @param strict enforce optional properties
	 * @return true if the sparse block is valid wrt the corresponding format
	 *         such as COO, CSR, MCSR.
	 */

	public abstract boolean checkValidity(int rlen, int clen, long nnz, boolean strict);

	/**
	 * Computes the exact size in memory of the materialized block
	 * @return the exact size in memory
	 */
	public abstract long getExactSizeInMemory();
	
	////////////////////////
	//obtain indexes/values/positions
	
	/**
	 * Get the sorted array of column indexes of all non-zero entries in 
	 * row r. Note that - for flexibility of the implementing format - the 
	 * returned array may be larger, where the range for row r is given by 
	 * [pos(r),pos(r)+size(r)).
	 * 
	 * @param r  row index starting at 0
	 * @return sorted array of column indexes
	 */
	public abstract int[] indexes(int r);
	
	/**
	 * Get the array of all non-zero entries in row r, sorted by their column
	 * indexes. Note that - for flexibility of the implementing format - the 
	 * returned array may be larger, where the range for row r is given by 
	 * [pos(r),pos(r)+size(r)).
	 * 
	 * @param r  row index starting at 0
	 * @return array of all non-zero entries in row r sorted by column indexes
	 */
	public abstract double[] values(int r);
	
	/**
	 * Get the starting position of row r in the indexes/values arrays returned
	 * by indexes(r) and values(r). 
	 * 
	 * @param r  row index starting at 0
	 * @return starting position of row r
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
	 * @return true, if number of non-zeros changed
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
	 * @param row sparse row
	 * @param deep  indicator to create deep copy of sparse row
	 */
	public abstract void set(int r, SparseRow row, boolean deep);
	
	/**
	 * Add a value to a matrix cell (r,c). This might update an existing 
	 * non-zero value, or insert a new non-zero value.
	 * 
	 * @param r  row index starting at 0
	 * @param c  column index starting at 0
	 * @param v  zero or non-zero value 
	 * @return true, if number of non-zeros changed
	 */
	public abstract boolean add(int r, int c, double v);
	
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
	 * Sets a dense array of non-zeros values into the column range [cl,cu) 
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
	 * Sets a sparse array of non-zeros values and indexes into the column range [cl,cu) 
	 * in row r. The passed value array may be larger.
	 * 
	 * @param r    row index starting at 0
	 * @param cl   lower column index starting at 0
	 * @param cu   upper column index starting at 0
	 * @param v    value array
	 * @param vix  column index array
	 * @param vpos start index in value and index arrays
	 * @param vlen number of relevant values 
	 */
	public abstract void setIndexRange(int r, int cl, int cu, double[] v, int[] vix, int vpos, int vlen);
	
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
	 * @return value of cell at position (r,c)
	 */
	public abstract double get(int r, int c);
	
	/**
	 * Get values of row r in the format of a sparse row. 
	 * 
	 * @param r  row index starting at 0
	 * @return values of row r as a sparse row
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
	 * @return position of the first column index lower than or equal to column c in row r
	 */
	public abstract int posFIndexLTE(int r, int c);
	
	/**
	 * Get position of first column index greater than or equal column c
	 * in row r. The position is relative to the indexes/values arrays 
	 * returned by indexes(r) and values(r). If no such value exists, 
	 * this call returns -1.
	 * 
	 * Note if CSR the pos(r) is subtracted from the result.
	 * 
	 * @param r row index starting at 0
	 * @param c column index starting at 0
	 * @return position of the first column index greater than or equal to column c in row r
	 */
	public abstract int posFIndexGTE(int r, int c);
	
	/**
	 * Get position of first column index greater than column c in row r. 
	 * The position is relative to the indexes/values arrays returned by 
	 * indexes(r) and values(r). If no such value exists, this call 
	 * returns -1.
	 * 
	 * @param r row index starting at 0
	 * @param c column index starting at 0
	 * @return position of the first column index greater than column c in row r
	 */
	public abstract int posFIndexGT(int r, int c);
	
	/** 
	 * Checks if the block contains at least one value of the given
	 * pattern. Implementations need to handle NaN patterns as well
	 * (note that NaN==NaN yields false).
	 * 
	 * @param pattern checked pattern
	 * @param rl row lower bound (inclusive)
	 * @param ru row upper bound (exclusive)
	 * @return true if pattern appears at least once, otherwise false
	 */
	public boolean contains(double pattern, int rl, int ru) {
		boolean NaNpattern = Double.isNaN(pattern);
		for(int i=rl; i<ru; i++) {
			if( isEmpty(i) ) continue;
			int apos = pos(i);
			int alen = size(i);
			double[] avals = values(i);
			for( int j=apos; j<apos+alen; j++ )
				if(avals[j]==pattern || (NaNpattern && Double.isNaN(avals[j])))
					return true;
		}
		return false;
	}
	
	public List<Integer> contains(double[] pattern, boolean earlyAbort) {
		int nnz = UtilFunctions.computeNnz(pattern, 0, pattern.length);
		List<Integer> ret = new ArrayList<>();
		int rlen = numRows();

		for( int i=0; i<rlen; i++ ) {
			int apos = pos(i);
			int alen = size(i);
			if(nnz != alen) continue;

			int[] aix = indexes(i);
			double[] avals = values(i);
			boolean lret = true;
			//safe comparison on long representations, incl NaN
			for(int k=apos; k<apos+alen && lret; k++)
				lret &= Double.compare(avals[k], pattern[aix[k]]) == 0;
			if( lret )
				ret.add(i);
			if(earlyAbort && ret.size()>0)
				return ret;
		}
		return ret;
	}
	
	////////////////////////
	//iterators
	
	/**
	 * Get a non-zero iterator over the entire sparse block. Note that
	 * the returned IJV object is reused across next calls and should 
	 * be directly consumed or deep copied. 
	 * 
	 * @return IJV iterator
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
	 * @return IJV iterator
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
	 * @return IJV iterator
	 */
	public Iterator<IJV> getIterator(int rl, int ru) {
		//default generic iterator, override if necessary
		return new SparseBlockIterator(rl, Math.min(ru,numRows()));
	}
	
	/**
	 * Get a non-zero iterator over the subblock [rl/cl, ru/cu). Note that
	 * the returned IJV object is reused across next calls and should 
	 * be directly consumed or deep copied. 
	 * 
	 * @param rl   inclusive lower row index starting at 0
	 * @param ru   exclusive upper row index starting at 0
	 * @param cl   inclusive lower column index starting at 0
	 * @param cu   exclusive upper column index starting at 0
	 * @return IJV iterator
	 */
	public Iterator<IJV> getIterator(int rl, int ru, int cl, int cu) {
		//default generic iterator, override if necessary
		return new SparseBlockIterator(rl, Math.min(ru,numRows()), cl, cu);
	}

	/**
	 * Get an iterator over the indices of non-empty rows within the entire sparse block.
	 * This iterator facilitates traversal over rows that contain at least one non-zero element,
	 * skipping entirely zero rows. The returned integers represent the indexes of non-empty rows.
	 *
	 * @return iterable
	 */
	public Iterable<Integer> getNonEmptyRows() {
		return new SparseNonEmptyRowIterable(0, numRows());
	}

	/**
	 * Get an iterator over the indices of non-zero rows within the sub-block [rl,ru).
	 * This iterator facilitates traversal over rows that contain at least one non-zero element,
	 * skipping entirely zero rows. The returned integers represent the indexes of non-empty rows.
	 *
	 * @param rl inclusive lower row index starting at 0
	 * @param ru exclusive upper row index starting at 0
	 * @return iterable
	 */
	public Iterable<Integer> getNonEmptyRows(int rl, int ru) {
		return new SparseNonEmptyRowIterable(rl, ru);
	}
	
	public abstract Iterator<Integer> getNonEmptyRowsIterator(int rl, int ru);
	
	@Override 
	public abstract String toString();
	

	@Override
	public boolean equals(Object o) {
		if(o instanceof SparseBlock)
			return equals((SparseBlock) o, Double.MIN_NORMAL * 1024);
		return false;
	}

	/**
	 * Verify if the values in this sparse block is equivalent to that sparse block, not taking into account the
	 * dimensions of the contained values.
	 * 
	 * @param o   Other block
	 * @param eps Epsilon allowed
	 * @return If the blocs are equivalent.
	 */
	public boolean equals(SparseBlock o, double eps) {
		for(int r = 0; r < numRows(); r++){
			if(isEmpty(r) != o.isEmpty(r))
				return false;
			if(isEmpty(r))
				continue;
			
			final int apos = pos(r);
			final int alen = apos + size(r);

			final int aposO = o.pos(r);
			final int alenO = aposO + o.size(r);
	
			if(! Arrays.equals(indexes(r), apos, alen,  o.indexes(r), aposO, alenO))
				return false;
			if(! Arrays.equals(values(r), apos, alen,  o.values(r), aposO, alenO))
				return false;
		}
		return true;
	}


	/**
	 * Get if the dense double array is equivalent to this sparse Block.
	 * 
	 * @param denseValues row major double values same dimensions of sparse Block.
	 * @param nCol        Number of columns in dense values (and hopefully in this sparse block)
	 * @return If the dense array is equivalent
	 */
	public boolean equals(double[] denseValues, int nCol) {
		return equals(denseValues, nCol, Double.MIN_NORMAL * 1024);
	}

	/**
	 * Get if the dense double array is equivalent to this sparse Block.
	 * 
	 * @param denseValues row major double values same dimensions of sparse Block.
	 * @param nCol        Number of columns in dense values (and hopefully in this sparse block)
	 * @param eps         Epsilon allowed to be off. Note we treat zero differently and it must be zero.
	 * @return If the dense array is equivalent
	 */
	public boolean equals(double[] denseValues, int nCol, double eps) {
		for(int r = 0; r < numRows(); r++) {
			final int off = r * nCol;
			final int offEnd = off + nCol;
			if(isEmpty(r)) {
				// all in row should be zero.
				for(int i = off; i < offEnd; i++)
					if(denseValues[i] != 0)
						return false;
			}
			else {
				final int apos = pos(r);
				final int alen = apos + size(r);
				final double[] avals = values(r);
				final int[] aix = indexes(r);
				int j = apos;
				int i = off;
				for(int k = 0; i < offEnd && j < alen; i++, k++) {
					if(aix[j] == k) {
						if(Math.abs(denseValues[i] - avals[j]) > eps)
							return false;
						j++;
					}
					else if(denseValues[i] != 0.0)
						return false;
				}
				for(; i < offEnd; i++)
					if(denseValues[i] != 0)
						return false;
			}
		}
		return true;
	}

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
		private int _cl = 0;
		private int _cu = Integer.MAX_VALUE;

		protected SparseBlockIterator(int ru) {
			_rlen = ru;
			_curRow = 0;
			findNextNonZeroRow(0);
		}
		
		protected SparseBlockIterator(int rl, int ru) {
			_rlen = ru;
			_curRow = rl;
			findNextNonZeroRow(0);
		}
		
		protected SparseBlockIterator(int rl, int ru, int cl, int cu) {
			_rlen = ru;
			_curRow = rl;
			_cl = cl;
			_cu = cu;
			findNextNonZeroRow(cl);
		}
		
		@Override
		public boolean hasNext() {
			return !_noNext;
		}

		@Override
		public IJV next( ) {
			retijv.set(_curRow, _curIndexes[_curColIx], _curValues[_curColIx]);
			if( _curColIx < pos(_curRow)+size(_curRow)-1 && _curIndexes[_curColIx+1] < _cu ) { 
				_curColIx++;
			}
			else {
				_curRow++;
				findNextNonZeroRow(_cl);
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
		private void findNextNonZeroRow(int cl) {
			while(_curRow < _rlen){
				if(isEmpty(_curRow)){
					_curRow++;
					continue;
				}

				int pos = (cl == 0)? 0 : posFIndexGTE(_curRow, cl);
				if(pos < 0){
					_curRow++;
					continue;
				}

				int sizeRow = size(_curRow);
				int endPos = (_cu == Integer.MAX_VALUE)? sizeRow : posFIndexGTE(_curRow, _cu);
				if(endPos < 0) endPos = sizeRow;

				if(pos < endPos){
					_curColIx = pos(_curRow)+pos;
					_curIndexes = indexes(_curRow);
					_curValues = values(_curRow);
					return;
				}
				_curRow++;
			}
			_noNext = true;
		}
	}
	
	//generic iterable for use in enhanced for loops: for(int i : s.getNonEmptyRows())
	private class SparseNonEmptyRowIterable implements Iterable<Integer> {
		private final int _rl; //row lower
		private final int _ru; //row upper

		protected SparseNonEmptyRowIterable(int rl, int ru) {
			_rl = rl;
			_ru  =ru;
		}

		@Override
		public Iterator<Integer> iterator() {
			//use specialized non-empty row iterators of sparse blocks
			return getNonEmptyRowsIterator(_rl, _ru);
		}
	}
}
