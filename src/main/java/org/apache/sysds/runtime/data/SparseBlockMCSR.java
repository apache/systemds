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

import java.util.Iterator;

import org.apache.sysds.utils.MemoryEstimates;

/**
 * SparseBlock implementation that realizes a 'modified compressed sparse row'
 * representation, where each compressed row is stored as a separate SparseRow
 * object which provides flexibility for unsorted row appends without the need 
 * for global reshifting of values/indexes but it incurs additional memory 
 * overhead per row for object/array headers per row which also slows down
 * memory-bound operations due to higher memory bandwidth requirements.
 * 
 */
public class SparseBlockMCSR extends SparseBlock
{
	private static final long serialVersionUID = -4743624499258436199L;
	
	private SparseRow[] _rows = null;
	
	/**
	 * Copy constructor sparse block abstraction. 
	 * 
	 * @param sblock sparse block to copy
	 */
	public SparseBlockMCSR(SparseBlock sblock)
	{
		//special case SparseBlockMCSR
		if( sblock instanceof SparseBlockMCSR ) { 
			SparseRow[] orows = ((SparseBlockMCSR)sblock)._rows;
			_rows = new SparseRow[orows.length];
			for( int i=0; i<_rows.length; i++ )
				if( orows[i] != null )
					_rows[i] = orows[i].copy(true);
		}
		//general case SparseBlock
		else { 
			_rows = new SparseRow[sblock.numRows()];
			for( int i=0; i<_rows.length; i++ ) {
				if( !sblock.isEmpty(i) ) {
					int apos = sblock.pos(i);
					int alen = sblock.size(i);
					if(alen == 0){
						// do nothing
					}
					else if(alen == 1)
						_rows[i] = new SparseRowScalar(sblock.indexes(i)[apos], sblock.values(i)[apos]);
					else{
						_rows[i] = new SparseRowVector(alen);
						((SparseRowVector)_rows[i]).setSize(alen);
						System.arraycopy(sblock.indexes(i), apos, _rows[i].indexes(), 0, alen);
						System.arraycopy(sblock.values(i), apos, _rows[i].values(), 0, alen);
					}
				}
			}
		}
	}
	
	/**
	 * Copy constructor old sparse row representation. 
	 * 
	 * @param rows array of sparse rows
	 * @param deep if true, deep copy
	 */
	public SparseBlockMCSR(SparseRow[] rows, boolean deep) {
		if( deep ) {
			_rows = new SparseRow[rows.length];
			for( int i=0; i<_rows.length; i++ ) {
				_rows[i] = (rows[i].size()==1) ? new SparseRowScalar(
					rows[i].indexes()[0], rows[i].values()[0]) : 
					new SparseRowVector(rows[i]);
			}
		}
		else {
			_rows = rows;
		}
	}
	
	public SparseBlockMCSR(int rlen){
		_rows = new SparseRow[rlen];
	}

	public SparseBlockMCSR(int rlen, int clen) {
		this(rlen);
	}
	
	/**
	 * Get the estimated in-memory size of the sparse block in MCSR 
	 * with the given dimensions w/o accounting for overallocation. 
	 * 
	 * @param nrows number of rows
	 * @param ncols number of columns
	 * @param sparsity sparsity ratio
	 * @return memory estimate
	 */
	public static long estimateSizeInMemory(long nrows, long ncols, double sparsity) {
		double nnz = Math.ceil(sparsity*nrows*ncols);
		double rlen = Math.min(nrows, nnz); // num sparse row objects
		double cnnz = Math.max(SparseRowVector.initialCapacity, nnz/rlen);
		
		//Each sparse row has a fixed overhead of 16B (object) + 12B (3 ints),
		//24B (int array), 24B (double array), i.e., in total 76B
		//Each non-zero value requires 12B for the column-index/value pair.
		//Overheads for arrays, objects, and references refer to 64bit JVMs
		//If nnz < rows we have guaranteed also empty rows.
		double size = 16; //object
		size += MemoryEstimates.objectArrayCost(nrows); //references
		long sparseRowSize = 16; // object
		sparseRowSize += 2*4; // 2 integers + padding
		sparseRowSize += MemoryEstimates.intArrayCost(0);
		sparseRowSize += MemoryEstimates.doubleArrayCost(0);
		sparseRowSize += 12*Math.max(1, cnnz); //avoid bias by down cast for ultra-sparse
		size += rlen * sparseRowSize; //sparse rows

		// robustness for long overflows
		return (long) Math.min(size, Long.MAX_VALUE);
	}

	@Override
	public long getExactSizeInMemory() {
		double size = 16; //object
		size += MemoryEstimates.objectArrayCost(_rows.length); //references

		for (SparseRow sr : _rows) {
			if (sr == null)
				continue;
			long sparseRowSize = 16; // object
			if( sr instanceof SparseRowScalar )
				sparseRowSize += 12;
			else { //SparseRowVector
				sparseRowSize += 2*4; // 2 integers
				sparseRowSize += MemoryEstimates.intArrayCost(0);
				sparseRowSize += MemoryEstimates.doubleArrayCost(0);
				sparseRowSize += 12*((SparseRowVector)sr).capacity();
			}
			size += sparseRowSize; //sparse rows
		}

		// robustness for long overflows
		return (long) Math.min(size, Long.MAX_VALUE);
	}

	///////////////////
	//SparseBlock implementation

	@Override
	public void allocate(int r) {
		if( !isAllocated(r) )
			_rows[r] = new SparseRowVector();
	}
	
	@Override
	public void allocate(int r, int nnz) {
		if( !isAllocated(r) )
			_rows[r] = (nnz == 1) ? new SparseRowScalar() :
				new SparseRowVector(nnz);
	}
	
	@Override
	public void allocate(int r, int ennz, int maxnnz) {
		if( !isAllocated(r) )
			_rows[r] = (ennz == 1) ? new SparseRowScalar() :
				new SparseRowVector(ennz, maxnnz);
	}
	
	@Override
	public void compact(int r) {
		if( isAllocated(r) ){
			if(_rows[r] instanceof SparseRowVector
			  && _rows[r].size() > SparseBlock.INIT_CAPACITY
			  && _rows[r].size() * SparseBlock.RESIZE_FACTOR1 
				  < ((SparseRowVector)_rows[r]).capacity() ) {
			  ((SparseRowVector)_rows[r]).compact();
		  }
		  else if(_rows[r] instanceof SparseRowScalar) {
				SparseRowScalar s = (SparseRowScalar) _rows[r];
				if(s.getValue() == 0)
					_rows[r] = null;
			}
		}
	}

	@Override
	public void compact() {
		for(int i = 0; i < numRows(); i++) {
			if(isAllocated(i)) {
				if(_rows[i] instanceof SparseRowVector) {
					_rows[i].compact();
					if(_rows[i].isEmpty()) _rows[i] = null;
				}
				else if(_rows[i] instanceof SparseRowScalar) {
					SparseRowScalar s = (SparseRowScalar) _rows[i];
					if(s.getValue() == 0) _rows[i] = null;
				}
			}
		}
	}

	@Override
	public SparseBlock.Type getSparseBlockType() {
		return Type.MCSR;
	}
	
	@Override
	public int numRows() {
		return _rows.length;
	}

	@Override
	public boolean isThreadSafe() {
		return true;
	}
	
	@Override
	public boolean isContiguous() {
		return false;
	}
	
	@Override
	public final boolean isAllocated(int r) {
		return _rows[r] != null;
	}

	@Override
	public boolean checkValidity(int rlen, int clen, long nnz, boolean strict) {

		//1. Correct meta data
		if( rlen < 0 || clen < 0 )
			throw new RuntimeException("Invalid block dimensions: ("+rlen+", "+clen+").");

		//2. Correct array lengths
		if( size() < nnz )
			throw new RuntimeException("Incorrect size: "+size()+" (expected: "+nnz+").");

		//3. Sorted column indices per row
		for( int i=0; i<rlen; i++ ) {
			if( isEmpty(i) ) continue;
			int apos = pos(i);
			int alen = size(i);
			int[] aix = indexes(i);
			double[] avals = values(i);

			int prevCol = -1;
			for (int k = apos; k < apos + alen; k++) {
				if(aix[k] < 0)
					throw new RuntimeException(
						"Invalid index, at row=" + i + ", pos=" + k);
				if(aix[k] <= prevCol)
					throw new RuntimeException(
						"Wrong sparse row ordering, at row=" + i + ", pos=" + k + " with column indexes " +
							prevCol + ">=" + aix[k]);
				if(avals[k] == 0)
					throw new RuntimeException(
						"The values array should not contain zeros " + "but zero at row: " + i + ", column pos: " + k);
				prevCol = aix[k];
			}
		}

		//3. A capacity that is no larger than nnz times resize factor
		for( int i=0; i<rlen; i++ ) {
			long max_size = (long)Math.max(nnz*RESIZE_FACTOR1, INIT_CAPACITY);
			if( !isEmpty(i) && values(i).length > max_size )
				throw new RuntimeException("The capacity is larger than nnz times a resize factor(=2). "
					+ "Actual length = " + values(i).length+", should not exceed "+max_size);
		}

		return true;
	}

	@Override 
	public void reset() {
		for( SparseRow row : _rows )
			if( row != null )
				row.reset(row.size(), Integer.MAX_VALUE);
	}
	
	@Override 
	public void reset(int ennz, int maxnnz) {
		for( SparseRow row : _rows )
			if( row != null )
				row.reset(ennz, maxnnz);
	}
	
	@Override 
	public void reset(int r, int ennz, int maxnnz) {
		if( isAllocated(r) )
			_rows[r].reset(ennz, maxnnz);
	}
	
	@Override
	public long size() {
		//recompute non-zeros to avoid redundant maintenance
		long nnz = 0;
		for( SparseRow row : _rows )
			if( row != null ) 
				nnz += row.size();
		return nnz;
	}

	@Override
	public int size(int r) {
		//prior check with isEmpty(r) expected
		return isAllocated(r) ? _rows[r].size() : 0;
	}
	
	@Override
	public long size(int rl, int ru) {
		long ret = 0;
		for( int i=rl; i<ru; i++ )
			ret += isAllocated(i) ? _rows[i].size() : 0;
		return ret;
	}

	@Override
	public long size(int rl, int ru, int cl, int cu) {
		long nnz = 0;
		for(int i = rl; i < ru; i++)
			if(!isEmpty(i)) {
				int start = posFIndexGTE(i, cl);
				int end = posFIndexLTE(i, cu - 1);
				nnz += (start != -1 && end != -1) ? (end - start + 1) : 0;
			}
		return nnz;
	}

	@Override
	public final boolean isEmpty(int r) {
		return _rows[r] == null || _rows[r].isEmpty();
	}
	
	@Override
	public int[] indexes(int r) {
		//prior check with isEmpty(r) expected
		return _rows[r].indexes();
	}

	@Override
	public double[] values(int r) {
		//prior check with isEmpty(r) expected
		return _rows[r].values();
	}

	@Override
	public int pos(int r) {
		//arrays per row (always start 0)
		return 0;
	}

	@Override
	public boolean set(int r, int c, double v) {
		if( !isAllocated(r) )
			_rows[r] = new SparseRowScalar();
		else if( _rows[r] instanceof SparseRowScalar && !_rows[r].isEmpty())
			_rows[r] = new SparseRowVector(_rows[r]);
		return _rows[r].set(c, v);
	}

	@Override
	public void set(int r, SparseRow row, boolean deep) {
		//copy values into existing row to avoid allocation
		if( isAllocated(r) && _rows[r] instanceof SparseRowVector
			&& ((SparseRowVector)_rows[r]).capacity() >= row.size() && deep )
			((SparseRowVector)_rows[r]).copy(row);
		//set new sparse row (incl allocation if required)
		else 
			_rows[r] = (deep && row != null) ?
				new SparseRowVector(row) : row;
	}
	
	@Override
	public boolean add(int r, int c, double v) {
		if( !isAllocated(r) )
			_rows[r] = new SparseRowScalar();
		else if( _rows[r] instanceof SparseRowScalar && !_rows[r].isEmpty()){
			SparseRowScalar s = (SparseRowScalar) _rows[r];
			if(s.getIndex() == c)
				return s.set(s.getIndex(), v + s.getValue());
			else
			  _rows[r] = new SparseRowVector(_rows[r]);
		}
		return _rows[r].add(c, v);
	}
	
	@Override
	public final void append(final int r, final int c, final double v) {
		// Perf verified in java -jar target/systemds-3.3.0-SNAPSHOT-perf.jar 1004 1000 100000
		if(v == 0)
			return;
		else if(_rows[r] == null)
			_rows[r] = new SparseRowScalar(c, v);
		else
			_rows[r] = _rows[r].append(c, v);
	}

	@Override
	public void setIndexRange(int r, int cl, int cu, double[] v, int vix, int vlen) {
		if( !isAllocated(r) )
			_rows[r] = new SparseRowVector();
		else if( _rows[r] instanceof SparseRowScalar )
			_rows[r] = new SparseRowVector(_rows[r]);
		//different sparse row semantics: upper bound inclusive
		((SparseRowVector)_rows[r]).setIndexRange(cl, cu-1, v, vix, vlen);
	}
	
	@Override
	public void setIndexRange(int r, int cl, int cu, double[] v, int[] vix, int vpos, int vlen) {
		if( !isAllocated(r) )
			_rows[r] = new SparseRowVector();
		else if( _rows[r] instanceof SparseRowScalar )
			_rows[r] = new SparseRowVector(_rows[r]);
		//different sparse row semantics: upper bound inclusive
		((SparseRowVector)_rows[r]).setIndexRange(cl, cu-1, v, vix, vpos, vlen);
	}

	@Override
	public void deleteIndexRange(int r, int cl, int cu) {
		//prior check with isEmpty(r) expected
		//different sparse row semantics: upper bound inclusive
		if( _rows[r] instanceof SparseRowScalar )
			_rows[r] = new SparseRowVector(_rows[r]);
		((SparseRowVector)_rows[r]).deleteIndexRange(cl, cu-1);
	}

	@Override
	public void sort() {
		for( SparseRow row : _rows )
			if( row != null && !row.isEmpty() )
				row.sort();
	}

	@Override
	public void sort(int r) {
		//prior check with isEmpty(r) expected
		_rows[r].sort();
	}

	@Override
	public double get(int r, int c) {
		if( !isAllocated(r) )
			return 0;
		return _rows[r].get(c); 
	}
	
	@Override
	public SparseRow get(int r) {
		return _rows[r]; 
	}

	@Override
	public int posFIndexLTE(int r, int c) {
		//prior check with isEmpty(r) expected
		if( _rows[r] instanceof SparseRowScalar )
			_rows[r] = new SparseRowVector(_rows[r]);
		return ((SparseRowVector)_rows[r]).searchIndexesFirstLTE(c);
	}

	@Override
	public int posFIndexGTE(int r, int c) {
		return _rows[r].searchIndexesFirstGTE(c);
	}

	@Override
	public int posFIndexGT(int r, int c) {
		return _rows[r].searchIndexesFirstGT(c);
	}

	public void setNnzEstimatePerRow(int nnzPerCol, int nCol){
		for(SparseRow s : _rows){
			if(s instanceof SparseRowVector){
				SparseRowVector sv = (SparseRowVector)s;
				sv.setEstimatedNzs(nnzPerCol);
			}
			else if(s == null){
				s = new SparseRowVector(nnzPerCol, nCol);
			}
		}
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		final int nRow = numRows();
		sb.append("SparseBlockMCSR: rlen=");
		sb.append(nRow);
		sb.append(", nnz=");
		sb.append(size());
		sb.append("\n");
		final int rowDigits = (int)Math.max(Math.ceil(Math.log10(nRow)),1) ;
		for( int i=0; i<nRow; i++ ) {
			if(isEmpty(i))
				continue;
			sb.append(String.format("%0"+rowDigits+"d %s\n", i, _rows[i].toString()));
		}
		
		return sb.toString();
	}
	
	@Override
	public Iterator<Integer> getNonEmptyRowsIterator(int rl, int ru) {
		return new NonEmptyRowsIteratorMCSR(rl, ru);
	}
	
	public class NonEmptyRowsIteratorMCSR implements Iterator<Integer> {
		private int _rpos;
		private final int _ru;
		
		public NonEmptyRowsIteratorMCSR(int rl, int ru) {
			_rpos = rl;
			_ru = ru;
		}
		
		@Override
		public boolean hasNext() {
			while( _rpos<_ru && isEmpty(_rpos) )
				_rpos++;
			return _rpos < _ru;
		}

		@Override
		public Integer next() {
			return _rpos++;
		}
	}
	
	/**
	 * Helper function for MCSR -&gt; {COO, CSR}
	 * @return the underlying array of {@link SparseRow}
	 */
	public SparseRow[] getRows() {
		return _rows;
	}
}
