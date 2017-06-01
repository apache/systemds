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

package org.apache.sysml.runtime.compress;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.compress.utils.LinearAlgebraUtils;
import org.apache.sysml.runtime.functionobjects.Builtin;
import org.apache.sysml.runtime.functionobjects.KahanFunction;
import org.apache.sysml.runtime.functionobjects.KahanPlus;
import org.apache.sysml.runtime.functionobjects.KahanPlusSq;
import org.apache.sysml.runtime.functionobjects.ReduceAll;
import org.apache.sysml.runtime.functionobjects.ReduceCol;
import org.apache.sysml.runtime.functionobjects.ReduceRow;
import org.apache.sysml.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysml.runtime.matrix.data.IJV;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.AggregateUnaryOperator;


/**
 * Base class for column groups encoded with various types of bitmap encoding.
 * 
 * 
 * NOTES:
 *  * OLE: separate storage segment length and bitmaps led to a 30% improvement
 *    but not applied because more difficult to support both data layouts at the
 *    same time (distributed/local as well as w/ and w/o low-level opt)
 */
public abstract class ColGroupOffset extends ColGroupValue 
{
	private static final long serialVersionUID = -1635828933479403125L;

	protected static final boolean CREATE_SKIPLIST = true;
	
	protected static final int READ_CACHE_BLKSZ = 2 * BitmapEncoder.BITMAP_BLOCK_SZ;
	public static final int WRITE_CACHE_BLKSZ = 2 * BitmapEncoder.BITMAP_BLOCK_SZ;
	public static boolean ALLOW_CACHE_CONSCIOUS_ROWSUMS = true;
	
	/** Bitmaps, one per uncompressed value in {@link #_values}. */
	protected int[] _ptr; //bitmap offsets per value
	protected char[] _data; //linearized bitmaps (variable length)
	protected boolean _zeros; //contains zero values
	
	protected int[] _skiplist;
	
	public ColGroupOffset() {
		super();
	}
	
	/**
	 * Main constructor. Stores the headers for the individual bitmaps.
	 * 
	 * @param colIndices
	 *            indices (within the block) of the columns included in this
	 *            column
	 * @param numRows
	 *            total number of rows in the parent block
	 * @param ubm
	 *            Uncompressed bitmap representation of the block
	 */
	public ColGroupOffset(int[] colIndices, int numRows, UncompressedBitmap ubm) {
		super(colIndices, numRows, ubm);
		_zeros = (ubm.getNumOffsets() < numRows);
	}

	/**
	 * Constructor for subclass methods that need to create shallow copies
	 * 
	 * @param colIndices
	 *            raw column index information
	 * @param numRows
	 *            number of rows in the block
	 * @param zeros
	 * 			  indicator if column group contains zero values
	 * @param values
	 *            set of distinct values for the block (associated bitmaps are
	 *            kept in the subclass)
	 */
	protected ColGroupOffset(int[] colIndices, int numRows, boolean zeros, double[] values) {
		super(colIndices, numRows, values);
		_zeros = zeros;
	}
	
	protected final int len(int k) {
		return _ptr[k+1] - _ptr[k];
	}

	protected void createCompressedBitmaps(int numVals, int totalLen, char[][] lbitmaps) {
		// compact bitmaps to linearized representation
		_ptr = new int[numVals+1];
		_data = new char[totalLen];
		for( int i=0, off=0; i<numVals; i++ ) {
			int len = lbitmaps[i].length;
			_ptr[i] = off;
			System.arraycopy(lbitmaps[i], 0, _data, off, len);
			off += len;
		}
		_ptr[numVals] = totalLen;
	}
	
	@Override
	public long estimateInMemorySize() {
		long size = super.estimateInMemorySize();
		
		// adding bitmaps size
		size += 16; //array references
		if (_data != null) {
			size += 32 + _ptr.length * 4; // offsets
			size += 32 + _data.length * 2;    // bitmaps
		}
	
		return size;
	}

	//generic decompression for OLE/RLE, to be overwritten for performance
	@Override
	public void decompressToBlock(MatrixBlock target, int rl, int ru) 
	{
		final int numCols = getNumCols();
		final int numVals = getNumValues();
		int[] colIndices = getColIndices();
		
		// Run through the bitmaps for this column group
		for (int i = 0; i < numVals; i++) {
			Iterator<Integer> decoder = getIterator(i);
			int valOff = i*numCols;

			while (decoder.hasNext()) {
				int row = decoder.next();
				if( row<rl ) continue;
				if( row>ru ) break;
				
				for (int colIx = 0; colIx < numCols; colIx++)
					target.appendValue(row, colIndices[colIx], _values[valOff+colIx]);
			}
		}
	}

	//generic decompression for OLE/RLE, to be overwritten for performance
	@Override
	public void decompressToBlock(MatrixBlock target, int[] colIndexTargets) 
	{
		final int numCols = getNumCols();
		final int numVals = getNumValues();
		
		// Run through the bitmaps for this column group
		for (int i = 0; i < numVals; i++) {
			Iterator<Integer> decoder = getIterator(i);
			int valOff = i*numCols;

			while (decoder.hasNext()) {
				int row = decoder.next();
				for (int colIx = 0; colIx < numCols; colIx++) {
					int origMatrixColIx = getColIndex(colIx);
					int targetColIx = colIndexTargets[origMatrixColIx];
					target.quickSetValue(row, targetColIx, _values[valOff+colIx]);
				}
			}
		}
	}
	
	//generic decompression for OLE/RLE, to be overwritten for performance
	@Override
	public void decompressToBlock(MatrixBlock target, int colpos) 
	{
		final int numCols = getNumCols();
		final int numVals = getNumValues();
		
		// Run through the bitmaps for this column group
		for (int i = 0; i < numVals; i++) {
			Iterator<Integer> decoder = getIterator(i);
			int valOff = i*numCols;

			while (decoder.hasNext()) {
				int row = decoder.next();
				target.quickSetValue(row, 0, _values[valOff+colpos]);
			}
		}
	}

	//generic get for OLE/RLE, to be overwritten for performance
	//potential: skip scan (segment length agg and run length) instead of decode
	@Override
	public double get(int r, int c) {
		//find local column index
		int ix = Arrays.binarySearch(_colIndexes, c);
		if( ix < 0 )
			throw new RuntimeException("Column index "+c+" not in bitmap group.");
		
		//find row index in value offset lists via scan
		final int numCols = getNumCols();
		final int numVals = getNumValues();
		for (int i = 0; i < numVals; i++) {
			Iterator<Integer> decoder = getIterator(i);
			int valOff = i*numCols;
			while (decoder.hasNext()) {
				int row = decoder.next();
				if( row == r )
					return _values[valOff+ix];
				else if( row > r )
					break; //current value
			}
		}		
		return 0;
	}

	protected final void sumAllValues(double[] b, double[] c)
	{
		final int numVals = getNumValues();
		final int numCols = getNumCols();
		
		//vectMultiplyAdd over cols instead of dotProduct over vals because
		//usually more values than columns
		for( int i=0, off=0; i<numCols; i++, off+=numVals )
			LinearAlgebraUtils.vectMultiplyAdd(b[i], _values, c, off, 0, numVals);
	}

	protected final double mxxValues(int bitmapIx, Builtin builtin)
	{
		final int numCols = getNumCols();
		final int valOff = bitmapIx * numCols;
		
		double val = Double.MAX_VALUE * ((builtin.getBuiltinCode()==BuiltinCode.MAX)?-1:1);
		for( int i = 0; i < numCols; i++ )
			val = builtin.execute2(val, _values[valOff+i]);
		
		return val;
	}

	public char[] getBitmaps() {
		return _data;
	}
	
	public int[] getBitmapOffsets() {
		return _ptr;
	}

	public boolean hasZeros() {
		return _zeros;
	}

	/**
	 * Utility function of sparse-unsafe operations.
	 * 
	 * @param ind row indicator vector of non zeros
	 * @return offsets
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	protected int[] computeOffsets(boolean[] ind)
		throws DMLRuntimeException 
	{
		//determine number of offsets
		int numOffsets = 0;
		for( int i=0; i<ind.length; i++ )
			numOffsets += ind[i] ? 1 : 0;
		
		//create offset lists
		int[] ret = new int[numOffsets];
		for( int i=0, pos=0; i<ind.length; i++ )
			if( ind[i] )
				ret[pos++] = i;
		
		return ret;
	}

	@Override
	public void readFields(DataInput in) 
		throws IOException 
	{
		_numRows = in.readInt();
		int numCols = in.readInt();
		int numVals = in.readInt();
		_zeros = in.readBoolean();
		
		//read col indices
		_colIndexes = new int[ numCols ];
		for( int i=0; i<numCols; i++ )
			_colIndexes[i] = in.readInt();
		
		//read distinct values
		_values = new double[numVals*numCols];
		for( int i=0; i<numVals*numCols; i++ )
			_values[i] = in.readDouble();
		
		//read bitmaps
		int totalLen = in.readInt();
		_ptr = new int[numVals+1];
		_data = new char[totalLen];		
		for( int i=0, off=0; i<numVals; i++ ) {
			int len = in.readInt();
			_ptr[i] = off;
			for( int j=0; j<len; j++ )
				_data[off+j] = in.readChar();
			off += len;
		}
		_ptr[numVals] = totalLen;
	}
	
	@Override
	public void write(DataOutput out) 
		throws IOException 
	{
		int numCols = getNumCols();
		int numVals = getNumValues();
		out.writeInt(_numRows);
		out.writeInt(numCols);
		out.writeInt(numVals);
		out.writeBoolean(_zeros);
		
		//write col indices
		for( int i=0; i<_colIndexes.length; i++ )
			out.writeInt( _colIndexes[i] );
		
		//write distinct values
		for( int i=0; i<_values.length; i++ )
			out.writeDouble(_values[i]);

		//write bitmaps (lens and data, offset later recreated)
		int totalLen = 0;
		for( int i=0; i<numVals; i++ )
			totalLen += len(i);
		out.writeInt(totalLen);	
		for( int i=0; i<numVals; i++ ) {
			int len = len(i);
			int off = _ptr[i];
			out.writeInt(len);
			for( int j=0; j<len; j++ )
				out.writeChar(_data[off+j]);
		}
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = 13; //header
		//col indices
		ret += 4 * _colIndexes.length; 
		//distinct values (groups of values)
		ret += 8 * _values.length;
		//actual bitmaps
		ret += 4; //total length
		for( int i=0; i<getNumValues(); i++ )
			ret += 4 + 2 * len(i);
		
		return ret;
	}
	

	
	@Override
	public void unaryAggregateOperations(AggregateUnaryOperator op, MatrixBlock result, int rl, int ru) 
		throws DMLRuntimeException 
	{
		//sum and sumsq (reduceall/reducerow over tuples and counts)
		if( op.aggOp.increOp.fn instanceof KahanPlus || op.aggOp.increOp.fn instanceof KahanPlusSq ) 
		{
			KahanFunction kplus = (op.aggOp.increOp.fn instanceof KahanPlus) ?
					KahanPlus.getKahanPlusFnObject() : KahanPlusSq.getKahanPlusSqFnObject();
			
			if( op.indexFn instanceof ReduceAll )
				computeSum(result, kplus);
			else if( op.indexFn instanceof ReduceCol )
				computeRowSums(result, kplus, rl, ru);
			else if( op.indexFn instanceof ReduceRow )
				computeColSums(result, kplus);
		}
		//min and max (reduceall/reducerow over tuples only)
		else if(op.aggOp.increOp.fn instanceof Builtin 
				&& (((Builtin)op.aggOp.increOp.fn).getBuiltinCode()==BuiltinCode.MAX 
				|| ((Builtin)op.aggOp.increOp.fn).getBuiltinCode()==BuiltinCode.MIN)) 
		{		
			Builtin builtin = (Builtin) op.aggOp.increOp.fn;

			if( op.indexFn instanceof ReduceAll )
				computeMxx(result, builtin, _zeros);
			else if( op.indexFn instanceof ReduceCol )
				computeRowMxx(result, builtin, rl, ru);
			else if( op.indexFn instanceof ReduceRow )
				computeColMxx(result, builtin, _zeros);
		}
	}
	
	protected abstract void computeSum(MatrixBlock result, KahanFunction kplus);
	
	protected abstract void computeRowSums(MatrixBlock result, KahanFunction kplus, int rl, int ru);
	
	protected abstract void computeColSums(MatrixBlock result, KahanFunction kplus);
	
	protected abstract void computeRowMxx(MatrixBlock result, Builtin builtin, int rl, int ru);

	protected abstract boolean[] computeZeroIndicatorVector();
	
	@Override
	public Iterator<IJV> getIterator(int rl, int ru, boolean inclZeros, boolean rowMajor) {
		if( rowMajor )
			return new OffsetRowIterator(rl, ru, inclZeros);
		else
			return new OffsetValueIterator(rl, ru, inclZeros);
	}
	
	/**
	 * @param k index of value tuple with associated bitmap
	 * @return an iterator over the row offsets in this bitmap
	 */
	public abstract Iterator<Integer> getIterator(int k);

	/**
	 * 
	 * @param k index of value tuple with associated bitmap
	 * @param rl row lower index, inclusive
	 * @param ru row upper index, exclusive
	 * @return an iterator over the row offsets in this bitmap
	 */
	public abstract Iterator<Integer> getIterator(int k, int rl, int ru);

	
	protected class OffsetValueIterator implements Iterator<IJV>
	{
		//iterator configuration
		private final int _rl;
		private final int _ru;
		private final boolean _inclZeros;
		
		//iterator state
		private final IJV _buff = new IJV();
		private Iterator<Integer> _viter = null;
		private int _vpos = -1;
		private int _rpos = -1;
		private int _cpos = -1;
		
		public OffsetValueIterator(int rl, int ru, boolean inclZeros) {
			_rl = rl;
			_ru = ru;
			_inclZeros = inclZeros;
			_vpos = -1;
			_rpos = -1;
			_cpos = 0;
			getNextValue();
		}

		@Override
		public boolean hasNext() {
			return (_rpos < _ru);
		}
		
		@Override
		public IJV next() {
			if( !hasNext() )
				throw new RuntimeException("No more offset entries.");
			_buff.set(_rpos, _colIndexes[_cpos], (_vpos >= getNumValues()) ? 
				0 : _values[_vpos*getNumCols()+_cpos]);
			getNextValue();
			return _buff;
		}
		
		private void getNextValue() {
			//advance to next value iterator if required
			if(_viter != null && _viter instanceof ZeroValueIterator && !_viter.hasNext() ) {
				_rpos = _ru; //end after zero iterator
				return;
			}
			else if( (_rpos< 0 || _cpos+1 >= getNumCols()) 
					&& !(_viter!=null && _viter.hasNext()) ) {
				do {
					_vpos++;
					if( _vpos < getNumValues() )
						_viter = getIterator(_vpos, _rl, _ru);
					else if( _inclZeros && _zeros)
						_viter = new ZeroValueIterator(_rl, _ru);
					else {
						_rpos = _ru; //end w/o zero iterator
						return;
					}
				}
				while(!_viter.hasNext());
				_rpos = -1;
			}
			
			//get next value from valid iterator
			if( _rpos < 0 || _cpos+1 >= getNumCols() ) {
				_rpos = _viter.next();
				_cpos = 0;
			}
			else {
				_cpos++;
			}
		}
	}
	
	protected class ZeroValueIterator implements Iterator<Integer>
	{
		private final boolean[] _zeros;
		private final int _ru;
		private int _rpos; 
		
		public ZeroValueIterator(int rl, int ru) {
			_zeros = computeZeroIndicatorVector();
			_ru = ru;
			_rpos = rl-1;
			getNextValue();
		}

		@Override
		public boolean hasNext() {
			return (_rpos < _ru);
		}

		@Override
		public Integer next() {
			int ret = _rpos;
			getNextValue();
			return ret;
		}
		
		private void getNextValue() {
			do { _rpos++; }
			while( _rpos < _ru && !_zeros[_rpos] );
		}
	}
	
	protected class OffsetRowIterator implements Iterator<IJV>
	{
		//iterator configuration
		private final int _rl;
		private final int _ru;
		private final boolean _inclZeros;
		
		//iterator state
		private final Iterator<Integer>[] _iters;
		private final IJV _ret = new IJV(); 
		private final HashMap<Integer,Integer> _ixbuff = 
			new HashMap<Integer,Integer>(); //<rowid-value>
		private int _rpos;
		private int _cpos;
		private int _vpos;
		
		@SuppressWarnings("unchecked")
		public OffsetRowIterator(int rl, int ru, boolean inclZeros) {
			_rl = rl;
			_ru = ru;
			_inclZeros = inclZeros;
			
			//initialize array of column group iterators
			_iters = new Iterator[getNumValues()];
			for( int k=0; k<getNumValues(); k++ )
				_iters[k] = getIterator(k, _rl, _ru);
			
			//initialize O(1)-lookup for next value
			for( int k=0; k<getNumValues(); k++ ) {
				_ixbuff.put(_iters[k].hasNext() ? 
						_iters[k].next() : _ru+k, k);
			}
			
			//get initial row
			_rpos = rl-1;
			_cpos = getNumCols()-1;
			getNextValue();
		}
		
		@Override
		public boolean hasNext() {
			return (_rpos < _ru);
		}
		
		@Override
		public IJV next() {
			if( !hasNext() )
				throw new RuntimeException("No more offset entries.");
			_ret.set(_rpos, _colIndexes[_cpos], 
				(_vpos<0) ? 0 : getValue(_vpos, _cpos));
			getNextValue();
			return _ret;
		}
		
		private void getNextValue() {
			do {
				//read iterators if necessary
				if( _cpos+1 >= getNumCols() ) {
					_rpos++; _cpos = -1; _vpos = -1;
					//direct lookup of single value to pull next index
					Integer ktmp = _ixbuff.remove(_rpos);
					if( ktmp != null ) {
						_ixbuff.put(_iters[ktmp].hasNext() ? 
								_iters[ktmp].next() : _ru+ktmp, ktmp);
						_vpos = ktmp;
					}
				}
				//check for end of row partition
				if( _rpos >= _ru )
					return;
				_cpos++;
			}
			while( !_inclZeros && (_vpos < 0 
				|| getValue(_vpos, _cpos)==0) );
		}
	}
}
