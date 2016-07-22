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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.TreeMap;
import java.util.Map.Entry;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.compress.utils.LinearAlgebraUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.ScalarOperator;


/**
 * Base class for column groups encoded with various types of bitmap encoding.
 * 
 * 
 * NOTES:
 *  * OLE: separate storage segment length and bitmaps led to a 30% improvement
 *    but not applied because more difficult to support both data layouts at the
 *    same time (distributed/local as well as w/ and w/o low-level opt)
 */
public abstract class ColGroupBitmap extends ColGroup 
{
	private static final long serialVersionUID = -1635828933479403125L;
	
	public static final boolean LOW_LEVEL_OPT = true;	
	//sorting of values by physical length helps by 10-20%, especially for serial, while
	//slight performance decrease for parallel incl multi-threaded, hence not applied for
	//distributed operations (also because compression time + garbage collection increases)
	private static final boolean SORT_VALUES_BY_LENGTH = true; 
	protected static final boolean CREATE_SKIPLIST = true;
	
	protected static final int READ_CACHE_BLKSZ = 2 * BitmapEncoder.BITMAP_BLOCK_SZ;
	protected static final int WRITE_CACHE_BLKSZ = 2 * BitmapEncoder.BITMAP_BLOCK_SZ;
	
	/** Distinct values associated with individual bitmaps. */
	protected double[] _values; //linearized <numcol vals> <numcol vals>

	/** Bitmaps, one per uncompressed value in {@link #values}. */
	protected int[] _ptr; //bitmap offsets per value
	protected char[] _data; //linearized bitmaps (variable length)
	
	protected int[] _skiplist;
	
	public ColGroupBitmap(CompressionType type) {
		super(type, (int[]) null, -1);
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
	public ColGroupBitmap(CompressionType type, int[] colIndices, int numRows, UncompressedBitmap ubm) 
	{
		super(type, colIndices, numRows);

		// Extract and store just the distinct values. The bitmaps themselves go
		// into the subclasses.
		final int numCols = ubm.getNumColumns();
		final int numVals = ubm.getNumValues();
		
		_values = new double[numVals*numCols];
		for (int i=0; i<numVals; i++) {
			//note: deep copied internally on getValues
			double[] tmp = ubm.getValues(i);
			System.arraycopy(tmp, 0, _values, i*numCols, numCols);
		}
	}

	/**
	 * Constructor for subclass methods that need to create shallow copies
	 * 
	 * @param colIndices
	 *            raw column index information
	 * @param numRows
	 *            number of rows in the block
	 * @param values
	 *            set of distinct values for the block (associated bitmaps are
	 *            kept in the subclass)
	 */
	protected ColGroupBitmap(CompressionType type, int[] colIndices, int numRows, double[] values) {
		super(type, colIndices, numRows);
		_values = values;
	}
	
	protected final int len(int k) {
		return _ptr[k+1] - _ptr[k];
	}

	/**
	 * 
	 * @param numVals
	 * @param totalLen
	 * @param lbitmaps
	 */
	protected void createCompressedBitmaps(int numVals, int totalLen, char[][] lbitmaps)
	{
		// compact bitmaps to linearized representation
		if( LOW_LEVEL_OPT && SORT_VALUES_BY_LENGTH
			&& _numRows > BitmapEncoder.BITMAP_BLOCK_SZ ) 
		{
			// sort value by num segments in descending order
			TreeMap<Integer,ArrayList<Integer>> tree = new TreeMap<Integer, ArrayList<Integer>>();
			for( int i=0; i<numVals; i++ ) {
				int revlen = totalLen-lbitmaps[i].length;
				if( !tree.containsKey(revlen) )
					tree.put(revlen, new ArrayList<Integer>());
				tree.get(revlen).add(i);
			}
			
			// compact bitmaps to linearized representation
			_ptr = new int[numVals+1];
			_data = new char[totalLen];
			int pos = 0, off = 0;
			for( Entry<Integer,ArrayList<Integer>> e : tree.entrySet() ) {
				for( Integer tmpix : e.getValue() ) {
					int len = lbitmaps[tmpix].length;
					_ptr[pos] = off;
					System.arraycopy(lbitmaps[tmpix], 0, _data, off, len);
					off += len;
					pos++;
				}
			}
			_ptr[numVals] = totalLen;
			
			// reorder values
			double[] lvalues = new double[_values.length];
			int off2 = 0; int numCols = _colIndexes.length;
			for( Entry<Integer,ArrayList<Integer>> e : tree.entrySet() ) {
				for( Integer tmpix : e.getValue() ) {
					System.arraycopy(_values, tmpix*numCols, lvalues, off2, numCols);				
					off2 += numCols;
				}
			}			
			_values = lvalues;
		}
		else
		{
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
	}
	
	@Override
	public long estimateInMemorySize() {
		long size = super.estimateInMemorySize();
		
		// adding the size of values
		size += 8; //array reference
		if (_values != null) {
			size += 32 + _values.length * 8; //values
		}
		
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
	public void decompressToBlock(MatrixBlock target) 
	{
		final int numCols = getNumCols();
		final int numVals = getNumValues();
		int[] colIndices = getColIndices();
		
		// Run through the bitmaps for this column group
		for (int i = 0; i < numVals; i++) {
			Iterator<Integer> decoder = getDecodeIterator(i);
			int valOff = i*numCols;

			while (decoder.hasNext()) {
				int row = decoder.next();
				for (int colIx = 0; colIx < numCols; colIx++) {
					target.appendValue(row, colIndices[colIx], _values[valOff+colIx]);
				}
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
			Iterator<Integer> decoder = getDecodeIterator(i);
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
			Iterator<Integer> decoder = getDecodeIterator(i);
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
			Iterator<Integer> decoder = getDecodeIterator(i);
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
	
	/**
	 * 
	 * @param bitmapIx
	 * @return
	 */
	protected double sumValues(int bitmapIx)
	{
		final int numCols = getNumCols();
		final int valOff = bitmapIx * numCols;
		
		double val = 0.0;
		for( int i = 0; i < numCols; i++ ) {
			val += _values[valOff+i];
		}
		
		return val;
	}
	
	protected double sumValues(int bitmapIx, double[] b)
	{
		final int numCols = getNumCols();
		final int valOff = bitmapIx * numCols;
		
		double val = 0;
		for( int i = 0; i < numCols; i++ ) {
			val += _values[valOff+i] * b[i];
		}
		
		return val;
	}
	
	/**
	 * 
	 * @param b
	 * @param c
	 */
	protected void sumAllValues(double[] b, double[] c)
	{
		final int numVals = getNumValues();
		final int numCols = getNumCols();
		
		//vectMultiplyAdd over cols instead of dotProduct over vals because
		//usually more values than columns
		for( int i=0, off=0; i<numCols; i++, off+=numVals )
			LinearAlgebraUtils.vectMultiplyAdd(b[i], _values, c, off, 0, numVals);
	}

	/**
	 * Method for use by subclasses. Applies a scalar operation to the value
	 * metadata stored in the superclass.
	 * 
	 * @param op
	 *            scalar operation to perform
	 * @return transformed copy of value metadata for this column group
	 * @throws DMLRuntimeException
	 */
	protected double[] applyScalarOp(ScalarOperator op)
			throws DMLRuntimeException 
	{
		//scan over linearized values
		double[] ret = new double[_values.length];
		for (int i = 0; i < _values.length; i++) {
			ret[i] = op.executeScalar(_values[i]);
		}

		return ret;
	}
	
	/**
	 * 
	 * @param op
	 * @param newVal
	 * @param numCols
	 * @return
	 * @throws DMLRuntimeException
	 */
	protected double[] applyScalarOp(ScalarOperator op, double newVal, int numCols)
			throws DMLRuntimeException 
	{
		//scan over linearized values
		double[] ret = new double[_values.length + numCols];
		for( int i = 0; i < _values.length; i++ ) {
			ret[i] = op.executeScalar(_values[i]);
		}
		
		//add new value to the end
		Arrays.fill(ret, _values.length, _values.length+numCols, newVal);
		
		return ret;
	}

	/**
	 * @return the number of distinct sets of values associated with the bitmaps
	 *         in this column group
	 */
	public int getNumValues() {
		return _values.length / _colIndexes.length;
	}

	/**
	 * 
	 * @return
	 */
	public double[] getValues() {
		return _values;
	}
	
	/**
	 * 
	 * @return
	 */
	public char[] getBitmaps() {
		return _data;
	}
	
	public int[] getBitmapOffsets() {
		return _ptr;
	}

	/**
	 * @param bmpIx
	 *            index of a specific compressed bitmap (stored in subclass,
	 *            index same as {@link #values})
	 * @return an object for iterating over the row offsets in this bitmap. Only
	 *         valid until the next call to this method. May be reused across
	 *         calls.
	 */
	public abstract Iterator<Integer> getDecodeIterator(int bmpIx);

	/**
	 * Utility function of sparse-unsafe operations.
	 * 
	 * @param ind
	 * @return
	 * @throws DMLRuntimeException
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
		long ret = 12; //header
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
}
