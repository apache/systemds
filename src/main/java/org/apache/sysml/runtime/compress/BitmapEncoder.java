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

import java.util.ArrayList;
import java.util.Arrays;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.compress.utils.DblArray;
import org.apache.sysml.runtime.compress.utils.DblArrayIntListHashMap;
import org.apache.sysml.runtime.compress.utils.DoubleIntListHashMap;
import org.apache.sysml.runtime.compress.utils.IntArrayList;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.SparseBlock;


/** 
 * Static functions for encoding bitmaps in various ways. 
 * 
 */
public class BitmapEncoder 
{
	/** Size of the blocks used in a blocked bitmap representation. */
	public static final int BITMAP_BLOCK_SZ = 65536;
	
	/**
	 * Generate uncompressed bitmaps for a set of columns in an uncompressed
	 * matrix block.
	 * 
	 * @param colIndices
	 *            indexes (within the block) of the columns to extract
	 * @param rawblock
	 *            an uncompressed matrix block; can be dense or sparse
	 * @return uncompressed bitmap representation of the columns
	 * @throws DMLRuntimeException 
	 */
	public static UncompressedBitmap extractBitmap(int[] colIndices, MatrixBlock rawblock) 
	{
		//note: no sparse column selection reader because low potential
		//single column selection
		if( colIndices.length==1 ) {
			return extractBitmap(colIndices[0], rawblock, 
					!CompressedMatrixBlock.MATERIALIZE_ZEROS);
		}
		//multiple column selection	(general case)
		else { 
			ReaderColumnSelection reader = null;
			if( rawblock.isInSparseFormat() && CompressedMatrixBlock.TRANSPOSE_INPUT )
				reader = new ReaderColumnSelectionSparse(rawblock, colIndices, 
						!CompressedMatrixBlock.MATERIALIZE_ZEROS);
			else
				reader = new ReaderColumnSelectionDense(rawblock, colIndices,
						!CompressedMatrixBlock.MATERIALIZE_ZEROS); 
			
			return extractBitmap(colIndices, rawblock, reader);
		}
	}

	/**
	 * 
	 * @param colIndices
	 * @param rawblock
	 * @param sampleIndexes
	 * @return
	 */
	public static UncompressedBitmap extractBitmapFromSample(int[] colIndices,
			MatrixBlock rawblock, int[] sampleIndexes) 
	{
		//note: no sparse column selection reader because low potential
		
		//single column selection
		if( colIndices.length==1 ) {
			return extractBitmap(colIndices[0], rawblock, sampleIndexes,
					!CompressedMatrixBlock.MATERIALIZE_ZEROS);
		}
		//multiple column selection	(general case)
		else {			
			return extractBitmap(colIndices, rawblock,
					new ReaderColumnSelectionDenseSample(rawblock, colIndices,
							sampleIndexes, !CompressedMatrixBlock.MATERIALIZE_ZEROS));	
		}
	}

	/**
	 * Encodes the bitmap as a series of run lengths and offsets.
	 * <p>
	 * <b>NOTE: This method must be kept in sync with {@link BitmapDecoderRLE}
	 * !</b>
	 * 
	 * @param offsets
	 *            uncompressed contents of the bitmap, expressed as a list of
	 *            the offsets of different bits
	 * @return compressed version of said bitmap
	 */
	public static char[] genRLEBitmap(int[] offsets) {
		if( offsets.length == 0 )
			return new char[0]; //empty list

		// Use an ArrayList for correctness at the expense of temp space
		ArrayList<Character> buf = new ArrayList<Character>();

		// 1 + (position of last 1 in the previous run of 1's)
		// We add 1 because runs may be of length zero.
		int lastRunEnd = 0;

		// Offset between the end of the previous run of 1's and the first 1 in
		// the current run. Initialized below.
		int curRunOff;

		// Length of the most recent run of 1's
		int curRunLen = 0;

		// Current encoding is as follows:
		// Negative entry: abs(Entry) encodes the offset to the next lone 1 bit.
		// Positive entry: Entry encodes offset to next run of 1's. The next
		// entry in the bitmap holds a run length.

		// Special-case the first run to simplify the loop below.
		int firstOff = offsets[0];

		// The first run may start more than a short's worth of bits in
		while (firstOff > Character.MAX_VALUE) {
			buf.add(Character.MAX_VALUE);
			buf.add((char) 0);
			firstOff -= Character.MAX_VALUE;
			lastRunEnd += Character.MAX_VALUE;
		}

		// Create the first run with an initial size of 1
		curRunOff = firstOff;
		curRunLen = 1;

		// Process the remaining offsets
		for (int i = 1; i < offsets.length; i++) {

			int absOffset = offsets[i];

			// 1 + (last position in run)
			int curRunEnd = lastRunEnd + curRunOff + curRunLen;

			if (absOffset > curRunEnd || curRunLen >= Character.MAX_VALUE) {
				// End of a run, either because we hit a run of 0's or because the 
				// number of 1's won't fit in 16 bits. Add run to bitmap and start a new one.
				buf.add((char) curRunOff);
				buf.add((char) curRunLen);

				lastRunEnd = curRunEnd;
				curRunOff = absOffset - lastRunEnd;

				while (curRunOff > Character.MAX_VALUE) {
					// SPECIAL CASE: Offset to next run doesn't fit into 16 bits.
					// Add zero-length runs until the offset is small enough.
					buf.add(Character.MAX_VALUE);
					buf.add((char) 0);
					lastRunEnd += Character.MAX_VALUE;
					curRunOff -= Character.MAX_VALUE;
				}
				
				curRunLen = 1;
			} else {
				// Middle of a run
				curRunLen++;
			}
		}

		// Close out the last run
		if (curRunLen >= 1) {
			buf.add((char) curRunOff);
			buf.add((char) curRunLen);
		}

		// Convert wasteful ArrayList to packed array.
		char[] ret = new char[buf.size()];
		for (int i = 0; i < buf.size(); i++) {
			ret[i] = buf.get(i);
		}
		return ret;
	}

	/**
	 * Encodes the bitmap in blocks of offsets. Within each block, the bits are
	 * stored as absolute offsets from the start of the block.
	 * 
	 * @param offsets
	 *            uncompressed contents of the bitmap, expressed as a list of
	 *            the offsets of different bits
	 * @return compressed version of said bitmap
	 */
	public static char[] genOffsetBitmap(int[] offsets) 
	{
		int lastOffset = offsets[offsets.length - 1];

		// Build up the blocks
		int numBlocks = (lastOffset / BITMAP_BLOCK_SZ) + 1;
		// To simplify the logic, we make two passes.
		// The first pass divides the offsets by block.
		int[] blockLengths = new int[numBlocks];
		Arrays.fill(blockLengths, 0);

		for (int ix = 0; ix < offsets.length; ix++) {
			int val = offsets[ix];
			int blockForVal = val / BITMAP_BLOCK_SZ;

			blockLengths[blockForVal]++;
		}

		// The second pass creates the blocks.
		int totalSize = numBlocks;
		for (int block = 0; block < numBlocks; block++) {
			totalSize += blockLengths[block];
		}
		char[] encodedBlocks = new char[totalSize];

		int inputIx = 0;
		int blockStartIx = 0;
		for (int block = 0; block < numBlocks; block++) {
			int blockSz = blockLengths[block];

			// First entry in the block is number of bits
			encodedBlocks[blockStartIx] = (char) blockSz;

			for (int i = 0; i < blockSz; i++) {
				encodedBlocks[blockStartIx + i + 1] = (char) 
						(offsets[inputIx+i] % BITMAP_BLOCK_SZ);
			}

			inputIx += blockSz;
			blockStartIx += blockSz + 1;
		}

		return encodedBlocks;
	}

	/**
	 * 
	 * @param colIndex
	 * @param rawblock
	 * @return
	 * @throws DMLRuntimeException 
	 */
	private static UncompressedBitmap extractBitmap(int colIndex, MatrixBlock rawblock, boolean skipZeros) 
	{
		//probe map for distinct items (for value or value groups)
		DoubleIntListHashMap distinctVals = new DoubleIntListHashMap();
		
		//scan rows and probe/build distinct items
		final int m = CompressedMatrixBlock.TRANSPOSE_INPUT ?
				rawblock.getNumColumns():rawblock.getNumRows();
		
		if( rawblock.isInSparseFormat() //SPARSE 
			&& CompressedMatrixBlock.TRANSPOSE_INPUT )	
		{
			SparseBlock a = rawblock.getSparseBlock();
			if( a != null && !a.isEmpty(colIndex) ) 
			{
				int apos = a.pos(colIndex);
				int alen = a.size(colIndex);
				int[] aix = a.indexes(colIndex);
				double[] avals = a.values(colIndex);
				
				IntArrayList lstPtr0 = new IntArrayList(); //for 0 values
				int last = -1;
				//iterate over non-zero entries but fill in zeros
				for( int j=apos; j<apos+alen; j++ ) 
				{
					//fill in zero values
					if( !skipZeros )
						for( int k=last+1; k<aix[j]; k++ ) 
							lstPtr0.appendValue(k);
					//handle non-zero value
					IntArrayList lstPtr = distinctVals.get(avals[j]);	
					if( lstPtr == null ) {
						lstPtr = new IntArrayList();
						distinctVals.appendValue(avals[j], lstPtr);
					}
					lstPtr.appendValue(aix[j]);
					last = aix[j];
				}
				//fill in remaining zero values
				if( !skipZeros ) {
					for( int k=last+1; k<m; k++ )
						lstPtr0.appendValue(k);
					if( lstPtr0.size()>0 )
						distinctVals.appendValue(0, lstPtr0);
				}
			}
			else if( !skipZeros ) { //full 0 column 
				IntArrayList lstPtr = new IntArrayList();
				for( int i=0; i<m; i++ )
					lstPtr.appendValue(i);
				distinctVals.appendValue(0, lstPtr);
			}
		}
		else //GENERAL CASE
		{
			for( int i=0; i<m; i++ ) {
				double val = CompressedMatrixBlock.TRANSPOSE_INPUT ? 
						rawblock.quickGetValue(colIndex, i):
						rawblock.quickGetValue(i, colIndex);
				if( val!=0 || !skipZeros ) {		
					IntArrayList lstPtr = distinctVals.get(val);	
					if( lstPtr == null ) {
						lstPtr = new IntArrayList();
						distinctVals.appendValue(val, lstPtr);
					}
					lstPtr.appendValue(i);
				}
			}
		}
		
		return new UncompressedBitmap(distinctVals);
	}
	
	/**
	 * 
	 * @param colIndex
	 * @param rawblock
	 * @param sampleIndexes
	 * @return
	 */
	private static UncompressedBitmap extractBitmap(int colIndex, MatrixBlock rawblock, int[] sampleIndexes, boolean skipZeros) 
	{
		//note: general case only because anyway binary search for small samples
		
		//probe map for distinct items (for value or value groups)
		DoubleIntListHashMap distinctVals = new DoubleIntListHashMap();
		
		//scan rows and probe/build distinct items
		final int m = sampleIndexes.length;
		for( int i=0; i<m; i++ ) {
			int rowIndex = sampleIndexes[i]; 
			double val = CompressedMatrixBlock.TRANSPOSE_INPUT ? 
					rawblock.quickGetValue(colIndex, rowIndex) : 
					rawblock.quickGetValue(rowIndex, colIndex); 
			if( val!=0 || !skipZeros ) {					
				IntArrayList lstPtr = distinctVals.get(val);	
				if( lstPtr == null ) {
					lstPtr = new IntArrayList();
					distinctVals.appendValue(val, lstPtr);
				}
				lstPtr.appendValue(i);
			}
		}

		return new UncompressedBitmap(distinctVals);
	}
	
	/**
	 * 
	 * @param colIndices
	 * @param rawblock
	 * @param rowReader
	 * @return
	 */
	private static UncompressedBitmap extractBitmap(int[] colIndices,
			MatrixBlock rawblock, ReaderColumnSelection rowReader) 
	{
		//probe map for distinct items (for value or value groups)
		DblArrayIntListHashMap distinctVals = new DblArrayIntListHashMap();
		
		//scan rows and probe/build distinct items
		DblArray cellVals = null;
		while ((cellVals = rowReader.nextRow()) != null) {
			IntArrayList lstPtr = distinctVals.get(cellVals);
			if (lstPtr == null) {
				//create new objects only on demand
				lstPtr = new IntArrayList();
				distinctVals.appendValue(new DblArray(cellVals), lstPtr);
			}
			lstPtr.appendValue(rowReader.getCurrentRowIndex());
		}
		
		return new UncompressedBitmap(distinctVals, colIndices.length);
	}
}
