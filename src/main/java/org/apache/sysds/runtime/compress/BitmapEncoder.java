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

package org.apache.sysds.runtime.compress;

import org.apache.sysds.runtime.compress.utils.AbstractBitmap;
import org.apache.sysds.runtime.compress.utils.Bitmap;
import org.apache.sysds.runtime.compress.utils.BitmapLossy;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.compress.utils.DblArrayIntListHashMap;
import org.apache.sysds.runtime.compress.utils.DoubleIntListHashMap;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Static functions for encoding bitmaps in various ways.
 */
public class BitmapEncoder {


	/**
	 * Generate uncompressed bitmaps for a set of columns in an uncompressed matrix block.
	 * 
	 * @param colIndices   Indexes (within the block) of the columns to extract
	 * @param rawBlock     An uncompressed matrix block; can be dense or sparse
	 * @param compSettings The compression settings used for the compression.
	 * @return uncompressed bitmap representation of the columns
	 */
	public static AbstractBitmap extractBitmap(int[] colIndices, MatrixBlock rawBlock,
		CompressionSettings compSettings) {
		// note: no sparse column selection reader because low potential
		// single column selection
		Bitmap res = null;
		if(colIndices.length == 1) {
			res = extractBitmap(colIndices[0], rawBlock, compSettings);
		}
		// multiple column selection (general case)
		else {
			ReaderColumnSelection reader = null;
			if(rawBlock.isInSparseFormat() && compSettings.transposeInput)
				reader = new ReaderColumnSelectionSparse(rawBlock, colIndices, compSettings);
			else
				reader = new ReaderColumnSelectionDense(rawBlock, colIndices, compSettings);

			res = extractBitmap(colIndices, rawBlock, reader);
		}
		if(compSettings.lossy) {
			return BitmapLossy.makeBitmapLossy(res);
		}
		else {
			return res;
		}
	}

	/**
	 * Extract Bitmap from a single column. It will always skip all zero values. It also counts the instances of zero.
	 * 
	 * @param colIndex     The index of the column
	 * @param rawBlock     The Raw matrix block (that can be transposed)
	 * @param compSettings The Compression settings used, in this instance to know if the raw block is transposed.
	 * @return Bitmap containing the Information of the column.
	 */
	private static Bitmap extractBitmap(int colIndex, MatrixBlock rawBlock, CompressionSettings compSettings) {
		// probe map for distinct items (for value or value groups)
		DoubleIntListHashMap distinctVals = new DoubleIntListHashMap();

		// scan rows and probe/build distinct items
		final int m = compSettings.transposeInput ? rawBlock.getNumColumns() : rawBlock.getNumRows();
		int numZeros = 0;

		if(rawBlock.isInSparseFormat() && compSettings.transposeInput) { // SPARSE and Transposed.
			SparseBlock a = rawBlock.getSparseBlock();
			if(a != null && !a.isEmpty(colIndex)) {
				int apos = a.pos(colIndex);
				int alen = a.size(colIndex);
				numZeros = m - alen;
				int[] aix = a.indexes(colIndex);
				double[] avals = a.values(colIndex);

				// IntArrayList lstPtr0 = new IntArrayList(); // for 0 values
				// int last = -1;
				// iterate over non-zero entries but fill in zeros
				for(int j = apos; j < apos + alen; j++) {
					// fill in zero values
					// if(!skipZeros)
					// for(int k = last + 1; k < aix[j]; k++)
					// lstPtr0.appendValue(k);
					// handle non-zero value
					IntArrayList lstPtr = distinctVals.get(avals[j]);
					if(lstPtr == null) {
						lstPtr = new IntArrayList();
						distinctVals.appendValue(avals[j], lstPtr);
					}
					lstPtr.appendValue(aix[j]);
					// last = aix[j];
				}
				// fill in remaining zero values
				// if(!skipZeros) {
				// for(int k = last + 1; k < m; k++)
				// lstPtr0.appendValue(k);
				// if(lstPtr0.size() > 0)
				// distinctVals.appendValue(0, lstPtr0);
				// }
			}
			// else if(!skipZeros) { // full 0 column
			// IntArrayList lstPtr = new IntArrayList();
			// for(int i = 0; i < m; i++)
			// lstPtr.appendValue(i);
			// distinctVals.appendValue(0, lstPtr);
			// }
		}
		else // GENERAL CASE
		{
			for(int i = 0; i < m; i++) {
				double val = compSettings.transposeInput ? rawBlock.quickGetValue(colIndex, i) : rawBlock
					.quickGetValue(i, colIndex);
				if(val != 0) {
					IntArrayList lstPtr = distinctVals.get(val);
					if(lstPtr == null) {
						lstPtr = new IntArrayList();
						distinctVals.appendValue(val, lstPtr);
					}
					lstPtr.appendValue(i);
				}
				else {
					numZeros++;
				}
			}
		}

		return Bitmap.makeBitmap(distinctVals, numZeros);
	}

	private static Bitmap extractBitmap(int[] colIndices, MatrixBlock rawBlock, ReaderColumnSelection rowReader) {
		// probe map for distinct items (for value or value groups)
		DblArrayIntListHashMap distinctVals = new DblArrayIntListHashMap();

		// scan rows and probe/build distinct items
		DblArray cellVals = null;

		int zero = 0;
		while((cellVals = rowReader.nextRow()) != null) {
			IntArrayList lstPtr = distinctVals.get(cellVals);
			if(lstPtr == null) {
				// create new objects only on demand
				lstPtr = new IntArrayList();
				distinctVals.appendValue(new DblArray(cellVals), lstPtr);
			}
			zero += DblArray.isZero(cellVals) ? 1 : 0;

			lstPtr.appendValue(rowReader.getCurrentRowIndex());
		}

		return Bitmap.makeBitmap(distinctVals, colIndices.length, zero);
	}

}
