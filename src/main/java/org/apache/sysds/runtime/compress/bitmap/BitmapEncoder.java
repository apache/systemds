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

package org.apache.sysds.runtime.compress.bitmap;

import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.readers.ReaderColumnSelection;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.compress.utils.DblArrayIntListHashMap;
import org.apache.sysds.runtime.compress.utils.DblArrayIntListHashMap.DArrayIListEntry;
import org.apache.sysds.runtime.compress.utils.DoubleIntListHashMap;
import org.apache.sysds.runtime.compress.utils.DoubleIntListHashMap.DIListEntry;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import scala.NotImplementedError;

/**
 * Static functions for extracting bitmaps from a MatrixBlock.
 */
public class BitmapEncoder {

	static Log LOG = LogFactory.getLog(BitmapEncoder.class.getName());

	public static ABitmap extractBitmap(IColIndex colIndices, MatrixBlock rawBlock, int estimatedNumberOfUniques,
		CompressionSettings cs) {
		return extractBitmap(colIndices, rawBlock, cs.transposed, estimatedNumberOfUniques, cs.sortTuplesByFrequency,
			cs.scaleFactors);
	}

	/**
	 * Generate uncompressed bitmaps for a set of columns in an uncompressed matrix block.
	 * 
	 * if the rawBlock is transposed and sparse it should be guaranteed that the rows specified are not empty, aka all
	 * zero.
	 * 
	 * @param colIndices               Indexes (within the block) of the columns to extract
	 * @param rawBlock                 An uncompressed matrix block; can be dense, sparse, empty, or null (not
	 *                                 Compressed!)
	 * @param transposed               Boolean specifying if the rawBlock was transposed.
	 * @param estimatedNumberOfUniques The number of estimated uniques inside this group. Used to allocated the
	 *                                 HashMaps.
	 * @param sortedEntries            Boolean specifying if the entries should be sorted based on frequency of tuples
	 * @param scaleFactors             For quantization-fused compression, scale factors per row, or a single value for entire matrix
	 * @return Uncompressed bitmap representation of the columns specified
	 */

	public static ABitmap extractBitmap(IColIndex colIndices, MatrixBlock rawBlock, boolean transposed,
		int estimatedNumberOfUniques, boolean sortedEntries) {
		// Overloaded method with scaleFactors defaulted to null
		return extractBitmap(colIndices, rawBlock, transposed, estimatedNumberOfUniques, sortedEntries, null);
	}

	public static ABitmap extractBitmap(IColIndex colIndices, MatrixBlock rawBlock, boolean transposed,
		int estimatedNumberOfUniques, boolean sortedEntries, double[] scaleFactors) {
		if(rawBlock == null || rawBlock.isEmpty())
			return null;

		final int numRows = transposed ? rawBlock.getNumColumns() : rawBlock.getNumRows();
		final int estimatedNumber = Math.max(estimatedNumberOfUniques, 8);
		if(colIndices.size() == 1) {
			if (scaleFactors != null) {
				throw new NotImplementedError(); // TODO: handle quantization-fused compression
			}
			return extractBitmapSingleColumn(colIndices.get(0), rawBlock, numRows, transposed, estimatedNumber,
				sortedEntries);
		} else {
			return extractBitmapMultiColumns(colIndices, rawBlock, numRows, transposed, estimatedNumber, sortedEntries,
				scaleFactors);
		}
	}

	private static ABitmap extractBitmapSingleColumn(int colIndex, MatrixBlock rawBlock, int numRows,
		boolean transposed, int est, boolean sort) {
		if(transposed) {
			if(rawBlock.isInSparseFormat() && rawBlock.getSparseBlock().isEmpty(colIndex))
				return null;
			return makeSingleColBitmap(extractSingleColT(colIndex, rawBlock, est), rawBlock.getNumColumns(), sort);
		}
		else
			return makeSingleColBitmap(extractSingleCol(colIndex, rawBlock, est), rawBlock.getNumRows(), sort);
	}

	private static DoubleIntListHashMap extractSingleCol(int colIndex, MatrixBlock rawBlock, int estimatedUnique) {
		final DoubleIntListHashMap distinctVals = new DoubleIntListHashMap(estimatedUnique);
		final int nRows = rawBlock.getNumRows();
		final int nCols = rawBlock.getNumColumns();
		final boolean sparse = rawBlock.isInSparseFormat();

		if(sparse) {
			final SparseBlock sb = rawBlock.getSparseBlock();
			for(int r = 0; r < nRows; r++) {
				if(sb.isEmpty(r))
					continue;
				final int apos = sb.pos(r);
				final int alen = sb.size(r) + apos;
				final int[] aix = sb.indexes(r);
				final int idx = Arrays.binarySearch(aix, apos, alen, colIndex);
				if(idx >= 0)
					distinctVals.appendValue(sb.values(r)[idx], r);
			}
		}
		else if(rawBlock.getDenseBlock().isContiguous()) {
			final double[] values = rawBlock.getDenseBlockValues();
			if(nCols == 1)
				// Since the only values contained is in this column index. simply extract it continuously.
				for(int i = 0; i < values.length; i++)
					distinctVals.appendValue(values[i], i);
			else
				// For loop down through the rows skipping all other values than the ones in the specified column index.
				for(int i = 0, off = colIndex; off < nRows * nCols; i++, off += nCols)
					distinctVals.appendValue(values[off], i);
		}
		else { // GENERAL CASE
				// This case is slow, because it does a binary search in each row of the sparse input. (if sparse)
				// and does get value in dense cases with multi blocks.
			for(int i = 0; i < nRows; i++)
				distinctVals.appendValue(rawBlock.get(i, colIndex), i);
		}
		return distinctVals;
	}

	private static DoubleIntListHashMap extractSingleColT(int colIndex, MatrixBlock rawBlock, int estimatedUnique) {
		// probe map for distinct items (for value or value groups)
		final DoubleIntListHashMap distinctVals = new DoubleIntListHashMap(estimatedUnique);

		if(rawBlock.isInSparseFormat()) { // SPARSE and Transposed.
			final SparseBlock a = rawBlock.getSparseBlock();
			// OBS, the main entry guarantees that the sparse column is not empty, if we get null pointers,
			// then the caller is doing it wrong.
			final int apos = a.pos(colIndex);
			final int alen = a.size(colIndex) + apos;
			final int[] aix = a.indexes(colIndex);
			final double[] avals = a.values(colIndex);
			for(int j = apos; j < alen; j++)
				distinctVals.appendValue(avals[j], aix[j]);
		}
		else if(rawBlock.getNumRows() == 1) {
			// there is no multi block if there is only one row
			// linear scan through the values in the input.
			final double[] values = rawBlock.getDenseBlockValues();
			for(int i = 0; i < values.length; i++)
				distinctVals.appendValue(values[i], i);
		}
		else { // GENERAL CASE / dense block + multi dense block case
			final DenseBlock db = rawBlock.getDenseBlock();
			final double[] values = db.values(colIndex);
			final int nCol = rawBlock.getNumColumns();
			// linear scan with offset.
			for(int i = 0, off = db.pos(colIndex); i < nCol; i++, off++)
				distinctVals.appendValue(values[off], i);
		}
		return distinctVals;
	}

	private static ABitmap extractBitmapMultiColumns(IColIndex colIndices, MatrixBlock rawBlock, int numRows,
		boolean transposed, int estimatedUnique, boolean sort, double[] scaleFactors) {
		final DblArrayIntListHashMap map = new DblArrayIntListHashMap(estimatedUnique);

		final ReaderColumnSelection reader = (scaleFactors == null) ? 
		ReaderColumnSelection.createReader(rawBlock, colIndices, transposed) : 
		ReaderColumnSelection.createQuantizedReader(rawBlock, colIndices, transposed, scaleFactors);

		DblArray cellVals = null;
		try {
			DblArray empty = new DblArray(new double[colIndices.size()]);
			while((cellVals = reader.nextRow()) != null) {
				if(!cellVals.equals(empty)) {
					map.appendValue(cellVals, reader.getCurrentRowIndex());
				}
			}

		}
		catch(Exception e) {
			throw new RuntimeException("failed extracting bitmap and adding. " + map + "  \n " + cellVals, e);
		}

		return makeMultiColBitmap(map, numRows, colIndices.size(), sort);
	}

	private static ABitmap makeMultiColBitmap(DblArrayIntListHashMap map, int numRows, int numCols, boolean sort) {
		final int numVals = map.size();
		if(numVals > 0) {
			List<DArrayIListEntry> mVals = map.extractValues();
			if(sort)
				Collections.sort(mVals, new CompSizeDArrayIListEntry());
			double[][] values = new double[numVals][];
			IntArrayList[] offsetsLists = new IntArrayList[numVals];
			int bitmapIx = 0;
			for(DArrayIListEntry val : mVals) {
				values[bitmapIx] = val.key.getData();
				offsetsLists[bitmapIx++] = val.value;
			}

			return new MultiColBitmap(offsetsLists, values, numRows);
		}
		else
			return null;
	}

	private static Bitmap makeSingleColBitmap(DoubleIntListHashMap distinctVals, int numRows, boolean sort) {
		final int numVals = distinctVals.size();
		if(numVals > 0) {
			List<DIListEntry> mVals = distinctVals.extractValues();
			if(sort)
				Collections.sort(mVals, new CompSizeDIListEntry());
			double[] values = new double[numVals];
			IntArrayList[] offsetsLists = new IntArrayList[numVals];
			int bitmapIx = 0;
			for(DIListEntry val : mVals) {
				values[bitmapIx] = val.key;
				offsetsLists[bitmapIx++] = val.value;
			}
			return new Bitmap(offsetsLists, values, numRows);
		}
		return null;
	}

	static class CompSizeDArrayIListEntry implements Comparator<DArrayIListEntry> {
		@Override
		public int compare(DArrayIListEntry o1, DArrayIListEntry o2) {
			final int v1 = o1.value.size();
			final int v2 = o2.value.size();
			return -Integer.compare(v1, v2);
		}
	}

	static class CompSizeDIListEntry implements Comparator<DIListEntry> {
		@Override
		public int compare(DIListEntry o1, DIListEntry o2) {
			final int v1 = o1.value.size();
			final int v2 = o2.value.size();
			return -Integer.compare(v1, v2);
		}
	}
}
