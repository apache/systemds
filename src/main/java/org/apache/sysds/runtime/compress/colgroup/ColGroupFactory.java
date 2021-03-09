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

package org.apache.sysds.runtime.compress.colgroup;

import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLCompressionException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.mapping.IMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.tree.AInsertionSorter;
import org.apache.sysds.runtime.compress.colgroup.tree.InsertionSorterFactory;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimatorExact;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.lib.BitmapEncoder;
import org.apache.sysds.runtime.compress.utils.ABitmap;
import org.apache.sysds.runtime.compress.utils.Bitmap;
import org.apache.sysds.runtime.compress.utils.BitmapLossy;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

/**
 * Factory pattern for constructing ColGroups.
 */
public class ColGroupFactory {
	private static final Log LOG = LogFactory.getLog(ColGroupFactory.class.getName());

	/**
	 * The actual compression method, that handles the logic of compressing multiple columns together. This method also
	 * have the responsibility of correcting any estimation errors previously made.
	 * 
	 * @param in           The input matrix, that could have been transposed if CompSettings was set to do that
	 * @param csi          The compression Information extracted from the estimation, this contains which groups of
	 *                     columns to compress together
	 * @param compSettings The compression settings to construct the compression based on.
	 * @param k            The degree of parallelism used.
	 * @return A Resulting array of ColGroups, containing the compressed information from the input matrix block.
	 */
	public static AColGroup[] compressColGroups(MatrixBlock in, CompressedSizeInfo csi,
		CompressionSettings compSettings, int k) {
		int[][] groups = csi.getGroups();
		if(k <= 1)
			return compressColGroups(in, groups, compSettings);
		else
			try {
				ExecutorService pool = CommonThreadPool.get(k);
				ArrayList<CompressTask> tasks = new ArrayList<>();
				for(int[] colIndexes : groups)
					tasks.add(new CompressTask(in, colIndexes, compSettings));

				ArrayList<AColGroup> ret = new ArrayList<>();
				for(Future<AColGroup> lrtask : pool.invokeAll(tasks))
					ret.add(lrtask.get());
				pool.shutdown();
				return ret.toArray(new AColGroup[groups.length]);
			}
			catch(InterruptedException | ExecutionException e) {
				// return compressColGroups(in, groups, compSettings);
				throw new DMLRuntimeException("Failed compression ", e);
			}

	}

	private static AColGroup[] compressColGroups(MatrixBlock in, int[][] groups, CompressionSettings compSettings) {
		AColGroup[] ret = new AColGroup[groups.length];
		for(int i = 0; i < groups.length; i++)
			ret[i] = compressColGroup(in, groups[i], compSettings);
		return ret;
	}

	// private static class CompressedColumn implements Comparable<CompressedColumn>
	// {
	// final int colIx;
	// final double compRatio;

	// public CompressedColumn(int colIx, double compRatio) {
	// this.colIx = colIx;
	// this.compRatio = compRatio;
	// }

	// public static PriorityQueue<CompressedColumn>
	// makePriorityQue(HashMap<Integer, Double> compRatios,
	// int[] colIndexes) {
	// PriorityQueue<CompressedColumn> compRatioPQ;

	// // first modification
	// compRatioPQ = new PriorityQueue<>();
	// for(int i = 0; i < colIndexes.length; i++)
	// compRatioPQ.add(new CompressedColumn(i, compRatios.get(colIndexes[i])));

	// return compRatioPQ;
	// }

	// @Override
	// public int compareTo(CompressedColumn o) {
	// return (int) Math.signum(compRatio - o.compRatio);
	// }
	// }

	private static class CompressTask implements Callable<AColGroup> {
		private final MatrixBlock _in;
		private final int[] _colIndexes;
		private final CompressionSettings _compSettings;

		protected CompressTask(MatrixBlock in, int[] colIndexes, CompressionSettings compSettings) {
			_in = in;
			_colIndexes = colIndexes;
			_compSettings = compSettings;
		}

		@Override
		public AColGroup call() {
			return compressColGroup(_in, _colIndexes, _compSettings);
		}
	}

	private static AColGroup compressColGroup(MatrixBlock in, int[] colIndexes, CompressionSettings compSettings) {
		return compressColGroupForced(in, colIndexes, compSettings);
		// return (compRatios == null) ? compressColGroupForced(in,
		// colIndexes,
		// compSettings) : compressColGroupCorrecting(in, compRatios, colIndexes,
		// compSettings);

	}

	private static AColGroup compressColGroupForced(MatrixBlock in, int[] colIndexes,
		CompressionSettings compSettings) {

		CompressedSizeEstimator estimator = new CompressedSizeEstimatorExact(in, compSettings, compSettings.transposed);

		ABitmap ubm = BitmapEncoder.extractBitmap(colIndexes, in, compSettings.transposed);
		CompressedSizeInfoColGroup sizeInfo = new CompressedSizeInfoColGroup(
			estimator.estimateCompressedColGroupSize(ubm, colIndexes), compSettings.validCompressions);

		int numRows = compSettings.transposed ? in.getNumColumns() : in.getNumRows();
		return compress(colIndexes, numRows, ubm, sizeInfo.getBestCompressionType(), compSettings, in);
	}

	// private static AColGroup compressColGroupCorrecting(MatrixBlock in,
	// HashMap<Integer, Double> compRatios,
	// int[] colIndexes, CompressionSettings compSettings) {

	// int[] allGroupIndices = colIndexes.clone();
	// CompressedSizeInfoColGroup sizeInfo;
	// ABitmap ubm = null;
	// PriorityQueue<CompressedColumn> compRatioPQ =
	// CompressedColumn.makePriorityQue(compRatios, colIndexes);
	// CompressedSizeEstimator estimator = new CompressedSizeEstimatorExact(in,
	// compSettings, compSettings.transposed);

	// while(true) {

	// // STEP 1.
	// // Extract the entire input column list and observe compression ratio
	// ubm = BitmapEncoder.extractBitmap(colIndexes, in, compSettings.transposed);

	// sizeInfo = new
	// CompressedSizeInfoColGroup(estimator.estimateCompressedColGroupSize(ubm,
	// colIndexes),
	// compSettings.validCompressions);

	// // Throw error if for some reason the compression observed is 0.
	// if(sizeInfo.getMinSize() == 0) {
	// throw new DMLRuntimeException("Size info of compressed Col Group is 0");
	// }

	// // STEP 2.
	// // Calculate the compression ratio compared to an uncompressed ColGroup type.
	// double compRatio = sizeInfo.getCompressionSize(CompressionType.UNCOMPRESSED)
	// / sizeInfo.getMinSize();

	// // STEP 3.
	// // Finish the search and close this compression if the group show good
	// compression.
	// if(compRatio > 1.0 || PartitionerType.isCost(compSettings.columnPartitioner))
	// {
	// int rlen = compSettings.transposed ? in.getNumColumns() : in.getNumRows();
	// return compress(colIndexes, rlen, ubm, sizeInfo.getBestCompressionType(),
	// compSettings, in);
	// }
	// else {
	// // STEP 4.
	// // Try to remove the least compressible column from the columns to compress.
	// // Then repeat from Step 1.
	// allGroupIndices[compRatioPQ.poll().colIx] = -1;

	// if(colIndexes.length - 1 == 0) {
	// return null;
	// }

	// colIndexes = new int[colIndexes.length - 1];
	// // copying the values that do not equal -1
	// int ix = 0;
	// for(int col : allGroupIndices)
	// if(col != -1)
	// colIndexes[ix++] = col;
	// }
	// }
	// }

	/**
	 * Method for compressing an ColGroup.
	 * 
	 * @param colIndexes     The Column indexes to compress
	 * @param rlen           The number of rows in the columns
	 * @param ubm            The Bitmap containing all the data needed for the compression (unless Uncompressed
	 *                       ColGroup)
	 * @param compType       The CompressionType selected
	 * @param cs             The compression Settings used for the given compression
	 * @param rawMatrixBlock The copy of the original input (maybe transposed) MatrixBlock
	 * @return A Compressed ColGroup
	 */
	public static AColGroup compress(int[] colIndexes, int rlen, ABitmap ubm, CompressionType compType,
		CompressionSettings cs, MatrixBlock rawMatrixBlock) {

		if(LOG.isTraceEnabled())
			LOG.trace("compressing to: " + compType);
		if(ubm.getOffsetList().length == 0)
			return new ColGroupEmpty(colIndexes, rlen);
		switch(compType) {
			case DDC:
				return compressDDC(colIndexes, rlen, ubm, cs);
			case RLE:
				return new ColGroupRLE(colIndexes, rlen, ubm, cs);
			case OLE:
				return new ColGroupOLE(colIndexes, rlen, ubm, cs);
			case SDC:
				return compressSDC(colIndexes, rlen, ubm, cs);
			case UNCOMPRESSED:
				return new ColGroupUncompressed(colIndexes, rawMatrixBlock, cs.transposed);
			default:
				throw new DMLCompressionException("Not implemented ColGroup Type compressed in factory.");
		}
	}

	private static AColGroup compressSDC(int[] colIndexes, int rlen, ABitmap ubm, CompressionSettings cs) {

		int numZeros = (int) ((long) rlen - (int) ubm.getNumOffsets());
		int largestOffset = 0;
		int largestIndex = 0;
		int index = 0;
		for(IntArrayList a : ubm.getOffsetList()) {
			if(a.size() > largestOffset) {
				largestOffset = a.size();
				largestIndex = index;
			}
			index++;
		}
		AColGroup cg;
		ADictionary dict = new Dictionary(((Bitmap) ubm).getValues());
		if(numZeros >= largestOffset && ubm.getOffsetList().length == 1)
			cg =  new ColGroupSDCSingleZeros(colIndexes, rlen, dict, ubm.getOffsetList()[0].extractValues(true), null);
		else if(numZeros >= largestOffset)
			cg =  setupMultiValueZeroColGroup(colIndexes, ubm, rlen, dict);
		else if(ubm.getOffsetList().length == 1 && ubm.getOffsetsList(0).size() == rlen)
			cg =  new ColGroupConst(colIndexes, rlen, dict);
		else {
			dict = moveFrequentToLastDictionaryEntry(dict, ubm, rlen, largestIndex);
			cg =  setupMultiValueColGroup(colIndexes, numZeros, largestOffset, ubm, rlen, largestIndex, dict);
			// return new ColGroupEmpty(colIndexes, rlen);
		}
		return cg;
	}

	private static AColGroup setupMultiValueZeroColGroup(int[] colIndexes, ABitmap ubm, int numRows, ADictionary dict) {
		IntArrayList[] offsets = ubm.getOffsetList();

		final int numOffsets = (int) ubm.getNumOffsets();
		AInsertionSorter s = InsertionSorterFactory.create(numOffsets, offsets.length, numRows);
		s.insert(offsets);
		int[] _indexes = s.getIndexes();
		IMapToData _data = s.getData();

		return new ColGroupSDCZeros(colIndexes, numRows, dict, _indexes, _data, null);
	}

	private static AColGroup setupMultiValueColGroup(int[] colIndexes, int numZeros, int largestOffset, ABitmap ubm,
		int numRows, int largestIndex, ADictionary dict) {
		IntArrayList[] offsets = ubm.getOffsetList();

		AInsertionSorter s = InsertionSorterFactory.create(numRows - largestOffset, offsets.length, numRows);
		
		s.insert(offsets, largestIndex);
		int[] _indexes = s.getIndexes();
		IMapToData _data = s.getData();
		AColGroup ret = new ColGroupSDC(colIndexes, numRows, dict, _indexes, _data, null);

		return ret;
	}

	private static ADictionary moveFrequentToLastDictionaryEntry(ADictionary dict, ABitmap ubm, int numRows,
		int largestIndex) {
		final double[] dictValues = dict.getValues();
		final int zeros = numRows - (int) ubm.getNumOffsets();
		final int nCol = ubm.getNumColumns();
		final int offsetToLargest = largestIndex * nCol;

		if(zeros == 0) {
			final double[] swap = new double[nCol];
			System.arraycopy(dictValues, offsetToLargest, swap, 0, nCol);
			for(int i = offsetToLargest; i < dictValues.length - nCol; i++) {
				dictValues[i] = dictValues[i + nCol];
			}
			System.arraycopy(swap, 0, dictValues, dictValues.length - nCol, nCol);
			return dict;
		}

		final int largestIndexSize = ubm.getOffsetsList(largestIndex).size();
		final double[] newDict = new double[dictValues.length + nCol];

		if(zeros > largestIndexSize)
			System.arraycopy(dictValues, 0, newDict, 0, dictValues.length);
		else {
			System.arraycopy(dictValues, 0, newDict, 0, offsetToLargest);
			System.arraycopy(dictValues,
				offsetToLargest + nCol,
				newDict,
				offsetToLargest,
				dictValues.length - offsetToLargest - nCol);
			System.arraycopy(dictValues, offsetToLargest, newDict, newDict.length - nCol, nCol);
		}
		return new Dictionary(newDict);
	}

	private static AColGroup compressDDC(int[] colIndexes, int rlen, ABitmap ubm, CompressionSettings cs) {

		boolean _zeros = ubm.getNumOffsets() < (long) rlen;
		ADictionary dict;
		if(ubm instanceof BitmapLossy)
			dict = new QDictionary((BitmapLossy) ubm).makeDoubleDictionary();
		else
			dict = new Dictionary( ((Bitmap) ubm).getValues());
		double[] values = dict.getValues();
		if(_zeros) {
			double[] appendedZero = new double[values.length + colIndexes.length];
			System.arraycopy(values, 0, appendedZero, 0, values.length);
			dict = new Dictionary(appendedZero);
		}
		else
			dict = new Dictionary(values);

		int numVals = ubm.getNumValues();
		IMapToData _data = MapToFactory.create(rlen, numVals + (_zeros ? 1 : 0));
		if(_zeros)
			_data.fill(numVals);

		// for(int i = 0; i < numVals; i++) {
		// int[] tmpList = ubm.getOffsetsList(i).extractValues();
		// int tmpListSize = ubm.getNumOffsets(i);
		// for(int k = 0; k < tmpListSize; k++)
		// _data[tmpList[k]] = (char) i;
		// }

		for(int i = 0; i < numVals; i++) {
			IntArrayList tmpList = ubm.getOffsetsList(i);
			final int sz = tmpList.size();
			for(int k = 0; k < sz; k++)
				_data.set(tmpList.get(k), i);
		}

		return new ColGroupDDC(colIndexes, rlen, dict, _data, null);

	}
}
