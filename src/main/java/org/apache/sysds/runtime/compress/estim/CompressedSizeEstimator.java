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

package org.apache.sysds.runtime.compress.estim;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.UncompressedBitmap;
import org.apache.sysds.runtime.compress.colgroup.ColGroup.CompressionType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

/**
 * Main abstract class for estimating size of compressions on columns.
 */
public abstract class CompressedSizeEstimator {

	private static final boolean LOCAL_DEBUG = false;
	private static final Level LOCAL_DEBUG_LEVEL = Level.DEBUG;
	static {
		if(LOCAL_DEBUG) {
			Logger.getLogger("org.apache.sysds.runtime.compress.estim").setLevel(LOCAL_DEBUG_LEVEL);
		}
	}
	protected static final Log LOG = LogFactory.getLog(CompressedSizeEstimator.class.getName());

	/** The Matrix Block to extract the compression estimates from */
	protected MatrixBlock _data;
	/** The number of rows in the matrix block, extracted to a field because the matrix could be transposed */
	protected final int _numRows;
	/** The number of columns in the matrix block, extracted to a field because the matrix could be transposed */
	protected final int _numCols;
	/** The compression settings to use, for estimating the size, and compress the ColGroups. */
	protected final CompressionSettings _compSettings;

	/**
	 * Main Constructor for Compression Estimator.
	 * 
	 * protected because the factory should be used to construct the CompressedSizeEstimator
	 * 
	 * @param data         The matrix block to extract information from
	 * @param compSettings The Compression settings used.
	 */
	protected CompressedSizeEstimator(MatrixBlock data, CompressionSettings compSettings) {
		_data = data;
		_numRows = compSettings.transposeInput ? _data.getNumColumns() : _data.getNumRows();
		_numCols = compSettings.transposeInput ? _data.getNumRows() : _data.getNumColumns();
		_compSettings = compSettings;
	}

	/**
	 * Single threaded version of extracting Compression Size info
	 * 
	 * @return The Compression Size info of each Column compressed isolated.
	 */
	public CompressedSizeInfo computeCompressedSizeInfos() {
		return computeCompressedSizeInfos(1);
	}

	/**
	 * Multi threaded version of extracting Compression Size info
	 * 
	 * @param k The concurrency degree.
	 * @return The Compression Size info of each Column compressed isolated.
	 */
	public CompressedSizeInfo computeCompressedSizeInfos(int k) {
		CompressedSizeInfoColGroup[] sizeInfos = estimateIndividualColumnGroupSizes(k);
		return computeCompressedSizeInfos(sizeInfos);
	}

	private CompressedSizeInfo computeCompressedSizeInfos(CompressedSizeInfoColGroup[] sizeInfos) {
		List<Integer> colsC = new ArrayList<>();
		List<Integer> colsUC = new ArrayList<>();
		HashMap<Integer, Double> compRatios = new HashMap<>();
		int nnzUC = 0;

		for(int col = 0; col < _numCols; col++) {
			double uncompSize = sizeInfos[col].getCompressionSize(CompressionType.UNCOMPRESSED);
			double minCompressedSize = (double) sizeInfos[col].getMinSize();
			double compRatio = uncompSize / minCompressedSize;

			if(compRatio > 1000) {

				LOG.warn("\n\tVery good CompressionRatio: " + compRatio + "\n\tUncompressedSize: " + uncompSize
					+ "\tCompressedSize: " + minCompressedSize + "\tType: " + sizeInfos[col].getBestCompressionType());
			}

			if(compRatio > 1) {
				colsC.add(col);
				compRatios.put(col, compRatio);
			}
			else {
				colsUC.add(col);
				// TODO nnzUC not incrementing as intended outside this function.
				nnzUC += sizeInfos[col].getEstNnz();
			}
		}

		// correction of column classification (reevaluate dense estimates if necessary)
		if(!MatrixBlock.evalSparseFormatInMemory(_numRows, colsUC.size(), nnzUC) && !colsUC.isEmpty()) {
			for(int i = 0; i < colsUC.size(); i++) {
				int col = colsUC.get(i);
				double uncompSize = MatrixBlock.estimateSizeInMemory(_numRows, 1, 1.0);
				// CompressedMatrixBlock.getUncompressedSize(numRows, 1, 1.0);
				double compRatio = uncompSize / sizeInfos[col].getMinSize();
				if(compRatio > 1) {
					colsC.add(col);
					colsUC.remove(i);
					i--;
					compRatios.put(col, compRatio);
					nnzUC -= sizeInfos[col].getEstNnz();
				}
			}
		}

		if(LOG.isTraceEnabled()) {
			LOG.trace("C: " + Arrays.toString(colsC.toArray(new Integer[0])));
			LOG.trace(
				"-- compression ratios: " + Arrays.toString(colsC.stream().map(c -> compRatios.get(c)).toArray()));
			LOG.trace("UC: " + Arrays.toString(colsUC.toArray(new Integer[0])));
			LOG.trace(
				"-- compression ratios: " + Arrays.toString(colsUC.stream().map(c -> compRatios.get(c)).toArray()));
		}

		return new CompressedSizeInfo(sizeInfos, colsC, colsUC, compRatios, nnzUC);

	}

	private CompressedSizeInfoColGroup[] estimateIndividualColumnGroupSizes(int k) {
		return (k > 1) ? CompressedSizeInfoColGroup(_numCols, k) : CompressedSizeInfoColGroup(_numCols);
	}

	/**
	 * Method used for compressing into one type of colGroup
	 * 
	 * @return CompressedSizeInfo on a compressed colGroup compressing the entire matrix into a single colGroup type.
	 */
	public CompressedSizeInfoColGroup estimateCompressedColGroupSize() {
		int[] colIndexes = makeColIndexes();
		return estimateCompressedColGroupSize(colIndexes);
	}

	/**
	 * Abstract method for extracting Compressed Size Info of specified columns, together in a single ColGroup
	 * 
	 * @param colIndexes The Colums to group together inside a ColGroup
	 * @return The CompressedSizeInformation associated with the selected ColGroups.
	 */
	public abstract CompressedSizeInfoColGroup estimateCompressedColGroupSize(int[] colIndexes);

	/**
	 * Method used to extract the CompressedSizeEstimationFactors from an constructed UncompressedBitMap. Note this
	 * method works both for the sample based estimator and the exact estimator, since the bitmap, can be extracted from
	 * a sample or from the entire dataset.
	 * 
	 * @param ubm the UncompressedBitMap, either extracted from a sample or from the entier dataset
	 * @return The size factors estimated from the Bit Map.
	 */
	public CompressedSizeEstimationFactors estimateCompressedColGroupSize(UncompressedBitmap ubm) {
		return CompressedSizeEstimationFactors.computeSizeEstimationFactors(ubm,
			_compSettings.validCompressions.contains(CompressionType.RLE),
			_numRows,
			ubm.getNumColumns());
	}

	// ------------------------------------------------
	// PARALLEL CODE
	// ------------------------------------------------

	private CompressedSizeInfoColGroup[] CompressedSizeInfoColGroup(int clen) {
		CompressedSizeInfoColGroup[] ret = new CompressedSizeInfoColGroup[clen];
		for(int col = 0; col < clen; col++)
			ret[col] = estimateCompressedColGroupSize(new int[] {col});
		return ret;
	}

	private CompressedSizeInfoColGroup[] CompressedSizeInfoColGroup(int clen, int k) {
		try {
			ExecutorService pool = CommonThreadPool.get(k);
			ArrayList<SizeEstimationTask> tasks = new ArrayList<>();
			for(int col = 0; col < clen; col++)
				tasks.add(new SizeEstimationTask(this, col));
			List<Future<CompressedSizeInfoColGroup>> rtask = pool.invokeAll(tasks);
			ArrayList<CompressedSizeInfoColGroup> ret = new ArrayList<>();
			for(Future<CompressedSizeInfoColGroup> lrtask : rtask)
				ret.add(lrtask.get());
			pool.shutdown();
			return ret.toArray(new CompressedSizeInfoColGroup[0]);
		}
		catch(InterruptedException | ExecutionException e) {
			throw new DMLRuntimeException(e);
		}
	}

	private static class SizeEstimationTask implements Callable<CompressedSizeInfoColGroup> {
		private final CompressedSizeEstimator _estimator;
		private final int _col;

		protected SizeEstimationTask(CompressedSizeEstimator estimator, int col) {
			_estimator = estimator;
			_col = col;
		}

		@Override
		public CompressedSizeInfoColGroup call() {
			return _estimator.estimateCompressedColGroupSize(new int[] {_col});
		}
	}

	// ------------------------------------------------
	// PARALLEL CODE END
	// ------------------------------------------------

	// UTIL

	private int[] makeColIndexes() {
		int[] colIndexes = new int[_numCols];
		for(int i = 0; i < _numCols; i++) {
			colIndexes[i] = i;
		}
		return colIndexes;
	}
}
