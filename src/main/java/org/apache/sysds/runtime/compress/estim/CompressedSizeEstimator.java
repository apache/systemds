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
import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.utils.ABitmap;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

/**
 * Main abstract class for estimating size of compressions on columns.
 */
public abstract class CompressedSizeEstimator {
	protected static final Log LOG = LogFactory.getLog(CompressedSizeEstimator.class.getName());

	/** The Matrix Block to extract the compression estimates from */
	final protected MatrixBlock _data;
	/** The number of rows in the matrix block, extracted to a field because the matrix could be transposed */
	final protected int _numRows;
	/** The number of columns in the matrix block, extracted to a field because the matrix could be transposed */
	final protected int _numCols;
	/** The compression settings to use, for estimating the size, and compress the ColGroups. */
	final protected CompressionSettings _cs;

	/**
	 * Boolean specifying if the _data is in transposed format. This is used to select the correct readers for the
	 * extraction of bitmaps for the columns.
	 */
	protected boolean _transposed;

	/**
	 * Main Constructor for Compression Estimator.
	 * 
	 * protected because the factory should be used to construct the CompressedSizeEstimator
	 * 
	 * @param data The matrix block to extract information from
	 * @param cs   The Compression settings used.
	 */
	protected CompressedSizeEstimator(MatrixBlock data, CompressionSettings cs) {
		_data = data;
		_transposed = cs.transposed;
		_numRows = _transposed ? _data.getNumColumns() : _data.getNumRows();
		_numCols = _transposed ? _data.getNumRows() : _data.getNumColumns();
		_cs = cs;
	}

	public int getNumRows() {
		return _numRows;
	}

	public int getNumColumns() {
		return _numCols;
	}

	public MatrixBlock getData(){
		return _data;
	}

	/**
	 * Multi threaded version of extracting Compression Size info
	 * 
	 * @param k The concurrency degree.
	 * @return The Compression Size info of each Column compressed isolated.
	 */
	public CompressedSizeInfo computeCompressedSizeInfos(int k) {
		List<CompressedSizeInfoColGroup> sizeInfos = Arrays.asList(estimateIndividualColumnGroupSizes(k));
		return new CompressedSizeInfo(sizeInfos);
	}

	/**
	 * Multi threaded version of extracting Compression Size info from list of specified columns
	 * 
	 * @return
	 */

	/**
	 * Multi threaded version of extracting Compression Size info from list of specified columns
	 * 
	 * @param columnLists The specified columns to extract.
	 * @param k           The parallelization degree
	 * @return The Compression information from the specified column groups.
	 */
	public List<CompressedSizeInfoColGroup> computeCompressedSizeInfos(Collection<int[]> columnLists, int k) {
		if(k == 1)
			return computeCompressedSizeInfos(columnLists);
		try {
			ExecutorService pool = CommonThreadPool.get(k);
			ArrayList<SizeEstimationTask> tasks = new ArrayList<>();
			for(int[] g : columnLists)
				tasks.add(new SizeEstimationTask(this, g));
			List<Future<CompressedSizeInfoColGroup>> rtask = pool.invokeAll(tasks);
			ArrayList<CompressedSizeInfoColGroup> ret = new ArrayList<>();
			for(Future<CompressedSizeInfoColGroup> lrtask : rtask)
				ret.add(lrtask.get());
			pool.shutdown();
			return ret;
		}
		catch(InterruptedException | ExecutionException e) {
			return computeCompressedSizeInfos(columnLists);
		}
	}

	/**
	 * Compression Size info from list of specified columns
	 * 
	 * @param columnLists The specified columns to extract.
	 * @return The Compression information from the specified column groups.
	 */
	public List<CompressedSizeInfoColGroup> computeCompressedSizeInfos(Collection<int[]> columnLists) {
		ArrayList<CompressedSizeInfoColGroup> ret = new ArrayList<>();
		for(int[] g : columnLists)
			ret.add(estimateCompressedColGroupSize(g));
		return ret;
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
	 * Method for extracting Compressed Size Info of specified columns, together in a single ColGroup
	 * 
	 * @param colIndexes The columns to group together inside a ColGroup
	 * @return The CompressedSizeInformation associated with the selected ColGroups.
	 */
	public CompressedSizeInfoColGroup estimateCompressedColGroupSize(int[] colIndexes){
		return estimateCompressedColGroupSize(colIndexes, 8, getNumRows());
	}

	/**
	 * A method to extract the Compressed Size Info for a given list of columns, This method further limits the
	 * estimated number of unique values, since in some cases the estimated number of uniques is estimated higher than
	 * the number estimated in sub groups of the given colIndexes.
	 * 
	 * @param colIndexes         The columns to extract compression information from
	 * @param estimate 			 An estimate of number of unique elements in these columns
	 * @param nrUniqueUpperBound The upper bound of unique elements allowed in the estimate, can be calculated from the
	 *                           number of unique elements estimated in sub columns multiplied together. This is
	 *                           flexible in the sense that if the sample is small then this unique can be manually
	 *                           edited like in CoCodeCostMatrixMult.
	 * 
	 * @return The CompressedSizeInfoColGroup fro the given column indexes.
	 */
	public abstract CompressedSizeInfoColGroup estimateCompressedColGroupSize(int[] colIndexes, int estimate, int nrUniqueUpperBound);

	/**
	 * Join two analyzed column groups together. without materializing the dictionaries of either side.
	 * 
	 * If either side was constructed without analysis then fall back to default materialization of double arrays.
	 * 
	 * @param g1 First group
	 * @param g2 Second group
	 * @return A joined compressed size estimation for the group.
	 */
	public CompressedSizeInfoColGroup estimateJoinCompressedSize(CompressedSizeInfoColGroup g1,
		CompressedSizeInfoColGroup g2) {
		final int[] joined = Util.join(g1.getColumns(), g2.getColumns());
		final int g1V = g1.getNumVals();
		final int g2V = g2.getNumVals();
		if(g1V * g2V < 0 || g1V * g2V > getNumRows())
			return null;
		else if((g1.getMap() == null && g2V != 0) || (g2.getMap() == null && g2V != 0))
			return estimateCompressedColGroupSize(joined, Math.max(g1V + 1, g2V+ 1), Math.min((g1V + 1) * (g2V + 1), getNumRows()));
		else
			return estimateJoinCompressedSize(joined, g1, g2);

	}

	protected abstract CompressedSizeInfoColGroup estimateJoinCompressedSize(int[] joinedcols,
		CompressedSizeInfoColGroup g1, CompressedSizeInfoColGroup g2);

	/**
	 * Method used to extract the CompressedSizeEstimationFactors from an constructed UncompressedBitmap. Note this
	 * method works both for the sample based estimator and the exact estimator, since the bitmap, can be extracted from
	 * a sample or from the entire dataset.
	 * 
	 * @param ubm        The UncompressedBitmap, either extracted from a sample or from the entire dataset
	 * @param colIndexes The columns that is compressed together.
	 * @return The size factors estimated from the Bit Map.
	 */
	public EstimationFactors estimateCompressedColGroupSize(ABitmap ubm, int[] colIndexes) {
		return estimateCompressedColGroupSize(ubm, colIndexes, _numRows, _cs);
	}

	public static EstimationFactors estimateCompressedColGroupSize(ABitmap ubm, int[] colIndexes, int nrRows,
		CompressionSettings cs) {
		return EstimationFactors.computeSizeEstimationFactors(ubm, cs.validCompressions.contains(CompressionType.RLE),
			colIndexes);
	}

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
			return CompressedSizeInfoColGroup(clen);
		}
	}

	private static class SizeEstimationTask implements Callable<CompressedSizeInfoColGroup> {
		private final CompressedSizeEstimator _estimator;
		private final int[] _cols;

		protected SizeEstimationTask(CompressedSizeEstimator estimator, int col) {
			_estimator = estimator;
			_cols = new int[] {col};
		}

		protected SizeEstimationTask(CompressedSizeEstimator estimator, int[] cols) {
			_estimator = estimator;
			_cols = cols;
		}

		@Override
		public CompressedSizeInfoColGroup call() {
			return _estimator.estimateCompressedColGroupSize(_cols);
		}
	}

	private int[] makeColIndexes() {
		int[] colIndexes = new int[_numCols];
		for(int i = 0; i < _numCols; i++) {
			colIndexes[i] = i;
		}
		return colIndexes;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append(" transposed: ");
		sb.append(_transposed);
		sb.append(" cols: ");
		sb.append(_numCols);
		sb.append(" rows: ");
		sb.append(_numRows);
		return sb.toString();
	}

}
