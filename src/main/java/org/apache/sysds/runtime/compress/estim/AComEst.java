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
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.SingleIndex;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.utils.stats.Timing;

/**
 * Main abstract class for estimating size of compressions on columns.
 */
public abstract class AComEst {
	protected static final Log LOG = LogFactory.getLog(AComEst.class.getName());

	/** The Matrix Block to extract the compression estimates from */
	final protected MatrixBlock _data;
	/** The compression settings to use, for estimating the size, and compress the ColGroups. */
	final protected CompressionSettings _cs;
	/** NNZ count in each column of the input */
	protected int[] nnzCols;

	/**
	 * Main Constructor for Compression Estimator.
	 * 
	 * Protected because the factory should be used to construct the AComEst
	 * 
	 * @param data The matrix block to extract information from
	 * @param cs   The Compression settings used.
	 */
	protected AComEst(MatrixBlock data, CompressionSettings cs) {
		_data = data;
		_cs = cs;
	}

	/**
	 * Get the number of rows in the overall compressing block.
	 * 
	 * @return The number of rows
	 */
	public int getNumRows() {
		return _cs.transposed ? _data.getNumColumns() : _data.getNumRows();
	}

	/**
	 * Get the number of columns in the overall compressing block.
	 * 
	 * @return The number of cols
	 */
	public int getNumColumns() {
		return _cs.transposed ? _data.getNumRows() : _data.getNumColumns();
	}

	/**
	 * Multi threaded version of extracting compression size info
	 * 
	 * @param k The concurrency degree.
	 * @return The Compression Size info of each Column compressed isolated.
	 */
	public final CompressedSizeInfo computeCompressedSizeInfos(int k) {
		final int _numCols = getNumColumns();
		if(LOG.isDebugEnabled()) {
			Timing time = new Timing(true);
			CompressedSizeInfo ret = new CompressedSizeInfo(CompressedSizeInfoColGroup(_numCols, k));
			LOG.debug("CompressedSizeInfo for each column [ms]:" + time.stop());
			return ret;
		}
		else
			return new CompressedSizeInfo(CompressedSizeInfoColGroup(_numCols, k));
	}

	/**
	 * Method for extracting Compressed Size Info of specified columns, together in a single ColGroup
	 * 
	 * @param colIndexes The columns to group together inside a ColGroup
	 * @return The CompressedSizeInformation associated with the selected ColGroups.
	 */
	public final CompressedSizeInfoColGroup getColGroupInfo(IColIndex colIndexes) {
		return getColGroupInfo(colIndexes, 8, worstCaseUpperBound(colIndexes));
	}

	/**
	 * A method to extract the Compressed Size Info for a given list of columns, This method further limits the estimated
	 * number of unique values, since in some cases the estimated number of uniques is estimated higher than the number
	 * estimated in sub groups of the given colIndexes.
	 * 
	 * @param colIndexes         The columns to extract compression information from
	 * @param estimate           An estimate of number of unique elements in these columns
	 * @param nrUniqueUpperBound The upper bound of unique elements allowed in the estimate, can be calculated from the
	 *                           number of unique elements estimated in sub columns multiplied together. This is flexible
	 *                           in the sense that if the sample is small then this unique can be manually edited like in
	 *                           CoCodeCostMatrixMult.
	 * 
	 * @return The CompressedSizeInfoColGroup for the given column indexes.
	 */
	public abstract CompressedSizeInfoColGroup getColGroupInfo(IColIndex colIndexes, int estimate,
		int nrUniqueUpperBound);

	/**
	 * Method for extracting info of specified columns as delta encodings (delta from previous rows values)
	 * 
	 * @param colIndexes The columns to group together inside a ColGroup
	 * @return The CompressedSizeInformation assuming delta encoding of the column.
	 */
	public final CompressedSizeInfoColGroup getDeltaColGroupInfo(IColIndex colIndexes) {
		return getDeltaColGroupInfo(colIndexes, 8, worstCaseUpperBound(colIndexes));
	}

	/**
	 * A method to extract the Compressed Size Info for a given list of columns, This method further limits the estimated
	 * number of unique values, since in some cases the estimated number of uniques is estimated higher than the number
	 * estimated in sub groups of the given colIndexes.
	 * 
	 * The Difference for this method is that it extract the values as delta values from the matrix block input.
	 * 
	 * @param colIndexes         The columns to extract compression information from
	 * @param estimate           An estimate of number of unique delta elements in these columns
	 * @param nrUniqueUpperBound The upper bound of unique elements allowed in the estimate, can be calculated from the
	 *                           number of unique elements estimated in sub columns multiplied together. This is flexible
	 *                           in the sense that if the sample is small then this unique can be manually edited like in
	 *                           CoCodeCostMatrixMult.
	 * 
	 * @return The CompressedSizeInfoColGroup for the given column indexes.
	 */
	public abstract CompressedSizeInfoColGroup getDeltaColGroupInfo(IColIndex colIndexes, int estimate,
		int nrUniqueUpperBound);

	/**
	 * combine two analyzed column groups together. without materializing the dictionaries of either side.
	 * 
	 * if the number of distinct elements in both sides multiplied is larger than Integer, return null.
	 * 
	 * If either side was constructed without analysis then fall back to default materialization of double arrays. O
	 * 
	 * @param g1 First group
	 * @param g2 Second group
	 * @return A combined compressed size estimation for the group.
	 */
	public final CompressedSizeInfoColGroup combine(CompressedSizeInfoColGroup g1, CompressedSizeInfoColGroup g2) {
		final IColIndex combinedColIndexes = g1.getColumns().combine(g2.getColumns());
		return combine(combinedColIndexes, g1, g2);
	}

	/**
	 * Combine two analyzed column groups together. without materializing the dictionaries of either side.
	 * 
	 * if the number of distinct elements in both sides multiplied is larger than Integer, return null.
	 * 
	 * If either side was constructed without analysis then fall back to default materialization of double arrays.
	 * 
	 * @param combinedColumns The combined column indexes.
	 * @param g1              First group
	 * @param g2              Second group
	 * @return A combined compressed size estimation for the columns specified using the combining algorithm
	 */
	public final CompressedSizeInfoColGroup combine(IColIndex combinedColumns, CompressedSizeInfoColGroup g1,
		CompressedSizeInfoColGroup g2) {
		final int nRows = g1.getNumRows();
		// num vals + 1 if the offsets does not contain all this indicate that the columns contains default tuples.
		final int g1V = g1.getNumVals() + (g1.getNumOffs() < nRows ? 1 : 0);
		final int g2V = g2.getNumVals() + (g2.getNumOffs() < nRows ? 1 : 0);
		// Get worst case upper bound on unique tuples
		// typically this is the number of rows in dense or the sum of nnz in the columns sparse
		final int worstCase = worstCaseUpperBound(combinedColumns);
		// Get max number of tuples based on the above.
		final long max = Math.min((long) g1V * g2V, worstCase);
		if(max > 1000000) // set the max combination to a million distinct
			return null; // This combination is clearly not a good idea return null to indicate that.
		else if(g1.getMap() == null || g2.getMap() == null)
			// the previous information did not contain maps, therefore fall back to extract from sample
			if(_cs.preferDeltaEncoding)
				return getDeltaColGroupInfo(combinedColumns, Math.max(g1V, g2V), (int) max);
			else
				return getColGroupInfo(combinedColumns, Math.max(g1V, g2V), (int) max);
		else // Default combine the previous subject to max value calculated.
			return combine(combinedColumns, g1, g2, (int) max);
	}

	/** Clear the pointer to the materialized list of nnz in columns */
	public void clearNNZ() {
		nnzCols = null;
	}

	/**
	 * Extract the worst case upper bound of unique tuples in specified columns.
	 * 
	 * Note we rely on this method being very cheep, therefore don't give perfect results, but cheep easy extracted
	 * approximations that should be guaranteed to be above or equal to the true value.
	 * 
	 * @param columns The columns to look at
	 * @return The worst case upper bound.
	 */
	protected abstract int worstCaseUpperBound(IColIndex columns);

	/**
	 * Combine two estimated column groups
	 * 
	 * @param combinedColumns The combined column indexes
	 * @param g1              The left side estimate
	 * @param g2              The right side estimate
	 * @param maxDistinct     The maximum distinct tuples possible to get with the two groups
	 * @return The combined column group estimate
	 */
	protected abstract CompressedSizeInfoColGroup combine(IColIndex combinedColumns, CompressedSizeInfoColGroup g1,
		CompressedSizeInfoColGroup g2, int maxDistinct);

	/**
	 * Collect the compressed size for all individual columns using the available k parallelism degree.
	 * 
	 * @param clen The number of total columns
	 * @param k    The parallelization degree
	 * @return A list of the individual columns compressibility.
	 */
	protected List<CompressedSizeInfoColGroup> CompressedSizeInfoColGroup(int clen, int k) {
		if(k <= 1)
			return CompressedSizeInfoColGroupSingleThread(clen);
		else
			return CompressedSizeInfoColGroupParallel(clen, k);
	}

	/**
	 * Compress the column groups using a single thread.
	 * 
	 * @param clen the number of total columns
	 * @return A list of the individual columns compressibility.
	 */
	private List<CompressedSizeInfoColGroup> CompressedSizeInfoColGroupSingleThread(int clen) {
		List<CompressedSizeInfoColGroup> ret = new ArrayList<>(clen);
		if(!_cs.transposed && !_data.isEmpty() && _data.isInSparseFormat())
			nnzCols = LibMatrixReorg.countNnzPerColumn(_data);
		for(int col = 0; col < clen; col++) {
			if(_cs.preferDeltaEncoding)
				ret.add(getDeltaColGroupInfo(new SingleIndex(col)));
			else
				ret.add(getColGroupInfo(new SingleIndex(col)));
		}
		return ret;
	}

	/**
	 * Collect the compressed size for all individual columns using the available k parallelism degree.
	 * 
	 * @param clen The number of total columns
	 * @param k    The parallelization degree
	 * @return A list of the individual columns compressibility.
	 */
	private List<CompressedSizeInfoColGroup> CompressedSizeInfoColGroupParallel(int clen, int k) {
		final ExecutorService pool = CommonThreadPool.get(k);
		try {
			if(!_cs.transposed && !_data.isEmpty() && _data.isInSparseFormat()) {
				LOG.debug("Extracting number of nonzeros in each column");
				List<Future<int[]>> nnzFutures = LibMatrixReorg.countNNZColumnsFuture(_data, k, pool);
				for(Future<int[]> t : nnzFutures)
					nnzCols = LibMatrixReorg.mergeNnzCounts(nnzCols, t.get());
			}

			CompressedSizeInfoColGroup[] res = new CompressedSizeInfoColGroup[clen];
			final int blkz = Math.max(1, clen / (k * 10));
			final ArrayList<Future<Object>> tasks = new ArrayList<>(clen / blkz + 1);

			if(blkz != 1)
				LOG.debug("Extracting column samples in blocks of " + blkz);

			for(int col = 0; col < clen; col += blkz) {
				final int start = col;
				final int end = Math.min(clen, col + blkz);
				final boolean useDelta = _cs.preferDeltaEncoding;
				tasks.add(pool.submit(() -> {
					for(int c = start; c < end; c++) {
						if(useDelta)
							res[c] = getDeltaColGroupInfo(new SingleIndex(c));
						else
							res[c] = getColGroupInfo(new SingleIndex(c));
					}
					return null;
				}));
			}

			for(Future<Object> f : tasks)
				f.get();

			return Arrays.asList(res);
		}
		catch(Exception e) {
			throw new DMLCompressionException("Multithreaded first extraction failed", e);
		}
		finally {
			pool.shutdown();
		}
	}

	@Override
	public String toString() {
		return this.getClass().getSimpleName();
	}
}
