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

package org.apache.sysds.runtime.matrix.data;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.UtilFunctions;

public class LibMatrixSketch {
	private static final long PAR_UNIQUE_NUMCELL_THRESHOLD = 1024 * 16;
	private static final long PAR_UNIQUE_MAX_LOCAL_BYTES_FRACTION = 4;
	private static final long PAR_UNIQUE_LOCAL_BYTES_OVERHEAD = 8;

	/**
	 * Computes unique values with the original single-threaded behavior.
	 * The overload with a parallelism argument keeps this path as the k=1 baseline.
	 *
	 * @param blkIn input matrix block
	 * @param dir unique direction
	 * @return matrix block containing unique values
	 */
	public static MatrixBlock getUniqueValues(MatrixBlock blkIn, Types.Direction dir) {
		return getUniqueValues(blkIn, dir, 1);
	}

	/**
	 * Computes unique values. For sufficiently large inputs and k > 1, this uses
	 * parallel local deduplication or its batched variant.
	 *
	 * @param blkIn input matrix block
	 * @param dir unique direction
	 * @param k requested degree of parallelism
	 * @return matrix block containing unique values
	 */
	public static MatrixBlock getUniqueValues(MatrixBlock blkIn, Types.Direction dir, int k) {
		// Similar to R's unique, this operation computes unique values according
		// to the requested direction.
		if( !satisfiesMultiThreadingConstraints(blkIn, dir, k) )
			return getUniqueValuesSequential(blkIn, dir);

		boolean localDedupMemorySafe = isLocalDedupMemoryBudgetSafe(blkIn, dir);
		switch(dir) {
			case RowCol:
				return localDedupMemorySafe ?
					getUniqueValuesRowColParallel(blkIn, k) :
					getUniqueValuesRowColBatchedParallel(blkIn, dir, k);
			case Row:
				return localDedupMemorySafe ?
					getUniqueRowValuesParallel(blkIn, k) :
					getUniqueRowValuesBatchedParallel(blkIn, dir, k);
			case Col:
				return localDedupMemorySafe ?
					getUniqueColumnValuesParallel(blkIn, k) :
					getUniqueColumnValuesBatchedParallel(blkIn, dir, k);
			default:
				throw new IllegalArgumentException("Unrecognized direction: " + dir);
		}
	}

	/**
	 * Single-threaded baseline implementation for all unique directions.
	 * This preserves the original row-wise and column-wise unique behavior.
	 *
	 * @param blkIn input matrix block
	 * @param dir unique direction
	 * @return matrix block containing unique values
	 */
	private static MatrixBlock getUniqueValuesSequential(MatrixBlock blkIn, Types.Direction dir) {
		int rlen = blkIn.getNumRows();
		int clen = blkIn.getNumColumns();

		MatrixBlock blkOut = null;
		switch (dir) {
			case RowCol:
				// TODO optimize for dense/sparse/compressed (once multi-column support added)

				// obtain set of unique items
				HashSet<Double> hashSet = new HashSet<>();
				for( int i=0; i<rlen; i++ ) {
					for( int j=0; j<clen; j++ )
						hashSet.add(blkIn.get(i, j));
				}

				// allocate output block and place values
				blkOut = createRowColOutput(hashSet);
				break;

			case Row:
				// 2-pass algorithm to avoid unnecessarily large mem requirements
				HashSet<Double> rowSet = new HashSet<>();
				int clen2 = 0;
				for( int i=0; i<rlen; i++ ) {
					rowSet.clear();
					for( int j=0; j<clen; j++ )
						rowSet.add(blkIn.get(i, j));
					clen2 = Math.max(clen2, rowSet.size());
				}

				// actual
				blkOut = allocateOutputBlock(rlen, clen2);
				for( int i=0; i<rlen; i++ ) {
					rowSet.clear();
					for( int j=0; j<clen; j++ )
						rowSet.add(blkIn.get(i, j));
					int pos = 0;
					for( Double val : rowSet )
						blkOut.set(i, pos++, val);
				}
				break;

			case Col:
				// 2-pass algorithm to avoid unnecessarily large mem requirements
				HashSet<Double> colSet = new HashSet<>();
				int rlen2 = 0;
				for( int j=0; j<clen; j++ ) {
					colSet.clear();
					for( int i=0; i<rlen; i++ )
						colSet.add(blkIn.get(i, j));
					rlen2 = Math.max(rlen2, colSet.size());
				}

				// actual
				blkOut = allocateOutputBlock(rlen2, clen);
				for( int j=0; j<clen; j++ ) {
					colSet.clear();
					for( int i=0; i<rlen; i++ )
						colSet.add(blkIn.get(i, j));
					int pos = 0;
					for( Double val : colSet )
						blkOut.set(pos++, j, val);
				}
				break;

			default:
				throw new IllegalArgumentException("Unrecognized direction: " + dir);
		}

		blkOut.recomputeNonZeros();
		blkOut.examSparsity();
		return blkOut;
	}

	/**
	 * Parallel unique for all matrix values. Rows are split into balanced partitions,
	 * each task builds a local set, and the caller merges all local sets afterwards.
	 *
	 * @param blkIn input matrix block
	 * @param k requested degree of parallelism
	 * @return one-column matrix block containing the unique values
	 */
	private static MatrixBlock getUniqueValuesRowColParallel(MatrixBlock blkIn, int k) {
		int numThreads = getNumThreads(k, blkIn.getNumRows());
		ExecutorService pool = CommonThreadPool.get(numThreads);
		try {
			ArrayList<UniqueValueTask> tasks = new ArrayList<>();
			for( int[] range : getBalancedRanges(blkIn.getNumRows(), numThreads) )
				tasks.add(new UniqueValueTask(blkIn, range[0], range[1]));

			// Merge local sets after the workers complete.
			HashSet<Double> hashSet = new HashSet<>();
			List<Future<HashSet<Double>>> rtasks = pool.invokeAll(tasks);
			for( int i = 0; i < rtasks.size(); i++ ) {
				HashSet<Double> localSet = rtasks.get(i).get();
				hashSet.addAll(localSet);
				localSet.clear();
				rtasks.set(i, null);
			}

			return createRowColOutput(hashSet);
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		finally {
			pool.shutdown();
		}
	}

	/**
	 * Parallel row-wise unique values. A first pass computes the output width,
	 * and a second pass materializes the row-local unique values.
	 *
	 * @param blkIn input matrix block
	 * @param k requested degree of parallelism
	 * @return matrix block containing row-wise unique values
	 */
	private static MatrixBlock getUniqueRowValuesParallel(MatrixBlock blkIn, int k) {
		int numThreads = getNumThreads(k, blkIn.getNumRows());
		ExecutorService pool = CommonThreadPool.get(numThreads);
		try {
			ArrayList<int[]> ranges = getBalancedRanges(blkIn.getNumRows(), numThreads);
			int clen2 = getMaxUniqueValues(pool, blkIn, Types.Direction.Row, ranges);
			MatrixBlock blkOut = allocateOutputBlock(blkIn.getNumRows(), clen2);
			fillUniqueValues(pool, blkIn, blkOut, Types.Direction.Row, ranges);

			blkOut.recomputeNonZeros();
			blkOut.examSparsity();
			return blkOut;
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		finally {
			pool.shutdown();
		}
	}

	/**
	 * Parallel column-wise unique values. A first pass computes the output height,
	 * and a second pass materializes the column-local unique values.
	 *
	 * @param blkIn input matrix block
	 * @param k requested degree of parallelism
	 * @return matrix block containing column-wise unique values
	 */
	private static MatrixBlock getUniqueColumnValuesParallel(MatrixBlock blkIn, int k) {
		int numThreads = getNumThreads(k, blkIn.getNumColumns());
		ExecutorService pool = CommonThreadPool.get(numThreads);
		try {
			ArrayList<int[]> ranges = getBalancedRanges(blkIn.getNumColumns(), numThreads);
			int rlen2 = getMaxUniqueValues(pool, blkIn, Types.Direction.Col, ranges);
			MatrixBlock blkOut = allocateOutputBlock(rlen2, blkIn.getNumColumns());
			fillUniqueValues(pool, blkIn, blkOut, Types.Direction.Col, ranges);

			blkOut.recomputeNonZeros();
			blkOut.examSparsity();
			return blkOut;
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		finally {
			pool.shutdown();
		}
	}

	/**
	 * Batched parallel unique for all matrix values.
	 *
	 * @param blkIn input matrix block
	 * @param dir unique direction
	 * @param k requested degree of parallelism
	 * @return one-column matrix block containing the unique values
	 */
	private static MatrixBlock getUniqueValuesRowColBatchedParallel(MatrixBlock blkIn, Types.Direction dir, int k) {
		BatchConfig config = getBatchConfig(blkIn, dir, k);
		if( config == null )
			return getUniqueValuesSequential(blkIn, dir);

		ExecutorService pool = CommonThreadPool.get(config._numThreads);
		try {
			HashSet<Double> hashSet = new HashSet<>();
			for( int pos = 0; pos < config._len; ) {
				ArrayList<UniqueValueTask> tasks = new ArrayList<>();
				for( int i = 0; i < config._numThreads && pos < config._len; i++ ) {
					int end = Math.min(pos + config._taskLen, config._len);
					tasks.add(new UniqueValueTask(blkIn, pos, end));
					pos = end;
				}

				List<Future<HashSet<Double>>> rtasks = pool.invokeAll(tasks);
				for( int i = 0; i < rtasks.size(); i++ ) {
					HashSet<Double> localSet = rtasks.get(i).get();
					hashSet.addAll(localSet);
					localSet.clear();
					rtasks.set(i, null);
				}
			}

			return createRowColOutput(hashSet);
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		finally {
			pool.shutdown();
		}
	}

	/**
	 * Batched parallel row-wise unique values.
	 *
	 * @param blkIn input matrix block
	 * @param dir unique direction
	 * @param k requested degree of parallelism
	 * @return matrix block containing row-wise unique values
	 */
	private static MatrixBlock getUniqueRowValuesBatchedParallel(MatrixBlock blkIn, Types.Direction dir, int k) {
		BatchConfig config = getBatchConfig(blkIn, dir, k);
		if( config == null )
			return getUniqueValuesSequential(blkIn, dir);

		ExecutorService pool = CommonThreadPool.get(config._numThreads);
		try {
			int clen2 = getMaxUniqueValuesBatched(pool, blkIn, dir, config);
			MatrixBlock blkOut = allocateOutputBlock(blkIn.getNumRows(), clen2);
			fillUniqueValuesBatched(pool, blkIn, blkOut, dir, config);

			blkOut.recomputeNonZeros();
			blkOut.examSparsity();
			return blkOut;
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		finally {
			pool.shutdown();
		}
	}

	/**
	 * Batched parallel column-wise unique values.
	 *
	 * @param blkIn input matrix block
	 * @param dir unique direction
	 * @param k requested degree of parallelism
	 * @return matrix block containing column-wise unique values
	 */
	private static MatrixBlock getUniqueColumnValuesBatchedParallel(MatrixBlock blkIn, Types.Direction dir, int k) {
		BatchConfig config = getBatchConfig(blkIn, dir, k);
		if( config == null )
			return getUniqueValuesSequential(blkIn, dir);

		ExecutorService pool = CommonThreadPool.get(config._numThreads);
		try {
			int rlen2 = getMaxUniqueValuesBatched(pool, blkIn, dir, config);
			MatrixBlock blkOut = allocateOutputBlock(rlen2, blkIn.getNumColumns());
			fillUniqueValuesBatched(pool, blkIn, blkOut, dir, config);

			blkOut.recomputeNonZeros();
			blkOut.examSparsity();
			return blkOut;
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		finally {
			pool.shutdown();
		}
	}

	/**
	 * Computes the maximum row-wise or column-wise unique count over balanced ranges.
	 */
	private static int getMaxUniqueValues(ExecutorService pool, MatrixBlock blkIn, Types.Direction dir,
		ArrayList<int[]> ranges) throws Exception {
		ArrayList<UniqueCountTask> tasks = new ArrayList<>();
		for( int[] range : ranges )
			tasks.add(new UniqueCountTask(blkIn, dir, range[0], range[1]));

		int ret = 0;
		List<Future<Integer>> rtasks = pool.invokeAll(tasks);
		for( int i = 0; i < rtasks.size(); i++ ) {
			ret = Math.max(ret, rtasks.get(i).get());
			rtasks.set(i, null);
		}
		return ret;
	}

	/**
	 * Fills row-wise or column-wise unique values over balanced ranges.
	 */
	private static void fillUniqueValues(ExecutorService pool, MatrixBlock blkIn, MatrixBlock blkOut,
		Types.Direction dir, ArrayList<int[]> ranges) throws Exception {
		ArrayList<UniqueOutputTask> tasks = new ArrayList<>();
		for( int[] range : ranges )
			tasks.add(new UniqueOutputTask(blkIn, blkOut, dir, range[0], range[1]));
		List<Future<Void>> rtasks = pool.invokeAll(tasks);
		for( int i = 0; i < rtasks.size(); i++ ) {
			rtasks.get(i).get();
			rtasks.set(i, null);
		}
	}

	/**
	 * Batched variant of getMaxUniqueValues.
	 */
	private static int getMaxUniqueValuesBatched(ExecutorService pool, MatrixBlock blkIn, Types.Direction dir,
		BatchConfig config) throws Exception {
		int ret = 0;
		for( int pos = 0; pos < config._len; ) {
			ArrayList<UniqueCountTask> tasks = new ArrayList<>();
			for( int i = 0; i < config._numThreads && pos < config._len; i++ ) {
				int end = Math.min(pos + config._taskLen, config._len);
				tasks.add(new UniqueCountTask(blkIn, dir, pos, end));
				pos = end;
			}

			List<Future<Integer>> rtasks = pool.invokeAll(tasks);
			for( int i = 0; i < rtasks.size(); i++ ) {
				ret = Math.max(ret, rtasks.get(i).get());
				rtasks.set(i, null);
			}
		}
		return ret;
	}

	/**
	 * Batched variant of fillUniqueValues.
	 */
	private static void fillUniqueValuesBatched(ExecutorService pool, MatrixBlock blkIn, MatrixBlock blkOut,
		Types.Direction dir, BatchConfig config) throws Exception {
		for( int pos = 0; pos < config._len; ) {
			ArrayList<UniqueOutputTask> tasks = new ArrayList<>();
			for( int i = 0; i < config._numThreads && pos < config._len; i++ ) {
				int end = Math.min(pos + config._taskLen, config._len);
				tasks.add(new UniqueOutputTask(blkIn, blkOut, dir, pos, end));
				pos = end;
			}

			List<Future<Void>> rtasks = pool.invokeAll(tasks);
			for( int i = 0; i < rtasks.size(); i++ ) {
				rtasks.get(i).get();
				rtasks.set(i, null);
			}
		}
	}

	/**
	 * Decides whether the input is large enough to justify local deduplication tasks.
	 *
	 * @param blkIn input matrix block
	 * @param dir unique direction
	 * @param k requested degree of parallelism
	 * @return true if the parallel path should be used
	 */
	private static boolean satisfiesMultiThreadingConstraints(MatrixBlock blkIn, Types.Direction dir, int k) {
		if( k <= 1 || ((long) blkIn.getNumRows()) * blkIn.getNumColumns() < PAR_UNIQUE_NUMCELL_THRESHOLD )
			return false;

		switch(dir) {
			case RowCol:
				return blkIn.getNumRows() > 1;
			case Row:
				return blkIn.getNumRows() > 1;
			case Col:
				return blkIn.getNumColumns() > 1;
			default:
				throw new IllegalArgumentException("Unrecognized direction: " + dir);
		}
	}

	/**
	 * Creates balanced half-open ranges [start, end) using the same utility pattern as
	 * other SystemDS matrix libraries.
	 *
	 * @param len number of rows or columns to partition
	 * @param k requested degree of parallelism
	 * @return list of balanced index ranges
	 */
	private static ArrayList<int[]> getBalancedRanges(int len, int k) {
		ArrayList<int[]> ranges = new ArrayList<>();
		ArrayList<Integer> blklens = UtilFunctions.getBalancedBlockSizesDefault(len, getNumThreads(k, len), false);
		for( int i = 0, lb = 0; i < blklens.size(); lb += blklens.get(i), i++ )
			ranges.add(new int[] {lb, lb + blklens.get(i)});
		return ranges;
	}

	/**
	 * Caps the number of workers by the number of row or column partitions available.
	 *
	 * @param k requested degree of parallelism
	 * @param len number of rows or columns to partition
	 * @return effective number of worker threads
	 */
	private static int getNumThreads(int k, int len) {
		return Math.max(1, Math.min(k, len));
	}

	/**
	 * Builds the execution plan for the batched parallel path.
	 *
	 * @param blkIn input matrix block
	 * @param dir unique direction
	 * @param k requested degree of parallelism
	 * @return batch configuration, or null if batching would not use parallelism
	 */
	private static BatchConfig getBatchConfig(MatrixBlock blkIn, Types.Direction dir, int k) {
		int len = getPartitionLength(blkIn, dir);
		long maxBatchIndexes = getMaxLocalDedupIndexes(blkIn, dir);
		if( maxBatchIndexes < 2 )
			return null;

		int numThreads = getNumThreads(k, (int) Math.min(len, maxBatchIndexes));
		if( numThreads <= 1 )
			return null;

		int taskLen = Math.max(1, (int) Math.min(Integer.MAX_VALUE, maxBatchIndexes / numThreads));
		return new BatchConfig(numThreads, taskLen, len);
	}

	/**
	 * Returns the number of rows or columns that define the partition direction.
	 *
	 * @param blkIn input matrix block
	 * @param dir unique direction
	 * @return number of partition indexes
	 */
	private static int getPartitionLength(MatrixBlock blkIn, Types.Direction dir) {
		return dir == Types.Direction.Col ? blkIn.getNumColumns() : blkIn.getNumRows();
	}

	/**
	 * Estimates how many partition indexes can be processed in one batch.
	 *
	 * @param blkIn input matrix block
	 * @param dir unique direction
	 * @return maximum number of rows or columns per batch
	 */
	private static long getMaxLocalDedupIndexes(MatrixBlock blkIn, Types.Direction dir) {
		long cellsPerIndex = dir == Types.Direction.Col ? blkIn.getNumRows() : blkIn.getNumColumns();
		if( cellsPerIndex <= 0 ||
			cellsPerIndex > Long.MAX_VALUE / Double.BYTES / PAR_UNIQUE_LOCAL_BYTES_OVERHEAD )
			return 0;

		long bytesPerIndex = cellsPerIndex * Double.BYTES * PAR_UNIQUE_LOCAL_BYTES_OVERHEAD;
		long maxLocalBytes = Runtime.getRuntime().maxMemory() / PAR_UNIQUE_MAX_LOCAL_BYTES_FRACTION;
		return maxLocalBytes / bytesPerIndex;
	}

	/**
	 * Conservative memory guard for full local deduplication. The estimate includes
	 * a small overhead factor for set objects.
	 *
	 * @param blkIn input matrix block
	 * @param dir unique direction
	 * @return true if local deduplication is small enough for the parallel path
	 */
	private static boolean isLocalDedupMemoryBudgetSafe(MatrixBlock blkIn, Types.Direction dir) {
		return getMaxLocalDedupIndexes(blkIn, dir) >= getPartitionLength(blkIn, dir);
	}

	/**
	 * Allocates and fills a one-column MatrixBlock from a set of unique scalar values.
	 *
	 * @param values unique scalar values
	 * @return one-column matrix block
	 */
	private static MatrixBlock createRowColOutput(HashSet<Double> values) {
		int rlen = values.size();
		MatrixBlock blkOut = allocateOutputBlock(rlen, 1);
		Iterator<Double> iter = values.iterator();
		for( int i = 0; i < rlen; i++ )
			blkOut.set(i, 0, iter.next());
		blkOut.recomputeNonZeros();
		blkOut.examSparsity();
		return blkOut;
	}

	/**
	 * Creates an output block and allocates storage only when at least one cell exists.
	 *
	 * @param rlen number of rows
	 * @param clen number of columns
	 * @return matrix block ready for writes when non-empty
	 */
	private static MatrixBlock allocateOutputBlock(int rlen, int clen) {
		MatrixBlock blkOut = new MatrixBlock(rlen, clen, false);
		if( rlen > 0 && clen > 0 )
			blkOut.allocateBlock();
		return blkOut;
	}

	/**
	 * Configuration for batched execution.
	 */
	private static class BatchConfig {
		private final int _numThreads;
		private final int _taskLen;
		private final int _len;

		private BatchConfig(int numThreads, int taskLen, int len) {
			_numThreads = numThreads;
			_taskLen = taskLen;
			_len = len;
		}
	}

	/**
	 * Worker that deduplicates all scalar values in a row range locally.
	 */
	private static class UniqueValueTask implements Callable<HashSet<Double>> {
		private final MatrixBlock _blkIn;
		private final int _rl;
		private final int _ru;

		private UniqueValueTask(MatrixBlock blkIn, int rl, int ru) {
			_blkIn = blkIn;
			_rl = rl;
			_ru = ru;
		}

		@Override
		public HashSet<Double> call() {
			HashSet<Double> ret = new HashSet<>();
			for( int i = _rl; i < _ru; i++ )
				for( int j = 0; j < _blkIn.getNumColumns(); j++ )
					ret.add(_blkIn.get(i, j));
			return ret;
		}
	}

	/**
	 * Worker that computes the largest row-wise or column-wise unique count in a range.
	 */
	private static class UniqueCountTask implements Callable<Integer> {
		private final MatrixBlock _blkIn;
		private final Types.Direction _dir;
		private final int _l;
		private final int _u;

		private UniqueCountTask(MatrixBlock blkIn, Types.Direction dir, int l, int u) {
			_blkIn = blkIn;
			_dir = dir;
			_l = l;
			_u = u;
		}

		@Override
		public Integer call() {
			HashSet<Double> hashSet = new HashSet<>();
			int ret = 0;
			if( _dir == Types.Direction.Row ) {
				for( int i = _l; i < _u; i++ ) {
					hashSet.clear();
					for( int j = 0; j < _blkIn.getNumColumns(); j++ )
						hashSet.add(_blkIn.get(i, j));
					ret = Math.max(ret, hashSet.size());
				}
			}
			else {
				for( int j = _l; j < _u; j++ ) {
					hashSet.clear();
					for( int i = 0; i < _blkIn.getNumRows(); i++ )
						hashSet.add(_blkIn.get(i, j));
					ret = Math.max(ret, hashSet.size());
				}
			}
			return ret;
		}
	}

	/**
	 * Worker that writes row-wise or column-wise unique values for a range.
	 */
	private static class UniqueOutputTask implements Callable<Void> {
		private final MatrixBlock _blkIn;
		private final MatrixBlock _blkOut;
		private final Types.Direction _dir;
		private final int _l;
		private final int _u;

		private UniqueOutputTask(MatrixBlock blkIn, MatrixBlock blkOut, Types.Direction dir, int l, int u) {
			_blkIn = blkIn;
			_blkOut = blkOut;
			_dir = dir;
			_l = l;
			_u = u;
		}

		@Override
		public Void call() {
			HashSet<Double> hashSet = new HashSet<>();
			if( _dir == Types.Direction.Row ) {
				for( int i = _l; i < _u; i++ ) {
					hashSet.clear();
					for( int j = 0; j < _blkIn.getNumColumns(); j++ )
						hashSet.add(_blkIn.get(i, j));
					int pos = 0;
					for( Double val : hashSet )
						_blkOut.set(i, pos++, val);
				}
			}
			else {
				for( int j = _l; j < _u; j++ ) {
					hashSet.clear();
					for( int i = 0; i < _blkIn.getNumRows(); i++ )
						hashSet.add(_blkIn.get(i, j));
					int pos = 0;
					for( Double val : hashSet )
						_blkOut.set(pos++, j, val);
				}
			}
			return null;
		}
	}
}
