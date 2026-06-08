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
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.UtilFunctions;

public class LibMatrixSketch {
	private static final long PAR_UNIQUE_NUMCELL_THRESHOLD = 1024 * 16;

	/**
	 * Computes unique values, rows, or columns with the original single-threaded behavior.
	 * The overload with a parallelism argument keeps this path as the k=1 baseline.
	 *
	 * @param blkIn input matrix block
	 * @param dir unique direction
	 * @return matrix block containing unique values, rows, or columns
	 */
	public static MatrixBlock getUniqueValues(MatrixBlock blkIn, Types.Direction dir) {
		return getUniqueValues(blkIn, dir, 1);
	}

	/**
	 * Computes unique values, rows, or columns. Parallel execution is used only for
	 * sufficiently large inputs with k > 1; otherwise the existing sequential path is used.
	 *
	 * @param blkIn input matrix block
	 * @param dir unique direction
	 * @param k requested degree of parallelism
	 * @return matrix block containing unique values, rows, or columns
	 */
	public static MatrixBlock getUniqueValues(MatrixBlock blkIn, Types.Direction dir, int k) {
		// similar to R's unique, this operation takes a matrix and computes the unique values
		// (or rows in case of multiple column inputs)
		if( !satisfiesMultiThreadingConstraints(blkIn, dir, k) )
			return getUniqueValuesSequential(blkIn, dir);

		switch(dir) {
			case RowCol:
				return getUniqueValuesRowColParallel(blkIn, k);
			case Row:
				return getUniqueRowsParallel(blkIn, k);
			case Col:
				return getUniqueColumnsParallel(blkIn, k);
			default:
				throw new IllegalArgumentException("Unrecognized direction: " + dir);
		}
	}

	/**
	 * Single-threaded baseline implementation for all unique directions.
	 * This preserves the original RowCol and Row behavior and adds the sequential Col path.
	 *
	 * @param blkIn input matrix block
	 * @param dir unique direction
	 * @return matrix block containing unique values, rows, or columns
	 */
	private static MatrixBlock getUniqueValuesSequential(MatrixBlock blkIn, Types.Direction dir) {
		int rlen = blkIn.getNumRows();
		int clen = blkIn.getNumColumns();

		MatrixBlock blkOut = null;
		switch (dir) {
			case RowCol:
				if( clen != 1 )
					throw new NotImplementedException("Unique only support single-column vectors yet");
				// TODO optimize for dense/sparse/compressed (once multi-column support added)

				// obtain set of unique items (dense input vector)
				HashSet<Double> hashSet = new HashSet<>();
				for( int i=0; i<rlen; i++ ) {
					hashSet.add(blkIn.get(i, 0));
				}

				// allocate output block and place values
				blkOut = createRowColOutput(hashSet);
				break;

			case Row:
				ArrayList<double[]> retainedRows = new ArrayList<>();

				for (int i=0; i<rlen; ++i) {

					// BitSet will not work because we need 2 pieces of info:
					// 1. the index and
					// 2. the value
					// A BitSet gives us only whether there is a value at a particular index, but not what that
					// specific value is.

					double[] currentRow = new double[clen];
					for (int j=0; j<clen; ++j) {
						double rawValue = blkIn.get(i, j);
						currentRow[j] = rawValue;
					}

					// no need to check for duplicates for the first row
					if (i == 0) {
						retainedRows.add(currentRow);
						continue;
					}

					// ensure we are not adding duplicate rows to retainedRows array
					int uniqueRowCount = 0;
					for (int m=0; m<retainedRows.size(); ++m) {

						double[] prevRow = retainedRows.get(m);

						int n = 0;
						while (n < clen) {
							if (prevRow[n] != currentRow[n]) {
								break;
							}
							n++;
						}

						// column check terminates early only if there is a column-level mismatch, ie rows are different
						if (n != clen) {
							uniqueRowCount++;
						}
					}

					// add current row to retainedRows iff it is unique from all prev retained rows
					if (uniqueRowCount == retainedRows.size()) {
						retainedRows.add(currentRow);
					}
				}

				blkOut = createRowOutput(retainedRows, blkIn.getNumColumns());

				break;

			case Col:
				LinkedHashMap<ColKey, double[]> retainedColumns = new LinkedHashMap<>();
				for( int j = 0; j < clen; j++ ) {
					double[] currentColumn = copyColumn(blkIn, j);
					retainedColumns.putIfAbsent(new ColKey(currentColumn), currentColumn);
				}
				blkOut = createColumnOutput(retainedColumns.values(), rlen);
				break;

			default:
				throw new IllegalArgumentException("Unrecognized direction: " + dir);
		}

		return blkOut;
	}

	/**
	 * Parallel unique for single-column vectors. Rows are split into balanced partitions,
	 * each task builds a local set, and the caller merges all local sets afterwards.
	 *
	 * @param blkIn input single-column matrix block
	 * @param k requested degree of parallelism
	 * @return one-column matrix block containing the unique values
	 */
	private static MatrixBlock getUniqueValuesRowColParallel(MatrixBlock blkIn, int k) {
		if( blkIn.getNumColumns() != 1 )
			throw new NotImplementedException("Unique only support single-column vectors yet");

		ExecutorService pool = CommonThreadPool.get(getNumThreads(k, blkIn.getNumRows()));
		try {
			ArrayList<UniqueValueTask> tasks = new ArrayList<>();
			for( int[] range : getBalancedRanges(blkIn.getNumRows(), k) )
				tasks.add(new UniqueValueTask(blkIn, range[0], range[1]));

			// Merge after local deduplication to avoid a shared synchronized set in the workers.
			HashSet<Double> hashSet = new HashSet<>();
			List<Future<HashSet<Double>>> rtasks = pool.invokeAll(tasks);
			for( Future<HashSet<Double>> task : rtasks )
				hashSet.addAll(task.get());

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
	 * Parallel unique rows. Each worker deduplicates its row partition locally, and the
	 * final merge scans the partition results in input order to keep the first occurrence.
	 *
	 * @param blkIn input matrix block
	 * @param k requested degree of parallelism
	 * @return matrix block containing exact unique rows
	 */
	private static MatrixBlock getUniqueRowsParallel(MatrixBlock blkIn, int k) {
		ExecutorService pool = CommonThreadPool.get(getNumThreads(k, blkIn.getNumRows()));
		try {
			ArrayList<UniqueRowTask> tasks = new ArrayList<>();
			for( int[] range : getBalancedRanges(blkIn.getNumRows(), k) )
				tasks.add(new UniqueRowTask(blkIn, range[0], range[1]));

			// Global merge is intentionally single-threaded and ordered for correctness.
			LinkedHashMap<RowKey, double[]> retainedRows = new LinkedHashMap<>();
			List<Future<LinkedHashMap<RowKey, double[]>>> rtasks = pool.invokeAll(tasks);
			for( Future<LinkedHashMap<RowKey, double[]>> task : rtasks )
				for( java.util.Map.Entry<RowKey, double[]> entry : task.get().entrySet() )
					retainedRows.putIfAbsent(entry.getKey(), entry.getValue());

			return createRowOutput(retainedRows.values(), blkIn.getNumColumns());
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		finally {
			pool.shutdown();
		}
	}

	/**
	 * Parallel unique columns. Each worker deduplicates a column partition locally, and
	 * the final merge scans partitions from left to right to keep the first occurrence.
	 *
	 * @param blkIn input matrix block
	 * @param k requested degree of parallelism
	 * @return matrix block containing exact unique columns
	 */
	private static MatrixBlock getUniqueColumnsParallel(MatrixBlock blkIn, int k) {
		ExecutorService pool = CommonThreadPool.get(getNumThreads(k, blkIn.getNumColumns()));
		try {
			ArrayList<UniqueColumnTask> tasks = new ArrayList<>();
			for( int[] range : getBalancedRanges(blkIn.getNumColumns(), k) )
				tasks.add(new UniqueColumnTask(blkIn, range[0], range[1]));

			// Global merge is intentionally single-threaded and ordered for correctness.
			LinkedHashMap<ColKey, double[]> retainedColumns = new LinkedHashMap<>();
			List<Future<LinkedHashMap<ColKey, double[]>>> rtasks = pool.invokeAll(tasks);
			for( Future<LinkedHashMap<ColKey, double[]>> task : rtasks )
				for( java.util.Map.Entry<ColKey, double[]> entry : task.get().entrySet() )
					retainedColumns.putIfAbsent(entry.getKey(), entry.getValue());

			return createColumnOutput(retainedColumns.values(), blkIn.getNumRows());
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		finally {
			pool.shutdown();
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
	 * Copies one row into an immutable key/value array for safe HashMap storage.
	 *
	 * @param blkIn input matrix block
	 * @param row row index to copy
	 * @return copied row values
	 */
	private static double[] copyRow(MatrixBlock blkIn, int row) {
		int clen = blkIn.getNumColumns();
		double[] ret = new double[clen];
		for( int j = 0; j < clen; j++ )
			ret[j] = blkIn.get(row, j);
		return ret;
	}

	/**
	 * Copies one column into an immutable key/value array for safe HashMap storage.
	 *
	 * @param blkIn input matrix block
	 * @param col column index to copy
	 * @return copied column values
	 */
	private static double[] copyColumn(MatrixBlock blkIn, int col) {
		int rlen = blkIn.getNumRows();
		double[] ret = new double[rlen];
		for( int i = 0; i < rlen; i++ )
			ret[i] = blkIn.get(i, col);
		return ret;
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
	 * Allocates and fills a MatrixBlock from retained row copies.
	 *
	 * @param rows unique row copies
	 * @param clen number of columns in the output
	 * @return matrix block containing the unique rows
	 */
	private static MatrixBlock createRowOutput(Collection<double[]> rows, int clen) {
		MatrixBlock blkOut = allocateOutputBlock(rows.size(), clen);
		int i = 0;
		for( double[] row : rows ) {
			for( int j = 0; j < clen; j++ )
				blkOut.set(i, j, row[j]);
			i++;
		}
		blkOut.recomputeNonZeros();
		blkOut.examSparsity();
		return blkOut;
	}

	/**
	 * Allocates and fills a MatrixBlock from retained column copies.
	 *
	 * @param columns unique column copies
	 * @param rlen number of rows in the output
	 * @return matrix block containing the unique columns
	 */
	private static MatrixBlock createColumnOutput(Collection<double[]> columns, int rlen) {
		MatrixBlock blkOut = allocateOutputBlock(rlen, columns.size());
		int j = 0;
		for( double[] column : columns ) {
			for( int i = 0; i < rlen; i++ )
				blkOut.set(i, j, column[i]);
			j++;
		}
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
	 * Computes a stable content hash for row and column keys using numeric equality.
	 *
	 * @param values copied row or column values
	 * @return content hash code
	 */
	private static int hashValues(double[] values) {
		int ret = 1;
		for( double value : values )
			ret = 31 * ret + (value == 0 ? 0 : Double.hashCode(value));
		return ret;
	}

	/**
	 * Compares copied row or column values with exact numeric equality.
	 *
	 * @param left first copied value array
	 * @param right second copied value array
	 * @return true if the arrays represent the same row or column
	 */
	private static boolean equalValues(double[] left, double[] right) {
		if( left.length != right.length )
			return false;
		for( int i = 0; i < left.length; i++ )
			if( left[i] != right[i] && Double.doubleToLongBits(left[i]) != Double.doubleToLongBits(right[i]) )
				return false;
		return true;
	}

	/**
	 * Worker that deduplicates a row partition of a single-column vector locally.
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
				ret.add(_blkIn.get(i, 0));
			return ret;
		}
	}

	/**
	 * Worker that deduplicates copied rows within a row partition before global merge.
	 */
	private static class UniqueRowTask implements Callable<LinkedHashMap<RowKey, double[]>> {
		private final MatrixBlock _blkIn;
		private final int _rl;
		private final int _ru;

		private UniqueRowTask(MatrixBlock blkIn, int rl, int ru) {
			_blkIn = blkIn;
			_rl = rl;
			_ru = ru;
		}

		@Override
		public LinkedHashMap<RowKey, double[]> call() {
			LinkedHashMap<RowKey, double[]> ret = new LinkedHashMap<>();
			for( int i = _rl; i < _ru; i++ ) {
				double[] row = copyRow(_blkIn, i);
				ret.putIfAbsent(new RowKey(row), row);
			}
			return ret;
		}
	}

	/**
	 * Worker that deduplicates copied columns within a column partition before global merge.
	 */
	private static class UniqueColumnTask implements Callable<LinkedHashMap<ColKey, double[]>> {
		private final MatrixBlock _blkIn;
		private final int _cl;
		private final int _cu;

		private UniqueColumnTask(MatrixBlock blkIn, int cl, int cu) {
			_blkIn = blkIn;
			_cl = cl;
			_cu = cu;
		}

		@Override
		public LinkedHashMap<ColKey, double[]> call() {
			LinkedHashMap<ColKey, double[]> ret = new LinkedHashMap<>();
			for( int j = _cl; j < _cu; j++ ) {
				double[] column = copyColumn(_blkIn, j);
				ret.putIfAbsent(new ColKey(column), column);
			}
			return ret;
		}
	}

	/**
	 * Content-based key for copied rows. The referenced array is never mutated after
	 * construction, so it is safe to reuse the copy as both key contents and output data.
	 */
	private static class RowKey {
		private final double[] _values;

		private RowKey(double[] values) {
			_values = values;
		}

		@Override
		public int hashCode() {
			return hashValues(_values);
		}

		@Override
		public boolean equals(Object obj) {
			return obj instanceof RowKey && equalValues(_values, ((RowKey) obj)._values);
		}
	}

	/**
	 * Content-based key for copied columns. The referenced array is never mutated after
	 * construction, so it is safe to reuse the copy as both key contents and output data.
	 */
	private static class ColKey {
		private final double[] _values;

		private ColKey(double[] values) {
			_values = values;
		}

		@Override
		public int hashCode() {
			return hashValues(_values);
		}

		@Override
		public boolean equals(Object obj) {
			return obj instanceof ColKey && equalValues(_values, ((ColKey) obj)._values);
		}
	}
}
