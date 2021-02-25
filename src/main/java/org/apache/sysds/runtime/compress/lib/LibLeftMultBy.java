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

package org.apache.sysds.runtime.compress.lib;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.ColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupOLE;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.colgroup.ColGroupValue;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class LibLeftMultBy {
	private static final Log LOG = LogFactory.getLog(LibLeftMultBy.class.getName());

	private static ThreadLocal<double[]> memPoolLeftMult = new ThreadLocal<double[]>() {
		@Override
		protected double[] initialValue() {
			return null;
		}
	};

	public static MatrixBlock leftMultByMatrixTransposed(CompressedMatrixBlock m1, CompressedMatrixBlock m2,
		MatrixBlock ret, int k) {
		prepareReturnMatrix(m1, m2, ret, true);
		leftMultByCompressedTransposedMatrix(m1
			.getColGroups(), m2, ret, k, m1.getNumColumns(), m1.getMaxNumValues(), m1.isOverlapping());

		ret.recomputeNonZeros();
		return ret;
	}

	public static MatrixBlock leftMultByMatrix(CompressedMatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int k) {
		prepareReturnMatrix(m1, m2, ret, false);
		return leftMultByMatrix(m1
			.getColGroups(), m2, ret, false, m1.getNumColumns(), m1.isOverlapping(), k, m1.getMaxNumValues());
	}

	private static MatrixBlock leftMultByMatrix(List<ColGroup> groups, MatrixBlock that, MatrixBlock ret,
		boolean doTranspose, int numCols, boolean overlapping, int k, Pair<Integer, int[]> v) {

		if(doTranspose) {
			ReorgOperator r_op = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), k);
			that = that.reorgOperations(r_op, new MatrixBlock(), 0, 0, 0);
		}

		return leftMultByMatrix(groups, that, ret, k, numCols, v, overlapping);

	}

	private static MatrixBlock prepareReturnMatrix(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret,
		boolean doTranspose) {
		int numRowsOutput = doTranspose ? m2.getNumColumns() : m2.getNumRows();
		int numColumnsOutput = m1.getNumColumns();
		if(ret == null)
			ret = new MatrixBlock(numRowsOutput, numColumnsOutput, false, numRowsOutput * numColumnsOutput);
		else if(!(ret.getNumColumns() == numColumnsOutput && ret.getNumRows() == numRowsOutput && ret.isAllocated()))
			ret.reset(numRowsOutput, numColumnsOutput, false, numRowsOutput * numColumnsOutput);
		return ret;
	}

	public static void leftMultByTransposeSelf(List<ColGroup> groups, MatrixBlock result, int k, int numColumns,
		Pair<Integer, int[]> v, boolean overlapping) {
		if(!overlapping)
			leftMultBySelfDiagonalColGroup(groups, result, numColumns, v);

		if(groups.size() > 1)
			if(k <= 1)
				leftMultByTransposeSelf(groups, result, v, 0, numColumns, 0, numColumns, overlapping);
			else
				try {
					ExecutorService pool = CommonThreadPool.get(k);
					ArrayList<MatrixMultTransposeReflectedTask> tasks = new ArrayList<>();
					int odd = numColumns % 2;
					for(int i = 0; i < numColumns / 2 + odd; i++) {
						// if(i < numColumns / 4 && numColumns > 100) {
						// int halfRemainingInRow = (i - numColumns) / 2;
						tasks.add(new MatrixMultTransposeReflectedTask(groups, result, i, i + 1, i, numColumns, v,
							overlapping));
						// tasks.add(new MatrixMultTransposeReflectedTask(groups, result, i, i + 1, i +
						// halfRemainingInRow,
						// numColumns, v, overlapping));
						// }
						// else
						// tasks.add(
						// new MatrixMultTransposeReflectedTask(groups, result, i, i + 1, i, numColumns, v,
						// overlapping));
					}

					List<Future<Object>> ret = pool.invokeAll(tasks);
					for(Future<Object> tret : ret)
						tret.get();
					pool.shutdown();
				}
				catch(InterruptedException | ExecutionException e) {
					throw new DMLRuntimeException(e);
				}

	}

	private static void leftMultBySelfDiagonalColGroup(List<ColGroup> groups, MatrixBlock result, int numColumns,
		Pair<Integer, int[]> v) {
		double[] outValues = result.getDenseBlockValues();
		for(ColGroup g : groups) {
			if(g instanceof ColGroupValue) {
				ColGroupValue gv = (ColGroupValue) g;
				int[] counts = gv.getCounts();
				double[] values = gv.getValues();
				int[] columns = gv.getColIndices();

				for(int i = 0; i < columns.length; i++) {
					int y = columns[i];
					for(int j = 0; j < columns.length; j++) {
						int x = columns[j];
						for(int h = 0; h < gv.getValues().length / columns.length; h++) {
							double a = values[h * columns.length + i];
							double b = values[h * columns.length + j];
							outValues[x + y * numColumns] += a * b * counts[h];
						}
					}
				}
			}
			else
				throw new NotImplementedException("Not Implemented diagonal on non ColGroupValue type.");

		}
	}

	public static MatrixBlock leftMultByVectorTranspose(List<ColGroup> colGroups, MatrixBlock vector,
		MatrixBlock result, boolean doTranspose, int k, Pair<Integer, int[]> v, boolean overlap) {
		// transpose vector if required
		MatrixBlock rowVector = vector;
		if(doTranspose) {
			rowVector = new MatrixBlock(1, vector.getNumRows(), false);
			LibMatrixReorg.transpose(vector, rowVector);
		}

		result.reset();
		result.allocateDenseBlock();

		// multi-threaded execution
		try {
			// compute uncompressed column group in parallel
			// ColGroupUncompressed uc = getUncompressedColGroup();
			// if(uc != null)
			// uc.leftMultByRowVector(rowVector, result, k);

			// compute remaining compressed column groups in parallel
			ExecutorService pool = CommonThreadPool.get(Math.min(colGroups.size(), k));
			ArrayList<LeftMatrixVectorMultTask> tasks = new ArrayList<>();

			tasks.add(new LeftMatrixVectorMultTask(colGroups, rowVector, result, v));

			List<Future<Object>> ret = pool.invokeAll(tasks);
			pool.shutdown();
			for(Future<Object> tmp : ret)
				tmp.get();

		}
		catch(InterruptedException | ExecutionException e) {
			LOG.error(e);
			throw new DMLRuntimeException(e);
		}

		// post-processing
		result.recomputeNonZeros();

		return result;
	}

	private static MatrixBlock leftMultByCompressedTransposedMatrix(List<ColGroup> colGroups,
		CompressedMatrixBlock that, MatrixBlock ret, int k, int numColumns, Pair<Integer, int[]> v,
		boolean overlapping) {
		ret.allocateDenseBlock();
		Pair<Integer, int[]> thatV = that.getMaxNumValues();
		if(k <= 1)
			leftMultByCompressedTransposeRowSection(colGroups,
				that.getColGroups(),
				ret,
				v,
				thatV,
				0,
				that.getNumColumns(),
				overlapping,
				k);
		else
			try {
				ExecutorService pool = CommonThreadPool.get(k);
				ArrayList<leftMultByCompressedTransposedMatrixTask> tasks = new ArrayList<>();
				int blklen = (int) (Math.ceil((double) that.getNumColumns() / k));
				int numBlocks = Math.max(that.getNumColumns() / blklen, 1);
				int numExtraThreads = Math.max(k / numBlocks, 1);

				for(int i = 0; i * blklen < that.getNumColumns(); i++)
					tasks.add(new leftMultByCompressedTransposedMatrixTask(colGroups, that.getColGroups(), ret, v,
						thatV, i * blklen, Math.min((i + 1) * blklen, that.getNumColumns()), overlapping,
						numExtraThreads));

				List<Future<Object>> futures = pool.invokeAll(tasks);
				for(Future<Object> tret : futures)
					tret.get(); // check for errors
				pool.shutdown();
			}
			catch(InterruptedException | ExecutionException e) {
				throw new DMLRuntimeException(e);
			}

		return ret;
	}

	private static MatrixBlock leftMultByMatrix(List<ColGroup> colGroups, MatrixBlock that, MatrixBlock ret, int k,
		int numColumns, Pair<Integer, int[]> v, boolean overlapping) {
		ret.allocateDenseBlock();
		if(that.isInSparseFormat())
			ret = leftMultBySparseMatrix(colGroups, that, ret, k, numColumns, v, overlapping);
		else
			ret = leftMultByDenseMatrix(colGroups, that, ret, k, numColumns, v, overlapping);

		ret.setNonZeros(ret.getNumColumns() * ret.getNumRows());
		return ret;
	}

	private static MatrixBlock leftMultByDenseMatrix(List<ColGroup> colGroups, MatrixBlock that, MatrixBlock ret, int k,
		int numColumns, Pair<Integer, int[]> v, boolean overlapping) {
		DenseBlock db = that.getDenseBlock();
		if(db == null)
			throw new DMLRuntimeException("Invalid LeftMult By Dense matrix, input matrix was sparse");

		double[] retV = ret.getDenseBlockValues();
		double[] thatV;
		int blockU;
		int blockL = 0;
		for(ColGroup grp : colGroups)
			if(grp instanceof ColGroupUncompressed)
				((ColGroupUncompressed) grp).leftMultByMatrix(that, ret);

		for(int b = 0; b < db.numBlocks(); b++) {
			int blockSize = db.blockSize(b);
			blockU = Math.min(blockL + blockSize, ret.getNumRows());
			thatV = db.valuesAt(b);

			if(k == 1) {

				for(int j = 0; j < colGroups.size(); j++) {
					colGroups.get(j).leftMultByMatrix(thatV,
						retV,
						colGroups.get(j).getValues(),
						that.getNumRows(),
						ret.getNumColumns(),
						0,
						ret.getNumRows(),
						0);
				}
			}
			else {
				try {
					ExecutorService pool = CommonThreadPool.get(k);
					// compute remaining compressed column groups in parallel
					ArrayList<LeftMatrixMatrixMultTask> tasks = new ArrayList<>();
					int rowBlockSize = 1;
					for(int blo = blockL; blo < blockU; blo += rowBlockSize) {
						tasks.add(new LeftMatrixMatrixMultTask(colGroups, thatV, retV, that.getNumRows(), numColumns,
							blo, Math.min(blo + rowBlockSize, blockU), blo - blockL, v));
					}

					List<Future<Object>> futures = pool.invokeAll(tasks);

					pool.shutdown();
					for(Future<Object> future : futures)
						future.get();
				}
				catch(InterruptedException | ExecutionException e) {
					throw new DMLRuntimeException(e);
				}
			}
			blockL += blockSize;
		}
		return ret;
	}

	private static double[] leftMultSelfVectorTranspose(List<ColGroup> colGroups, double[] rowVector, double[] result,
		int cl, int cu, int j) {
		// j is the current decompressed rowVector.
		Arrays.fill(result, 0);
		for(ColGroup grp : colGroups) {
			int[] columns = grp.getColIndices();
			if(columns[columns.length - 1] >= cl && columns[0] < cu)
				if(Arrays.binarySearch(columns, j) < 0) // if the colGroup is not it self.
					grp.leftMultByRowVector(rowVector, result);

		}

		return result;
	}

	private static MatrixBlock leftMultBySparseMatrix(List<ColGroup> colGroups, MatrixBlock that, MatrixBlock ret,
		int k, int numColumns, Pair<Integer, int[]> v, boolean overlapping) {

		SparseBlock sb = that.getSparseBlock();
		if(sb == null)
			throw new DMLRuntimeException("Invalid Left Mult by Sparse matrix, input matrix was dense");

		for(ColGroup grp : colGroups) {
			if(grp instanceof ColGroupUncompressed)
				((ColGroupUncompressed) grp).leftMultByMatrix(that, ret);
		}

		double[][] materialized = new double[colGroups.size()][];
		boolean containsOLE = false;
		for(int i = 0; i < colGroups.size(); i++) {
			materialized[i] = colGroups.get(i).getValues();
			if(colGroups.get(i) instanceof ColGroupOLE) {
				containsOLE = true;
			}
		}
		if(k == 1) {
			double[] tmpA = containsOLE ? new double[CompressionSettings.BITMAP_BLOCK_SZ * 2] : null;

			for(int j = 0; j < colGroups.size(); j++) {
				for(int r = 0; r < that.getNumRows(); r++) {
					if(!sb.isEmpty(r)) {
						colGroups.get(j).leftMultBySparseMatrix(sb,
							ret.getDenseBlockValues(),
							materialized[j],
							that.getNumRows(),
							numColumns,
							r,
							tmpA);
					}
				}
			}
		}
		else {
			ExecutorService pool = CommonThreadPool.get(k);
			ArrayList<LeftMatrixSparseMatrixMultTask> tasks = new ArrayList<>();
			try {
				// long thatnnz = that.getNonZeros();
				// int rowBlockSize = that.getNumRows() / k;
				int rowBlockSize = (int) Math.ceil(1000.0 / (that.getNonZeros() / that.getNumRows()));
				// rowBlockSize = 1;
				for(int r = 0; r * rowBlockSize < that.getNumRows(); r++) {
					if(overlapping) {
						tasks.add(new LeftMatrixSparseMatrixMultTask(colGroups, materialized, sb,
							ret.getDenseBlockValues(), that.getNumRows(), numColumns, v, r * rowBlockSize,
							Math.min((r + 1) * rowBlockSize, that.getNumRows())));
					}
					else {
						for(int i = 0; i < colGroups.size(); i++) {
							tasks.add(new LeftMatrixSparseMatrixMultTask(colGroups.get(i), materialized, i, sb,
								ret.getDenseBlockValues(), that.getNumRows(), numColumns, v, r * rowBlockSize,
								Math.min((r + 1) * rowBlockSize, that.getNumRows())));
						}
					}
				}

				List<Future<Object>> futures = pool.invokeAll(tasks);
				pool.shutdown();
				for(Future<Object> future : futures)
					future.get();
				memPoolLeftMult.remove();
			}
			catch(InterruptedException | ExecutionException e) {
				throw new DMLRuntimeException(e);
			}
		}

		return ret;

	}

	private static void leftMultByTransposeSelf(List<ColGroup> groups, MatrixBlock result, Pair<Integer, int[]> v,
		int rl, int ru, int cl, int cu, boolean overlapping) {
		if(overlapping)
			leftMultByTransposeSelfOverlapping(groups, result, v, rl, ru, cl, cu);
		else
			leftMultByTransposeSelfNormal(groups, result, v, rl, ru, cl, cu);
	}

	private static void leftMultByTransposeSelfNormal(List<ColGroup> groups, MatrixBlock result, Pair<Integer, int[]> v,
		int rl, int ru, int cl, int cu) {
		// It should be possible to get better performance exploiting if the matrix is not overlapping.
		// TODO: exploit specfic column groups (DDC most likely) to gain better performance.
		// calculate : count * v^2

		// preallocated dense tmp matrix blocks

		final int numRows = groups.get(0).getNumRows();
		double[] lhs = memPoolLeftMult.get() != null ? memPoolLeftMult.get(): new double[numRows];
		double[] tmpret = new double[result.getNumColumns()];

		for(int j = rl; j < ru; j++) {
			if(!rowIsDone(result.getDenseBlockValues(), result.getNumColumns(), j)) {

				ColGroup.decompressColumnToArray(lhs, j, groups);
				leftMultSelfVectorTranspose(groups, lhs, tmpret, Math.max(cl, rl), cu, j);
				copyToUpperTriangle(result
					.getDenseBlockValues(), tmpret, j, result.getNumColumns(), Math.max(cl, rl), cu);
				Arrays.fill(lhs, 0);
			}
		}
	}

	private static boolean rowIsDone(double[] values, int nrColumns, int row) {
		int offset = nrColumns * row + row;
		for(int i = row; i < nrColumns; i++) {
			if(values[offset++] == 0.0)
				return false;
		}
		return true;
	}

	private static void copyToUpperTriangle(double[] ret, double[] tmp, int row, int totalRows, int cl, int cu) {
		int offOut = row * totalRows;
		for(int i = cl; i < cu; i++) {
			ret[offOut + i] += tmp[i];
		}
	}

	private static void leftMultByTransposeSelfOverlapping(List<ColGroup> groups, MatrixBlock result,
		Pair<Integer, int[]> v, int rl, int ru, int cl, int cu) {

		final int numRows = groups.get(0).getNumRows();
		double[] lhs = new double[numRows];
		double[] tmpret = new double[result.getNumColumns()];

		for(int j = rl; j < ru; j++) {
			ColGroup.decompressColumnToArray(lhs, j, groups);
			leftMultSelfVectorTranspose(groups, lhs, tmpret, Math.max(cl, rl), cu, -1);
			copyToUpperTriangle(result.getDenseBlockValues(), tmpret, j, result.getNumColumns(), Math.max(cl, rl), cu);
			Arrays.fill(lhs, 0);
		}

	}

	private static void leftMultByCompressedTransposeRowSection(List<ColGroup> thisGroups, List<ColGroup> thatGroups,
		MatrixBlock result, Pair<Integer, int[]> v, Pair<Integer, int[]> thatV, int rl, int ru, boolean overlapping,
		int k) {
		if(k > 1 && !overlapping)
			leftMultByCompressedTransposeRowSectionParallel(thisGroups, thatGroups, result, v, thatV, rl, ru, k);
		else
			leftMultByCompressedTransposeRowSectionSingleThread(thisGroups, thatGroups, result, v, thatV, rl, ru);

	}

	private static void leftMultByCompressedTransposeRowSectionParallel(List<ColGroup> thisGroups,
		List<ColGroup> thatGroups, MatrixBlock result, Pair<Integer, int[]> v, Pair<Integer, int[]> thatV, int rl,
		int ru, int k) {

		// preallocated dense tmp matrix blocks
		MatrixBlock lhs = new MatrixBlock(1, thisGroups.get(0).getNumRows(), false);
		MatrixBlock tmpret = new MatrixBlock(1, result.getNumColumns(), false);
		lhs.allocateDenseBlock();
		tmpret.allocateDenseBlock();

		ExecutorService pool = CommonThreadPool.get(k);
		ArrayList<leftMultByVectorTransposeTask> tasks = new ArrayList<>();
		for(int j = rl; j < ru; j++) {
			ColGroup.decompressColumnToBlock(lhs, j, thatGroups);
			if(!lhs.isEmptyBlock(false)) {

				try {
					int groupBatch = Math.max(thisGroups.size() / k, 1);

					for(int i = 0; i * groupBatch < thisGroups.size(); i++) {
						tasks.add(new leftMultByVectorTransposeTask(thisGroups, lhs, tmpret, i * groupBatch,
							Math.min(thisGroups.size(), (i + 1) * groupBatch), v));
					}
					for(Future<Object> future : pool.invokeAll(tasks))
						future.get();
				}
				catch(InterruptedException | ExecutionException e) {
					throw new DMLRuntimeException(e);
				}

				double[] tmpRetValues = tmpret.getDenseBlockValues();
				double[] resultValues = result.getDenseBlockValues();
				int offset = tmpret.getNumColumns() * j;
				for(int i = 0; i < tmpret.getNumColumns(); i++, offset++) {
					resultValues[offset] += tmpRetValues[i];
					tmpRetValues[i] = 0;
				}
			}
			lhs.reset();
			tasks.clear();
		}
		pool.shutdown();

		// post processing
		ColGroupValue.cleanupThreadLocalMemory();
	}

	private static void leftMultByCompressedTransposeRowSectionSingleThread(List<ColGroup> thisGroups,
		List<ColGroup> thatGroups, MatrixBlock result, Pair<Integer, int[]> v, Pair<Integer, int[]> thatV, int rl,
		int ru) {
		final int numRows = thisGroups.get(0).getNumRows();

		// preallocated dense tmp matrix blocks
		MatrixBlock lhs = new MatrixBlock(1, numRows, false);
		MatrixBlock tmpret = new MatrixBlock(1, result.getNumColumns(), false);

		lhs.allocateDenseBlock();
		tmpret.allocateDenseBlock();

		for(int j = rl; j < ru; j++) {
			ColGroup.decompressColumnToBlock(lhs, j, thatGroups);
			if(!lhs.isEmptyBlock(false)) {
				for(ColGroup grp : thisGroups) {
					grp.leftMultByRowVector(lhs.getDenseBlockValues(), tmpret.getDenseBlockValues());
				}
				double[] tmpRetValues = tmpret.getDenseBlockValues();
				double[] resultValues = result.getDenseBlockValues();
				int offset = tmpret.getNumColumns() * j;
				for(int i = 0; i < tmpret.getNumColumns(); i++, offset++) {
					resultValues[offset] += tmpRetValues[i];
					tmpRetValues[i] = 0;
				}
			}
			lhs.reset();
		}

	}

	private static class LeftMatrixVectorMultTask implements Callable<Object> {
		private final List<ColGroup> _groups;
		private final MatrixBlock _vect;
		private final MatrixBlock _ret;
		private final Pair<Integer, int[]> _v;

		protected LeftMatrixVectorMultTask(List<ColGroup> groups, MatrixBlock vect, MatrixBlock ret,
			Pair<Integer, int[]> v) {
			_groups = groups;
			_vect = vect;
			_ret = ret;
			_v = v;
		}

		@Override
		public Object call() {
			try {
				ColGroupValue.setupThreadLocalMemory(_v.getLeft() + 1);
				for(int i = 0; i < _groups.size(); i++) {
					_groups.get(i).leftMultByRowVector(_vect.getDenseBlockValues(), _ret.getDenseBlockValues());
				}
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
			return null;
		}
	}

	private static class LeftMatrixMatrixMultTask implements Callable<Object> {
		private final List<ColGroup> _group;
		private final double[] _that;
		private final double[] _ret;
		private final int _numRows;
		private final int _numCols;
		private final int _rl;
		private final int _ru;
		private final int _vOff;
		private final Pair<Integer, int[]> _v;

		protected LeftMatrixMatrixMultTask(List<ColGroup> group, double[] that, double[] ret, int numRows, int numCols,
			int rl, int ru, int vOff, Pair<Integer, int[]> v) {
			_group = group;
			_that = that;
			_ret = ret;
			_numRows = numRows;
			_numCols = numCols;
			_rl = rl;
			_ru = ru;
			_vOff = vOff;
			_v = v;
		}

		@Override
		public Object call() {
			// setup memory pool for reuse

			double[][] materialized = new double[_group.size()][];
			for(int i = 0; i < _group.size(); i++) {
				materialized[i] = _group.get(i).getValues();
			}
			try {
				ColGroupValue.setupThreadLocalMemory(_v.getLeft() + 1);
				for(int j = 0; j < _group.size(); j++) {
					_group.get(j).leftMultByMatrix(_that, _ret, materialized[j], _numRows, _numCols, _rl, _ru, _vOff);
				}
				ColGroupValue.cleanupThreadLocalMemory();

			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
			return null;
		}
	}

	private static class LeftMatrixSparseMatrixMultTask implements Callable<Object> {
		private final List<ColGroup> _groups;
		private final ColGroup _group;
		private final int _i; // Used to identify the index for the materialized values.
		private final SparseBlock _that;
		private final double[] _ret;
		private final int _numRows;
		private final int _numCols;
		private final Pair<Integer, int[]> _v;
		private final double[][] _materialized;
		private final int _rl;
		private final int _ru;

		protected LeftMatrixSparseMatrixMultTask(List<ColGroup> group, double[][] materialized, SparseBlock that,
			double[] ret, int numRows, int numCols, Pair<Integer, int[]> v, int rl, int ru) {
			_groups = group;
			_group = null;
			_i = -1;
			_materialized = materialized;
			_that = that;
			_ret = ret;
			_numRows = numRows;
			_numCols = numCols;
			_v = v;
			_rl = rl;
			_ru = ru;
		}

		protected LeftMatrixSparseMatrixMultTask(ColGroup group, double[][] materialized, int i, SparseBlock that,
			double[] ret, int numRows, int numCols, Pair<Integer, int[]> v, int rl, int ru) {
			_groups = null;
			_group = group;
			_i = i;
			_materialized = materialized;
			_that = that;
			_ret = ret;
			_numRows = numRows;
			_numCols = numCols;
			_v = v;
			_rl = rl;
			_ru = ru;
		}

		@Override
		public Object call() {
			// Temporary Array to store 2 * block size in
			double[] tmpA = memPoolLeftMult.get();
			if(tmpA == null) {
				if(_groups != null) {
					tmpA = new double[Math.min(CompressionSettings.BITMAP_BLOCK_SZ * 2, _groups.get(0).getNumRows())];
				}
				else {
					tmpA = new double[Math.min(CompressionSettings.BITMAP_BLOCK_SZ * 2, _group.getNumRows())];
				}
				memPoolLeftMult.set(tmpA);
			}
			else {
				Arrays.fill(tmpA, 0.0);
			}

			ColGroupValue.setupThreadLocalMemory(_v.getLeft() + 1);
			try {
				if(_groups != null) {
					for(int j = 0; j < _groups.size(); j++) {
						double[] materializedV = _materialized[j];
						for(int r = _rl; r < _ru; r++) {
							if(!_that.isEmpty(r)) {
								_groups.get(j)
									.leftMultBySparseMatrix(_that, _ret, materializedV, _numRows, _numCols, r, tmpA);

							}
						}
					}
				}
				else if(_group != null) {
					for(int r = _rl; r < _ru; r++) {
						if(!_that.isEmpty(r)) {
							_group.leftMultBySparseMatrix(_that, _ret, _materialized[_i], _numRows, _numCols, r, tmpA);
						}
					}
				}
			}
			catch(Exception e) {
				e.printStackTrace();
				throw new DMLRuntimeException(e);
			}
			return null;
		}
	}

	/**
	 * The reflected part means that this task is executed on the rl to ru specified ... AND! the reflection on the
	 * bottom half of the matrix. This makes each task have equal size in execution because the TSMM only require us to
	 * calculate the upper half of the output matrix. if the number of columns is uneven the first column does not get
	 * extra work.
	 */
	private static class MatrixMultTransposeReflectedTask implements Callable<Object> {
		private final List<ColGroup> _groups;
		private final MatrixBlock _ret;
		private int _cl;
		private int _cu;
		private int _rl;
		private int _ru;
		private final Pair<Integer, int[]> _v;
		private final boolean _overlapping;

		protected MatrixMultTransposeReflectedTask(List<ColGroup> groups, MatrixBlock ret, int rl, int ru, int cl,
			int cu, Pair<Integer, int[]> v, boolean overlapping) {
			_groups = groups;
			_ret = ret;
			_cl = cl;
			_cu = cu;
			_rl = rl;
			_ru = ru;
			_v = v;
			_overlapping = overlapping;
		}

		@Override
		public Object call() {
			ColGroupValue.setupThreadLocalMemory(_v.getLeft() + 1);
			leftMultByTransposeSelf(_groups, _ret, _v, _rl, _ru, _cl, _cu, _overlapping);
			int nCol = _ret.getNumColumns();
			double[] tmpA = memPoolLeftMult.get();
			if(tmpA == null) {
				tmpA = new double[_groups.get(0).getNumRows()];
				memPoolLeftMult.set(tmpA);
			}
			else
				Arrays.fill(tmpA, 0);
			if(nCol % 2 == 1) {
				_rl = _rl == 0 ? 0 : _rl - 1;
				_ru = _ru - 1;
			}
			if(_rl != _ru)
				leftMultByTransposeSelf(_groups, _ret, _v, nCol - _ru, nCol - _rl, _cl, _cu, _overlapping);
			return null;
		}
	}

	private static class leftMultByCompressedTransposedMatrixTask implements Callable<Object> {
		private final List<ColGroup> _groups;
		private final List<ColGroup> _thatGroups;
		private final MatrixBlock _ret;
		private final int _rl;
		private final int _ru;
		private final Pair<Integer, int[]> _v;
		private final Pair<Integer, int[]> _thatV;
		private final boolean _overlapping;
		private final int _extraThreads;

		protected leftMultByCompressedTransposedMatrixTask(List<ColGroup> thisGroups, List<ColGroup> thatGroups,
			MatrixBlock ret, Pair<Integer, int[]> v, Pair<Integer, int[]> thatV, int rl, int ru, boolean overlapping,
			int extraThreads) {
			_groups = thisGroups;
			_thatGroups = thatGroups;
			_ret = ret;
			_rl = rl;
			_ru = ru;
			_v = v;
			_thatV = thatV;
			_overlapping = overlapping;
			_extraThreads = extraThreads;
		}

		@Override
		public Object call() {
			ColGroupValue.setupThreadLocalMemory(Math.max(_v.getLeft(), _thatV.getLeft()) + 1);
			leftMultByCompressedTransposeRowSection(_groups,
				_thatGroups,
				_ret,
				_v,
				_thatV,
				_rl,
				_ru,
				_overlapping,
				_extraThreads);
			return null;
		}
	}

	private static class leftMultByVectorTransposeTask implements Callable<Object> {
		private final List<ColGroup> _grps;
		private final MatrixBlock _rowVector;
		private final MatrixBlock _result;
		private final int _gl;
		private final int _gu;
		private final Pair<Integer, int[]> _v;

		protected leftMultByVectorTransposeTask(List<ColGroup> grps, MatrixBlock rowVector, MatrixBlock result, int gl,
			int gu, Pair<Integer, int[]> v) {
			_grps = grps;
			_rowVector = rowVector;
			_result = result;
			_gl = gl;
			_gu = gu;
			_v = v;
		}

		@Override
		public Object call() {
			ColGroupValue.setupThreadLocalMemory(_v.getLeft() + 1);
			for(int i = _gl; i < _gu; i++) {
				_grps.get(i).leftMultByRowVector(_rowVector.getDenseBlockValues(), _result.getDenseBlockValues());
			}
			return null;
		}
	}
}
