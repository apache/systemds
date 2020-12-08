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
import org.apache.sysds.runtime.compress.utils.LinearAlgebraUtils;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class LibLeftMultBy {
	private static final Log LOG = LogFactory.getLog(LibLeftMultBy.class.getName());

	private static ThreadLocal<double[]> memPoolOLE = new ThreadLocal<double[]>() {
		@Override
		protected double[] initialValue() {
			return null;
		}
	};

	public static MatrixBlock leftMultByMatrix(List<ColGroup> groups, MatrixBlock that, MatrixBlock ret,
		boolean doTranspose, boolean allocTmp, int numCols, boolean overlapping, int k, Pair<Integer, int[]> v) {
		int numRowsOutput = doTranspose ? that.getNumColumns() : that.getNumRows();
		if(ret == null)
			ret = new MatrixBlock(numRowsOutput, numCols, false, numRowsOutput * numCols);
		else if(!(ret.getNumColumns() == numCols && ret.getNumRows() == numRowsOutput && ret.isAllocated()))
			ret.reset(numRowsOutput, numCols, false, numRowsOutput * numCols);

		if(that instanceof CompressedMatrixBlock) {
			if(doTranspose) {
				return leftMultByCompressedTransposedMatrix(groups,
					(CompressedMatrixBlock) that,
					ret,
					k,
					numCols,
					v,
					overlapping);
			}
			else {
				LOG.error("Decompression Left side Matrix (Should not really happen)");
				that = ((CompressedMatrixBlock) that).decompress(k);
			}
		}
		else if(doTranspose) {
			ReorgOperator r_op = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), k);
			that = that.reorgOperations(r_op, new MatrixBlock(), 0, 0, 0);
		}

		return leftMultByMatrix(groups, that, ret, k, numCols, v, overlapping);

	}

	public static void leftMultByTransposeSelf(List<ColGroup> groups, MatrixBlock result, int k, int numColumns,
		Pair<Integer, int[]> v, boolean overlapping) {

		if(k <= 1) {
			int cl = 0;
			int cu = numColumns;
			leftMultByTransposeSelfOverlapping(groups, result, v, cl, cu, overlapping);
		}
		else {
			try {
				ExecutorService pool = CommonThreadPool.get(k);
				ArrayList<MatrixMultTransposeTaskOverlapping> tasks = new ArrayList<>();
				int blklen = (int) (Math.ceil((double) numColumns / k));
				for(int i = 0; i * blklen < numColumns; i++)
					tasks.add(new MatrixMultTransposeTaskOverlapping(groups, result, i * blklen,
						Math.min((i + 1) * blklen, numColumns), v, overlapping));
				List<Future<Object>> ret = pool.invokeAll(tasks);
				for(Future<Object> tret : ret)
					tret.get(); // check for errors
				pool.shutdown();
			}
			catch(InterruptedException | ExecutionException e) {
				throw new DMLRuntimeException(e);
			}
		}
	}

	public static MatrixBlock leftMultByCompressedTransposedMatrix(List<ColGroup> colGroups, CompressedMatrixBlock that,
		MatrixBlock ret, int k, int numColumns, Pair<Integer, int[]> v, boolean overlapping) {

		if(ret == null)
			ret = new MatrixBlock(that.getNumColumns(), numColumns, true, -1);
		else
			ret.reset(that.getNumColumns(), numColumns, true, -1);
		ret.allocateDenseBlock();
		Pair<Integer, int[]> thatV = that.getMaxNumValues();
		LOG.error("Left mult by compressed transposed matrix: threads" + k);
		if(k <= 1) {
			int rl = 0;
			int ru = that.getNumColumns();
			leftMultByTranspose(colGroups, that.getColGroups(), ret, v, thatV, rl, ru, overlapping, 1);
		}
		else {
			try {
				ExecutorService pool = CommonThreadPool.get(k);
				ArrayList<leftMultByCompressedTransposedMatrixTask> tasks = new ArrayList<>();
				int blklen = (int) (Math.ceil((double) that.getNumColumns() / k));
				int numBlocks = that.getNumColumns() / blklen;
				int numExtraThreads = k / numBlocks;
				LOG.error("overlapping  : " + overlapping);
				// if(!overlapping) {
				// for(int i = 0; i < that.getNumColumns(); i++) {
				// tasks.add(new leftMultByCompressedTransposedMatrixTask(colGroups, that.getColGroups(), ret, v,
				// thatV, i * blklen, Math.min((i + 1) * blklen, that.getNumColumns()), overlapping, ));
				// }
				// }
				// else {
				for(int i = 0; i * blklen < that.getNumColumns(); i++)
					tasks.add(new leftMultByCompressedTransposedMatrixTask(colGroups, that.getColGroups(), ret, v,
						thatV, i * blklen, Math.min((i + 1) * blklen, that.getNumColumns()), overlapping,
						numExtraThreads));
				// }
				List<Future<Object>> futures = pool.invokeAll(tasks);
				LOG.error("tasks: " + futures.size() + "  Each task has threads: " + numExtraThreads);
				for(Future<Object> tret : futures)
					tret.get(); // check for errors
				pool.shutdown();
			}
			catch(InterruptedException | ExecutionException e) {
				throw new DMLRuntimeException(e);
			}
		}
		return ret;
	}

	private static MatrixBlock leftMultByMatrix(List<ColGroup> colGroups, MatrixBlock that, MatrixBlock ret, int k,
		int numColumns, Pair<Integer, int[]> v, boolean overlapping) {
		ret.allocateDenseBlock();
		if(that.isInSparseFormat()) {
			ret = leftMultBySparseMatrix(colGroups, that, ret, k, numColumns, v, overlapping);
		}
		else {
			ret = leftMultByDenseMatrix(colGroups, that, ret, k, numColumns, v, overlapping);
		}

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
				// Pair<Integer, int[]> v = getMaxNumValues(colGroups);

				ColGroupValue.setupThreadLocalMemory(v.getLeft() + 1);
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
				ColGroupValue.cleanupThreadLocalMemory();
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

	private static MatrixBlock leftMultByVectorTranspose(List<ColGroup> colGroups, MatrixBlock vector,
		MatrixBlock result, boolean doTranspose, boolean allocTmp, Pair<Integer, int[]> v, boolean overlap) {

		MatrixBlock rowVector = vector;
		// Note that transpose here is a metadata operation since the input is a vector.
		if(doTranspose) {
			rowVector = new MatrixBlock(1, vector.getNumRows(), false);
			LibMatrixReorg.transpose(vector, rowVector);
		}

		// initialize and allocate the result
		result.reset();
		result.allocateDenseBlock();

		// setup memory pool for reuse
		if(allocTmp) {
			// Pair<Integer, int[]> v = getMaxNumValues(colGroups);
			ColGroupValue.setupThreadLocalMemory(v.getLeft() + 1); // +1 for efficiency in DDC groups.
			for(int i = 0; i < colGroups.size(); i++) {
				colGroups.get(i).leftMultByRowVector(rowVector.getDenseBlockValues(),
					result.getDenseBlockValues(),
					v.getRight()[i]);
			}
			ColGroupValue.cleanupThreadLocalMemory();
		}
		else {

			for(ColGroup grp : colGroups) {
				grp.leftMultByRowVector(rowVector.getDenseBlockValues(), result.getDenseBlockValues(), -1);
			}
		}

		// delegate matrix-vector operation to each column group

		// post-processing
		// if(allocTmp)
		result.recomputeNonZeros();

		return result;
	}

	public static MatrixBlock leftMultByVectorTranspose(List<ColGroup> colGroups, MatrixBlock vector,
		MatrixBlock result, boolean doTranspose, int k, Pair<Integer, int[]> v, boolean overlap) {
		// transpose vector if required
		MatrixBlock rowVector = vector;
		if(doTranspose) {
			rowVector = new MatrixBlock(1, vector.getNumRows(), false);
			LibMatrixReorg.transpose(vector, rowVector);
		}

		// initialize and allocate the result
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

			// if(overlap){
			tasks.add(new LeftMatrixVectorMultTask(colGroups, rowVector, result, v));
			// } else{
			// ArrayList<ColGroup>[] grpParts = createStaticTaskPartitioning(colGroups, 4 * k, true);
			// for(ArrayList<ColGroup> groups : grpParts)
			// tasks.add(new LeftMatrixVectorMultTask(groups, rowVector, result, v));
			// }

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

			ColGroupValue.setupThreadLocalMemory(v.getLeft() + 1);
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
			ColGroupValue.cleanupThreadLocalMemory();
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
				memPoolOLE.remove();
			}
			catch(InterruptedException | ExecutionException e) {
				throw new DMLRuntimeException(e);
			}
		}

		return ret;

	}

	// private static void leftMultByTransposeSelfNonOverlapping(List<ColGroup> groups, MatrixBlock result,
	// Pair<Integer, int[]> v, int gl, int gu) {

	// // TODO exploit potential multiplcation in compressed format.

	// final int numRows = groups.get(0).getNumRows();

	// // preallocated dense tmp matrix blocks
	// MatrixBlock lhs = new MatrixBlock(1, numRows, false);
	// MatrixBlock tmpret = new MatrixBlock(1, result.getNumColumns(), false);
	// lhs.allocateDenseBlock();
	// tmpret.allocateDenseBlock();

	// // setup memory pool for reuse
	// ColGroupValue.setupThreadLocalMemory(v.getLeft() + 1);

	// // approach: for each colgroup, extract uncompressed columns one at-a-time
	// // vector-matrix multiplies against remaining col groups
	// for(int i = gl; i < gu; i++) {
	// // get current group and relevant col groups
	// ColGroup group = groups.get(i);
	// int[] ixgroup = group.getColIndices();
	// List<ColGroup> tmpList = groups.subList(i, groups.size());

	// // if(group instanceof ColGroupDDC // single DDC group
	// // && ixgroup.length == 1 && !containsUC && numRows < CompressionSettings.BITMAP_BLOCK_SZ) {
	// // // compute vector-matrix partial result
	// // leftMultByVectorTranspose(tmpList, (ColGroupDDC) group, tmpret);

	// // // write partial results (disjoint non-zeros)
	// // LinearAlgebraUtils.copyNonZerosToUpperTriangle(result, tmpret, ixgroup[0]);
	// // }
	// // else {
	// // for all uncompressed lhs columns vectors
	// for(int j = 0; j < result.getNumColumns(); j++) {
	// ColGroup.decompressToBlock(lhs, j, groups);

	// if(!lhs.isEmptyBlock(false)) {
	// // tmpret.reset();
	// // compute vector-matrix partial result
	// // leftMultByMatrix(groups,lhs, tmpret, false, true, 0, 0, overlapping, 1, v );
	// leftMultByVectorTranspose(groups, lhs, tmpret, false, true, v, overlapping);
	// // LOG.error(tmpret);

	// // write partial results (disjoint non-zeros)
	// LinearAlgebraUtils.copyNonZerosToUpperTriangle(result, tmpret, j);
	// }
	// lhs.reset();
	// // }
	// }
	// }

	// // post processing
	// ColGroupValue.cleanupThreadLocalMemory();
	// }

	private static void leftMultByTransposeSelfOverlapping(List<ColGroup> groups, MatrixBlock result,
		Pair<Integer, int[]> v, int cl, int cu, boolean overlapping) {
		// It should be possible to get better performance exploiting if the matrix is not overlapping.
		// TODO: exploit specfic column groups (DDC most likely) to gain better performance.
		// Idea multiplying with one self simply use count of values, and then
		// calculate : count * v^2

		final int numRows = groups.get(0).getNumRows();

		// preallocated dense tmp matrix blocks
		MatrixBlock lhs = new MatrixBlock(1, numRows, false);
		MatrixBlock tmpret = new MatrixBlock(1, result.getNumColumns(), false);
		lhs.allocateDenseBlock();
		tmpret.allocateDenseBlock();

		// setup memory pool for reuse
		ColGroupValue.setupThreadLocalMemory(v.getLeft() + 1);

		for(int j = cl; j < cu; j++) {
			ColGroup.decompressToBlock(lhs, j, groups);
			if(!lhs.isEmptyBlock(false)) {
				leftMultByVectorTranspose(groups, lhs, tmpret, false, true, v, overlapping);
				LinearAlgebraUtils.copyNonZerosToUpperTriangle(result, tmpret, j);
			}
			lhs.reset();
		}

		// post processing
		ColGroupValue.cleanupThreadLocalMemory();
	}

	private static void leftMultByTranspose(List<ColGroup> thisGroups, List<ColGroup> thatGroups, MatrixBlock result,
		Pair<Integer, int[]> v, Pair<Integer, int[]> thatV, int rl, int ru, boolean overlapping, int k) {

		final int numRows = thisGroups.get(0).getNumRows();

		// preallocated dense tmp matrix blocks
		MatrixBlock lhs = new MatrixBlock(1, numRows, false);
		MatrixBlock tmpret = new MatrixBlock(1, result.getNumColumns(), false);
		lhs.allocateDenseBlock();
		tmpret.allocateDenseBlock();
		if(k > 1)
			ColGroupValue.setupThreadLocalMemory(Math.max(v.getLeft(), thatV.getLeft()) + 1);

		ExecutorService pool = (k > 1) ? CommonThreadPool.get(k) : null;
		ArrayList<leftMultByVectorTransposeTask> tasks = (k > 1) ? new ArrayList<>() : null;
		for(int j = rl; j < ru; j++) {
			ColGroup.decompressToBlock(lhs, j, thatGroups);
			if(!lhs.isEmptyBlock(false)) {
				if(!overlapping && k > 1) {
					try {
						int groupBatch = thisGroups.size() / k;

						for(int i = 0; i * groupBatch < thisGroups.size(); i++) {
							tasks.add(new leftMultByVectorTransposeTask(thisGroups, lhs, tmpret, i * groupBatch,
								Math.min(thisGroups.size(), (i + 1) * groupBatch), v));
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
				else {
					for(ColGroup grp : thisGroups) {
						grp.leftMultByRowVector(lhs.getDenseBlockValues(), tmpret.getDenseBlockValues(), -1);
					}
				}
				for(int i = 0; i < tmpret.getNumColumns(); i++) {
					result.appendValue(j, i, tmpret.quickGetValue(0, i));
				}
			}
			lhs.reset();
		}

		// post processing
		ColGroupValue.cleanupThreadLocalMemory();

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
			// setup memory pool for reuse
			try {
				ColGroupValue.setupThreadLocalMemory(_v.getLeft() + 1);
				for(int i = 0; i < _groups.size(); i++) {
					_groups.get(i)
						.leftMultByRowVector(_vect.getDenseBlockValues(), _ret.getDenseBlockValues(), _v.getRight()[i]);
				}

				ColGroupValue.cleanupThreadLocalMemory();
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
			// Pair<Integer, int[]> v = getMaxNumValues(_group);
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
			double[] tmpA = memPoolOLE.get();
			if(tmpA == null) {
				if(_groups != null) {
					tmpA = new double[Math.min(CompressionSettings.BITMAP_BLOCK_SZ * 2, _groups.get(0).getNumRows())];
				}
				else {
					tmpA = new double[Math.min(CompressionSettings.BITMAP_BLOCK_SZ * 2, _group.getNumRows())];
				}
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
								// LOG.error(_that.get(r));
								// _v.getRight()[j],
								_groups.get(j)
									.leftMultBySparseMatrix(_that, _ret, materializedV, _numRows, _numCols, r, tmpA);
								// Arrays.fill(tmpA, 0.0);
							}
						}
					}
				}
				else if(_group != null) {
					for(int r = _rl; r < _ru; r++) {
						if(!_that.isEmpty(r)) {
							// _v.getRight()[0],
							_group.leftMultBySparseMatrix(_that, _ret, _materialized[_i], _numRows, _numCols, r, tmpA);
							// Arrays.fill(tmpA, 0.0);
						}
					}
				}
			}
			catch(Exception e) {
				e.printStackTrace();
				throw new DMLRuntimeException(e);
			}
			ColGroupValue.cleanupThreadLocalMemory();
			return null;
		}
	}

	// private static class MatrixMultTransposeTaskNonOverlapping implements Callable<Object> {
	// private final List<ColGroup> _groups;
	// private final MatrixBlock _ret;
	// private final int _gl;
	// private final int _gu;
	// private final Pair<Integer, int[]> _v;

	// protected MatrixMultTransposeTaskNonOverlapping(List<ColGroup> groups, MatrixBlock ret, int gl, int gu,
	// Pair<Integer, int[]> v, boolean overlapping) {
	// _groups = groups;
	// _ret = ret;
	// _gl = gl;
	// _gu = gu;
	// _v = v;
	// }

	// @Override
	// public Object call() {
	// leftMultByTransposeSelfNonOverlapping(_groups, _ret, _v, _gl, _gu);
	// return null;
	// }
	// }

	private static class MatrixMultTransposeTaskOverlapping implements Callable<Object> {
		private final List<ColGroup> _groups;
		private final MatrixBlock _ret;
		private final int _gl;
		private final int _gu;
		private final Pair<Integer, int[]> _v;
		private final boolean _overlapping;

		protected MatrixMultTransposeTaskOverlapping(List<ColGroup> groups, MatrixBlock ret, int gl, int gu,
			Pair<Integer, int[]> v, boolean overlapping) {
			_groups = groups;
			_ret = ret;
			_gl = gl;
			_gu = gu;
			_v = v;
			_overlapping = overlapping;
		}

		@Override
		public Object call() {
			leftMultByTransposeSelfOverlapping(_groups, _ret, _v, _gl, _gu, _overlapping);
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
			leftMultByTranspose(_groups, _thatGroups, _ret, _v, _thatV, _rl, _ru, _overlapping, _extraThreads);
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
				_grps.get(i).leftMultByRowVector(_rowVector.getDenseBlockValues(), _result.getDenseBlockValues(), -1);
			}
			ColGroupValue.cleanupThreadLocalMemory();
			return null;
		}
	}
}
