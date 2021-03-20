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
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupOLE;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.colgroup.ColGroupValue;
import org.apache.sysds.runtime.compress.colgroup.pre.IPreAggregate;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class CLALibLeftMultBy {
	// private static final Log LOG = LogFactory.getLog(CLALibLeftMultBy.class.getName());

	private static ThreadLocal<double[]> memPoolLeftMult = new ThreadLocal<double[]>() {
		@Override
		protected double[] initialValue() {
			return null;
		}
	};

	public static MatrixBlock leftMultByMatrixTransposed(CompressedMatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int k){
		MatrixBlock transposed = new MatrixBlock(m2.getNumColumns(), m2.getNumRows(), false);
		LibMatrixReorg.transpose(m2, transposed);
		ret = leftMultByMatrix(m1, transposed, ret, k );
		ret.recomputeNonZeros();
		return ret;
		// return LibMatrixReorg.transpose(ret, new MatrixBlock(ret.getNumColumns(), ret.getNumRows(), false));
	}

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
		ret = leftMultByMatrix(m1
			.getColGroups(), m2, ret, false, m1.getNumColumns(), m1.isOverlapping(), k, m1.getMaxNumValues());
		ret.recomputeNonZeros();
		return ret;
	}

	private static MatrixBlock leftMultByMatrix(List<AColGroup> groups, MatrixBlock that, MatrixBlock ret,
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

	public static void leftMultByTransposeSelf(List<AColGroup> groups, MatrixBlock result, int k, int numColumns,
		Pair<Integer, int[]> v, boolean overlapping) {

		result.allocateDenseBlock();

		if(!overlapping && groups.get(0).getNumCols() == numColumns) {
			leftMultBySelfDiagonalColGroup(groups, result, numColumns);
			return;
		}

		if(k <= 1) {
			leftMultByCompressedTransposedMatrix(groups, groups, result, true);
		}
		else {
			try {
				ExecutorService pool = CommonThreadPool.get(k);
				ArrayList<Callable<Object>> tasks = new ArrayList<>();
				for(int i = 0; i < groups.size(); i++) {
					if(groups.get(i).getNumCols() > 1 || groups.get(i).getCompType() == CompressionType.CONST)
						tasks.add(new LeftMultByCompressedTransposedMatrixTask(groups, groups.get(i), result, 0,
							groups.size(), true));
					else {
						AColGroup g = groups.get(i);
						int[] indexes = g.getColIndices();
						int row = indexes[0];
						// int rowEnd = indexes.length > 0 ? indexes[indexes.length -1] + 1 : row + 1;
						tasks.add(new MatrixMultTransposeReflectedTask(groups, result, row, row + 1, 0, numColumns, v,
							overlapping));
					}
				}
				// for(int j = 0; j < groups.size()-1; j++)
				// tasks.add(new LeftMultByCompressedTransposedMatrixTask(groups, groups.get(i), result, v, j, j+1));

				for(Future<Object> tret : pool.invokeAll(tasks))
					tret.get();
				pool.shutdown();
			}
			catch(InterruptedException | ExecutionException e) {
				throw new DMLRuntimeException(e);
			}
		}

		copyToUpperTriangle(result.getDenseBlockValues(), numColumns);

		result.recomputeNonZeros();
	}

	private static void copyToUpperTriangle(final double[] c, final int cols) {
		for(int i = 0, offC = 0; i < cols; i++, offC += cols)
			for(int j = i, offR = i * cols; j < cols; j++, offR += cols) {
				final double d = c[i + offR];
				if(d != 0)
					c[offC + j] = d;
			}

	}

	private static void leftMultBySelfDiagonalColGroup(List<AColGroup> groups, MatrixBlock result, int numColumns) {
		double[] outValues = result.getDenseBlockValues();
		for(AColGroup g : groups) {
			if(g instanceof ColGroupValue) {
				ColGroupValue gv = (ColGroupValue) g;
				gv.leftMultBySelfDiagonalColGroup(outValues, numColumns);
			}
			else
				throw new NotImplementedException("Not Implemented diagonal on non ColGroupValue type.");

		}
	}

	private static MatrixBlock leftMultByCompressedTransposedMatrix(List<AColGroup> colGroups,
		CompressedMatrixBlock that, MatrixBlock ret, int k, int numColumns, Pair<Integer, int[]> v,
		boolean overlapping) {

		ret.allocateDenseBlock();
		List<AColGroup> thatCGs = that.getColGroups();
		Pair<Integer, int[]> thatV = that.getMaxNumValues();

		if(k <= 1)
			leftMultByCompressedTransposedMatrix(colGroups, thatCGs, ret, false);
		else
			try {
				ExecutorService pool = CommonThreadPool.get(k);
				ArrayList<Callable<Object>> tasks = new ArrayList<>();
				for(int i = 0; i < thatCGs.size(); i++) {

					if(thatCGs.get(i).getNumCols() > 1 || thatCGs.get(i).getCompType() == CompressionType.CONST)
						tasks.add(new LeftMultByCompressedTransposedMatrixTask(colGroups, thatCGs.get(i), ret, 0,
							colGroups.size(), false));
					else {
						int row = thatCGs.get(i).getColIndices()[0];
						tasks.add(new LeftMultByCompressedTransposedMatrixTask2(colGroups, thatCGs, ret, v, thatV, row,
							row + 1, overlapping, 1));
					}
				}

				for(Future<Object> tret : pool.invokeAll(tasks))
					tret.get();
				pool.shutdown();
			}
			catch(InterruptedException | ExecutionException e) {
				throw new DMLRuntimeException(e);
			}

		ret.recomputeNonZeros();
		return ret;
	}

	private static class LeftMultByCompressedTransposedMatrixTask implements Callable<Object> {
		private final List<AColGroup> _groups;
		private final AColGroup _left;
		private final MatrixBlock _ret;
		private final int _start;
		private final int _end;
		private final boolean _self;

		protected LeftMultByCompressedTransposedMatrixTask(List<AColGroup> groups, AColGroup left, MatrixBlock ret,
			int start, int end, boolean self) {
			_groups = groups;
			_left = left;
			_ret = ret;
			_start = start;
			_end = end;
			_self = self;
		}

		@Override
		public Object call() {
			try {
				IPreAggregate.setupThreadLocalMemory(1024);
				if(_left instanceof ColGroupValue) {
					leftMultByCompressedTransposedMatrix((ColGroupValue) _left,
						_groups,
						_ret.getDenseBlockValues(),
						_ret.getNumRows(),
						_ret.getNumColumns(),
						_start,
						_end,
						_self);
				}
				else
					throw new NotImplementedException("Not implemented uncompressed instance of left mult compressed");
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
			return null;
		}
	}

	private static void leftMultByCompressedTransposedMatrix(List<AColGroup> thisCG, List<AColGroup> thatCG,
		MatrixBlock ret, boolean self) {
		double[] c = ret.getDenseBlockValues();

		for(AColGroup lhs : thatCG) {
			if(lhs instanceof ColGroupValue) {
				leftMultByCompressedTransposedMatrix((ColGroupValue) lhs,
					thisCG,
					c,
					ret.getNumRows(),
					ret.getNumColumns(),
					0,
					thisCG.size(),
					self);
			}
			else
				throw new NotImplementedException("Not implemented uncompressed instance of left mult compressed");
		}
	}

	private static void leftMultByCompressedTransposedMatrix(ColGroupValue lhs, List<AColGroup> thisCG, double[] c,
		int rows, int cols, int colGroupStart, int colGroupEnd, boolean self) {
		if(self) {
			final int[] lhsCols = lhs.getColIndices();
			final int lMinCol = lhsCols[0];
			for(; colGroupStart < colGroupEnd; colGroupStart++) {
				final ColGroupValue rhsV = (ColGroupValue) thisCG.get(colGroupStart);
				final int[] rhsCols = rhsV.getColIndices();
				final int rMinCol = rhsCols[0];
				if(self && lMinCol <= rMinCol)
					rhsV.leftMultByAggregatedColGroup(lhs, c, rows, cols);
			}
		}
		else
			for(; colGroupStart < colGroupEnd; colGroupStart++) {
				final ColGroupValue rhsV = (ColGroupValue) thisCG.get(colGroupStart);
				rhsV.leftMultByAggregatedColGroup(lhs, c, rows, cols);
			}

	}

	private static MatrixBlock leftMultByMatrix(List<AColGroup> colGroups, MatrixBlock that, MatrixBlock ret, int k,
		int numColumns, Pair<Integer, int[]> v, boolean overlapping) {
		if(that.isEmpty()) {
			ret.setNonZeros(0);
			return ret;
		}

		ret.allocateDenseBlock();
		if(that.isInSparseFormat())
			ret = leftMultBySparseMatrix(colGroups, that, ret, k, numColumns, v, overlapping);
		else
			ret = leftMultByDenseMatrix(colGroups, that, ret, k, numColumns, v, overlapping);
		ret.setNonZeros(ret.getNumColumns() * ret.getNumRows());
		return ret;
	}

	private static MatrixBlock leftMultByDenseMatrix(List<AColGroup> colGroups, MatrixBlock that, MatrixBlock ret,
		int k, int numColumns, Pair<Integer, int[]> v, boolean overlapping) {
		DenseBlock db = that.getDenseBlock();
		if(db == null)
			throw new DMLRuntimeException("Invalid LeftMult By Dense matrix, input matrix was sparse");

		double[] retV = ret.getDenseBlockValues();
		double[] thatV;
		int blockU;
		int blockL = 0;
		for(AColGroup grp : colGroups)
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
					ArrayList<Callable<Object>> tasks = new ArrayList<>();
					int rowBlockSize = 1;
					if(overlapping) {

						for(int blo = blockL; blo < blockU; blo += rowBlockSize) {
							tasks.add(new LeftMatrixMatrixMultTask(colGroups, thatV, retV, that.getNumRows(),
								numColumns, blo, Math.min(blo + rowBlockSize, blockU), blo - blockL, v));
						}
					}
					else {
						for(int blo = blockL; blo < blockU; blo += rowBlockSize) {
							for(AColGroup g : colGroups) {
								tasks.add(new LeftMatrixColGroupMultTask(g, thatV, retV, that.getNumRows(), numColumns,
									blo, Math.min(blo + rowBlockSize, blockU), blo - blockL, v));
							}
						}
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

	private static void leftMultSelfVectorTranspose(List<AColGroup> colGroups, double[] rowVector, double[] result,
		int cl, int cu, int j, int nCol) {
		for(AColGroup grp : colGroups) {
			int[] columns = grp.getColIndices();
			// if(columns[columns.length - 1] >= cl && columns[0] < cu)
			if(Arrays.binarySearch(columns, j) < 0) // if the colGroup is not it self.
				grp.leftMultByRowVector(rowVector, result, j * nCol);
			else
				grp.leftMultBySelfDiagonalColGroup(result, nCol);
		}
	}

	private static MatrixBlock leftMultBySparseMatrix(List<AColGroup> colGroups, MatrixBlock that, MatrixBlock ret,
		int k, int numColumns, Pair<Integer, int[]> v, boolean overlapping) {

		SparseBlock sb = that.getSparseBlock();
		if(sb == null)
			throw new DMLRuntimeException("Invalid Left Mult by Sparse matrix, input matrix was dense");

		for(AColGroup grp : colGroups) {
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

	private static void leftMultByTransposeSelf(List<AColGroup> groups, MatrixBlock result, Pair<Integer, int[]> v,
		int rl, int ru, int cl, int cu, boolean overlapping) {
		if(overlapping)
			leftMultByTransposeSelfOverlapping(groups, result, v, rl, ru, cl, cu);
		else
			leftMultByTransposeSelfNormal(groups, result, v, rl, ru, cl, cu);
	}

	private static void leftMultByTransposeSelfNormal(List<AColGroup> groups, MatrixBlock result,
		Pair<Integer, int[]> v, int rl, int ru, int cl, int cu) {
		// preallocated dense tmp matrix blocks

		final int numRows = groups.get(0).getNumRows();
		double[] lhs = memPoolLeftMult.get() != null ? memPoolLeftMult.get() : new double[numRows];
		double[] resArray = result.getDenseBlockValues();
		final int nCol = result.getNumColumns();
		for(int j = rl; j < ru; j++) {
			if(!rowIsDone(result.getDenseBlockValues(), result.getNumColumns(), j)) {
				AColGroup.decompressColumnToArray(lhs, j, groups);
				leftMultSelfVectorTranspose(groups, lhs, resArray, Math.max(cl, rl), cu, j, nCol);
				// copyToUpperTriangle(resArray, tmpret, j, result.getNumColumns(), Math.max(cl, rl), cu);
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

	// private static void copyToUpperTriangle(double[] ret, double[] tmp, int row, int totalRows, int cl, int cu) {
	// int offOut = row * totalRows;
	// for(int i = cl; i < cu; i++) {
	// ret[offOut + i] += tmp[i];
	// }
	// }

	private static void leftMultByTransposeSelfOverlapping(List<AColGroup> groups, MatrixBlock result,
		Pair<Integer, int[]> v, int rl, int ru, int cl, int cu) {
		throw new NotImplementedException("Since the rework this overlapping is not supported.");
		// final int numRows = groups.get(0).getNumRows();
		// double[] lhs = new double[numRows];
		// double[] resArray = result.getDenseBlockValues();
		// final int nCol = result.getNumColumns();
		// for(int j = rl; j < ru; j++) {
		// ColGroup.decompressColumnToArray(lhs, j, groups);
		// leftMultSelfVectorTranspose(groups, lhs, resArray, Math.max(cl, rl), cu, -1, nCol);
		// // copyToUpperTriangle(resArray, tmpret, j, result.getNumColumns(), Math.max(cl, rl), cu);
		// Arrays.fill(lhs, 0);
		// }

	}

	private static void leftMultByCompressedTransposeRowSection(List<AColGroup> thisGroups, List<AColGroup> thatGroups,
		MatrixBlock result, Pair<Integer, int[]> v, Pair<Integer, int[]> thatV, int rl, int ru, boolean overlapping,
		int k) {
		if(k > 1 && !overlapping)
			leftMultByCompressedTransposeRowSectionParallel(thisGroups, thatGroups, result, v, thatV, rl, ru, k);
		else
			leftMultByCompressedTransposeRowSectionSingleThread(thisGroups, thatGroups, result, v, thatV, rl, ru);

	}

	private static void leftMultByCompressedTransposeRowSectionParallel(List<AColGroup> thisGroups,
		List<AColGroup> thatGroups, MatrixBlock result, Pair<Integer, int[]> v, Pair<Integer, int[]> thatV, int rl,
		int ru, int k) {

		// preallocated dense tmp matrix blocks
		MatrixBlock lhs = new MatrixBlock(1, thisGroups.get(0).getNumRows(), false);
		MatrixBlock tmpret = new MatrixBlock(1, result.getNumColumns(), false);
		lhs.allocateDenseBlock();
		tmpret.allocateDenseBlock();

		ExecutorService pool = CommonThreadPool.get(k);
		ArrayList<leftMultByVectorTransposeTask> tasks = new ArrayList<>();
		for(int j = rl; j < ru; j++) {
			AColGroup.decompressColumnToBlock(lhs, j, thatGroups);
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

	private static void leftMultByCompressedTransposeRowSectionSingleThread(List<AColGroup> thisGroups,
		List<AColGroup> thatGroups, MatrixBlock result, Pair<Integer, int[]> v, Pair<Integer, int[]> thatV, int rl,
		int ru) {
		final int numRows = thisGroups.get(0).getNumRows();

		// preallocated dense tmp matrix blocks
		MatrixBlock lhs = new MatrixBlock(1, numRows, false);
		MatrixBlock tmpret = new MatrixBlock(1, result.getNumColumns(), false);

		lhs.allocateDenseBlock();
		tmpret.allocateDenseBlock();

		for(int j = rl; j < ru; j++) {
			AColGroup.decompressColumnToBlock(lhs, j, thatGroups);
			if(!lhs.isEmptyBlock(false)) {
				for(AColGroup grp : thisGroups) {
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

	// private static class LeftMatrixVectorMultTask implements Callable<Object> {
	// 	private final List<AColGroup> _groups;
	// 	private final MatrixBlock _vect;
	// 	private final MatrixBlock _ret;
	// 	private final Pair<Integer, int[]> _v;

	// 	protected LeftMatrixVectorMultTask(List<AColGroup> groups, MatrixBlock vect, MatrixBlock ret,
	// 		Pair<Integer, int[]> v) {
	// 		_groups = groups;
	// 		_vect = vect;
	// 		_ret = ret;
	// 		_v = v;
	// 	}

	// 	@Override
	// 	public Object call() {
	// 		try {
	// 			ColGroupValue.setupThreadLocalMemory(_v.getLeft() + 1);
	// 			for(int i = 0; i < _groups.size(); i++) {
	// 				_groups.get(i).leftMultByRowVector(_vect.getDenseBlockValues(), _ret.getDenseBlockValues());
	// 			}
	// 		}
	// 		catch(Exception e) {
	// 			throw new DMLRuntimeException(e);
	// 		}
	// 		return null;
	// 	}
	// }

	private static class LeftMatrixMatrixMultTask implements Callable<Object> {
		private final List<AColGroup> _group;
		private final double[] _that;
		private final double[] _ret;
		private final int _numRows;
		private final int _numCols;
		private final int _rl;
		private final int _ru;
		private final int _vOff;
		private final Pair<Integer, int[]> _v;

		protected LeftMatrixMatrixMultTask(List<AColGroup> group, double[] that, double[] ret, int numRows, int numCols,
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

			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
			return null;
		}
	}

	private static class LeftMatrixColGroupMultTask implements Callable<Object> {
		private final AColGroup _group;
		private final double[] _that;
		private final double[] _ret;
		private final int _numRows;
		private final int _numCols;
		private final int _rl;
		private final int _ru;
		private final int _vOff;
		private final Pair<Integer, int[]> _v;

		protected LeftMatrixColGroupMultTask(AColGroup group, double[] that, double[] ret, int numRows, int numCols,
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

			try {
				ColGroupValue.setupThreadLocalMemory(_v.getLeft() + 1);
				_group.leftMultByMatrix(_that, _ret, _group.getValues(), _numRows, _numCols, _rl, _ru, _vOff);
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
			return null;
		}
	}

	private static class LeftMatrixSparseMatrixMultTask implements Callable<Object> {
		private final List<AColGroup> _groups;
		private final AColGroup _group;
		private final int _i; // Used to identify the index for the materialized values.
		private final SparseBlock _that;
		private final double[] _ret;
		private final int _numRows;
		private final int _numCols;
		private final Pair<Integer, int[]> _v;
		private final double[][] _materialized;
		private final int _rl;
		private final int _ru;

		protected LeftMatrixSparseMatrixMultTask(List<AColGroup> group, double[][] materialized, SparseBlock that,
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

		protected LeftMatrixSparseMatrixMultTask(AColGroup group, double[][] materialized, int i, SparseBlock that,
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
		private final List<AColGroup> _groups;
		private final MatrixBlock _ret;
		private int _cl;
		private int _cu;
		private int _rl;
		private int _ru;
		private final Pair<Integer, int[]> _v;
		private final boolean _overlapping;

		protected MatrixMultTransposeReflectedTask(List<AColGroup> groups, MatrixBlock ret, int rl, int ru, int cl,
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
			// ColGroupValue.setupThreadLocalMemory(_v.getLeft() + 1);
			leftMultByTransposeSelf(_groups, _ret, _v, _rl, _ru, _cl, _cu, _overlapping);

			// int nCol = _ret.getNumColumns();
			// double[] tmpA = memPoolLeftMult.get();
			// if(tmpA == null) {
			// tmpA = new double[_groups.get(0).getNumRows()];
			// memPoolLeftMult.set(tmpA);
			// }
			// else
			// Arrays.fill(tmpA, 0);
			// if(nCol % 2 == 1) {
			// _rl = _rl == 0 ? 0 : _rl - 1;
			// _ru = _ru - 1;
			// }
			// if(_rl != _ru)
			// leftMultByTransposeSelf(_groups, _ret, _v, nCol - _ru, nCol - _rl, _cl, _cu, _overlapping);
			return null;
		}
	}

	private static class LeftMultByCompressedTransposedMatrixTask2 implements Callable<Object> {
		private final List<AColGroup> _groups;
		private final List<AColGroup> _thatGroups;
		private final MatrixBlock _ret;
		private final int _rl;
		private final int _ru;
		private final Pair<Integer, int[]> _v;
		private final Pair<Integer, int[]> _thatV;
		private final boolean _overlapping;
		private final int _extraThreads;

		protected LeftMultByCompressedTransposedMatrixTask2(List<AColGroup> thisGroups, List<AColGroup> thatGroups,
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
		private final List<AColGroup> _grps;
		private final MatrixBlock _rowVector;
		private final MatrixBlock _result;
		private final int _gl;
		private final int _gu;
		private final Pair<Integer, int[]> _v;

		protected leftMultByVectorTransposeTask(List<AColGroup> grps, MatrixBlock rowVector, MatrixBlock result, int gl,
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
