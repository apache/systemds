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
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupValue;
import org.apache.sysds.runtime.compress.colgroup.pre.IPreAggregate;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class CLALibLeftMultBy {
	private static final Log LOG = LogFactory.getLog(CLALibLeftMultBy.class.getName());

	public static MatrixBlock leftMultByMatrixTransposed(CompressedMatrixBlock m1, MatrixBlock m2, MatrixBlock ret,
		int k) {
		MatrixBlock transposed = new MatrixBlock(m2.getNumColumns(), m2.getNumRows(), false);
		LibMatrixReorg.transpose(m2, transposed);
		ret = leftMultByMatrix(m1, transposed, ret, k);
		ret.recomputeNonZeros();
		return ret;
		// return LibMatrixReorg.transpose(ret, new MatrixBlock(ret.getNumColumns(), ret.getNumRows(), false));
	}

	public static MatrixBlock leftMultByMatrixTransposed(CompressedMatrixBlock m1, CompressedMatrixBlock m2,
		MatrixBlock ret, int k) {
		prepareReturnMatrix(m1, m2, ret, true);
		leftMultByCompressedTransposedMatrix(m1.getColGroups(), m2, ret, k, m1.getNumColumns(), m1.getMaxNumValues(),
			m1.isOverlapping());

		ret.recomputeNonZeros();
		return ret;
	}

	public static MatrixBlock leftMultByMatrix(CompressedMatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int k) {
		prepareReturnMatrix(m1, m2, ret, false);
		ret = leftMultByMatrix(m1.getColGroups(), m2, ret, false, m1.getNumColumns(), m1.isOverlapping(), k,
			m1.getMaxNumValues());
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


	public static void leftMultByTransposeSelf(CompressedMatrixBlock mb, MatrixBlock result, int k){

	}

	public static void leftMultByTransposeSelf(List<AColGroup> groups, MatrixBlock result, int k, int numColumns,
		Pair<Integer, int[]> v, boolean overlapping) {

		result.allocateDenseBlock();

		if(!overlapping && groups.get(0).getNumCols() == numColumns) {
			leftMultBySelfDiagonalColGroup(groups, result, numColumns);
			return;
		}

		if(k <= 1 || overlapping){
			if(overlapping)
				LOG.warn("Inefficient TSMM with overlapping matrix Could be implemented multi-threaded but is not yet.");
			leftMultByCompressedTransposedMatrix(groups, groups, result);
		}
		else {
			try {
				ExecutorService pool = CommonThreadPool.get(k);
				ArrayList<Callable<Object>> tasks = new ArrayList<>();
				for(int i = 0; i < groups.size(); i++) {
					tasks.add(
						new LeftMultByCompressedTransposedMatrixTask(groups, groups.get(i), result, i, groups.size()));
				}

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
		for(AColGroup g : groups)
			g.tsmm(outValues, numColumns);

	}

	private static MatrixBlock leftMultByCompressedTransposedMatrix(List<AColGroup> colGroups,
		CompressedMatrixBlock that, MatrixBlock ret, int k, int numColumns, Pair<Integer, int[]> v,
		boolean overlapping) {

		ret.allocateDenseBlock();
		List<AColGroup> thatCGs = that.getColGroups();
		Pair<Integer, int[]> thatV = that.getMaxNumValues();

		if(k <= 1)
			leftMultByCompressedTransposedMatrix(colGroups, thatCGs, ret);
		else
			try {
				ExecutorService pool = CommonThreadPool.get(k);
				ArrayList<Callable<Object>> tasks = new ArrayList<>();
				for(int i = 0; i < thatCGs.size(); i++) {

					if(thatCGs.get(i).getNumCols() > 1 || thatCGs.get(i).getCompType() == CompressionType.CONST)
						tasks.add(new LeftMultByCompressedTransposedMatrixTask(colGroups, thatCGs.get(i), ret, 0,
							colGroups.size()));
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

		protected LeftMultByCompressedTransposedMatrixTask(List<AColGroup> groups, AColGroup left, MatrixBlock ret,
			int start, int end) {
			_groups = groups;
			_left = left;
			_ret = ret;
			_start = start;
			_end = end;
		}

		@Override
		public Object call() {
			try {
				IPreAggregate.setupThreadLocalMemory(1024);
				leftMultByCompressedTransposedMatrix(_left, _groups, _ret.getDenseBlockValues(), _ret.getNumRows(),
					_ret.getNumColumns(), _start, _end);

			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
			return null;
		}
	}

	private static void leftMultByCompressedTransposedMatrix(List<AColGroup> thisCG, List<AColGroup> thatCG,
		MatrixBlock ret) {
		double[] c = ret.getDenseBlockValues();

		for(AColGroup lhs : thatCG) {
			leftMultByCompressedTransposedMatrix(lhs, thisCG, c, ret.getNumRows(), ret.getNumColumns(), 0,
				thisCG.size());
		}
	}

	private static void leftMultByCompressedTransposedMatrix(AColGroup lhs, List<AColGroup> thisCG, double[] c,
		int rows, int cols, int colGroupStart, int colGroupEnd) {

		for(; colGroupStart < colGroupEnd; colGroupStart++) {
			thisCG.get(colGroupStart).leftMultByAColGroup(lhs, c, rows, cols);
		}

	}

	private static MatrixBlock leftMultByMatrix(List<AColGroup> colGroups, MatrixBlock that, MatrixBlock ret, int k,
		int numColumns, Pair<Integer, int[]> v, boolean overlapping) {

		if(that.isEmpty()) {
			ret.setNonZeros(0);
			return ret;
		}

		ret.allocateDenseBlock();
		double[] retV = ret.getDenseBlockValues();

		// for(int b = 0; b < db.numBlocks(); b++) {
		// int blockSize = db.blockSize(b);
		// blockU = Math.min(blockL + blockSize, ret.getNumRows());

		if(k == 1) {
			LOG.trace("Single treaded left matrix multiplication");
			for(int j = 0; j < colGroups.size(); j++) {
				colGroups.get(j).leftMultByMatrix(that, retV, numColumns);
			}
		}
		else {
			try {
				ExecutorService pool = CommonThreadPool.get(k);
				// compute remaining compressed column groups in parallel
				ArrayList<Callable<Object>> tasks = new ArrayList<>();
				int rowBlockSize = 1;
				if(overlapping) {
					for(int blo = 0; blo < that.getNumRows(); blo += rowBlockSize) {
						tasks.add(new LeftMatrixMatrixMultTask(colGroups, that, retV, numColumns, blo,
							Math.min(blo + rowBlockSize, that.getNumRows()), v));
					}
				}
				else {
					for(int blo = 0; blo < that.getNumRows(); blo += rowBlockSize) {
						for(AColGroup g : colGroups) {
							tasks.add(new LeftMatrixColGroupMultTask(g, that, retV, numColumns, blo,
								Math.min(blo + rowBlockSize, that.getNumRows()), v));
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
		ret.recomputeNonZeros();
		return ret;
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
					grp.leftMultByMatrix(lhs, tmpret.getDenseBlockValues(), result.getNumColumns());
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

	private static class LeftMatrixMatrixMultTask implements Callable<Object> {
		private final List<AColGroup> _group;
		private final MatrixBlock _that;
		private final double[] _ret;
		private final int _numCols;
		private final int _rl;
		private final int _ru;
		private final Pair<Integer, int[]> _v;

		protected LeftMatrixMatrixMultTask(List<AColGroup> group, MatrixBlock that, double[] ret, int numCols, int rl,
			int ru, Pair<Integer, int[]> v) {
			_group = group;
			_that = that;
			_ret = ret;
			_numCols = numCols;
			_rl = rl;
			_ru = ru;
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
				for(int j = 0; j < _group.size(); j++)
					_group.get(j).leftMultByMatrix(_that, _ret, _numCols, _rl, _ru);
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
			return null;
		}
	}

	private static class LeftMatrixColGroupMultTask implements Callable<Object> {
		private final AColGroup _group;
		private final MatrixBlock _that;
		private final double[] _ret;
		private final int _numCols;
		private final int _rl;
		private final int _ru;
		private final Pair<Integer, int[]> _v;

		protected LeftMatrixColGroupMultTask(AColGroup group, MatrixBlock that, double[] ret, int numCols, int rl,
			int ru, Pair<Integer, int[]> v) {
			_group = group;
			_that = that;
			_ret = ret;
			_numCols = numCols;
			_rl = rl;
			_ru = ru;
			_v = v;
		}

		@Override
		public Object call() {

			try {
				ColGroupValue.setupThreadLocalMemory(_v.getLeft() + 1);
				_group.leftMultByMatrix(_that, _ret, _numCols, _rl, _ru);
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
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
			leftMultByCompressedTransposeRowSection(_groups, _thatGroups, _ret, _v, _thatV, _rl, _ru, _overlapping,
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
				_grps.get(i).leftMultByMatrix(_rowVector, _result.getDenseBlockValues(), _result.getNumColumns());
			}
			return null;
		}
	}
}
