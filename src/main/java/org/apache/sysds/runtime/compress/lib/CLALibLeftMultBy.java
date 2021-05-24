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
import org.apache.sysds.runtime.compress.colgroup.ColGroupValue;
import org.apache.sysds.runtime.compress.utils.LinearAlgebraUtils;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class CLALibLeftMultBy {
	private static final Log LOG = LogFactory.getLog(CLALibLeftMultBy.class.getName());

	public static MatrixBlock leftMultByMatrixTransposed(CompressedMatrixBlock m1, MatrixBlock m2, MatrixBlock ret,
		int k) {
		if(m2.isEmpty())
			return ret;
		MatrixBlock transposed = new MatrixBlock(m2.getNumColumns(), m2.getNumRows(), false);
		LibMatrixReorg.transpose(m2, transposed);
		ret = leftMultByMatrix(m1, transposed, ret, k);
		ret.recomputeNonZeros();
		return ret;
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
		if(m2.isEmpty())
			return ret;
		ret = leftMultByMatrix(m1.getColGroups(), m2, ret, k, m1.getNumColumns(), m1.getMaxNumValues(),
			m1.isOverlapping());
		ret.recomputeNonZeros();
		return ret;
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

		if(overlapping) {
			LOG.warn("Inefficient TSMM with overlapping matrix could be implemented multi-threaded but is not yet.");
			leftMultByCompressedTransposedMatrix(groups, groups, result);
		}
		else if(k <= 1) {
			for(int i = 0; i < groups.size(); i++)
				leftMultByCompressedTransposedMatrix(groups.get(i), groups, result, i, groups.size());
		}
		else {
			try {
				ExecutorService pool = CommonThreadPool.get(k);
				ArrayList<Callable<Object>> tasks = new ArrayList<>();
				for(int i = 0; i < groups.size(); i++) {
					final AColGroup g = groups.get(i);
					tasks.add(new LeftMultByCompressedTransposedMatrixTask(groups, g, result, i, groups.size()));
				}
				for(Future<Object> tret : pool.invokeAll(tasks))
					tret.get();
				pool.shutdown();
			}
			catch(InterruptedException | ExecutionException e) {
				throw new DMLRuntimeException(e);
			}
		}

		// Move values in the lower part of the matrix to the upper part
		copyToUpperTriangle(result.getDenseBlockValues(), numColumns);
		// calculate the number of non zeros, and allocate all value locations by copying upper triangle back to bottom.
		long nnz = LinearAlgebraUtils.copyUpperToLowerTriangle(result);
		result.setNonZeros(nnz);
		// Evaluate if the output should be sparsely allocated.
		result.examSparsity();
	}

	private static void copyToUpperTriangle(final double[] c, final int cols) {
		for(int i = 0, offC = 0; i < cols; i++, offC += cols)
			for(int j = i, offR = i * cols; j < cols; j++, offR += cols) {
				final double prev = c[offC + j];
				if(prev == 0)
					c[offC + j] = c[i + offR];
			}

	}

	private static MatrixBlock leftMultByCompressedTransposedMatrix(List<AColGroup> colGroups,
		CompressedMatrixBlock that, MatrixBlock ret, int k, int numColumns, Pair<Integer, int[]> v,
		boolean overlapping) {

		ret.allocateDenseBlock();
		List<AColGroup> thatCGs = that.getColGroups();

		if(k <= 1 || overlapping || that.isOverlapping()) {
			if(overlapping || that.isOverlapping())
				LOG.warn("Inefficient Compressed multiplication with overlapping matrix"
					+ " could be implemented multi-threaded but is not yet.");
			leftMultByCompressedTransposedMatrix(colGroups, thatCGs, ret);
		}
		else
			try {
				ExecutorService pool = CommonThreadPool.get(k);
				ArrayList<Callable<Object>> tasks = new ArrayList<>();
				for(int i = 0; i < thatCGs.size(); i++) {
					tasks.add(new LeftMultByCompressedTransposedMatrixTask(colGroups, thatCGs.get(i), ret));
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

		protected LeftMultByCompressedTransposedMatrixTask(List<AColGroup> groups, AColGroup left, MatrixBlock ret) {
			_groups = groups;
			_left = left;
			_ret = ret;
			_start = 0;
			_end = groups.size();
		}

		@Override
		public Object call() {
			try {
				leftMultByCompressedTransposedMatrix(_left, _groups, _ret, _start, _end);
			}
			catch(Exception e) {
				e.printStackTrace();
				throw new DMLRuntimeException(e);
			}
			return null;
		}
	}

	private static void leftMultByCompressedTransposedMatrix(List<AColGroup> thisCG, List<AColGroup> thatCG,
		MatrixBlock ret) {
		for(AColGroup lhs : thatCG)
			leftMultByCompressedTransposedMatrix(lhs, thisCG, ret, 0, thisCG.size());
	}

	private static void leftMultByCompressedTransposedMatrix(AColGroup lhs, List<AColGroup> thisCG, MatrixBlock ret,
		int colGroupStart, int colGroupEnd) {

		for(; colGroupStart < colGroupEnd; colGroupStart++) {
			AColGroup rhs = thisCG.get(colGroupStart);
			if(rhs != lhs)
				rhs.leftMultByAColGroup(lhs, ret);
			else
				rhs.tsmm(ret.getDenseBlockValues(), ret.getNumColumns());
		}

	}

	private static MatrixBlock leftMultByMatrix(List<AColGroup> colGroups, MatrixBlock that, MatrixBlock ret, int k,
		int numColumns, Pair<Integer, int[]> v, boolean overlapping) {

		if(that.isEmpty()) {
			ret.setNonZeros(0);
			return ret;
		}

		ret.allocateDenseBlock();

		if(k == 1)
			for(int j = 0; j < colGroups.size(); j++)
				colGroups.get(j).leftMultByMatrix(that, ret);
		else {
			try {
				ExecutorService pool = CommonThreadPool.get(k);
				// compute remaining compressed column groups in parallel
				ArrayList<Callable<Object>> tasks = new ArrayList<>();
				int rowBlockSize = 1;
				if(overlapping) {
					for(int blo = 0; blo < that.getNumRows(); blo += rowBlockSize) {
						tasks.add(new LeftMatrixMatrixMultTask(colGroups, that, ret, blo,
							Math.min(blo + rowBlockSize, that.getNumRows()), v));
					}
				}
				else {
					for(AColGroup g : colGroups) {
						// if(g instanceof ColGroupDDC) {
						// tasks.add(new LeftMatrixColGroupMultTask(g, that, ret, 0, that.getNumRows(), v));
						// }
						// else {

						for(int blo = 0; blo < that.getNumRows(); blo += rowBlockSize) {
							tasks.add(new LeftMatrixColGroupMultTask(g, that, ret, blo,
								Math.min(blo + rowBlockSize, that.getNumRows()), v));
						}
						// }
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

	private static class LeftMatrixMatrixMultTask implements Callable<Object> {
		private final List<AColGroup> _group;
		private final MatrixBlock _that;
		private final MatrixBlock _ret;
		private final int _rl;
		private final int _ru;
		private final Pair<Integer, int[]> _v;

		protected LeftMatrixMatrixMultTask(List<AColGroup> group, MatrixBlock that, MatrixBlock ret, int rl, int ru,
			Pair<Integer, int[]> v) {
			_group = group;
			_that = that;
			_ret = ret;
			_rl = rl;
			_ru = ru;
			_v = v;
		}

		@Override
		public Object call() {
			try {
				ColGroupValue.setupThreadLocalMemory(_v.getLeft());
				for(int j = 0; j < _group.size(); j++)
					_group.get(j).leftMultByMatrix(_that, _ret, _rl, _ru);
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
		private final MatrixBlock _ret;
		private final int _rl;
		private final int _ru;
		private final Pair<Integer, int[]> _v;

		protected LeftMatrixColGroupMultTask(AColGroup group, MatrixBlock that, MatrixBlock ret, int rl, int ru,
			Pair<Integer, int[]> v) {
			_group = group;
			_that = that;
			_ret = ret;
			_rl = rl;
			_ru = ru;
			_v = v;
		}

		@Override
		public Object call() {

			try {
				ColGroupValue.setupThreadLocalMemory(_v.getLeft());
				_group.leftMultByMatrix(_that, _ret, _rl, _ru);
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
			return null;
		}
	}
}
