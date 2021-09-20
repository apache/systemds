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
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupSDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupSDCSingle;
import org.apache.sysds.runtime.compress.colgroup.ColGroupValue;
import org.apache.sysds.runtime.compress.utils.LinearAlgebraUtils;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class CLALibLeftMultBy {
	private static final Log LOG = LogFactory.getLog(CLALibLeftMultBy.class.getName());

	public static MatrixBlock leftMultByMatrixTransposed(CompressedMatrixBlock m1, MatrixBlock m2, MatrixBlock ret,
		int k) {
		if(m2.isEmpty())
			return ret;
		MatrixBlock transposed = new MatrixBlock(m2.getNumColumns(), m2.getNumRows(), false);
		LibMatrixReorg.transpose(m2, transposed, k);
		ret = leftMultByMatrix(m1, transposed, ret, k);
		ret.recomputeNonZeros();
		return ret;
	}

	public static MatrixBlock leftMultByMatrixTransposed(CompressedMatrixBlock right, CompressedMatrixBlock left,
		MatrixBlock ret, int k) {
		LOG.warn("Compressed Compressed matrix multiplication");
		ret = prepareReturnMatrix(right, left, ret, true);
		leftMultByCompressedTransposedMatrix(right, left, ret, k);

		ret.recomputeNonZeros();
		return ret;
	}

	public static MatrixBlock leftMultByMatrix(CompressedMatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int k) {
		ret = prepareReturnMatrix(m1, m2, ret, false);

		if(m2.isEmpty())
			return ret;

		ret = leftMultByMatrix(m1.getColGroups(), m2, ret, k, m1.isOverlapping());
		ret.recomputeNonZeros();
		return ret;
	}

	/**
	 * Prepare the output matrix.
	 * 
	 * @param m1          The right hand side matrix
	 * @param m2          The left hand side matrix
	 * @param ret         The output matrix to reallocate
	 * @param doTranspose Boolean specifying if the m2 (left side) matrix should be considered transposed
	 * @return the result matrix allocated.
	 */
	private static MatrixBlock prepareReturnMatrix(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret,
		boolean doTranspose) {
		final int numRowsOutput = doTranspose ? m2.getNumColumns() : m2.getNumRows();
		final int numColumnsOutput = m1.getNumColumns();
		if(ret == null)
			ret = new MatrixBlock(numRowsOutput, numColumnsOutput, false, numRowsOutput * numColumnsOutput);
		else if(!(ret.getNumColumns() == numColumnsOutput && ret.getNumRows() == numRowsOutput && ret.isAllocated()))
			ret.reset(numRowsOutput, numColumnsOutput, false, numRowsOutput * numColumnsOutput);

		ret.allocateDenseBlock();
		return ret;
	}

	public static void leftMultByTransposeSelf(CompressedMatrixBlock cmb, MatrixBlock result, int k) {
		final boolean overlapping = cmb.isOverlapping();
		final List<AColGroup> groups = cmb.getColGroups();

		if(overlapping) {
			LOG.warn("Inefficient TSMM with overlapping matrix could be implemented multi-threaded but is not yet.");
			multAllColGroups(groups, groups, result);
		}
		else {
			final boolean containsSDC = CLALibUtils.containsSDC(groups);
			final double[] constV = containsSDC ? new double[cmb.getNumColumns()] : null;
			final List<AColGroup> filteredGroups = CLALibUtils.filterSDCGroups(groups, constV);
			final double[] colSums = containsSDC ? new double[cmb.getNumColumns()] : null;
			final int numColumns = cmb.getNumColumns();

			if(containsSDC)
				for(int i = 0; i < groups.size(); i++) {
					AColGroup gi = groups.get(i);
					if(!(gi instanceof ColGroupSDC || gi instanceof ColGroupSDCSingle))
						gi.computeColSums(colSums);
				}

			if(k <= 1)
				tsmmColGroups(groups, filteredGroups, result);
			else
				tsmmColGroupsParallel(groups, filteredGroups, result, k);

			double[] retV = result.getDenseBlockValues();

			// Move values in the lower part of the matrix to the upper part
			copyToUpperTriangle(retV, numColumns);

			// add the correction layer for the subtracted common values.
			if(colSums != null) {
				outerProduct(colSums, constV, retV);
				addToUpperTriangle(retV, numColumns);
			}
		}

		long nnz = LinearAlgebraUtils.copyUpperToLowerTriangle(result);
		result.setNonZeros(nnz);
		result.examSparsity();
	}

	private static void copyToUpperTriangle(final double[] c, final int cols) {
		for(int i = 0, offC = 0; i < cols; i++, offC += cols)
			for(int j = (i + 1), offR = (i + 1) * cols; j < cols; j++, offR += cols) {
				final double prev = c[offC + j];
				if(prev == 0)
					c[offC + j] = c[i + offR];
				c[i + offR] = 0;
			}
	}

	private static void addToUpperTriangle(final double[] c, final int cols) {
		for(int i = 0, offC = 0; i < cols; i++, offC += cols)
			for(int j = (i + 1), offR = (i + 1) * cols; j < cols; j++, offR += cols)
				c[offC + j] += c[i + offR];

	}

	private static MatrixBlock leftMultByCompressedTransposedMatrix(CompressedMatrixBlock right,
		CompressedMatrixBlock left, MatrixBlock ret, int k) {

		final List<AColGroup> thisCGs = right.getColGroups();
		final List<AColGroup> thatCGs = left.getColGroups();

		final boolean thisOverlapping = right.isOverlapping();
		final boolean thatOverlapping = left.isOverlapping();
		final boolean anyOverlap = thisOverlapping || thatOverlapping;

		if(k <= 1 || anyOverlap) {
			if(anyOverlap)
				LOG.warn("Inefficient Compressed multiplication with overlapping matrix"
					+ " could be implemented multi-threaded but is not yet.");
			multAllColGroups(thisCGs, thatCGs, ret);
		}
		else {
			try {
				ExecutorService pool = CommonThreadPool.get(k);
				ArrayList<Callable<Object>> tasks = new ArrayList<>();
				for(int i = 0; i < thatCGs.size(); i++)
					for(int j = 0; j < thisCGs.size(); j++)
						tasks.add(new LeftMultByCompressedTransposedMatrixTask(thisCGs.get(j), thatCGs.get(i), ret));

				for(Future<Object> tret : pool.invokeAll(tasks))
					tret.get();
				pool.shutdown();
			}
			catch(InterruptedException | ExecutionException e) {
				throw new DMLRuntimeException(e);
			}
		}

		ret.recomputeNonZeros();
		return ret;
	}

	private static class LeftMultByCompressedTransposedMatrixTask implements Callable<Object> {
		private final AColGroup _right;
		private final AColGroup _left;
		private final MatrixBlock _ret;

		protected LeftMultByCompressedTransposedMatrixTask(AColGroup right, AColGroup left, MatrixBlock ret) {
			_right = right;
			_left = left;
			_ret = ret;
		}

		@Override
		public Object call() {
			try {
				if(_right != _left)
					_right.leftMultByAColGroup(_left, _ret);
				else
					_right.tsmm(_ret);
			}
			catch(Exception e) {
				e.printStackTrace();
				throw new DMLRuntimeException(e);
			}
			return null;
		}
	}

	private static void multAllColGroups(List<AColGroup> right, List<AColGroup> left, MatrixBlock ret) {
		for(int i = 0; i < left.size(); i++) {
			AColGroup leftCG = left.get(i);
			for(int j = 0; j < right.size(); j++) {
				AColGroup rightCG = right.get(j);
				if(rightCG != leftCG)
					rightCG.leftMultByAColGroup(leftCG, ret);
				else
					rightCG.tsmm(ret);
			}
		}
	}

	private static void tsmmColGroups(List<AColGroup> groups, List<AColGroup> filteredGroups, MatrixBlock ret) {
		for(int i = 0; i < groups.size(); i++) {
			groups.get(i).tsmm(ret);
			tsmmColGroupsIndexI(groups, filteredGroups, ret, i);
		}
	}

	private static void tsmmColGroupsParallel(List<AColGroup> groups, List<AColGroup> filteredGroups, MatrixBlock ret,
		int k) {
		try {
			ExecutorService pool = CommonThreadPool.get(k);
			ArrayList<Callable<Object>> tasks = new ArrayList<>();

			final int numColGroups = groups.size();
			for(int i = 0; i < numColGroups; i++) {
				tasks.add(new tsmmSelfColGroupTask(groups.get(i), ret));
				for(int j = i + 1; j < numColGroups; j++)
					tasks.add(new tsmmColGroupTask(groups, filteredGroups, ret, i, j, j + 1));
			}

			for(Future<Object> tret : pool.invokeAll(tasks))
				tret.get();
			pool.shutdown();
		}
		catch(InterruptedException | ExecutionException e) {
			throw new DMLRuntimeException(e);
		}
	}

	private static void tsmmColGroupsIndexI(List<AColGroup> groups, List<AColGroup> filteredGroups, MatrixBlock ret,
		int i) {
		tsmmColGroupsIndexIStartEnd(groups, filteredGroups, ret, i, i + 1, groups.size());
	}

	private static void tsmmColGroupsIndexIStartEnd(List<AColGroup> groups, List<AColGroup> filteredGroups,
		MatrixBlock ret, int i, int start, int end) {
		final AColGroup full_lhs = groups.get(i);
		final AColGroup lhs = filteredGroups.get(i);
		boolean isSDC = full_lhs instanceof ColGroupSDC || full_lhs instanceof ColGroupSDCSingle;
		for(int id = start; id < end; id++) {
			final AColGroup full_rhs = groups.get(id);
			final AColGroup rhs = filteredGroups.get(id);
			if(isSDC && (full_rhs instanceof ColGroupSDC || full_rhs instanceof ColGroupSDCSingle))
				full_lhs.leftMultByAColGroup(full_rhs, ret);
			else
				lhs.leftMultByAColGroup(rhs, ret);

		}
	}

	private static MatrixBlock leftMultByMatrix(List<AColGroup> colGroups, MatrixBlock that, MatrixBlock ret, int k,
		boolean overlapping) {

		if(that.isEmpty()) {
			ret.setNonZeros(0);
			return ret;
		}

		final int numColumnsOut = ret.getNumColumns();
		final boolean containsSDC = CLALibUtils.containsSDC(colGroups);

		// a constant colgroup summing the default values.
		double[] constV = containsSDC ? new double[numColumnsOut] : null;
		final List<AColGroup> filteredGroups = CLALibUtils.filterSDCGroups(colGroups, constV);
		if(colGroups == filteredGroups)
			constV = null;
		final double[] rowSums = containsSDC ? new double[that.getNumRows()] : null;

		if(k == 1) {
			leftMultByMatrixPrimitive(filteredGroups, that, ret, 0, that.getNumRows(), rowSums);
		}
		else {
			try {
				final ExecutorService pool = CommonThreadPool.get(k);
				final ArrayList<Callable<MatrixBlock>> tasks = new ArrayList<>();
				final int rowBlockSize = that.getNumRows() <= k ? 1 : Math.min(Math.max(that.getNumRows() / k * 2, 1),
					8);

				if(overlapping) {
					for(AColGroup g : filteredGroups) {
						MatrixBlock tmpRet = new MatrixBlock(ret.getNumRows(), ret.getNumColumns(), false);
						tmpRet.allocateDenseBlock();
						for(int blo = 0; blo < that.getNumRows(); blo += rowBlockSize)
							tasks.add(new LeftMatrixColGroupMultTaskOld(g, that, tmpRet, blo,
								Math.min(blo + rowBlockSize, that.getNumRows())));

					}
					List<Future<MatrixBlock>> futures = pool.invokeAll(tasks);
					pool.shutdown();
					BinaryOperator op = new BinaryOperator(Plus.getPlusFnObject());
					for(Future<MatrixBlock> future : futures)
						ret.binaryOperationsInPlace(op, future.get());
				}
				else {
					final int numberSplits = Math.max((k / (ret.getNumRows() / rowBlockSize)), 1);
					if(numberSplits == 1) {
						for(int blo = 0; blo < that.getNumRows(); blo += rowBlockSize) {
							tasks.add(new LeftMatrixColGroupMultTaskNew(filteredGroups, that, ret, blo,
								Math.min(blo + rowBlockSize, that.getNumRows()), rowSums));
						}
					}
					else {
						List<List<AColGroup>> split = split(filteredGroups, numberSplits);
						for(int blo = 0; blo < that.getNumRows(); blo += rowBlockSize) {
							for(int i = 0; i < split.size(); i++) {
								List<AColGroup> gr = split.get(i);
								if(i == 0) {
									// the first thread also have the responsibility to calculate the som of the left
									// hand side.
									tasks.add(new LeftMatrixColGroupMultTaskNew(gr, that, ret, blo,
										Math.min(blo + rowBlockSize, that.getNumRows()), rowSums));
								}
								else {
									tasks.add(new LeftMatrixColGroupMultTaskNew(gr, that, ret, blo,
										Math.min(blo + rowBlockSize, that.getNumRows()), null));
								}
							}
						}
					}

					List<Future<MatrixBlock>> futures = pool.invokeAll(tasks);

					pool.shutdown();
					for(Future<MatrixBlock> future : futures)
						future.get();
				}

			}
			catch(InterruptedException | ExecutionException e) {
				throw new DMLRuntimeException(e);
			}
		}

		// add the correction layer for the subtracted common values.
		if(rowSums != null)
			outerProduct(rowSums, constV, ret.getDenseBlockValues());

		ret.recomputeNonZeros();
		return ret;
	}

	private static List<List<AColGroup>> split(List<AColGroup> groups, int splits) {
		Collections.sort(groups, Comparator.comparing(AColGroup::getNumValues).reversed());

		List<List<AColGroup>> ret = new ArrayList<>();
		for(int i = 0; i < splits; i++)
			ret.add(new ArrayList<>());

		for(int j = 0; j < groups.size(); j++)
			ret.get(j % splits).add(groups.get(j));

		return ret;
	}

	private static void outerProduct(final double[] leftRowSum, final double[] rightColumnSum, final double[] result) {
		for(int row = 0; row < leftRowSum.length; row++) {
			final int offOut = rightColumnSum.length * row;
			final double vLeft = leftRowSum[row];
			for(int col = 0; col < rightColumnSum.length; col++) {
				result[offOut + col] += vLeft * rightColumnSum[col];
			}
		}
	}

	private static class LeftMatrixColGroupMultTaskOld implements Callable<MatrixBlock> {
		private final AColGroup _group;
		private final MatrixBlock _that;
		private final MatrixBlock _ret;
		private final int _rl;
		private final int _ru;

		protected LeftMatrixColGroupMultTaskOld(AColGroup group, MatrixBlock that, MatrixBlock ret, int rl, int ru) {
			_group = group;
			_that = that;
			_ret = ret;
			_rl = rl;
			_ru = ru;
		}

		@Override
		public MatrixBlock call() {
			try {
				_group.leftMultByMatrix(_that, _ret, _rl, _ru);
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
			return _ret;
		}
	}

	private static class LeftMatrixColGroupMultTaskNew implements Callable<MatrixBlock> {
		private final List<AColGroup> _groups;
		private final MatrixBlock _that;
		private final MatrixBlock _ret;
		private final int _rl;
		private final int _ru;
		private final double[] _rowSums;

		protected LeftMatrixColGroupMultTaskNew(List<AColGroup> groups, MatrixBlock that, MatrixBlock ret, int rl,
			int ru, double[] rowSums) {
			_groups = groups;
			_that = that;
			_ret = ret;
			_rl = rl;
			_ru = ru;
			_rowSums = rowSums;
		}

		@Override
		public MatrixBlock call() {
			try {
				leftMultByMatrixPrimitive(_groups, _that, _ret, _rl, _ru, _rowSums);
			}
			catch(Exception e) {
				e.printStackTrace();
				throw new DMLRuntimeException(e);
			}
			return _ret;
		}
	}

	private static class tsmmColGroupTask implements Callable<Object> {
		private final List<AColGroup> _groups;
		private final List<AColGroup> _filteredGroups;
		private final MatrixBlock _ret;
		private final int _index;
		private final int _start;
		private final int _end;

		protected tsmmColGroupTask(List<AColGroup> groups, List<AColGroup> filteredGroups, MatrixBlock ret, int i,
			int start, int end) {
			_groups = groups;
			_filteredGroups = filteredGroups;
			_ret = ret;
			_index = i;
			_start = start;
			_end = end;
		}

		@Override
		public MatrixBlock call() {
			try {
				tsmmColGroupsIndexIStartEnd(_groups, _filteredGroups, _ret, _index, _start, _end);
			}
			catch(Exception e) {
				e.printStackTrace();
				throw new DMLRuntimeException(e);
			}
			return _ret;
		}
	}

	private static class tsmmSelfColGroupTask implements Callable<Object> {
		private final AColGroup _g;
		private final MatrixBlock _ret;

		protected tsmmSelfColGroupTask(AColGroup g, MatrixBlock ret) {
			_g = g;
			_ret = ret;
		}

		@Override
		public MatrixBlock call() {
			try {
				_g.tsmm(_ret);
			}
			catch(Exception e) {
				e.printStackTrace();
				throw new DMLRuntimeException(e);
			}
			return _ret;
		}
	}

	private static void leftMultByMatrixPrimitive(List<AColGroup> colGroups, MatrixBlock that, MatrixBlock ret, int rl,
		int ru, double[] rowSums) {
		if(that.isInSparseFormat())
			leftMultByMatrixPrimitiveSparse(colGroups, that, ret, rl, ru, rowSums);
		else
			leftMultByMatrixPrimitiveDense(colGroups, that, ret, rl, ru, rowSums);
	}

	private static void leftMultByMatrixPrimitiveSparse(List<AColGroup> colGroups, MatrixBlock that, MatrixBlock ret,
		int rl, int ru, double[] rowSum) {

		for(int i = rl; i < ru; i++) {
			for(int j = 0; j < colGroups.size(); j++) {
				colGroups.get(j).leftMultByMatrix(that, ret, i, i + 1);
			}
			if(rowSum != null) {
				final SparseBlock sb = that.getSparseBlock();
				if(!sb.isEmpty(i)) {
					final int apos = sb.pos(i);
					final int alen = sb.size(i) + apos;
					final double[] aval = sb.values(i);
					for(int j = apos; j < alen; j++)
						rowSum[i] += aval[j];
				}
			}
		}
	}

	private static void leftMultByMatrixPrimitiveDense(List<AColGroup> colGroups, MatrixBlock that, MatrixBlock ret,
		int rl, int ru, double[] rowSum) {

		final int numColsOut = ret.getNumColumns();
		// Allocate a ColGroupValue array for the Column Groups of Value Type and multiply out any other columns.
		final List<ColGroupValue> ColGroupValues = preFilterAndMultiply(colGroups, that, ret, rl, ru);

		// The number of rows to process together
		final int rowBlockSize = 1;
		// The number of column groups to process together
		// the value should ideally be set so that the colgroups fits into cache together with a row block.
		// currently we only try to avoid having a dangling small number of column groups in the last block.
		final int colGroupBlocking = ColGroupValues.size() % 16 < 4 ? 20 : 16;

		// Allocate pre Aggregate Array List
		final MatrixBlock[] preAgg = populatePreAggregate(colGroupBlocking);

		// Allocate temporary Result matrix.
		MatrixBlock tmpRes = new MatrixBlock(rowBlockSize, numColsOut, false);

		// For each column group block
		for(int g = 0; g < ColGroupValues.size(); g += colGroupBlocking) {
			final int gEnd = Math.min(g + colGroupBlocking, ColGroupValues.size());

			// For each column group in the current block allocate the preaggregate array.
			for(int j = g; j < gEnd && j < ColGroupValues.size(); j++) {
				ColGroupValue cg = ColGroupValues.get(j);
				int nVals = cg.getNumValues();
				preAgg[j % colGroupBlocking].reset(rowBlockSize, nVals, false);
			}

			int colBlockSize = 32000;

			// For each row block
			for(int h = rl; h < ru; h += rowBlockSize) {
				// For each column block
				final int rowUpper = Math.min(h + rowBlockSize, ru);
				for(int i = 0; i < that.getNumColumns(); i += colBlockSize) {
					final int colUpper = Math.min(i + colBlockSize, that.getNumColumns());
					// Pre Aggregate each column group in block
					for(int j = g; j < gEnd && j < ColGroupValues.size(); j++) {
						ColGroupValues.get(j).preAggregateDense(that, preAgg[j % colGroupBlocking], h, rowUpper, i,
							colUpper);
					}
					if(rowSum != null) {
						final double[] thatV = that.getDenseBlockValues();
						for(int r = h; r < rowUpper; r++) {
							final int rowOff = r * that.getNumColumns();
							for(int c = rowOff + i; c < rowOff + colUpper; c++)
								rowSum[r] += thatV[c];
						}
					}
				}
				// Multiply out the preAggregate to the output matrix.
				for(int j = g; j < gEnd && j < ColGroupValues.size(); j++) {
					ColGroupValue vj = ColGroupValues.get(j);
					MatrixBlock preAggJ = preAgg[j % colGroupBlocking];
					preAggJ.recomputeNonZeros();
					tmpRes.reset(rowBlockSize, vj.getNumCols(), false);
					MatrixBlock tmp = vj.leftMultByPreAggregateMatrix(preAggJ, tmpRes);
					vj.addMatrixToResult(tmp, ret, h, Math.min(h + rowBlockSize, ru));
					preAggJ.reset();
				}
			}
		}

	}

	private static MatrixBlock[] populatePreAggregate(int colGroupBlocking) {
		final MatrixBlock[] preAgg = new MatrixBlock[colGroupBlocking];
		// poplate the preAgg array.
		for(int j = 0; j < colGroupBlocking; j++) {
			MatrixBlock m = new MatrixBlock(1, 1, false);
			m.allocateDenseBlock();
			preAgg[j] = m;
		}
		return preAgg;
	}

	private static List<ColGroupValue> preFilterAndMultiply(List<AColGroup> colGroups, MatrixBlock that,
		MatrixBlock ret, int rl, int ru) {
		final List<ColGroupValue> ColGroupValues = new ArrayList<>(colGroups.size());
		for(int j = 0; j < colGroups.size(); j++) {
			AColGroup a = colGroups.get(j);
			if(a instanceof ColGroupValue)
				ColGroupValues.add((ColGroupValue) a);
			else
				a.leftMultByMatrix(that, ret, rl, ru);
		}
		Collections.sort(ColGroupValues, Comparator.comparing(AColGroup::getNumValues).reversed());
		return ColGroupValues;
	}
}
