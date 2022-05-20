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
import org.apache.sysds.runtime.compress.colgroup.APreAgg;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class CLALibLeftMultBy {
	private static final Log LOG = LogFactory.getLog(CLALibLeftMultBy.class.getName());

	/**
	 * Left multiplication with a CompressedMatrixBlock on the right following the equation:
	 * 
	 * ret = t(left) %*% right
	 * 
	 * @param right A CompressedMatrixBlock on the right side of the multiplication.
	 * @param left  A not transposed MatrixBlock.
	 * @param ret   The result output matrix, this allocation of the object can be used if appropriate, otherwise a new
	 *              matrix Block is allocated to be returned. This argument can also be null.
	 * @param k     The number of threads allowed to be used
	 * @return The result of the matrix multiplication
	 */
	public static MatrixBlock leftMultByMatrixTransposed(CompressedMatrixBlock right, MatrixBlock left, MatrixBlock ret,
		int k) {
		if(left.isEmpty() || right.isEmpty())
			return prepareEmptyReturnMatrix(right, left, ret, true);
		if(left.getNumColumns() > 1)
			LOG.warn("Transposing matrix block for transposed left matrix multiplication");
		MatrixBlock transposed = new MatrixBlock(left.getNumColumns(), left.getNumRows(), false);
		LibMatrixReorg.transpose(left, transposed, k);
		ret = leftMultByMatrix(right, transposed, ret, k);
		return ret;
	}

	/**
	 * Left multiplication with two CompressedMatrixBlock following the equation:
	 * 
	 * ret = t(left) %*% right
	 * 
	 * @param right A CompressedMatrixBlock on the right side of the multiplication.
	 * @param left  A not transposed CompressedMatrixBlock, but logically inside the function it is considered
	 *              transposed.
	 * @param ret   The result output matrix, this allocation of the object can be used if appropriate, otherwise a new
	 *              matrix Block is allocated to be returned. This argument can also be null.
	 * @param k     The number of threads allowed to be used
	 * @return The result of the matrix multiplication
	 */
	public static MatrixBlock leftMultByMatrixTransposed(CompressedMatrixBlock right, CompressedMatrixBlock left,
		MatrixBlock ret, int k) {
		if(left.isEmpty() || right.isEmpty())
			return prepareEmptyReturnMatrix(right, left, ret, true);
		ret = prepareReturnMatrix(right, left, ret, true);
		leftMultByCompressedTransposedMatrix(right, left, ret, k);
		return ret;
	}

	/**
	 * Left multiplication with two CompressedMatrixBlock following the equation:
	 * 
	 * ret = left %*% right
	 * 
	 * @param right A CompressedMatrixBlock on the right side of the multiplication.
	 * @param left  A MatrixBlock on the left side of the equation
	 * @param ret   The result output matrix, this allocation of the object can be used if appropriate, otherwise a new
	 *              matrix Block is allocated to be returned. This argument can also be null.
	 * @param k     The number of threads allowed to be used
	 * @return The result of the matrix multiplication
	 */
	public static MatrixBlock leftMultByMatrix(CompressedMatrixBlock right, MatrixBlock left, MatrixBlock ret, int k) {
		if(left.isEmpty() || right.isEmpty())
			return prepareEmptyReturnMatrix(right, left, ret, false);
		ret = prepareReturnMatrix(right, left, ret, false);
		ret = LMM(right.getColGroups(), left, ret, k, right.isOverlapping());
		return ret;
	}

	private static MatrixBlock prepareEmptyReturnMatrix(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret,
		boolean doTranspose) {
		final int numRowsOutput = doTranspose ? m2.getNumColumns() : m2.getNumRows();
		final int numColumnsOutput = m1.getNumColumns();
		if(ret == null)
			ret = new MatrixBlock(numRowsOutput, numColumnsOutput, true, 0);
		else if(!(ret.getNumColumns() == numColumnsOutput && ret.getNumRows() == numRowsOutput && ret.isAllocated()))
			ret.reset(numRowsOutput, numColumnsOutput, true, 0);
		return ret;
	}

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

	private static MatrixBlock leftMultByCompressedTransposedMatrix(CompressedMatrixBlock right,
		CompressedMatrixBlock left, MatrixBlock ret, int k) {

		final int sd = right.getNumRows(); // shared dim
		final int cr = right.getNumColumns();
		final int rl = left.getNumColumns();

		final List<AColGroup> rightCG = right.getColGroups();
		final List<AColGroup> leftCG = left.getColGroups();

		final boolean containsRight = CLALibUtils.shouldPreFilter(rightCG);
		double[] cR = containsRight ? new double[cr] : null;
		final List<AColGroup> fRight = CLALibUtils.filterGroups(rightCG, cR);

		final boolean containsLeft = CLALibUtils.shouldPreFilter(leftCG);
		double[] cL = containsLeft ? new double[rl] : null;
		final List<AColGroup> fLeft = CLALibUtils.filterGroups(leftCG, cL);

		for(int j = 0; j < fLeft.size(); j++)
			for(int i = 0; i < fRight.size(); i++)
				fRight.get(i).leftMultByAColGroup(fLeft.get(j), ret, sd);

		double[] retV = ret.getDenseBlockValues();
		if(containsLeft && containsRight)
			// if both -- multiply the left and right vectors scaling by number of shared dim
			outerProductWithScaling(cL, cR, sd, retV);
		if(containsLeft) // if left -- multiply left with right sum
			outerProduct(cL, CLALibUtils.getColSum(fRight, cr, sd), retV);
		if(containsRight)// if right -- multiply right with left sum
			outerProduct(CLALibUtils.getColSum(fLeft, rl, sd), cR, retV);
		ret.recomputeNonZeros();

		return ret;
	}

	private static MatrixBlock LMM(List<AColGroup> colGroups, MatrixBlock that, MatrixBlock ret, int k,
		boolean overlapping) {
		final int numColumnsOut = ret.getNumColumns();
		final int lr = that.getNumRows();
		final boolean shouldFilter = CLALibUtils.shouldPreFilter(colGroups);
		final List<AColGroup> noPreAggGroups = new ArrayList<>();
		final List<APreAgg> preAggGroups = new ArrayList<>();
		if(shouldFilter) {
			final double[] constV = new double[numColumnsOut];
			CLALibUtils.filterGroupsAndSplitPreAgg(colGroups, constV, noPreAggGroups, preAggGroups);
			// Sort so that the big expensive preAgg groups are first.
			Collections.sort(preAggGroups, Comparator.comparing(AColGroup::getNumValues).reversed());

			double[] rowSums;
			if(!noPreAggGroups.isEmpty() || !preAggGroups.isEmpty()) {
				final int sizeSum = preAggGroups.size() + noPreAggGroups.size();
				rowSums = new double[lr];
				if(k == 1 || sizeSum == 1)
					LMMTaskExec(noPreAggGroups, preAggGroups, that, ret, 0, lr, rowSums, k);
				else
					LMMParallel(noPreAggGroups, preAggGroups, that, ret, rowSums, overlapping, k);
			}
			else
				rowSums = that.rowSum(k).getDenseBlockValues();

			// add the correction layer for the subtracted common values.
			if(rowSums != null && constV != null) {
				if(ret.isEmpty())
					ret.allocateDenseBlock();
				else
					ret.sparseToDense();

				outerProduct(rowSums, constV, ret.getDenseBlockValues());
			}
		}
		else {
			CLALibUtils.splitPreAgg(colGroups, noPreAggGroups, preAggGroups);
			// Sort so that the big expensive preAgg groups are first.
			Collections.sort(preAggGroups, Comparator.comparing(AColGroup::getNumValues).reversed());
			if(k == 1 || colGroups.size() == 1)
				LMMTaskExec(noPreAggGroups, preAggGroups, that, ret, 0, lr, null, k);
			else
				LMMParallel(noPreAggGroups, preAggGroups, that, ret, null, overlapping, k);
		}

		ret.recomputeNonZeros();
		ret.examSparsity();
		return ret;
	}

	private static void LMMParallel(List<AColGroup> npa, List<APreAgg> pa, MatrixBlock that, MatrixBlock ret,
		double[] rowSums, boolean overlapping, int k) {
		final ExecutorService pool = CommonThreadPool.get(k);
		try {
			final ArrayList<Callable<MatrixBlock>> tasks = new ArrayList<>();

			final int rl = that.getNumRows();
			final int rowBlockSize = Math.max(rl / k, 1);
			final int nG = npa.size() + pa.size();

			final boolean useTmp = overlapping && nG > 1;
			// skip value to parallelize the pa groups without allocating new arrays
			final int s = Math.min(pa.size(), k);
			if(!useTmp) {
				// Put results directly into ret
				for(int blo = 0; blo < rl; blo += rowBlockSize) {
					final int end = Math.min(blo + rowBlockSize, rl);

					for(AColGroup g : npa) // all groups get their own task
						tasks.add(new LMMNoPreAggTask(g, that, ret, blo, end));

					for(int off = 0; off < s; off++) { // only allocate k tasks at max
						if(off == s - 1)
							tasks.add(new LMMPreAggTask(pa, that, ret, blo, end, off, s, rowSums, 1));
						else
							tasks.add(new LMMPreAggTask(pa, that, ret, blo, end, off, s, null, 1));
					}

					if(pa.isEmpty() && rowSums != null) // row sums task
						tasks.add(new LMMRowSums(that, blo, end, rowSums));

				}

				for(Future<MatrixBlock> future : pool.invokeAll(tasks))
					future.get();
			}
			else {
				// allocate temp
				final int nCol = ret.getNumColumns();
				final int nRow = ret.getNumRows();
				for(int blo = 0; blo < rl; blo += rowBlockSize) {
					final int end = Math.min(blo + rowBlockSize, rl);

					for(AColGroup g : npa) // all groups get their own task
						tasks.add(new LMMNoPreAggTask(g, that, nRow, nCol, blo, end));

					for(int off = 0; off < s; off++) { // only allocate k tasks at max
						if(off == s - 1)
							tasks.add(new LMMPreAggTask(pa, that, nRow, nCol, blo, end, off, s, rowSums, 1));
						else
							tasks.add(new LMMPreAggTask(pa, that, nRow, nCol, blo, end, off, s, null, 1));
					}

					if(pa.isEmpty() && rowSums != null) // row sums task
						tasks.add(new LMMRowSums(that, blo, end, rowSums));

				}

				BinaryOperator op = new BinaryOperator(Plus.getPlusFnObject());
				for(Future<MatrixBlock> future : pool.invokeAll(tasks)) {
					MatrixBlock mb = future.get();
					mb.examSparsity();
					ret.binaryOperationsInPlace(op, mb);
				}
			}

		}
		catch(InterruptedException | ExecutionException e) {
			pool.shutdown();
			throw new DMLRuntimeException(e);
		}
		pool.shutdown();
	}

	private static void LMMTaskExec(List<AColGroup> npa, List<APreAgg> pa, MatrixBlock that, MatrixBlock ret, int rl,
		int ru, double[] rowSums, int k) {
		if(npa.isEmpty() && pa.isEmpty()) {
			rowSum(that, rowSums, rl, ru, 0, that.getNumColumns());
			return;
		}
		for(int r = rl; r < ru; r += 4) {
			final int re = Math.min(r + 4, ru);
			// Process MMs.
			for(int i = 0; i < npa.size(); i++)
				LMMNoPreAgg(npa.get(i), that, ret, r, re);

			if(pa.size() > 0)
				LMMWithPreAgg(pa, that, ret, r, re, 0, 1, rowSums, k);
		}
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

	private static void outerProductWithScaling(final double[] leftRowSum, final double[] rightColumnSum,
		final int scaling, final double[] result) {
		for(int row = 0; row < leftRowSum.length; row++) {
			final int offOut = rightColumnSum.length * row;
			final double vLeft = leftRowSum[row] * scaling;
			for(int col = 0; col < rightColumnSum.length; col++) {
				result[offOut + col] += vLeft * rightColumnSum[col];
			}
		}
	}

	private static void LMMNoPreAgg(AColGroup g, MatrixBlock that, MatrixBlock ret, int rl, int ru) {
		g.leftMultByMatrixNoPreAgg(that, ret, rl, ru, 0, that.getNumColumns());
	}

	private static void LMMWithPreAgg(List<APreAgg> preAggCGs, MatrixBlock that, MatrixBlock ret, int rl, int ru,
		int off, int skip, double[] rowSums, int k) {
		if(!that.isInSparseFormat())
			LMMWithPreAggDense(preAggCGs, that, ret, rl, ru, off, skip, rowSums);
		else
			LMMWithPreAggSparse(preAggCGs, that, ret, rl, ru, off, skip, rowSums);
	}

	private static void LMMWithPreAggSparse(List<APreAgg> preAggCGs, MatrixBlock that, MatrixBlock ret, int rl, int ru,
		int off, int skip, double[] rowSum) {
		// row multiplication
		final MatrixBlock tmpRes = new MatrixBlock(1, ret.getNumColumns(), false);
		final int maxV = preAggCGs.get(off).getNumValues();
		final MatrixBlock preA = new MatrixBlock(1, maxV, false);
		// final DenseBlock db = preA.getDenseBlock();
		preA.allocateDenseBlock();
		final double[] preAV = preA.getDenseBlockValues();
		tmpRes.allocateDenseBlock();
		final SparseBlock sb = that.getSparseBlock();

		for(int j = off; j < preAggCGs.size(); j += skip) {
			for(int r = rl; r < ru; r++) {
				if(sb.isEmpty(r))
					continue;
				final int rcu = r + 1;
				final int nCol = preAggCGs.get(j).getNumCols();
				final int nVal = preAggCGs.get(j).getNumValues();
				if(nCol == 1 || (sb.size(r) * nCol < sb.size(r) + nCol * nVal))
					LMMNoPreAgg(preAggCGs.get(j), that, ret, r, rcu);
				else {
					final APreAgg g = preAggCGs.get(j);
					preA.reset(1, g.getPreAggregateSize(), false);
					g.preAggregateSparse(sb, preAV, r, rcu);
					g.mmWithDictionary(preA, tmpRes, ret, 1, r, rcu);
				}
			}
		}

		rowSumSparse(that.getSparseBlock(), rowSum, rl, ru, 0, that.getNumColumns());
	}

	private static void LMMWithPreAggDense(List<APreAgg> preAggCGs, MatrixBlock that, MatrixBlock ret, int rl, int ru,
		int off, int skip, double[] rowSum) {

		/** The column block size for preAggregating column groups */
		final int colBZ = 1024;
		// The number of rows to process together
		final int rowBlockSize = 4;
		// The number of column groups to process together
		// the value should ideally be set so that the colGroups fits into cache together with a row block.
		// currently we only try to avoid having a dangling small number of column groups in the last block.
		// final int colGroupBlocking = preAggCGs.size() ;// % 16 < 4 ? 20 : 16;
		final int colGroupBlocking = 8;
		// final int colGroupBlocking = 4;
		final int nColGroups = preAggCGs.size();

		// Allocate pre Aggregate Array List
		final MatrixBlock[] preAgg = populatePreAggregate(colGroupBlocking);

		// Allocate temporary Result matrix
		// guaranteed to be large enough for all groups
		final MatrixBlock tmpRes = new MatrixBlock(rowBlockSize, ret.getNumColumns(), false);

		final int lc = that.getNumColumns();
		// For each row block
		for(int rlt = rl; rlt < ru; rlt += rowBlockSize) {
			final int rut = Math.min(rlt + rowBlockSize, ru);
			// For each column group block
			for(int gl = off; gl < nColGroups; gl += colGroupBlocking * skip) {
				final int gu = Math.min(gl + (colGroupBlocking * skip), nColGroups);
				// For each column group in the current block allocate the preaggregate array.
				for(int j = gl, p = 0; j < gu; j += skip, p++) {
					final int preAggNCol = preAggCGs.get(j).getPreAggregateSize();
					preAgg[p].reset(rut - rlt, preAggNCol, false);
				}

				// PreAggregate current block of column groups
				for(int cl = 0; cl < lc; cl += colBZ) {
					final int cu = Math.min(cl + colBZ, lc);
					for(int j = gl, p = 0; j < gu; j += skip, p++)
						preAggCGs.get(j).preAggregateDense(that, preAgg[p].getDenseBlockValues(), rlt, rut, cl, cu);
					if(gu == nColGroups)
						rowSum(that, rowSum, rlt, rut, cl, cu);
				}

				// Multiply out the PreAggregate to the output matrix.
				for(int j = gl, p = 0; j < gu; j += skip, p++) {
					final APreAgg cg = preAggCGs.get(j);
					final MatrixBlock preAggThis = preAgg[p];
					cg.mmWithDictionary(preAggThis, tmpRes, ret, 1, rlt, rut);
				}
			}
		}
	}

	public static double[] rowSum(MatrixBlock mb, int rl, int ru, int cl, int cu) {
		double[] ret = new double[ru];
		rowSum(mb, ret, rl, ru, cl, cu);
		return ret;
	}

	private static void rowSum(MatrixBlock mb, double[] rowSum, int rl, int ru, int cl, int cu) {
		if(mb.isInSparseFormat())
			rowSumSparse(mb.getSparseBlock(), rowSum, rl, ru, cl, cu);
		else
			rowSumDense(mb, rowSum, rl, ru, cl, cu);
	}

	private static void rowSumSparse(SparseBlock sb, double[] rowSum, int rl, int ru, int cl, int cu) {
		if(rowSum != null) {
			for(int i = rl; i < ru; i++) {
				if(sb.isEmpty(i))
					continue;
				final int apos = sb.pos(i);
				final int alen = sb.size(i) + apos;
				final double[] aval = sb.values(i);
				final int[] aix = sb.indexes(i);
				if(cl == 0 && aix[alen - 1] < cu)
					for(int j = apos; j < alen; j++)
						rowSum[i] += aval[j];
				else {
					int j = apos;
					while(j < alen && aix[j] < cl)
						j++;
					while(j < alen && aix[j] < cu)
						rowSum[i] += aval[j++];
				}
			}
		}
	}

	private static void rowSumDense(MatrixBlock that, double[] rowSum, int rl, int ru, int cl, int cu) {
		if(rowSum != null) {
			final DenseBlock db = that.getDenseBlock();
			for(int r = rl; r < ru; r++) {
				final double[] thatV = db.values(r);
				final int rowOff = db.pos(r);
				for(int c = rowOff + cl; c < rowOff + cu; c++)
					rowSum[r] += thatV[c];
			}
		}
	}

	private static MatrixBlock[] populatePreAggregate(int colGroupBlocking) {
		final MatrixBlock[] preAgg = new MatrixBlock[colGroupBlocking];
		// populate the preAgg array.
		for(int j = 0; j < colGroupBlocking; j++) {
			final MatrixBlock m = new MatrixBlock(1, 1, false);
			m.allocateDenseBlock();
			preAgg[j] = m;
		}
		return preAgg;
	}

	private static class LMMPreAggTask implements Callable<MatrixBlock> {
		private final List<APreAgg> _pa;
		private final MatrixBlock _that;
		private final MatrixBlock _ret;
		private final int _rl;
		private final int _ru;
		private final double[] _rowSums;
		private final int _off;
		private final int _skip;
		private final int _k;

		protected LMMPreAggTask(List<APreAgg> pa, MatrixBlock that, int retR, int retC, int rl, int ru, int off, int skip,
			double[] rowSums, int k) {
			_pa = pa;
			_that = that;
			_ret = new MatrixBlock(retR, retC, false);
			_ret.allocateDenseBlock();
			_rl = rl;
			_ru = ru;
			_rowSums = rowSums;
			_off = off;
			_skip = skip;
			_k = k;
		}

		protected LMMPreAggTask(List<APreAgg> pa, MatrixBlock that, MatrixBlock ret, int rl, int ru, int off, int skip,
			double[] rowSums, int k) {
			_pa = pa;
			_that = that;
			_ret = ret;
			_rl = rl;
			_ru = ru;
			_rowSums = rowSums;
			_off = off;
			_skip = skip;
			_k = k;
		}

		@Override
		public MatrixBlock call() {
			try {
				LMMWithPreAgg(_pa, _that, _ret, _rl, _ru, _off, _skip, _rowSums, _k);
			}
			catch(Exception e) {
				e.printStackTrace();
				throw new DMLRuntimeException(e);
			}
			return _ret;
		}
	}

	private static class LMMNoPreAggTask implements Callable<MatrixBlock> {
		private final AColGroup _cg;
		private final MatrixBlock _that;
		private final MatrixBlock _ret;
		private final int _rl;
		private final int _ru;

		protected LMMNoPreAggTask(AColGroup cg, MatrixBlock that, int retR, int retC, int rl, int ru) {
			_cg = cg;
			_that = that;
			_ret = new MatrixBlock(retR, retC, false);
			_ret.allocateDenseBlock();
			_rl = rl;
			_ru = ru;
		}

		protected LMMNoPreAggTask(AColGroup cg, MatrixBlock that, MatrixBlock ret, int rl, int ru) {
			_cg = cg;
			_that = that;
			_ret = ret;
			_rl = rl;
			_ru = ru;
		}

		@Override
		public MatrixBlock call() {
			try {
				LMMNoPreAgg(_cg, _that, _ret, _rl, _ru);
			}
			catch(Exception e) {
				e.printStackTrace();
				throw new DMLRuntimeException(e);
			}
			return _ret;
		}
	}

	private static class LMMRowSums implements Callable<MatrixBlock> {
		private final MatrixBlock _that;
		private final int _rl;
		private final int _ru;
		private final double[] _rowSums;

		protected LMMRowSums(MatrixBlock that, int rl, int ru, double[] rowSums) {
			_that = that;
			_rl = rl;
			_ru = ru;
			_rowSums = rowSums;
		}

		@Override
		public MatrixBlock call() {
			try {
				rowSumDense(_that, _rowSums, _rl, _ru, 0, _that.getNumColumns());
			}
			catch(Exception e) {
				e.printStackTrace();
				throw new DMLRuntimeException(e);
			}
			return null;
		}
	}
}
