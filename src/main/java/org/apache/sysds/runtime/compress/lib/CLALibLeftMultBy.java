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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.APreAgg;
import org.apache.sysds.runtime.compress.colgroup.dictionary.AIdentityDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IdentityDictionary;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.LibMatrixBincell;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.utils.stats.Timing;

public final class CLALibLeftMultBy {
	private static final Log LOG = LogFactory.getLog(CLALibLeftMultBy.class.getName());

	// /** Reusable cache intermediate double array for temporary lmm */
	// private static ThreadLocal<Pair<Boolean, double[]>> cacheIntermediate = null;

	private CLALibLeftMultBy() {
		// private constructor
	}

	/**
	 * Left multiplication with a CompressedMatrixBlock on the right following the equation:
	 * 
	 * <p>
	 * ret = t(left) %*% right
	 * </p>
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
	 * <p>
	 * ret = t(left) %*% right
	 * </p>
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
		try {
			if(left.isEmpty() || right.isEmpty())
				return prepareEmptyReturnMatrix(right, left, ret, true);
			ret = prepareReturnMatrix(right, left, ret, true);
			leftMultByCompressedTransposedMatrix(right, left, ret, k);
			return ret;
		}
		catch(Exception e) {
			throw new DMLCompressionException("Failed CLA Compressed Transposed LMM", e);
		}
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
		try {
			// return LibMatrixMult.matrixMult(left, right.getUncompressed(), ret, k); // uncompressed example

			if(left.isEmpty() //
				|| right.isEmpty())
				return prepareEmptyReturnMatrix(right, left, ret, false);

			if(CLALibSelectionMult.isSelectionMatrix(left))
				return CLALibSelectionMult.leftSelection(right, left, ret, k);

			ret = prepareReturnMatrix(right, left, ret, false);
			ret = LMM(right.getColGroups(), left, ret, k, right.isOverlapping());

			return ret;
		}
		catch(Exception e) {
			throw new DMLCompressionException("Failed CLA LMM", e);
		}
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
		CompressedMatrixBlock left, final MatrixBlock ret, int k) throws Exception {
		if(k > 1)
			return leftMultByCompressedTransposedMatrixParallel(right, left, ret, k);
		else
			return leftMultByCompressedTransposedMatrixSingleThread(right, left, ret);
	}

	private static MatrixBlock leftMultByCompressedTransposedMatrixParallel(CompressedMatrixBlock right,
		CompressedMatrixBlock left, final MatrixBlock ret, int k) throws Exception {

		final int sd = right.getNumRows(); // shared dim
		final int cr = right.getNumColumns();
		final int rl = left.getNumColumns();

		final List<AColGroup> rightCG = right.getColGroups();
		final List<AColGroup> leftCG = left.getColGroups();

		final boolean containsRight = CLALibUtils.shouldPreFilter(rightCG);
		final double[] cR = containsRight ? new double[cr] : null;
		final List<AColGroup> fRight = CLALibUtils.filterGroups(rightCG, cR);

		final boolean containsLeft = CLALibUtils.shouldPreFilter(leftCG);
		final double[] cL = containsLeft ? new double[rl] : null;
		final List<AColGroup> fLeft = CLALibUtils.filterGroups(leftCG, cL);

		// Force dense output
		ret.allocateDenseBlock();
		ret.setNonZeros((long) ret.getNumRows() * ret.getNumColumns());

		final ExecutorService pool = CommonThreadPool.get(k);

		try {
			final List<Future<MatrixBlock>> t = new ArrayList<>();
			for(int j = 0; j < fLeft.size(); j++) {
				final int jj = j;
				t.add(pool.submit(() -> {
					MatrixBlock retT = new MatrixBlock(ret.getNumRows(), ret.getNumColumns(), false);
					retT.allocateDenseBlock();
					for(int i = 0; i < fRight.size(); i++) {
						fRight.get(i).leftMultByAColGroup(fLeft.get(jj), retT, sd);
					}
					retT.examSparsity(true);
					return retT;
				}));
			}

			if(containsLeft && containsRight)
				// if both -- multiply the left and right vectors scaling by number of shared dim
				outerProductWithScaling(cL, cR, sd, ret);
			if(containsLeft) // if left -- multiply left with right sum
				for(Future<?> f : outerProductParallelTasks(cL, CLALibUtils.getColSum(fRight, cr, sd), ret, pool))
					f.get();

			if(containsRight)// if right -- multiply right with left sum
				for(Future<?> f : outerProductParallelTasks(CLALibUtils.getColSum(fLeft, rl, sd), cR, ret, pool))
					f.get();

			for(Future<MatrixBlock> f : t) {
				MatrixBlock mb = f.get();
				if(!mb.isEmpty()) {
					if(mb.isInSparseFormat())
						LibMatrixBincell.bincellOpInPlaceRight(ret, mb, new BinaryOperator(Plus.getPlusFnObject()));
					else if(mb.getDenseBlock().isContiguous()) {
						final double[] retV = ret.getDenseBlockValues();
						LibMatrixMult.vectAdd(mb.getDenseBlockValues(), retV, 0, 0, retV.length);
					}
					else
						LibMatrixBincell.bincellOpInPlaceRight(ret, mb, new BinaryOperator(Plus.getPlusFnObject()));
				}
			}
			ret.recomputeNonZeros(k);
		}
		finally {
			pool.shutdown();
		}
		return ret;
	}

	private static MatrixBlock leftMultByCompressedTransposedMatrixSingleThread(CompressedMatrixBlock right,
		CompressedMatrixBlock left, final MatrixBlock ret) {
		final int sd = right.getNumRows(); // shared dim
		final int cr = right.getNumColumns();
		final int rl = left.getNumColumns();

		final List<AColGroup> rightCG = right.getColGroups();
		final List<AColGroup> leftCG = left.getColGroups();

		final boolean containsRight = CLALibUtils.shouldPreFilter(rightCG);
		final double[] cR = containsRight ? new double[cr] : null;
		final List<AColGroup> fRight = CLALibUtils.filterGroups(rightCG, cR);

		final boolean containsLeft = CLALibUtils.shouldPreFilter(leftCG);
		final double[] cL = containsLeft ? new double[rl] : null;
		final List<AColGroup> fLeft = CLALibUtils.filterGroups(leftCG, cL);

		// Force dense output
		ret.setNonZeros((long) ret.getNumRows() * ret.getNumColumns());
		ret.allocateDenseBlock();

		for(int j = 0; j < fLeft.size(); j++)
			for(int i = 0; i < fRight.size(); i++)
				fRight.get(i).leftMultByAColGroup(fLeft.get(j), ret, sd);

		if(containsLeft && containsRight)
			// if both -- multiply the left and right vectors scaling by number of shared dim
			outerProductWithScaling(cL, cR, sd, ret);
		if(containsLeft) // if left -- multiply left with right sum
			outerProductSingleThread(cL, CLALibUtils.getColSum(fRight, cr, sd), ret);
		if(containsRight)// if right -- multiply right with left sum
			outerProductSingleThread(CLALibUtils.getColSum(fLeft, rl, sd), cR, ret);

		ret.recomputeNonZeros();
		return ret;
	}

	private static MatrixBlock LMM(List<AColGroup> colGroups, MatrixBlock that, MatrixBlock ret, int k,
		boolean overlapping) throws Exception {
		final int numColumnsOut = ret.getNumColumns();
		final int lr = that.getNumRows();
		final boolean shouldFilter = CLALibUtils.shouldPreFilter(colGroups);
		final List<AColGroup> noPreAggGroups = new ArrayList<>();
		final List<APreAgg> preAggGroups = new ArrayList<>();

		if(shouldFilter) {
			// Timing t = new Timing();
			final double[] constV;
			// if(CLALibUtils.alreadyPreFiltered(colGroups, ret.getNumColumns())) {
			// constV = CLALibUtils.filterGroupsAndSplitPreAggOneConst(colGroups, noPreAggGroups, preAggGroups);
			// }
			// else {
			constV = new double[numColumnsOut]; // millions of columns...
			CLALibUtils.filterGroupsAndSplitPreAgg(colGroups, constV, noPreAggGroups, preAggGroups);
			// }

			// final double filterGroupsTime = t.stop();

			// Sort so that the big expensive preAgg groups are first to balance threads
			// if(k * 2 < colGroups.size())
			// Collections.sort(preAggGroups, Comparator.comparing(AColGroup::getNumValues).reversed());

			final double[] rowSums;
			if(!noPreAggGroups.isEmpty() || !preAggGroups.isEmpty()) {
				final int sizeSum = preAggGroups.size() + noPreAggGroups.size();
				rowSums = new double[lr];
				if(k == 1 || sizeSum == 1)
					LMMTaskExec(noPreAggGroups, preAggGroups, that, ret, 0, lr, rowSums);
				else
					LMMParallel(noPreAggGroups, preAggGroups, that, ret, rowSums, overlapping, k);
			}
			else
				rowSums = that.rowSum(k).getDenseBlockValues();

			// add the correction layer for the subtracted common values.
			if(rowSums != null) {
				if(ret.isEmpty())
					ret.allocateDenseBlock();
				else
					ret.sparseToDense();
				outerProduct(rowSums, constV, ret, k);
			}
		}
		else {
			CLALibUtils.splitPreAgg(colGroups, noPreAggGroups, preAggGroups);
			// Sort so that the big expensive preAgg groups are first.
			// Collections.sort(preAggGroups, Comparator.comparing(AColGroup::getNumValues).reversed());
			if(k == 1 || colGroups.size() == 1)
				LMMTaskExec(noPreAggGroups, preAggGroups, that, ret, 0, lr, null);
			else
				LMMParallel(noPreAggGroups, preAggGroups, that, ret, null, overlapping, k);

		}

		ret.recomputeNonZeros(k);
		ret.examSparsity(k);
		return ret;
	}

	private static void LMMParallel(List<AColGroup> npa, List<APreAgg> pa, MatrixBlock that, MatrixBlock ret,
		double[] rowSums, boolean overlapping, int k) throws Exception {
		final ExecutorService pool = CommonThreadPool.get(k);
		try {
			final int nG = npa.size() + pa.size();
			final boolean useTmp = (overlapping && nG > 1) //
				|| (nG * 2 < k && ret.getNumColumns() < 1000);

			// skip value to parallelize the pa groups without allocating new arrays
			if(!useTmp)
				LMMParallelNoTempOut(npa, pa, that, ret, rowSums, overlapping, k, pool);
			else
				LMMParallelTempOut(npa, pa, that, ret, rowSums, overlapping, k, pool);
		}
		finally {
			pool.shutdown();
		}
	}

	private static void LMMParallelNoTempOut(List<AColGroup> npa, List<APreAgg> pa, MatrixBlock that, MatrixBlock ret,
		double[] rowSums, boolean overlapping, int k, ExecutorService pool) throws Exception {

		final int s = Math.min(pa.size(), k);
		final int rt = that.getNumRows();
		final int ct = that.getNumColumns();
		final int rowBlockSize = Math.max(rt / k, 1);

		// skip value to parallelize the pa groups without allocating new arrays

		final ArrayList<Future<?>> tasks = new ArrayList<>();
		// Put results directly into ret
		for(int blo = 0; blo < rt; blo += rowBlockSize) {
			final int start = blo;
			final int end = Math.min(blo + rowBlockSize, rt);
			LLMNoTempOutRowBlockTasks(npa, pa, that, ret, rowSums, pool, s, ct, tasks, start, end, k);
		}

		for(Future<?> future : tasks)
			future.get();

	}

	private static void LLMNoTempOutRowBlockTasks(List<AColGroup> npa, List<APreAgg> pa, MatrixBlock that,
		MatrixBlock ret, double[] rowSums, ExecutorService pool, final int s, final int ct,
		final ArrayList<Future<?>> tasks, final int start, final int end, int k) {
		for(AColGroup g : npa) // all non aggregate groups task
			noTmpNoAggGroups(that, ret, pool, ct, tasks, start, end, g, k);

		for(int off = 0; off < s; off++) {
			// all pre-aggregate group tasks
			// s ensure that there is no more than k number of tasks.
			final int offT = off;
			tasks.add(pool.submit(() -> LMMWithPreAgg(pa, that, ret, start, end, 0, ct, offT, s, null)));
		}

		if(rowSums != null) // row sums task
			tasks.add(pool.submit(() -> rowSum(that, rowSums, start, end, 0, ct)));
	}

	private static void noTmpNoAggGroups(MatrixBlock that, MatrixBlock ret, ExecutorService pool, final int ct,
		final ArrayList<Future<?>> tasks, final int start, final int end, AColGroup g, int k) {
		final List<Future<MatrixBlock>> npaSubTask = new ArrayList<>();
		final int retNRow = ret.getNumRows();
		final int retNCol = ret.getNumColumns();
		if(retNCol < 1000000) {

			final int colBlockSize = Math.max(ct / Math.max(k, 2), 64000);

			for(int bloC = 0; bloC < ct; bloC += colBlockSize) {
				final int startC = bloC;
				final int endC = Math.min(bloC + colBlockSize, ct);
				npaSubTask.add(pool.submit(() -> {
					Timing t = new Timing();
					final double[] tmp = new double[retNRow * retNCol];
					final MatrixBlock tmpBlock = new MatrixBlock(retNRow, retNCol, tmp);
					g.leftMultByMatrixNoPreAgg(that, tmpBlock, start, end, startC, endC);
					LOG.debug("noPreAggTiming: " + t);
					return tmpBlock;
				}));
			}

			tasks.add(pool.submit(() -> addInPlaceFuture(ret, npaSubTask)));
		}
		else {
			tasks.add(pool.submit(() -> g.leftMultByMatrixNoPreAgg(that, ret, start, end, 0, ct)));
		}
	}

	private static Object addInPlaceFuture(MatrixBlock ret, List<Future<MatrixBlock>> npaSubTask) throws Exception {
		for(Future<MatrixBlock> f : npaSubTask)
			addInPlace(f.get(), ret);
		return null;
	}

	private static void LMMParallelTempOut(List<AColGroup> npa, List<APreAgg> pa, MatrixBlock that, MatrixBlock ret,
		double[] rowSums, boolean overlapping, int k, ExecutorService pool) throws Exception {

		final int rt = that.getNumRows();
		final int ct = that.getNumColumns();
		// perfect parallel over rows left.
		final int rowBlockSize = Math.max(rt / k, 1);
		final int threadsUsedOnRows = (int) Math.ceil((double) rt / rowBlockSize);
		k = Math.max(1, k / threadsUsedOnRows);
		// parallel over column blocks ... should be bigger than largest distinct.
		// final int colBlockSize = Math.max(ct, 1);
		final int s = Math.min(npa.size() + pa.size(), k);
		k = Math.max(1, k / s); //

		// We set it to minimum 4k
		final int colBlockSize = Math.max(ct / k, 64000);
		final int threadsUsedOnColBlocks = (int) Math.ceil((double) ct / colBlockSize);
		k = k / threadsUsedOnColBlocks;

		final ArrayList<Future<MatrixBlock>> tasks = new ArrayList<>();
		// allocate temp
		final int retCols = ret.getNumColumns();
		final int retRows = ret.getNumRows();
		for(int blo = 0; blo < rt; blo += rowBlockSize) {
			final int start = blo;
			final int end = Math.min(blo + rowBlockSize, rt);

			for(AColGroup g : npa) // all groups get their own task
				tasks.add(pool.submit(new LMMNoPreAggTask(g, that, retRows, retCols, start, end)));

			for(int off = 0; off < s; off++) { // only allocate k tasks at max
				final int offT = off;

				if(that.isInSparseFormat()) {
					tasks.add(pool.submit(new LMMPreAggTask(pa, that, retRows, retCols, start, end, 0, ct, offT, s, null)));
				}
				else {
					for(int bloC = 0; bloC < ct; bloC += colBlockSize) {
						final int startC = bloC;
						final int endC = Math.min(startC + colBlockSize, ct);
						tasks.add(pool
							.submit(new LMMPreAggTask(pa, that, retRows, retCols, start, end, startC, endC, offT, s, null)));
					}
				}
			}

			if(rowSums != null) // row sums task
				tasks.add(pool.submit(new LMMRowSums(that, start, end, rowSums)));
		}

		addInPlaceFuture(ret, tasks);
	}

	private static Object addInPlace(MatrixBlock a, MatrixBlock out) throws Exception {
		if(a != null) {
			final DenseBlock dba = a.getDenseBlock();
			final DenseBlock dbb = out.getDenseBlock();
			final int blocks = dba.numBlocks();
			for(int b = 0; b < blocks; b++) {
				final double[] av = dba.valuesAt(b);
				final double[] bv = dbb.valuesAt(b);
				final int len = av.length;
				for(int i = 0; i < len; i++) {
					bv[i] += av[i];
				}
			}
		}
		return null;
	}

	private static void LMMTaskExec(List<AColGroup> npa, List<APreAgg> pa, MatrixBlock that, MatrixBlock ret, int rl,
		int ru, double[] rowSums) throws Exception {
		final int cu = that.getNumColumns();
		if(npa.isEmpty() && pa.isEmpty()) {
			rowSum(that, rowSums, rl, ru, 0, cu);
			return;
		}
		for(int r = rl; r < ru; r += 4) {
			final int re = Math.min(r + 4, ru);
			// Process MMs.
			for(int i = 0; i < npa.size(); i++) {
				npa.get(i).leftMultByMatrixNoPreAgg(that, ret, r, re, 0, cu);
			}
			if(pa.size() > 0)
				LMMWithPreAgg(pa, that, ret, r, re, 0, cu, 0, 1, rowSums);
		}
	}

	private static void outerProduct(final double[] leftRowSum, final double[] rightColumnSum, final MatrixBlock result,
		int k) throws InterruptedException, ExecutionException {
		if(k > 1)
			outerProductParallel(leftRowSum, rightColumnSum, result, k);
		else
			outerProductSingleThread(leftRowSum, rightColumnSum, result);
	}

	private static void outerProductParallel(final double[] leftRowSum, final double[] rightColumnSum,
		final MatrixBlock result, int k) throws InterruptedException, ExecutionException {
		final ExecutorService pool = CommonThreadPool.get(k);
		try {
			for(Future<?> t : outerProductParallelTasks(leftRowSum, rightColumnSum, result, pool))
				t.get();
		}
		finally {
			pool.shutdown();
		}
	}

	private static void outerProductRange(final double[] leftRowSum, final double[] rightColumnSum,
		final MatrixBlock result, int rl, int ru, int cl, int cu) {
		if(result.getDenseBlock().isContiguous())
			outerProductRangeContiguous(leftRowSum, rightColumnSum, result.getDenseBlockValues(), rl, ru, cl, cu);
		else
			outerProductRangeGeneric(leftRowSum, rightColumnSum, result.getDenseBlock(), rl, ru, cl, cu);
	}

	private static void outerProductRangeContiguous(final double[] leftRowSum, final double[] rightColumnSum,
		final double[] result, int rl, int ru, int cl, int cu) {
		for(int row = rl; row < ru; row++) {
			final int offOut = rightColumnSum.length * row;
			final double vLeft = leftRowSum[row];
			if(vLeft != 0) {
				for(int col = cl; col < cu; col++) {
					result[offOut + col] += vLeft * rightColumnSum[col];
				}
			}
		}
	}

	private static void outerProductRangeGeneric(final double[] leftRowSum, final double[] rightColumnSum,
		final DenseBlock res, int rl, int ru, int cl, int cu) {
		for(int row = rl; row < ru; row++) {
			final int offOut = res.pos(row);
			final double[] result = res.values(row);
			final double vLeft = leftRowSum[row];
			if(vLeft != 0) {
				for(int col = cl; col < cu; col++) {
					result[offOut + col] += vLeft * rightColumnSum[col];
				}
			}
		}
	}

	private static void outerProductSingleThread(final double[] leftRowSum, final double[] rightColumnSum,
		MatrixBlock result) {
		final int blkz = 1024;
		for(int row = 0; row < leftRowSum.length; row += blkz) {
			final int rl = row;
			final int ru = Math.min(leftRowSum.length, row + blkz);
			final int colBz = outerProdGetColBz(blkz, row, rl, ru);

			for(int col = 0; col < rightColumnSum.length; col += colBz) {
				final int cl = col;
				final int cu = Math.min(rightColumnSum.length, col + colBz);
				outerProductRange(leftRowSum, rightColumnSum, result, rl, ru, cl, cu);
			}
		}
	}

	private static List<Future<?>> outerProductParallelTasks(final double[] leftRowSum, final double[] rightColumnSum,
		final MatrixBlock result, ExecutorService pool) {
		// windows of 1024 each
		final int blkz = 1024;
		final List<Future<?>> tasks = new ArrayList<>();
		for(int row = 0; row < leftRowSum.length; row += blkz) {
			final int rl = row;
			final int ru = Math.min(leftRowSum.length, row + blkz);
			final int colBz = outerProdGetColBz(blkz, row, rl, ru);

			for(int col = 0; col < rightColumnSum.length; col += colBz) {
				final int cl = col;
				final int cu = Math.min(rightColumnSum.length, col + colBz);
				tasks.add(pool.submit(() -> {
					outerProductRange(leftRowSum, rightColumnSum, result, rl, ru, cl, cu);
				}));
			}
		}
		return tasks;
	}

	private static int outerProdGetColBz(final int blkz, int row, final int rl, final int ru) {
		final int colBz;
		if(ru < row + blkz)
			colBz = 1024 * 1024 - ((ru - rl) * 1024) + 1024;
		else
			colBz = blkz;
		return colBz;
	}

	private static void outerProductWithScaling(final double[] leftRowSum, final double[] rightColumnSum,
		final int scaling, final MatrixBlock result) {
		if(result.getDenseBlock().isContiguous())
			outerProductWithScalingContiguous(leftRowSum, rightColumnSum, scaling, result.getDenseBlockValues());
		else
			outerProductWithScalingGeneric(leftRowSum, rightColumnSum, scaling, result.getDenseBlock());
	}

	private static void outerProductWithScalingContiguous(final double[] leftRowSum, final double[] rightColumnSum,
		final int scaling, final double[] result) {
		for(int row = 0; row < leftRowSum.length; row++) {
			final int offOut = rightColumnSum.length * row;
			final double vLeft = leftRowSum[row] * scaling;
			for(int col = 0; col < rightColumnSum.length; col++) {
				result[offOut + col] += vLeft * rightColumnSum[col];
			}
		}
	}

	private static void outerProductWithScalingGeneric(final double[] leftRowSum, final double[] rightColumnSum,
		final int scaling, final DenseBlock res) {
		for(int row = 0; row < leftRowSum.length; row++) {
			final int offOut = res.pos(row);
			final double[] result = res.values(row);
			final double vLeft = leftRowSum[row] * scaling;
			for(int col = 0; col < rightColumnSum.length; col++) {
				result[offOut + col] += vLeft * rightColumnSum[col];
			}
		}
	}

	private static void LMMWithPreAgg(List<APreAgg> preAggCGs, MatrixBlock that, MatrixBlock ret, int rl, int ru, int cl,
		int cu, int off, int skip, double[] rowSums) {
		try {
			if(!that.isInSparseFormat())
				LMMWithPreAggDense(preAggCGs, that, ret, rl, ru, cl, cu, off, skip, rowSums);
			else
				LMMWithPreAggSparse(preAggCGs, that, ret, rl, ru, cl, cu, off, skip, rowSums);
		}
		catch(Exception e) {
			throw new RuntimeException("Failed LLM pre aggregate", e);
		}
	}

	private static void LMMWithPreAggSparse(List<APreAgg> preAggCGs, MatrixBlock that, MatrixBlock ret, int rl, int ru,
		int cl, int cu, int off, int skip, double[] rowSum) throws Exception {

		final MatrixBlock preA = new MatrixBlock();
		final MatrixBlock fTmp = new MatrixBlock();
		final SparseBlock sb = that.getSparseBlock();
		for(int j = off; j < preAggCGs.size(); j += skip) { // selected column groups for this thread.
			final int nCol = preAggCGs.get(j).getNumCols();
			final int nVal = preAggCGs.get(j).getNumValues();
			final APreAgg g = preAggCGs.get(j);

			for(int r = rl; r < ru; r++) {
				preAggSparseRow(that, ret, cl, cu, preA, fTmp, sb, nCol, nVal, g, r);
			}
		}

		if(rowSum != null)
			rowSumSparse(that.getSparseBlock(), rowSum, rl, ru, cl, cu);

	}

	private static void preAggSparseRow(MatrixBlock that, MatrixBlock ret, int cl, int cu, final MatrixBlock preA,
		final MatrixBlock fTmp, final SparseBlock sb, final int nCol, final int nVal, final APreAgg g, int r) {
		if(sb.isEmpty(r))
			return;
		final int rcu = r + 1;

		// if(sb.size(r) * nCol < sb.size(r) + (long) nCol * nVal) {
		// g.leftMultByMatrixNoPreAgg(that, ret, r, rcu, cl, cu);
		// }
		// else {
		if(!preA.isAllocated()) {
			preA.reset(1, nVal);
			preA.allocateDenseBlock();
		}
		else
			preA.reset(1, nVal);
		allocateOrResetTmpRes(ret, fTmp, 1);

		final double[] preAV = preA.getDenseBlockValues();
		preA.setNonZeros(g.getPreAggregateSize());
		fTmp.setNonZeros(1);
		g.preAggregateSparse(sb, preAV, r, rcu, cl, cu);
		g.mmWithDictionary(preA, fTmp, ret, 1, r, rcu);
		// }
	}

	private static void allocateOrResetTmpRes(final MatrixBlock ret, final MatrixBlock fTmp, int rows) {
		if(!fTmp.isAllocated()) {
			fTmp.reset(rows, ret.getNumColumns());
			fTmp.allocateDenseBlock();
		}
		else
			fTmp.reset(rows, ret.getNumColumns());
	}

	private static void LMMWithPreAggDense(final List<APreAgg> preAggCGs, final MatrixBlock that, final MatrixBlock ret,
		final int rl, final int ru, final int cl, final int cu, final int off, final int skip, final double[] rowSum)
		throws InterruptedException, ExecutionException {
		// Timing t = new Timing();

		// ExecutorService pool = CommonThreadPool.get(k);
		/** The column block size for preAggregating column groups */
		// final int colBZ = 1024;
		final int colBZ = 2048;
		// final int colBZ = Math.max(1024, lc/2);
		// The number of rows to process together
		final int rowBlockSize = 4;
		// The number of column groups to process together
		// the value should ideally be set so that the colGroups fits into cache together with a row block.
		// currently we only try to avoid having a dangling small number of column groups in the last block.
		// final int colGroupBlocking = preAggCGs.size();// % 16 < 4 ? 20 : 16;
		// final int colGroupBlocking = 8;
		final int colGroupBlocking = 4;
		final int nColGroups = preAggCGs.size();

		// Allocate pre Aggregate Array List
		final double[][] preAgg = new double[colGroupBlocking][];

		// Allocate temporary Result matrix
		// guaranteed to be large enough for all groups
		MatrixBlock tmpRes = new MatrixBlock();

		// For each row block
		for(int rlt = rl; rlt < ru; rlt += rowBlockSize) {
			final int rut = Math.min(rlt + rowBlockSize, ru);
			// For each column group block
			for(int gl = off; gl < nColGroups; gl += colGroupBlocking * skip) {
				final int gu = Math.min(gl + (colGroupBlocking * skip), nColGroups);
				// For each column group in the current block allocate the pre aggregate array.
				// or reset the pre aggregate.
				for(int j = gl, p = 0; j < gu; j += skip, p++)
					preAllocate(preAggCGs, j, rut, rlt, preAgg, p);

				for(int clt = cl; clt < cu; clt += colBZ) {
					final int cut = Math.min(clt + colBZ, cu);
					for(int j = gl, p = 0; j < gu; j += skip, p++)
						preAggregate(that, ret, preAggCGs, rut, rlt, clt, cut, j, preAgg, p);
					if(gu == nColGroups)
						rowSum(that, rowSum, rlt, rut, clt, cut);
				}

				// Multiply out the PreAggregate to the output matrix.
				for(int j = gl, p = 0; j < gu; j += skip, p++) {
					final APreAgg cg = preAggCGs.get(j);
					if(cg.getDictionary() instanceof AIdentityDictionary)
						continue;

					allocateOrResetTmpRes(ret, tmpRes, rowBlockSize);
					postMultiply(ret, tmpRes, preAgg, p, cg, rut, rlt);
				}
			}
		}

		// LOG.debug("SingleCallLMMTime: " + t.stop());
	}

	private static void preAllocate(List<APreAgg> preAggCGs, int j, int rut, int rlt, double[][] preAgg, int p) {
		final APreAgg cg = preAggCGs.get(j);
		if(cg.getDictionary() instanceof AIdentityDictionary)
			return;
		final int preAggNCol = cg.getPreAggregateSize();

		final int len = (rut - rlt) * preAggNCol;
		if(preAgg[p] == null || preAgg[p].length < len)
			preAgg[p] = new double[len];
		else
			Arrays.fill(preAgg[p], 0, (rut - rlt) * preAggNCol, 0);
	}

	private static void preAggregate(MatrixBlock that, MatrixBlock ret, List<APreAgg> preAggCGs, int rut, int rlt,
		int clt, int cut, int j, double[][] preAgg, int p) {
		final APreAgg cg = preAggCGs.get(j);
		if(cg.getDictionary() instanceof IdentityDictionary)
			cg.leftMMIdentityPreAggregateDense(that, ret, rlt, rut, clt, cut);
		else
			cg.preAggregateDense(that, preAgg[p], rlt, rut, clt, cut);
	}

	private static void postMultiply(MatrixBlock ret, MatrixBlock tmpRes, double[][] preAgg, int p, APreAgg cg, int rut,
		int rlt) {
		final int preAggNCol = cg.getPreAggregateSize();
		final MatrixBlock preAggThis = new MatrixBlock((rut - rlt), preAggNCol, preAgg[p]);
		cg.mmWithDictionary(preAggThis, tmpRes, ret, 1, rlt, rut);
	}

	public static double[] rowSum(MatrixBlock mb, int rl, int ru, int cl, int cu) {
		double[] ret = new double[ru];
		rowSum(mb, ret, rl, ru, cl, cu);
		return ret;
	}

	private static void rowSum(MatrixBlock mb, double[] rowSum, int rl, int ru, int cl, int cu) {
		if(mb.isEmpty())
			throw new DMLCompressionException("Invalid empty block to rowsum");
		else if(rowSum == null) // no sum to make since the rowSum result is null.
			return;
		else if(mb.isInSparseFormat())
			rowSumSparse(mb.getSparseBlock(), rowSum, rl, ru, cl, cu);
		else
			rowSumDense(mb, rowSum, rl, ru, cl, cu);
	}

	private static void rowSumSparse(SparseBlock sb, double[] rowSum, int rl, int ru, int cl, int cu) {
		for(int i = rl; i < ru; i++)
			rowSumSparseSingleRow(sb, rowSum, cl, cu, i);
	}

	private static void rowSumSparseSingleRow(SparseBlock sb, double[] rowSum, int cl, int cu, int i) {
		if(sb.isEmpty(i))
			return;
		final int apos = sb.pos(i);
		final int alen = sb.size(i) + apos;
		final double[] aval = sb.values(i);
		final int[] aix = sb.indexes(i);
		int j = apos;
		while(j < alen && aix[j] < cl)
			j++;
		if(aix[alen - 1] < cu)
			while(j < alen)
				rowSum[i] += aval[j++];
		else
			while(j < alen && aix[j] < cu)
				rowSum[i] += aval[j++];
	}

	private static void rowSumDense(MatrixBlock that, double[] rowSum, int rl, int ru, int cl, int cu) {

		final DenseBlock db = that.getDenseBlock();
		if(db.isContiguous()) {
			final double[] thatV = db.values(0);
			for(int r = rl; r < ru; r++)
				rowSumDenseSingleRow(rowSum, cl, cu, db, thatV, r);
		}
		else {
			for(int r = rl; r < ru; r++) {
				final double[] thatV = db.values(r);
				rowSumDenseSingleRow(rowSum, cl, cu, db, thatV, r);
			}
		}

	}

	private static void rowSumDenseSingleRow(double[] rowSum, int cl, int cu, final DenseBlock db, final double[] thatV,
		int r) {
		final int rowOff = db.pos(r);
		double tmp = 0;
		for(int c = rowOff + cl; c < rowOff + cu; c++)
			tmp += thatV[c];
		rowSum[r] += tmp;
	}

	private static class LMMPreAggTask implements Callable<MatrixBlock> {
		private final List<APreAgg> _pa;
		private final MatrixBlock _that;
		private final int _retR;
		private final int _retC;
		private final int _rl;
		private final int _ru;
		private final int _cl;
		private final int _cu;
		private final double[] _rowSums;
		private final int _off;
		private final int _skip;

		protected LMMPreAggTask(List<APreAgg> pa, MatrixBlock that, int retR, int retC, int rl, int ru, int cl, int cu,
			int off, int skip, double[] rowSums) {
			_pa = pa;
			_that = that;
			_retR = retR;
			_retC = retC;
			_rl = rl;
			_ru = ru;
			_cl = cl;
			_cu = cu;
			_rowSums = rowSums;
			_off = off;
			_skip = skip;
		}

		@Override
		public MatrixBlock call() throws Exception {
			final double[] tmpArr = new double[_retR * _retC];
			MatrixBlock _ret = new MatrixBlock(_retR, _retC, tmpArr);
			LMMWithPreAgg(_pa, _that, _ret, _rl, _ru, _cl, _cu, _off, _skip, _rowSums);
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

		@Override
		public MatrixBlock call() throws Exception {
			_cg.leftMultByMatrixNoPreAgg(_that, _ret, _rl, _ru, 0, _that.getNumColumns());
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
		public MatrixBlock call() throws Exception {
			if(_that.isInSparseFormat())
				rowSumSparse(_that.getSparseBlock(), _rowSums, _rl, _ru, 0, _that.getNumColumns());
			else
				rowSumDense(_that, _rowSums, _rl, _ru, 0, _that.getNumColumns());
			return null;
		}
	}
}
