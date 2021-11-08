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
import org.apache.sysds.runtime.compress.colgroup.AColGroupValue;
import org.apache.sysds.runtime.compress.colgroup.APreAgg;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
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

		// fall back solution?
		// MatrixBlock leftUc = left.getUncompressed();
		// leftMultByMatrixTransposed(right, leftUc, ret, k);

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
		ret = leftMultByMatrix(right.getColGroups(), left, ret, k, right.isOverlapping());
		return ret;
	}

	public static void leftMultByTransposeSelf(CompressedMatrixBlock cmb, MatrixBlock ret, int k) {
		// final boolean overlapping = cmb.isOverlapping();
		final List<AColGroup> groups = cmb.getColGroups();
		final int numColumns = cmb.getNumColumns();
		final int numRows = cmb.getNumRows();
		final boolean containsSDC = CLALibUtils.containsSDCOrConst(groups);
		final double[] constV = containsSDC ? new double[numColumns] : null;
		final List<AColGroup> filteredGroups = CLALibUtils.filterGroups(groups, constV);

		// TODO add parallel again
		tsmmColGroups(filteredGroups, ret, numRows);

		if(constV != null)
			addCorrectionLayer(filteredGroups, ret, numRows, numColumns, constV);

		long nnz = LibMatrixMult.copyUpperToLowerTriangle(ret);
		ret.setNonZeros(nnz);
		ret.examSparsity();
	}

	private static void addCorrectionLayer(List<AColGroup> filteredGroups, MatrixBlock result, int nRows, int nCols,
		double[] constV) {
		final double[] retV = result.getDenseBlockValues();
		final double[] filteredColSum = getColSum(filteredGroups, nCols, nRows);
		outerProductUpperTriangle(constV, filteredColSum, retV);
		outerProductUpperTriangleWithScaling(filteredColSum, constV, nRows, retV);
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

		final boolean containsRight = CLALibUtils.containsSDCOrConst(rightCG);
		double[] cR = containsRight ? new double[cr] : null;
		final List<AColGroup> fRight = CLALibUtils.filterGroups(rightCG, cR);

		final boolean containsLeft = CLALibUtils.containsSDCOrConst(leftCG);
		double[] cL = containsLeft ? new double[rl] : null;
		final List<AColGroup> fLeft = CLALibUtils.filterGroups(leftCG, cL);

		for(int i = 0; i < fRight.size(); i++)
			for(int j = 0; j < fLeft.size(); j++)
				fRight.get(i).leftMultByAColGroup(fLeft.get(j), ret);

		double[] retV = ret.getDenseBlockValues();
		if(containsLeft && containsRight)
			// if both -- multiply the left and right vectors scaling by number of shared dim
			outerProductWithScaling(cL, cR, sd, retV);
		if(containsLeft) // if left -- multiply left with right sum
			outerProduct(cL, getColSum(fRight, cr, sd), retV);
		if(containsRight)// if right -- multiply right with left sum
			outerProduct(getColSum(fLeft, rl, sd), cR, retV);
		ret.recomputeNonZeros();

		return ret;
	}

	private static void tsmmColGroups(List<AColGroup> filteredGroups, MatrixBlock ret, int nRows) {
		for(int i = 0; i < filteredGroups.size(); i++) {
			final AColGroup g = filteredGroups.get(i);
			g.tsmm(ret, nRows);
			for(int j = i + 1; j < filteredGroups.size(); j++) {
				final AColGroup h = filteredGroups.get(j);
				g.tsmmAColGroup(h, ret);
			}
		}
	}

	private static void outerProductUpperTriangle(final double[] leftRowSum, final double[] rightColumnSum,
		final double[] result) {
		for(int row = 0; row < leftRowSum.length; row++) {
			final int offOut = rightColumnSum.length * row;
			final double vLeft = leftRowSum[row];
			for(int col = row; col < rightColumnSum.length; col++) {
				result[offOut + col] += vLeft * rightColumnSum[col];
			}
		}
	}

	private static void outerProductUpperTriangleWithScaling(final double[] leftRowSum, final double[] rightColumnSum,
		final int scale, final double[] result) {
		// note this scaling is a bit different since it is encapsulating two scalar multiplications via an addition in
		// the outer loop.
		for(int row = 0; row < leftRowSum.length; row++) {
			final int offOut = rightColumnSum.length * row;
			final double vLeft = leftRowSum[row] + rightColumnSum[row] * scale;
			for(int col = row; col < rightColumnSum.length; col++) {
				result[offOut + col] += vLeft * rightColumnSum[col];
			}
		}
	}

	private static MatrixBlock leftMultByMatrix(List<AColGroup> colGroups, MatrixBlock that, MatrixBlock ret, int k,
		boolean overlapping) {

		if(that.isEmpty()) {
			ret.setNonZeros(0);
			return ret;
		}

		final int numColumnsOut = ret.getNumColumns();
		final boolean containsSDC = CLALibUtils.containsSDCOrConst(colGroups);
		final int lr = that.getNumRows();

		// a constant colgroup summing the default values.
		double[] constV = containsSDC ? new double[numColumnsOut] : null;
		final List<AColGroup> filteredGroups = CLALibUtils.filterGroups(colGroups, constV);
		if(colGroups == filteredGroups)
			constV = null;
		double[] rowSums;

		if(!filteredGroups.isEmpty()) {
			if(k == 1)
				rowSums = leftMultByMatrixPrimitive(filteredGroups, that, ret, 0, lr, containsSDC ? new double[lr] : null);
			else
				rowSums = leftMultByMatrixParallel(filteredGroups, that, ret, containsSDC, overlapping, k);
		}
		else if(constV != null)
			rowSums = that.rowSum(k).getDenseBlockValues();
		else
			rowSums = null;

		// add the correction layer for the subtracted common values.
		if(rowSums != null && constV != null) {
			ret.sparseToDense();
			outerProduct(rowSums, constV, ret.getDenseBlockValues());
		}

		ret.recomputeNonZeros();
		return ret;
	}

	private static double[] leftMultByMatrixParallel(List<AColGroup> filteredGroups, MatrixBlock that, MatrixBlock ret,
		boolean calculateRowSums, boolean overlapping, int k) {
		LOG.debug("Parallel left matrix multiplication");
		try {
			final ExecutorService pool = CommonThreadPool.get(k);
			final ArrayList<Callable<MatrixBlock>> tasks = new ArrayList<>();
			final int rl = that.getNumRows();
			final int rowBlockSize = rl <= k ? 1 : Math.min(Math.max(rl / k * 2, 1), 8);
			final double[] rowSums = calculateRowSums ? new double[rl] : null;
			final int numberSplits = Math.max((k / (rl / rowBlockSize)), 1);

			if(numberSplits == 1) {
				// no need to handle overlapping here, since outputs are in distinct locations
				for(int blo = 0; blo < rl; blo += rowBlockSize)
					tasks.add(new LeftMatrixColGroupMultTask(filteredGroups, that, ret, blo,
						Math.min(blo + rowBlockSize, rl), rowSums));

				for(Future<MatrixBlock> future : pool.invokeAll(tasks))
					future.get();
			}
			else {
				final List<List<AColGroup>> split = split(filteredGroups, numberSplits);
				final boolean useTmp = overlapping && filteredGroups.size() > 1;

				for(int blo = 0; blo < rl; blo += rowBlockSize) {
					final int start = blo;
					final int end = Math.min(blo + rowBlockSize, rl);
					for(int i = 0; i < split.size(); i++) {
						List<AColGroup> gr = split.get(i);
						// The first thread also have the responsibility to calculate the som of the left hand side.
						final MatrixBlock tmpRet = useTmp ? new MatrixBlock(rl, ret.getNumColumns(), false) : ret;
						if(tmpRet.getDenseBlock() == null)
							tmpRet.allocateDenseBlock();
						if(i == 0)
							tasks.add(new LeftMatrixColGroupMultTask(gr, that, tmpRet, start, end, rowSums));
						else
							tasks.add(new LeftMatrixColGroupMultTask(gr, that, tmpRet, start, end, null));
					}
				}
				if(useTmp) {
					BinaryOperator op = new BinaryOperator(Plus.getPlusFnObject());
					for(Future<MatrixBlock> future : pool.invokeAll(tasks))
						ret.binaryOperationsInPlace(op, future.get());
				}
				else
					for(Future<MatrixBlock> future : pool.invokeAll(tasks))
						future.get();
			}
			pool.shutdown();
			return rowSums;
		}
		catch(InterruptedException | ExecutionException e) {
			throw new DMLRuntimeException(e);
		}
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

	private static class LeftMatrixColGroupMultTask implements Callable<MatrixBlock> {
		private final List<AColGroup> _groups;
		private final MatrixBlock _that;
		private final MatrixBlock _ret;
		private final int _rl;
		private final int _ru;
		private final double[] _rowSums;

		protected LeftMatrixColGroupMultTask(List<AColGroup> groups, MatrixBlock that, MatrixBlock ret, int rl, int ru,
			double[] rowSums) {
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

	private static double[] leftMultByMatrixPrimitive(List<AColGroup> colGroups, MatrixBlock that, MatrixBlock ret,
		int rl, int ru, double[] rowSums) {
		if(that.isInSparseFormat())
			leftMultByMatrixPrimitiveSparse(colGroups, that, ret, rl, ru, rowSums);
		else
			leftMultByMatrixPrimitiveDense(colGroups, that, ret, rl, ru, rowSums);
		ret.setNonZeros(ret.getNumRows() * ret.getNumColumns()); // always assume dense, this is corrected later
		return rowSums;
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
		final List<APreAgg> preAggCGs = preFilterAndMultiply(colGroups, that, ret, rl, ru);

		// The number of rows to process together
		final int rowBlockSize = 1;
		// The number of column groups to process together
		// the value should ideally be set so that the colGroups fits into cache together with a row block.
		// currently we only try to avoid having a dangling small number of column groups in the last block.
		final int colGroupBlocking = preAggCGs.size() % 16 < 4 ? 20 : 16;

		// Allocate pre Aggregate Array List
		final MatrixBlock[] preAgg = populatePreAggregate(colGroupBlocking);

		// Allocate temporary Result matrix.
		final MatrixBlock tmpRes = new MatrixBlock(rowBlockSize, numColsOut, false);

		final int lc = that.getNumColumns();

		// For each column group block
		for(int g = 0; g < preAggCGs.size(); g += colGroupBlocking) {
			final int gEnd = Math.min(g + colGroupBlocking, preAggCGs.size());

			// For each column group in the current block allocate the preaggregate array.
			for(int j = g; j < gEnd && j < preAggCGs.size(); j++) {
				AColGroupValue cg = preAggCGs.get(j);
				int nVals = cg.getNumValues();
				preAgg[j % colGroupBlocking].reset(rowBlockSize, nVals, false);
			}

			int colBlockSize = 32000;

			// For each row block
			for(int h = rl; h < ru; h += rowBlockSize) {
				// For each column block
				final int rowUpper = Math.min(h + rowBlockSize, ru);
				for(int i = 0; i < lc; i += colBlockSize) {
					final int colUpper = Math.min(i + colBlockSize, lc);
					// Pre Aggregate each column group in block
					for(int j = g; j < gEnd && j < preAggCGs.size(); j++) {
						preAggCGs.get(j).preAggregateDense(that, preAgg[j % colGroupBlocking], h, rowUpper, i, colUpper);
					}
					if(rowSum != null) {
						final double[] thatV = that.getDenseBlockValues();
						for(int r = h; r < rowUpper; r++) {
							final int rowOff = r * lc;
							for(int c = rowOff + i; c < rowOff + colUpper; c++)
								rowSum[r] += thatV[c];
						}
					}
				}

				// Multiply out the preAggregate to the output matrix.
				for(int j = g; j < gEnd && j < preAggCGs.size(); j++) {
					AColGroupValue vj = preAggCGs.get(j);
					MatrixBlock preAggJ = preAgg[j % colGroupBlocking];
					preAggJ.recomputeNonZeros();
					tmpRes.reset(rowBlockSize, vj.getNumCols(), false);
					MatrixBlock tmp = vj.leftMultByPreAggregateMatrix(preAggJ, tmpRes);
					vj.addMatrixToResult(tmp, ret, h, Math.min(h + rowBlockSize, ru));
					preAggJ.reset();
				}
			}
		}

		if(preAggCGs.size() == 0 && rowSum != null) {
			final double[] thatV = that.getDenseBlockValues();
			for(int r = rl; r < ru; r++) {
				final int rowOff = r * lc;
				final int thatOffEnd = rowOff + lc;
				for(int c = rowOff; c < thatOffEnd; c++)
					rowSum[r] += thatV[c];
			}
		}
	}

	private static MatrixBlock[] populatePreAggregate(int colGroupBlocking) {
		final MatrixBlock[] preAgg = new MatrixBlock[colGroupBlocking];
		// populate the preAgg array.
		for(int j = 0; j < colGroupBlocking; j++) {
			MatrixBlock m = new MatrixBlock(1, 1, false);
			m.allocateDenseBlock();
			preAgg[j] = m;
		}
		return preAgg;
	}

	private static List<APreAgg> preFilterAndMultiply(List<AColGroup> colGroups, MatrixBlock that, MatrixBlock ret,
		int rl, int ru) {
		final List<APreAgg> ColGroupValues = new ArrayList<>(colGroups.size());
		for(int j = 0; j < colGroups.size(); j++) {
			AColGroup a = colGroups.get(j);
			if(a instanceof APreAgg)
				ColGroupValues.add((APreAgg) a);
			else
				a.leftMultByMatrix(that, ret, rl, ru);
		}
		Collections.sort(ColGroupValues, Comparator.comparing(AColGroup::getNumValues).reversed());
		return ColGroupValues;
	}

	private static double[] getColSum(List<AColGroup> groups, int nCols, int nRows) {
		return AColGroup.colSum(groups, new double[nCols], nRows);
	}
}
