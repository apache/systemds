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
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUtils;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUtils.P;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

/**
 * This lib is responsible for selecting and extracting specific rows or columns from a compressed matrix.
 * 
 * The operation performed is like a left matrix multiplication where the left side only have max 1 non zero per row.
 * 
 */
public interface CLALibSelectionMult {
	public final Log LOG = LogFactory.getLog(CLALibSelectionMult.class.getName());

	/**
	 * Left selection where the left matrix is sparse with a max 1 non zero per row and that non zero is a 1.
	 * 
	 * @param right Right hand side compressed matrix
	 * @param left  Left hand side matrix
	 * @param ret   Output matrix to put the result into.
	 * @param k     The parallelization degree.
	 * @return The selected rows and columns of the input matrix
	 */
	public static MatrixBlock leftSelection(CompressedMatrixBlock right, MatrixBlock left, MatrixBlock ret, int k) {
		try {

			if(right.getNonZeros() <= -1)
				right.recomputeNonZeros();

			boolean sparseOut = right.getSparsity() < 0.3;
			ret = allocateReturn(right, left, ret, sparseOut);

			final List<AColGroup> preFilter = right.getColGroups();
			final boolean shouldFilter = CLALibUtils.shouldPreFilter(preFilter);
			if(shouldFilter)
				filteredLeftSelection(left, ret, k, sparseOut, preFilter);
			else
				normalLeftSelection(left, ret, k, sparseOut, preFilter);

			ret.recomputeNonZeros(k);

			return ret;
		}
		catch(Exception e) {
			throw new DMLCompressionException("Failed left selection Multiplication", e);
		}
	}

	/**
	 * Analyze if the given matrix is a selection matrix if on the left side of a matrix multiplication.
	 * 
	 * @param mb The given matrix that should be on the left side
	 * @return If it is selective
	 */
	public static boolean isSelectionMatrix(MatrixBlock mb) {
		// See if the input is potentially only containing one nonzero per row.
		if(mb.isEmpty())
			return false;
		else if(mb.getNonZeros() <= mb.getNumRows() && mb.isInSparseFormat()) {

			SparseBlock sb = mb.getSparseBlock();
			// verify every row only contain one 1 value.
			for(int i = 0; i < mb.getNumRows(); i++) {
				if(sb.isEmpty(i))
					continue;
				else if(sb.size(i) != 1)
					return false;
				else if(!(sb instanceof SparseBlockCSR)) {
					double[] values = sb.values(i);
					final int spos = sb.pos(i);
					final int sEnd = spos + sb.size(i);
					for(int j = spos; j < sEnd; j++) {
						if(values[j] != 1) {
							return false;
						}
					}
				}
			}
			if(sb instanceof SparseBlockCSR) {
				for(double d : sb.values(0))
					if(d != 1)
						return false;
			}

			return true;
		}
		return false;
	}

	private static MatrixBlock allocateReturn(CompressedMatrixBlock right, MatrixBlock left, MatrixBlock ret,
		boolean sparseOut) {
		if(ret == null)
			ret = new MatrixBlock();
		// sparseOut = false;
		ret.reset(left.getNumRows(), right.getNumColumns(), sparseOut);
		ret.allocateBlock();
		return ret;
	}

	private static void normalLeftSelection(MatrixBlock left, MatrixBlock ret, int k, boolean sparseOut,
		final List<AColGroup> preFilter) throws Exception {
		final int rowLeft = left.getNumRows();
		final boolean pointsNeeded = areSortedCoordinatesNeeded(preFilter);
		if(k <= 1 || rowLeft < 1000)
			leftSelectionSingleThread(preFilter, left, ret, rowLeft, pointsNeeded, sparseOut);
		else
			leftSelectionParallel(preFilter, left, ret, k, rowLeft, pointsNeeded, sparseOut);
	}

	private static void filteredLeftSelection(MatrixBlock left, MatrixBlock ret, int k, boolean sparseOut,
		final List<AColGroup> preFilter) throws Exception {
		final double[] constV = new double[ret.getNumColumns()];
		final List<AColGroup> morphed = CLALibUtils.filterGroups(preFilter, constV);
		normalLeftSelection(left, ret, k, sparseOut, morphed);
		double[] rowSums = left.rowSum(k).getDenseBlockValues();

		outerProduct(rowSums, constV, ret, sparseOut);
	}

	private static void leftSelectionSingleThread(List<AColGroup> right, MatrixBlock left, MatrixBlock ret,
		final int rowLeft, final boolean pointsNeeded, final boolean sparseOut) {
		P[] points = pointsNeeded ? ColGroupUtils.getSortedSelection(left.getSparseBlock(), 0, rowLeft) : null;
		for(AColGroup g : right)
			g.selectionMultiply(left, points, ret, 0, rowLeft);
		if(sparseOut)
			ret.getSparseBlock().sort();
	}

	private static void leftSelectionParallel(List<AColGroup> right, MatrixBlock left, MatrixBlock ret, int k,
		final int rowLeft, final boolean pointsNeeded, final boolean sparseOut)
		throws InterruptedException, ExecutionException {
		final ExecutorService pool = CommonThreadPool.get(k);
		try {

			List<Future<?>> tasks = new ArrayList<>();
			final int blkz = Math.max(rowLeft / k, 1000);
			for(int i = 0; i < rowLeft; i += blkz) {
				final int start = i;
				final int end = Math.min(rowLeft, i + blkz);
				P[] points = pointsNeeded ? ColGroupUtils.getSortedSelection(left.getSparseBlock(), start, end) : null;
				tasks.add(pool.submit(() -> {
					for(AColGroup g : right)
						g.selectionMultiply(left, points, ret, start, end);
					if(sparseOut) {
						SparseBlock sb = ret.getSparseBlock();
						for(int j = start; j < end; j++) {
							if(!sb.isEmpty(j))
								sb.sort(j);
						}
					}
				}));
			}

			for(Future<?> t : tasks)
				t.get();
		}
		finally {
			pool.shutdown();
		}
	}

	private static boolean areSortedCoordinatesNeeded(List<AColGroup> right) {
		for(AColGroup g : right) {
			if(g.getCompType() == CompressionType.SDC)
				return true;
		}
		return false;
	}

	private static void outerProduct(double[] rows, double[] cols, MatrixBlock ret, boolean sparse) {
		if(sparse)
			outerProductSparse(rows, cols, ret);
		else
			outerProductDense(rows, cols, ret);
	}

	private static void outerProductDense(double[] rows, double[] cols, MatrixBlock ret) {
		DenseBlock db = ret.getDenseBlock();
		for(int r = 0; r < rows.length; r++) {
			final double rv = rows[r];
			final double[] dbV = db.values(r);
			final int pos = db.pos(r);
			if(rv != 0)
				for(int c = 0; c < cols.length; c++)
					dbV[pos + c] += rv * cols[c];
		}
	}

	private static void outerProductSparse(double[] rows, double[] cols, MatrixBlock ret) {
		final SparseBlock sb = ret.getSparseBlock();

		final IntArrayList skipCols = new IntArrayList();
		for(int c = 0; c < cols.length; c++)
			if(cols[c] != 0)
				skipCols.appendValue(c);

		final int skipSz = skipCols.size();
		if(skipSz == 0)
			return;

		final int[] skipC = skipCols.extractValues();
		for(int r = 0; r < rows.length; r++) {
			final double rv = rows[r];
			if(rv != 0) {
				for(int ci = 0; ci < skipSz; ci++) {
					final int c = skipC[ci];
					sb.add(r, c, rv * cols[c]);
				}
			}
		}
	}
}
