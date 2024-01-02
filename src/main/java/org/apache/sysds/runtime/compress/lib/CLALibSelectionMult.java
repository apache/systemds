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

import java.util.List;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * This lib is responsible for selecting and extracting specific rows or columns from a compressed matrix.
 * 
 * The operation performed is like a left matrix multiplication where the left side only have max 1 non zero per row.
 * 
 */
public class CLALibSelectionMult {
	protected static final Log LOG = LogFactory.getLog(CLALibSelectionMult.class.getName());

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
		if(right.getNonZeros() <= -1)
			right.recomputeNonZeros();

		boolean sparseOut = right.getSparsity() < 0.3;
		ret.reset(left.getNumRows(), right.getNumColumns(), sparseOut);
		ret.allocateBlock();
		final List<AColGroup> preFilter = right.getColGroups();
		final boolean shouldFilter = CLALibUtils.shouldPreFilter(preFilter);
		if(shouldFilter) {
			final double[] constV = new double[ret.getNumColumns()];
			// final List<AColGroup> noPreAggGroups = new ArrayList<>();
			// final List<APreAgg> preAggGroups = new ArrayList<>();
			final List<AColGroup> morphed = CLALibUtils.filterGroups(preFilter, constV);

			if(sparseOut) {
				leftSparseSelection(morphed, left, ret, k);
				double[] rowSums = left.rowSum(k).getDenseBlockValues();
				outerProductSparse(rowSums, constV, ret);
			}
			else {
				leftDenseSelection(morphed, left, ret, k);
			}

		}
		else {
			if(sparseOut)
				leftSparseSelection(preFilter, left, ret, k);
			else
				leftDenseSelection(preFilter, left, ret, k);
		}

		ret.recomputeNonZeros(k);
		return ret;
	}

	private static void leftSparseSelection(List<AColGroup> right, MatrixBlock left, MatrixBlock ret, int k) {
		for(AColGroup g : right)
			g.sparseSelection(left, ret, 0, left.getNumRows());
		left.getSparseBlock().sort();
	}

	private static void leftDenseSelection(List<AColGroup> right, MatrixBlock left, MatrixBlock ret, int k) {
		throw new NotImplementedException();
	}

	private static void outerProductSparse(double[] rows, double[] cols, MatrixBlock ret) {
		SparseBlock sb = ret.getSparseBlock();

		IntArrayList skipCols = new IntArrayList();
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
