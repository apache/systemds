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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class CLALibTSMM {
	private static final Log LOG = LogFactory.getLog(CLALibTSMM.class.getName());

	/**
	 * Self left Matrix multiplication (tsmm)
	 * 
	 * t(x) %*% x
	 * 
	 * @param cmb Compressed matrix to multiply
	 * @param ret The output matrix to put the result into
	 * @param k   The parallelization degree allowed
	 */
	public static void leftMultByTransposeSelf(CompressedMatrixBlock cmb, MatrixBlock ret, int k) {
		final List<AColGroup> groups = cmb.getColGroups();
		final int numColumns = cmb.getNumColumns();
		final int numRows = cmb.getNumRows();
		final boolean shouldFilter = CLALibUtils.shouldPreFilter(groups);
		final boolean overlapping = cmb.isOverlapping();
		if(shouldFilter) {
			final double[] constV = new double[numColumns];
			final List<AColGroup> filteredGroups = CLALibUtils.filterGroups(groups, constV);
			tsmmColGroups(filteredGroups, ret, numRows, overlapping, k);
			addCorrectionLayer(filteredGroups, ret, numRows, numColumns, constV);
		}
		else
			tsmmColGroups(groups, ret, numRows, overlapping, k);

		ret.setNonZeros(LibMatrixMult.copyUpperToLowerTriangle(ret));
		ret.examSparsity();
	}

	private static void addCorrectionLayer(List<AColGroup> filteredGroups, MatrixBlock result, int nRows, int nCols,
		double[] constV) {
		final double[] retV = result.getDenseBlockValues();
		final double[] filteredColSum = CLALibUtils.getColSum(filteredGroups, nCols, nRows);
		addCorrectionLayer(constV, filteredColSum, nRows, retV);
	}

	public static void addCorrectionLayer(double[] constV, double[] correctedSum, int nRow, double[] ret) {
		outerProductUpperTriangle(constV, correctedSum, ret);
		outerProductUpperTriangleWithScaling(correctedSum, constV, nRow, ret);
	}

	private static void tsmmColGroups(List<AColGroup> groups, MatrixBlock ret, int nRows, boolean overlapping, int k) {
		if(k <= 1)
			tsmmColGroupsSingleThread(groups, ret, nRows);
		else if(overlapping)
			tsmmColGroupsMultiThreadOverlapping(groups, ret, nRows, k);
		else
			tsmmColGroupsMultiThread(groups, ret, nRows, k);
	}

	private static void tsmmColGroupsSingleThread(List<AColGroup> groups, MatrixBlock ret, int nRows) {
		for(int i = 0; i < groups.size(); i++) {
			final AColGroup g = groups.get(i);
			g.tsmm(ret, nRows); // self
			for(int j = i + 1; j < groups.size(); j++) {
				final AColGroup h = groups.get(j);
				g.tsmmAColGroup(h, ret); // all remaining others
			}
		}
	}

	private static void tsmmColGroupsMultiThreadOverlapping(List<AColGroup> groups, MatrixBlock ret, int nRows, int k) {
		LOG.warn("fallback to single threaded for now");
		tsmmColGroupsSingleThread(groups, ret, nRows);
	}

	private static void tsmmColGroupsMultiThread(List<AColGroup> groups, MatrixBlock ret, int nRows, int k) {
		final ExecutorService pool = CommonThreadPool.get(k);
		final ArrayList<Callable<MatrixBlock>> tasks = new ArrayList<>((groups.size() * (1 + groups.size())) / 2);
		for(int i = 0; i < groups.size(); i++) {
			final AColGroup g = groups.get(i);
			tasks.add(new TSMMTask(g, ret, nRows)); // self
			for(int j = i + 1; j < groups.size(); j++)
				tasks.add(new TSMMColGroupTask(g, groups.get(j), ret)); // all remaining others
		}

		try {
			for(Future<MatrixBlock> future : pool.invokeAll(tasks))
				future.get();
		}
		catch(InterruptedException | ExecutionException e) {
			pool.shutdown();
			throw new DMLRuntimeException(e);
		}
		pool.shutdown();
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

	private static class TSMMTask implements Callable<MatrixBlock> {
		private final AColGroup _g;
		private final MatrixBlock _ret;
		private final int _nRows;

		protected TSMMTask(AColGroup g, MatrixBlock ret, int nRows) {
			_g = g;
			_ret = ret;
			_nRows = nRows;
		}

		@Override
		public MatrixBlock call() {
			try {
				_g.tsmm(_ret, _nRows);
				return _ret;
			}
			catch(Exception e) {
				e.printStackTrace();
				throw new DMLRuntimeException(e);
			}
		}
	}

	private static class TSMMColGroupTask implements Callable<MatrixBlock> {
		private final AColGroup _g;
		private final AColGroup _h;
		private final MatrixBlock _ret;

		protected TSMMColGroupTask(AColGroup g, AColGroup h, MatrixBlock ret) {
			_g = g;
			_h = h;
			_ret = ret;
		}

		@Override
		public MatrixBlock call() {
			try {
				_g.tsmmAColGroup(_h, _ret);
				return _ret;
			}
			catch(Exception e) {
				e.printStackTrace();
				throw new DMLRuntimeException(e);
			}
		}
	}
}
