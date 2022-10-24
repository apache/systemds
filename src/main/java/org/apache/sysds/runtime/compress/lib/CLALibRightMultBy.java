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
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDC;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.utils.DMLCompressionStatistics;

public class CLALibRightMultBy {
	private static final Log LOG = LogFactory.getLog(CLALibRightMultBy.class.getName());

	public static MatrixBlock rightMultByMatrix(CompressedMatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int k) {
		final boolean allowOverlap = ConfigurationManager.getDMLConfig()
			.getBooleanValue(DMLConfig.COMPRESSED_OVERLAPPING);
		return rightMultByMatrix(m1, m2, ret, k, allowOverlap);
	}

	public static MatrixBlock rightMultByMatrix(CompressedMatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int k,
		boolean allowOverlap) {

		final int rr = m1.getNumRows();
		final int rc = m2.getNumColumns();

		if(m1.isEmpty() || m2.isEmpty()) {
			LOG.trace("Empty right multiply");
			if(ret == null)
				ret = new MatrixBlock(rr, rc, 0);
			else
				ret.reset(rr, rc, 0);
			return ret;
		}
		else {
			if(m2 instanceof CompressedMatrixBlock)
				m2 = ((CompressedMatrixBlock) m2).getUncompressed("Uncompressed right side of right MM", k);

			if(!allowOverlap) {
				LOG.trace("Overlapping output not allowed in call to Right MM");
				return RMM(m1, m2, k);
			}

			final CompressedMatrixBlock retC = RMMOverlapping(m1, m2, k);
			// final double cs = retC.getInMemorySize();
			// final double us = MatrixBlock.estimateSizeDenseInMemory(rr, rc);
			// if(cs > us)
			// return retC.getUncompressed("Overlapping rep to big: " + cs + " vs uncompressed " + us);
			// else
			if(retC.isEmpty())
				return retC;
			else {
				if(retC.isOverlapping())
					retC.setNonZeros((long) rr * rc); // set non zeros to fully dense in case of overlapping.
				else
					retC.recomputeNonZeros(); // recompute if non overlapping compressed out.
				return retC;
			}
		}

	}

	private static CompressedMatrixBlock RMMOverlapping(CompressedMatrixBlock m1, MatrixBlock that, int k) {
		final int rl = m1.getNumRows();
		final int cr = that.getNumColumns();
		final int rr = that.getNumRows(); // shared dim
		final List<AColGroup> colGroups = m1.getColGroups();
		final List<AColGroup> retCg = new ArrayList<>();
		final CompressedMatrixBlock ret = new CompressedMatrixBlock(rl, cr);

		final boolean shouldFilter = CLALibUtils.shouldPreFilter(colGroups);

		double[] constV = shouldFilter ? new double[rr] : null;
		final List<AColGroup> filteredGroups = CLALibUtils.filterGroups(colGroups, constV);
		if(colGroups == filteredGroups)
			constV = null;

		if(k == 1)
			RMMSingle(filteredGroups, that, retCg);
		else
			RMMParallel(filteredGroups, that, retCg, k);

		if(constV != null) {
			final MatrixBlock cb = new MatrixBlock(1, constV.length, constV);
			final MatrixBlock cbRet = new MatrixBlock(1, that.getNumColumns(), false);
			LibMatrixMult.matrixMult(cb, that, cbRet);
			if(!cbRet.isEmpty())
				addConstant(cbRet, retCg);
		}

		ret.allocateColGroupList(retCg);

		if(retCg.size() > 1)
			ret.setOverlapping(true);

		CLALibUtils.addEmptyColumn(retCg, cr);

		return ret;
	}

	private static void addConstant(MatrixBlock constantRow, List<AColGroup> out) {
		final int nCol = constantRow.getNumColumns();
		int bestCandidate = -1;
		int bestCandidateValuesSize = Integer.MAX_VALUE;
		for(int i = 0; i < out.size(); i++) {
			AColGroup g = out.get(i);
			if(g instanceof ColGroupDDC && g.getNumCols() == nCol && g.getNumValues() < bestCandidateValuesSize)
				bestCandidate = i;
		}

		constantRow.sparseToDense();

		if(bestCandidate != -1) {
			AColGroup bc = out.get(bestCandidate);
			out.remove(bestCandidate);
			AColGroup ng = bc.binaryRowOpRight(new BinaryOperator(Plus.getPlusFnObject(), 1),
				constantRow.getDenseBlockValues(), true);
			out.add(ng);
		}
		else
			out.add(ColGroupConst.create(constantRow.getDenseBlockValues()));
	}

	private static MatrixBlock RMM(CompressedMatrixBlock m1, MatrixBlock that, int k) {
		// this version returns a decompressed result.
		final int rl = m1.getNumRows();
		final int cr = that.getNumColumns();
		final int rr = that.getNumRows(); // shared dim
		final List<AColGroup> colGroups = m1.getColGroups();
		final List<AColGroup> retCg = new ArrayList<>();

		final boolean shouldFilter = CLALibUtils.shouldPreFilter(colGroups);

		// start allocation of output.
		MatrixBlock ret = new MatrixBlock(rl, cr, false);
		final Future<MatrixBlock> f = ret.allocateBlockAsync();

		double[] constV = shouldFilter ? new double[rr] : null;
		final List<AColGroup> filteredGroups = CLALibUtils.filterGroups(colGroups, constV);
		if(colGroups == filteredGroups)
			constV = null;

		if(k == 1)
			RMMSingle(filteredGroups, that, retCg);
		else
			RMMParallel(filteredGroups, that, retCg, k);

		if(constV != null) {
			MatrixBlock constVMB = new MatrixBlock(1, constV.length, constV);
			MatrixBlock mmTemp = new MatrixBlock(1, cr, false);
			LibMatrixMult.matrixMult(constVMB, that, mmTemp);
			constV = mmTemp.isEmpty() ? null : mmTemp.getDenseBlockValues();
		}

		final Timing time = new Timing(true);

		ret = asyncRet(f);
		CLALibDecompress.decompressDenseMultiThread(ret, retCg, constV, 0, k);

		if(DMLScript.STATISTICS) {
			final double t = time.stop();
			DMLCompressionStatistics.addDecompressTime(t, k);
		}

		return ret;
	}

	private static <T> T asyncRet(Future<T> in) {
		try {
			return in.get();
		}
		catch(Exception e) {
			throw new DMLRuntimeException(e);
		}
	}

	private static boolean RMMSingle(List<AColGroup> filteredGroups, MatrixBlock that, List<AColGroup> retCg) {
		boolean containsNull = false;
		final int[] allCols = Util.genColsIndices(that.getNumColumns());
		for(AColGroup g : filteredGroups) {
			AColGroup retG = g.rightMultByMatrix(that, allCols);
			if(retG != null)
				retCg.add(retG);
			else
				containsNull = true;
		}
		return containsNull;
	}

	private static boolean RMMParallel(List<AColGroup> filteredGroups, MatrixBlock that, List<AColGroup> retCg, int k) {
		final ExecutorService pool = CommonThreadPool.get(k);
		boolean containsNull = false;
		try {
			int[] allCols = Util.genColsIndices(that.getNumColumns());
			List<Callable<AColGroup>> tasks = new ArrayList<>(filteredGroups.size());
			for(AColGroup g : filteredGroups)
				tasks.add(new RightMatrixMultTask(g, that, allCols));
			for(Future<AColGroup> fg : pool.invokeAll(tasks)) {
				AColGroup g = fg.get();
				if(g != null)
					retCg.add(g);
				else
					containsNull = true;
			}
		}
		catch(InterruptedException | ExecutionException e) {
			throw new DMLRuntimeException(e);
		}
		pool.shutdown();
		return containsNull;
	}

	private static class RightMatrixMultTask implements Callable<AColGroup> {
		private final AColGroup _colGroup;
		private final MatrixBlock _b;
		private final int[] _allCols;

		protected RightMatrixMultTask(AColGroup colGroup, MatrixBlock b, int[] allCols) {
			_colGroup = colGroup;
			_b = b;
			_allCols = allCols;
		}

		@Override
		public AColGroup call() {
			try {
				return _colGroup.rightMultByMatrix(_b, _allCols);
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
		}
	}
}
