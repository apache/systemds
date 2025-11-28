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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ASDC;
import org.apache.sysds.runtime.compress.colgroup.ASDCZero;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

public final class CLALibRightMultBy {
	private static final Log LOG = LogFactory.getLog(CLALibRightMultBy.class.getName());

	private CLALibRightMultBy() {
		// private constructor
	}

	public static MatrixBlock rightMultByMatrix(CompressedMatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int k) {
		final boolean allowOverlap = ConfigurationManager.getDMLConfig()
			.getBooleanValue(DMLConfig.COMPRESSED_OVERLAPPING);
		return rightMultByMatrix(m1, m2, ret, k, allowOverlap);
	}

	public static MatrixBlock rightMultByMatrix(CompressedMatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int k,
		boolean allowOverlap) {

		try {
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

				// if(betterIfDecompressed(m1)) {
				// 	// perform uncompressed multiplication.
				// 	return decompressingMatrixMult(m1, m2, k);
				// }

				if(!allowOverlap) {
					LOG.trace("Overlapping output not allowed in call to Right MM");
					return RMM(m1, m2, k);
				}

				final CompressedMatrixBlock retC = RMMOverlapping(m1, m2, k);

				if(retC.isEmpty())
					return retC;
				else {
					if(retC.isOverlapping())
						retC.setNonZeros((long) rr * rc); // set non zeros to fully dense in case of overlapping.
					else
						retC.recomputeNonZeros(k); // recompute if non overlapping compressed out.
					return retC;
				}
			}
		}
		catch(Exception e) {
			throw new RuntimeException("Failed Right MM", e);
		}
	}

	private static MatrixBlock decompressingMatrixMult(CompressedMatrixBlock m1, MatrixBlock m2, int k)
		throws Exception {
		final ExecutorService pool = CommonThreadPool.get(k);
		try {
			final int rl = m1.getNumRows();
			final int cr = m2.getNumColumns();
			// final int rr = m2.getNumRows(); // shared dim
			final MatrixBlock ret = new MatrixBlock(rl, cr, false);
			ret.allocateBlock();

			// MatrixBlock m1uc = m1.decompress(k);
			final List<Future<Long>> tasks = new ArrayList<>();
			final List<AColGroup> groups = m1.getColGroups();
			final int blkI = Math.max((int) Math.ceil((double) rl / k), 16);
			final int blkJ = blkI > 16 ? cr : Math.max((cr / k), 512); // make it a multiplicative of 8.
			for(int i = 0; i < rl; i += blkI) {
				final int startI = i;
				final int endI = Math.min(i + blkI, rl);
				for(int j = 0; j < cr; j += blkJ) {
					final int startJ = j;
					final int endJ = Math.min(j + blkJ, cr);
					tasks.add(pool.submit(() -> {
						for(AColGroup g : groups)
							g.rightDecompressingMult(m2, ret, startI, endI, rl, startJ, endJ);
						return ret.recomputeNonZeros(startI, endI - 1, startJ, endJ - 1);
					}));
				}
			}
			long nnz = 0;
			for(Future<Long> t : tasks)
				nnz += t.get();

			ret.setNonZeros(nnz);
			ret.examSparsity();
			return ret;
		}
		finally {
			pool.shutdown();
		}

	}

	private static boolean betterIfDecompressed(CompressedMatrixBlock m) {
		for(AColGroup g : m.getColGroups()) {
			// TODO add subpport for decompressing RMM to ASDC and ASDCZero
			if(!(g instanceof ColGroupUncompressed || g instanceof ASDC || g instanceof ASDCZero) &&
				g.getNumValues() * 2 >= m.getNumRows()) {
				return true;
			}
		}
		return false;
	}

	private static CompressedMatrixBlock RMMOverlapping(CompressedMatrixBlock m1, MatrixBlock that, int k)
		throws Exception {

		final int rl = m1.getNumRows();
		final int cr = that.getNumColumns();
		final int rr = that.getNumRows(); // shared dim
		final List<AColGroup> colGroups = m1.getColGroups();
		final List<AColGroup> retCg = new ArrayList<>();
		final CompressedMatrixBlock ret = new CompressedMatrixBlock(rl, cr);

		final boolean shouldFilter = CLALibUtils.shouldPreFilter(colGroups);
		final double[] constV;
		final List<AColGroup> filteredGroups;

		if(shouldFilter) {
			constV = new double[rr];
			filteredGroups = CLALibUtils.filterGroups(colGroups, constV);
		}
		else {
			filteredGroups = colGroups;
			constV = null;
		}

		if(k == 1 || filteredGroups.size() == 1)
			RMMSingle(filteredGroups, that, retCg);
		else
			RMMParallel(filteredGroups, that, retCg, k);

		if(constV != null) {
			final MatrixBlock cb = new MatrixBlock(1, constV.length, constV);
			final MatrixBlock cbRet = new MatrixBlock(1, that.getNumColumns(), false);
			LibMatrixMult.matrixMult(cb, that, cbRet); // mm on row vector left.
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
		constantRow.sparseToDense();
		out.add(ColGroupConst.create(constantRow.getDenseBlockValues()));
	}

	private static MatrixBlock RMM(CompressedMatrixBlock m1, MatrixBlock that, int k) throws Exception {

		// Timing t = new Timing();
		// this version returns a decompressed result.
		final int rl = m1.getNumRows();
		final int cr = that.getNumColumns();
		final int rr = that.getNumRows(); // shared dim
		final List<AColGroup> colGroups = m1.getColGroups();

		final boolean shouldFilter = CLALibUtils.shouldPreFilter(colGroups);

		// start allocation of output.
		MatrixBlock ret = new MatrixBlock(rl, cr, false);
		final Future<MatrixBlock> f = ret.allocateBlockAsync();

		double[] constV;
		final List<AColGroup> filteredGroups;

		if(shouldFilter) {
			if(CLALibUtils.alreadyPreFiltered(colGroups, cr)) {
				filteredGroups = new ArrayList<>(colGroups.size() - 1);
				constV = CLALibUtils.filterGroupsAndSplitPreAggOneConst(colGroups, filteredGroups);
			}
			else {
				constV = new double[rr];
				filteredGroups = CLALibUtils.filterGroups(colGroups, constV);
			}
		}
		else {
			filteredGroups = colGroups;
			constV = null;
		}

		final List<AColGroup> retCg = new ArrayList<>(filteredGroups.size());
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

		ret = f.get();
		CLALibDecompress.decompressDense(ret, retCg, constV, 0, k, true);

		return ret;
	}

	private static boolean RMMSingle(List<AColGroup> filteredGroups, MatrixBlock that, List<AColGroup> retCg) {
		boolean containsNull = false;
		final IColIndex allCols = ColIndexFactory.create(that.getNumColumns());
		for(AColGroup g : filteredGroups) {
			AColGroup retG = g.rightMultByMatrix(that, allCols, 1);
			if(retG != null)
				retCg.add(retG);
			else
				containsNull = true;
		}
		return containsNull;
	}

	private static boolean RMMParallel(List<AColGroup> filteredGroups, MatrixBlock that, List<AColGroup> retCg, int k)
		throws Exception {
		final ExecutorService pool = CommonThreadPool.get(k);
		boolean containsNull = false;
		try {
			final IColIndex allCols = ColIndexFactory.create(that.getNumColumns());
			List<Callable<AColGroup>> tasks = new ArrayList<>(filteredGroups.size());
			for(AColGroup g : filteredGroups)
				tasks.add(new RightMatrixMultTask(g, that, allCols, k));
			for(Future<AColGroup> fg : pool.invokeAll(tasks)) {
				AColGroup g = fg.get();
				if(g != null)
					retCg.add(g);
				else
					containsNull = true;
			}
		}
		finally {
			pool.shutdown();
		}
		return containsNull;
	}

	private static class RightMatrixMultTask implements Callable<AColGroup> {
		private final AColGroup _colGroup;
		private final MatrixBlock _b;
		private final IColIndex _allCols;
		private final int _k;

		protected RightMatrixMultTask(AColGroup colGroup, MatrixBlock b, IColIndex allCols, int k) {
			_colGroup = colGroup;
			_b = b;
			_allCols = allCols;
			_k = k;
		}

		@Override
		public AColGroup call() throws Exception {
			return _colGroup.rightMultByMatrix(_b, _allCols, _k);
		}
	}
}
