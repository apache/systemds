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
import java.util.HashSet;
import java.util.List;
import java.util.Set;
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
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
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
				m2 = ((CompressedMatrixBlock) m2).getUncompressed("Uncompressed right side of right MM");

			if(!allowOverlap) {
				LOG.trace("Overlapping output not allowed in call to Right MM");
				return RMM(m1, m2, k);
			}

			final CompressedMatrixBlock retC = RMMOverlapping(m1, m2, k);
			final double cs = retC.getInMemorySize();
			final double us = MatrixBlock.estimateSizeDenseInMemory(rr, rc);
			if(cs > us)
				return retC.getUncompressed("Overlapping rep to big: " + cs + " vs uncompressed " + us);
			else if(retC.isEmpty())
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

		boolean containsNull = false;
		if(k == 1)
			containsNull = RMMSingle(filteredGroups, that, retCg);
		else
			containsNull = RMMParallel(filteredGroups, that, retCg, k);

		if(constV != null) {
			AColGroup cRet = ColGroupFactory.genColGroupConst(constV).rightMultByMatrix(that);
			if(cRet != null)
				retCg.add(cRet);
		}

		ret.allocateColGroupList(retCg);

		if(retCg.size() > 1)
			ret.setOverlapping(true);

		addEmptyColumn(retCg, cr, rl, containsNull);

		return ret;
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
			ColGroupConst cRet = (ColGroupConst) ColGroupFactory.genColGroupConst(constV).rightMultByMatrix(that);
			constV = cRet.getValues(); // overwrite constV
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
		for(AColGroup g : filteredGroups) {
			AColGroup retG = g.rightMultByMatrix(that);
			if(retG != null)
				retCg.add(retG);
			else
				containsNull = true;
		}
		return containsNull;
	}

	private static boolean RMMParallel(List<AColGroup> filteredGroups, MatrixBlock that, List<AColGroup> retCg, int k) {
		ExecutorService pool = CommonThreadPool.get(k);
		boolean containsNull = false;
		try {
			List<Callable<AColGroup>> tasks = new ArrayList<>(filteredGroups.size());
			for(AColGroup g : filteredGroups)
				tasks.add(new RightMatrixMultTask(g, that));
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
		return containsNull;
	}

	private static void addEmptyColumn(List<AColGroup> retCg, int cr, int rl, boolean containsNull) {
		if(containsNull) {
			final ColGroupEmpty cge = findEmptyColumnsAndMakeEmptyColGroup(retCg, cr, rl);
			if(cge != null)
				retCg.add(cge);
		}
	}

	private static ColGroupEmpty findEmptyColumnsAndMakeEmptyColGroup(List<AColGroup> colGroups, int nCols, int nRows) {
		Set<Integer> emptyColumns = new HashSet<>(nCols);
		for(int i = 0; i < nCols; i++)
			emptyColumns.add(i);

		for(AColGroup g : colGroups)
			for(int c : g.getColIndices())
				emptyColumns.remove(c);

		if(emptyColumns.size() != 0) {
			int[] emptyColumnsFinal = emptyColumns.stream().mapToInt(Integer::intValue).toArray();
			return new ColGroupEmpty(emptyColumnsFinal);
		}
		else
			return null;
	}

	private static class RightMatrixMultTask implements Callable<AColGroup> {
		private final AColGroup _colGroup;
		private final MatrixBlock _b;

		protected RightMatrixMultTask(AColGroup colGroup, MatrixBlock b) {
			_colGroup = colGroup;
			_b = b;
		}

		@Override
		public AColGroup call() {
			try {
				return _colGroup.rightMultByMatrix(_b);
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
		}
	}
}
