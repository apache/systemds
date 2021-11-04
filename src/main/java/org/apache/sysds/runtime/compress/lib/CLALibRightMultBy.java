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
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class CLALibRightMultBy {
	private static final Log LOG = LogFactory.getLog(CLALibRightMultBy.class.getName());

	public static MatrixBlock rightMultByMatrix(CompressedMatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int k) {
		final boolean allowOverlap = ConfigurationManager.getDMLConfig()
			.getBooleanValue(DMLConfig.COMPRESSED_OVERLAPPING);
		return rightMultByMatrix(m1, m2, ret, k, allowOverlap);
	}

	public static MatrixBlock rightMultByMatrix(CompressedMatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int k,
		boolean allowOverlap) {

		if(m2.isEmpty()) {
			LOG.trace("Empty right multiply");
			if(ret == null)
				ret = new MatrixBlock(m1.getNumRows(), m2.getNumColumns(), 0);
			else
				ret.reset(m1.getNumRows(), m2.getNumColumns(), 0);
		}
		else {
			if(m2 instanceof CompressedMatrixBlock)
				m2 = ((CompressedMatrixBlock) m2).getUncompressed("Uncompressed right side of right MM");

			ret = rightMultByMatrixOverlapping(m1, m2, k);

			if(ret instanceof CompressedMatrixBlock) {
				if(!allowOverlap)
					ret = ((CompressedMatrixBlock) ret).getUncompressed("Overlapping not allowed");
				else {
					final double compressedSize = ret.getInMemorySize();
					final double uncompressedSize = MatrixBlock.estimateSizeDenseInMemory(ret.getNumRows(),
						ret.getNumColumns());
					if(compressedSize > uncompressedSize)
						ret = ((CompressedMatrixBlock) ret).getUncompressed(
							"Overlapping rep to big: " + compressedSize + " vs Uncompressed " + uncompressedSize);
				}
			}
		}

		ret.recomputeNonZeros();

		return ret;
	}

	private static MatrixBlock rightMultByMatrixOverlapping(CompressedMatrixBlock m1, MatrixBlock that, int k) {
		final int rl = m1.getNumRows();
		final int cr = that.getNumColumns();
		final int rr = that.getNumRows(); // shared dim
		final List<AColGroup> colGroups = m1.getColGroups();
		final List<AColGroup> retCg = new ArrayList<>();
		final CompressedMatrixBlock ret = new CompressedMatrixBlock(rl, cr);

		final boolean containsSDC = CLALibUtils.containsSDCOrConst(colGroups);

		double[] constV = containsSDC ? new double[rr] : null;
		final List<AColGroup> filteredGroups = CLALibUtils.filterGroups(colGroups, constV);
		if(colGroups == filteredGroups)
			constV = null;

		boolean containsNull = false;
		if(k == 1)
			containsNull = rightMultByMatrixOverlappingSingleThread(filteredGroups, that, retCg);
		else
			containsNull = rightMultByMatrixOverlappingMultiThread(filteredGroups, that, retCg, k);

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

	private static boolean rightMultByMatrixOverlappingSingleThread(List<AColGroup> filteredGroups, MatrixBlock that,
		List<AColGroup> retCg) {
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

	private static boolean rightMultByMatrixOverlappingMultiThread(List<AColGroup> filteredGroups, MatrixBlock that,
		List<AColGroup> retCg, int k) {
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
