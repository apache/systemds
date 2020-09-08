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

package org.apache.sysds.runtime.compress.cocode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class PlanningCoCoder {

	private static final Log LOG = LogFactory.getLog(PlanningCoCoder.class.getName());

	public enum PartitionerType {
		BIN_PACKING, STATIC,
	}

	/**
	 * Main entry point of CoCode.
	 * 
	 * This package groups together ColGroups across columns, to improve compression further,
	 * 
	 * @param sizeEstimator The size estimator used for estimating ColGroups potential sizes.
	 * @param colInfos      The information already gathered on the individual ColGroups of columns.
	 * @param numRows       The number of rows in the input matrix.
	 * @param k             The concurrency degree allowed for this operation.
	 * @param cs            The Compression Settings used in the compression.
	 * @return The Estimated (hopefully) best groups of ColGroups.
	 */
	public static List<int[]> findCoCodesByPartitioning(CompressedSizeEstimator sizeEstimator,
		CompressedSizeInfo colInfos, int numRows, int k, CompressionSettings cs) {
		// filtering out non-group-able columns as singleton groups
		// weight is the ratio of its cardinality to the number of rows

		List<Integer> cols = colInfos.colsC;
		CompressedSizeInfoColGroup[] colGroups = colInfos.compressionInfo;

		int numCols = cols.size();
		List<Integer> groupCols = new ArrayList<>();
		HashMap<Integer, GroupableColInfo> groupColsInfo = new HashMap<>();
		for(int i = 0; i < numCols; i++) {
			int colIx = cols.get(i);
			double cardinality = colGroups[colIx].getEstCard();
			double weight = cardinality / numRows;
			groupCols.add(colIx);
			groupColsInfo.put(colIx, new GroupableColInfo(weight, colGroups[colIx].getMinSize()));
		}

		// use column group partitioner to create partitions of columns
		List<int[]> bins = createColumnGroupPartitioner(cs.columnPartitioner)
			.partitionColumns(groupCols, groupColsInfo, cs);

		// brute force grouping within each partition
		return (k > 1) ? getCocodingGroupsBruteForce(bins,
			groupColsInfo,
			sizeEstimator,
			numRows,
			k) : getCocodingGroupsBruteForce(bins, groupColsInfo, sizeEstimator, numRows);
	}

	private static List<int[]> getCocodingGroupsBruteForce(List<int[]> bins,
		HashMap<Integer, GroupableColInfo> groupColsInfo, CompressedSizeEstimator estimator, int rlen) {
		List<int[]> retGroups = new ArrayList<>();
		for(int[] bin : bins) {
			// building an array of singleton CoCodingGroup
			ArrayList<PlanningCoCodingGroup> sgroups = new ArrayList<>();
			for(int col : bin)
				sgroups.add(new PlanningCoCodingGroup(col, groupColsInfo.get(col)));
			// brute force co-coding
			PlanningCoCodingGroup[] outputGroups = findCocodesBruteForce(estimator,
				rlen,
				sgroups.toArray(new PlanningCoCodingGroup[0]));
			for(PlanningCoCodingGroup grp : outputGroups)
				retGroups.add(grp.getColIndices());
		}

		return retGroups;
	}

	private static List<int[]> getCocodingGroupsBruteForce(List<int[]> bins,
		HashMap<Integer, GroupableColInfo> groupColsInfo, CompressedSizeEstimator estimator, int rlen, int k) {
		List<int[]> retGroups = new ArrayList<>();
		try {
			ExecutorService pool = CommonThreadPool.get(k);
			ArrayList<CocodeTask> tasks = new ArrayList<>();
			for(int[] bin : bins) {
				// building an array of singleton CoCodingGroup
				ArrayList<PlanningCoCodingGroup> sgroups = new ArrayList<>();
				for(int col : bin)
					sgroups.add(new PlanningCoCodingGroup(col, groupColsInfo.get(col)));
				tasks.add(new CocodeTask(estimator, sgroups, rlen));
			}
			List<Future<PlanningCoCodingGroup[]>> rtask = pool.invokeAll(tasks);
			for(Future<PlanningCoCodingGroup[]> lrtask : rtask)
				for(PlanningCoCodingGroup grp : lrtask.get())
					retGroups.add(grp.getColIndices());
			pool.shutdown();
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}

		return retGroups;
	}

	/**
	 * Identify columns to code together. Uses a greedy approach that merges pairs of column groups into larger groups.
	 * Each phase of the greedy algorithm considers all combinations of pairs to merge.
	 * 
	 * TODO Find better faster ways of finding cocodes than brute force.
	 * 
	 * @param sizeEstimator  compressed size estimator
	 * @param numRowsWeight  number of rows weight
	 * @param singltonGroups planning co-coding groups
	 * @return A PlanningCoCodingGroup.
	 */
	private static PlanningCoCodingGroup[] findCocodesBruteForce(CompressedSizeEstimator estimator, int numRows,
		PlanningCoCodingGroup[] singletonGroups) {
		if(LOG.isTraceEnabled())
			LOG.trace("Cocoding: process " + singletonGroups.length);

		List<PlanningCoCodingGroup> workset = new ArrayList<>(Arrays.asList(singletonGroups));

		// establish memo table for extracted column groups
		PlanningMemoTable memo = new PlanningMemoTable();

		// process merging iterations until no more change
		boolean changed = true;
		while(changed && workset.size() > 1) {
			// find best merge, incl memoization
			PlanningCoCodingGroup tmp = null;
			for(int i = 0; i < workset.size(); i++) {
				for(int j = i + 1; j < workset.size(); j++) {
					PlanningCoCodingGroup c1 = workset.get(i);
					PlanningCoCodingGroup c2 = workset.get(j);
					memo.incrStats(1, 0, 0);

					// pruning filter: skip dominated candidates
					if(-Math.min(c1.getEstSize(), c2.getEstSize()) > memo.getOptChangeInSize())
						continue;

					// memoization or newly created group (incl bitmap extraction)
					PlanningCoCodingGroup c1c2 = memo.getOrCreate(c1, c2, estimator, numRows);

					// keep best merged group only
					if(tmp == null || c1c2.getChangeInSize() < tmp.getChangeInSize() ||
						(c1c2.getChangeInSize() == tmp.getChangeInSize() &&
							c1c2.getColIndices().length < tmp.getColIndices().length))
						tmp = c1c2;
				}
			}

			// modify working set
			if(tmp != null && tmp.getChangeInSize() < 0) {
				workset.remove(tmp.getLeftGroup());
				workset.remove(tmp.getRightGroup());
				workset.add(tmp);
				memo.remove(tmp);

				if(LOG.isTraceEnabled()) {
					LOG.trace("--merge groups: " + Arrays.toString(tmp.getLeftGroup().getColIndices()) + " and "
						+ Arrays.toString(tmp.getRightGroup().getColIndices()));
				}
			}
			else {
				changed = false;
			}
		}

		if(LOG.isTraceEnabled())
			LOG.trace("--stats: " + Arrays.toString(memo.getStats()));

		return workset.toArray(new PlanningCoCodingGroup[0]);
	}

	private static ColumnGroupPartitioner createColumnGroupPartitioner(PartitionerType type) {
		switch(type) {
			case BIN_PACKING:
				return new ColumnGroupPartitionerBinPacking();

			case STATIC:
				return new ColumnGroupPartitionerStatic();

			default:
				throw new RuntimeException("Unsupported column group partitioner: " + type.toString());
		}
	}

	public static class GroupableColInfo {
		public final double cardRatio;
		public final long size;

		public GroupableColInfo(double lcardRatio, long lsize) {
			cardRatio = lcardRatio;
			size = lsize;
		}
	}

	private static class CocodeTask implements Callable<PlanningCoCodingGroup[]> {
		private CompressedSizeEstimator _estim = null;
		private ArrayList<PlanningCoCodingGroup> _sgroups = null;
		private int _rlen = -1;

		protected CocodeTask(CompressedSizeEstimator estim, ArrayList<PlanningCoCodingGroup> sgroups, int rlen) {
			_estim = estim;
			_sgroups = sgroups;
			_rlen = rlen;
		}

		@Override
		public PlanningCoCodingGroup[] call() {
			// brute force co-coding
			return findCocodesBruteForce(_estim, _rlen, _sgroups.toArray(new PlanningCoCodingGroup[0]));
		}
	}
}
