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
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.cost.ICostEstimate;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class CoCodeGreedy extends AColumnCoCoder {

	protected CoCodeGreedy(CompressedSizeEstimator sizeEstimator, ICostEstimate costEstimator, CompressionSettings cs) {
		super(sizeEstimator, costEstimator, cs);
	}

	@Override
	protected CompressedSizeInfo coCodeColumns(CompressedSizeInfo colInfos, int k) {
		colInfos.setInfo(join(colInfos.compressionInfo, _sest, _cest, _cs, k));
		return colInfos;
	}

	protected static List<CompressedSizeInfoColGroup> join(List<CompressedSizeInfoColGroup> inputColumns,
		CompressedSizeEstimator sEst, ICostEstimate cEst, CompressionSettings cs, int k) {
		Memorizer mem = new Memorizer(sEst);
		for(CompressedSizeInfoColGroup g : inputColumns)
			mem.put(g);

		return coCodeBruteForce(inputColumns, cEst, mem, k);
	}

	private static List<CompressedSizeInfoColGroup> coCodeBruteForce(List<CompressedSizeInfoColGroup> inputColumns,
		ICostEstimate cEst, Memorizer mem, int k) {

		final List<ColIndexes> workSet = new ArrayList<>(inputColumns.size());
		final boolean workloadCost = cEst instanceof ComputationCostEstimator;

		// assume that we can at max reduce 90 % of cost if joined
		// assume that we can max reduce 65% of compute cost if joined
		final double costFilterThreshold = (workloadCost ? 0.65 : 0.9);

		for(int i = 0; i < inputColumns.size(); i++)
			workSet.add(new ColIndexes(inputColumns.get(i).getColumns()));

		if(k > 1)
			parallelFirstJoin(workSet, mem, cEst, costFilterThreshold, k);

		// Process merging iterations until no more change
		while(workSet.size() > 1) {
			double changeInCost = 0;
			CompressedSizeInfoColGroup tmp = null;
			ColIndexes selected1 = null, selected2 = null;
			for(int i = 0; i < workSet.size(); i++) {
				for(int j = i + 1; j < workSet.size(); j++) {
					final ColIndexes c1 = workSet.get(i);
					final ColIndexes c2 = workSet.get(j);
					final double costC1 = cEst.getCostOfColumnGroup(mem.get(c1));
					final double costC2 = cEst.getCostOfColumnGroup(mem.get(c2));

					mem.incst1();

					// Pruning filter : skip dominated candidates
					// Since even if the entire size of one of the column lists is removed,
					// it still does not improve compression.
					// In the case of workload we relax the requirement for the filter.
					// if(-Math.min(costC1, costC2) > changeInCost)
					if(-Math.min(costC1, costC2) * costFilterThreshold > changeInCost)
						continue;

					// Join the two column groups.
					// and Memorize the new join.
					final CompressedSizeInfoColGroup c1c2Inf = mem.getOrCreate(c1, c2);
					final double costC1C2 = cEst.getCostOfColumnGroup(c1c2Inf);
					final double newCostIfJoined = costC1C2 - costC1 - costC2;

					// Select the best join of either the currently selected
					// or keep the old one.
					if((tmp == null && newCostIfJoined < changeInCost) || tmp != null && (newCostIfJoined < changeInCost ||
						newCostIfJoined == changeInCost && c1c2Inf.getColumns().length < tmp.getColumns().length)) {
						changeInCost = newCostIfJoined;
						tmp = c1c2Inf;
						selected1 = c1;
						selected2 = c2;
					}
				}
			}

			if(tmp != null) {
				workSet.remove(selected1);
				workSet.remove(selected2);
				mem.remove(selected1, selected2);
				workSet.add(new ColIndexes(tmp.getColumns()));
			}
			else
				break;
		}
		
		if(LOG.isDebugEnabled())
			LOG.debug("Memorizer stats:" + mem.stats());
		mem.resetStats();

		List<CompressedSizeInfoColGroup> ret = new ArrayList<>(workSet.size());
		for(ColIndexes w : workSet)
			ret.add(mem.get(w));

		return ret;
	}

	protected static void parallelFirstJoin(List<ColIndexes> workSet, Memorizer mem, ICostEstimate cEst,
		double costFilterThreshold, int k) {
		try {
			final ExecutorService pool = CommonThreadPool.get(k);
			final List<JoinTask> tasks = new ArrayList<>();
			final int size = workSet.size();
			for(int i = 0; i < size; i++)
				for(int j = i + 1; j < size; j++)
					tasks.add(new JoinTask(workSet.get(i), workSet.get(j), mem));

			for(Future<Object> t : pool.invokeAll(tasks))
				t.get();
			pool.shutdown();
		}
		catch(Exception e) {
			throw new DMLCompressionException("Failed parallelize first level all join all", e);
		}
	}

	protected static class JoinTask implements Callable<Object> {
		private final ColIndexes _c1, _c2;
		private final Memorizer _m;

		protected JoinTask(ColIndexes c1, ColIndexes c2, Memorizer m) {
			_c1 = c1;
			_c2 = c2;
			_m = m;
		}

		@Override
		public Object call() {
			try {
				_m.getOrCreate(_c1, _c2);
				return null;
			}
			catch(Exception e) {
				throw new DMLCompressionException(
					"Failed to join columns : " + Arrays.toString(_c1._indexes) + " + " + Arrays.toString(_c2._indexes), e);
			}
		}
	}
}
