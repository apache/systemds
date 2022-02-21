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
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.cost.ACostEstimate;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class CoCodeGreedy extends AColumnCoCoder {

	private final Memorizer mem;

	protected CoCodeGreedy(CompressedSizeEstimator sizeEstimator, ACostEstimate costEstimator, CompressionSettings cs) {
		super(sizeEstimator, costEstimator, cs);
		mem = new Memorizer(sizeEstimator);
	}

	@Override
	protected CompressedSizeInfo coCodeColumns(CompressedSizeInfo colInfos, int k) {
		colInfos.setInfo(combine(colInfos.compressionInfo, _sest, _cest, _cs, k));
		return colInfos;
	}

	protected List<CompressedSizeInfoColGroup> combine(List<CompressedSizeInfoColGroup> inputColumns,
		CompressedSizeEstimator sEst, ACostEstimate cEst, CompressionSettings cs, int k) {

		for(CompressedSizeInfoColGroup g : inputColumns)
			mem.put(g);

		return coCodeBruteForce(inputColumns, cEst, k);
	}

	private List<CompressedSizeInfoColGroup> coCodeBruteForce(List<CompressedSizeInfoColGroup> inputColumns,
		ACostEstimate cEst, int k) {

		final List<ColIndexes> workSet = new ArrayList<>(inputColumns.size());

		// assume that we can at max reduce 90 % of cost if combined
		final double costFilterThreshold = 0.9;

		final ExecutorService pool = CommonThreadPool.get(k);
		for(int i = 0; i < inputColumns.size(); i++) {
			CompressedSizeInfoColGroup g = inputColumns.get(i);
			workSet.add(new ColIndexes(g.getColumns()));
		}

		// if(k > 1)
		// 	parallelFirstCombine(workSet, mem, cEst, pool);

		// Process merging iterations until no more change
		while(workSet.size() > 1) {
			double changeInCost = 0;
			CompressedSizeInfoColGroup tmp = null;
			ColIndexes selected1 = null, selected2 = null;
			for(int i = 0; i < workSet.size(); i++) {
				for(int j = i + 1; j < workSet.size(); j++) {
					final ColIndexes c1 = workSet.get(i);
					final ColIndexes c2 = workSet.get(j);
					final double costC1 = cEst.getCost(mem.get(c1));
					final double costC2 = cEst.getCost(mem.get(c2));

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
					final double costC1C2 = cEst.getCost(c1c2Inf);
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

				Collections.sort(workSet, new CompareColumns());

				final ColIndexes a = new ColIndexes(tmp.getColumns());
				// if(k > 1) {
				// 	final List<CombineTask> tasks = new ArrayList<>();
				// 	final int size = workSet.size();
				// 	try {
				// 		// combine the first k columns...
				// 		// just to parallelize at least the first couple of options.
				// 		// This potentially filters out some of the options quickly.
				// 		for(int j = 0; j < Math.min(k, size); j++)
				// 			tasks.add(new CombineTask(a, workSet.get(j), mem));

				// 		for(Future<Object> t : pool.invokeAll(tasks))
				// 			t.get();
				// 	}
				// 	catch(Exception e) {
				// 		throw new DMLCompressionException("Failed parallelize first level all join all", e);
				// 	}
				// }
				workSet.add(a);
			}
			else
				break;
		}

		if(LOG.isDebugEnabled())
			LOG.debug("Memorizer stats:" + mem.stats());
		mem.resetStats();

		pool.shutdown();
		List<CompressedSizeInfoColGroup> ret = new ArrayList<>(workSet.size());
		for(ColIndexes w : workSet)
			ret.add(mem.get(w));

		return ret;
	}

	// protected static void parallelFirstCombine(List<ColIndexes> workSet, Memorizer mem, ACostEstimate cEst,
	// 	ExecutorService pool) {
	// 	try {
	// 		final List<CombineTask> tasks = new ArrayList<>();
	// 		final int size = workSet.size();
	// 		for(int i = 0; i < size; i++)
	// 			for(int j = i + 1; j < size; j++)
	// 				tasks.add(new CombineTask(workSet.get(i), workSet.get(j), mem));

	// 		for(Future<Object> t : pool.invokeAll(tasks))
	// 			t.get();
	// 	}
	// 	catch(Exception e) {
	// 		throw new DMLCompressionException("Failed parallelize first level all join all", e);
	// 	}
	// }

	protected static class CombineTask implements Callable<Object> {
		private final ColIndexes _c1, _c2;
		private final Memorizer _m;

		protected CombineTask(ColIndexes c1, ColIndexes c2, Memorizer m) {
			_c1 = c1;
			_c2 = c2;
			_m = m;
		}

		@Override
		public Object call() {
			_m.getOrCreate(_c1, _c2);
			return null;

		}
	}

	class CompareColumns implements Comparator<ColIndexes> {

		@Override
		public int compare(ColIndexes arg0, ColIndexes arg1) {
			final double c1 = _cest.getCost(mem.get(arg0));
			final double c2 = _cest.getCost(mem.get(arg1));
			if(c1 > c2)
				return -1;
			else if(c1 == c2)
				return 0;
			else
				return 1;
		}

	}
}
