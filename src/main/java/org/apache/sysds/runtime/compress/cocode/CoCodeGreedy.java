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
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.cost.ACostEstimate;
import org.apache.sysds.runtime.compress.estim.AComEst;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

public class CoCodeGreedy extends AColumnCoCoder {

	private final MemorizerV2 mem;

	protected CoCodeGreedy(AComEst sizeEstimator, ACostEstimate costEstimator, CompressionSettings cs) {
		super(sizeEstimator, costEstimator, cs);
		mem = new MemorizerV2(sizeEstimator, sizeEstimator.getNumColumns());
	}

	protected CoCodeGreedy(AComEst sizeEstimator, ACostEstimate costEstimator, CompressionSettings cs, MemorizerV2 mem) {
		super(sizeEstimator, costEstimator, cs);
		this.mem = mem;
	}

	@Override
	protected CompressedSizeInfo coCodeColumns(CompressedSizeInfo colInfos, int k) {
		colInfos.setInfo(combine(colInfos.compressionInfo, k));
		return colInfos;
	}

	protected List<CompressedSizeInfoColGroup> combine(List<CompressedSizeInfoColGroup> inputColumns, int k) {
		for(CompressedSizeInfoColGroup g : inputColumns)
			mem.put(g);
		return coCodeBruteForce(inputColumns, k);
	}

	private List<CompressedSizeInfoColGroup> coCodeBruteForce(List<CompressedSizeInfoColGroup> inputColumns, int k) {

		final List<ColIndexes> workSet = new ArrayList<>(inputColumns.size());
		k = k <= 0 ? InfrastructureAnalyzer.getLocalParallelism() : k;
		final ExecutorService pool = k > 1 ? CommonThreadPool.get(k) : null;
		try {
			for(int i = 0; i < inputColumns.size(); i++) {
				CompressedSizeInfoColGroup g = inputColumns.get(i);
				workSet.add(new ColIndexes(g.getColumns()));
			}

			if(k > 1)
				parallelFirstCombine(workSet, pool);

			// second layer to keep the second best combination
			double secondChange = 0;
			CompressedSizeInfoColGroup secondTmp = null;
			ColIndexes secondSelectedJ = null, secondSelected1 = null, secondSelected2 = null;

			// Process merging iterations until no more change
			while(workSet.size() > 1) {
				if(secondChange != 0)
					mem.incst4();
				// maintain selected
				double changeInCost = secondChange;
				CompressedSizeInfoColGroup tmp = secondTmp;
				ColIndexes selectedJ = secondSelectedJ, selected1 = secondSelected1, selected2 = secondSelected2;

				for(int i = 0; i < workSet.size(); i++) {
					for(int j = i + 1; j < workSet.size(); j++) {
						final ColIndexes c1 = workSet.get(i);
						final ColIndexes c2 = workSet.get(j);
						final CompressedSizeInfoColGroup c1i = mem.get(c1);
						final CompressedSizeInfoColGroup c2i = mem.get(c2);

						final double costC1 = _cest.getCost(c1i);
						final double costC2 = _cest.getCost(c2i);

						mem.incst1();
						final int maxCombined = c1i.getNumVals() * c2i.getNumVals();

						// Pruning filter : skip dominated candidates
						// Since even if the entire size of one of the column lists is removed,
						// it still does not improve compression.
						// In the case of workload we relax the requirement for the filter.
						if(-Math.min(costC1, costC2) > changeInCost // change in cost cannot possibly be better.
							|| (maxCombined < 0) // int overflow
							|| (maxCombined > c1i.getNumRows())) // higher combined number of rows.
							continue;

						// Combine the two column groups.
						// and Memorize the new Combine.
						final IColIndex c = c1._indexes.combine(c2._indexes);
						final ColIndexes cI = new ColIndexes(c);
						final CompressedSizeInfoColGroup c1c2Inf = mem.getOrCreate(cI, c1, c2);
						final double costC1C2 = _cest.getCost(c1c2Inf);
						final double newCostIfJoined = costC1C2 - costC1 - costC2;

						// Select the best Combine of either the currently selected
						// or keep the old one.
						if(newCostIfJoined < 0) {
							if(tmp == null) {
								changeInCost = newCostIfJoined;
								tmp = c1c2Inf;
								selectedJ = cI;
								selected1 = c1;
								selected2 = c2;
							}
							else if((newCostIfJoined < changeInCost ||
								newCostIfJoined == changeInCost && c1c2Inf.getColumns().size() < tmp.getColumns().size())) {

								if(selected1 != secondSelected1 && selected2 != secondSelected2) {
									secondTmp = tmp;
									secondSelectedJ = selectedJ;
									secondSelected1 = selected1;
									secondSelected2 = selected2;
									secondChange = changeInCost;
								}

								changeInCost = newCostIfJoined;
								tmp = c1c2Inf;
								selectedJ = cI;
								selected1 = c1;
								selected2 = c2;
							}
						}
					}
				}

				if(tmp != null) {
					// remove from workset
					workSet.remove(selected1);
					workSet.remove(selected2);
					mem.remove(selected1, selected2); // remove all memorized values of the combined columns

					// ColIndexes combined = new ColIndexes(tmp.getColumns());
					mem.put(selectedJ, tmp); // add back the new combination to memorizer
					workSet.add(selectedJ);
					if(selectedJ.contains(secondSelected1, secondSelected2)) {
						secondTmp = null;
						secondSelectedJ = null;
						secondSelected1 = null;
						secondSelected2 = null;
						secondChange = 0;
					}

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
		finally {
			if(pool != null)
				pool.shutdown();
		}
	}

	protected void parallelFirstCombine(List<ColIndexes> workSet, ExecutorService pool) {
		try {
			final List<CombineTask> tasks = new ArrayList<>();
			final int size = workSet.size();
			for(int i = 0; i < size; i++)
				for(int j = i + 1; j < size; j++)
					tasks.add(new CombineTask(workSet.get(i), workSet.get(j)));

			if(pool != null)
				for(Future<Object> t : pool.invokeAll(tasks))
					t.get();
			else 
				for(CombineTask t: tasks)
					t.call();
		}
		catch(Exception e) {
			throw new DMLCompressionException("Failed parallelize first level all join all", e);
		}
	}

	protected class CombineTask implements Callable<Object> {
		private final ColIndexes _c1, _c2;

		protected CombineTask(ColIndexes c1, ColIndexes c2) {
			_c1 = c1;
			_c2 = c2;
		}

		@Override
		public Object call() throws Exception {
			final CompressedSizeInfoColGroup c1i = mem.get(_c1);
			final CompressedSizeInfoColGroup c2i = mem.get(_c2);
			if(c1i != null && c2i != null) {
				final int maxCombined = c1i.getNumVals() * c2i.getNumVals();

				if(maxCombined < 0 // int overflow
					|| maxCombined > c1i.getNumRows() // higher than number of rows
					|| maxCombined > 100000) // higher than 100k ... then lets not precalculate it.
					return null;

				final IColIndex c = _c1._indexes.combine(_c2._indexes);
				final ColIndexes cI = new ColIndexes(c);
				mem.getOrCreate(cI, _c1, _c2);
			}
			return null;
		}
	}
}
