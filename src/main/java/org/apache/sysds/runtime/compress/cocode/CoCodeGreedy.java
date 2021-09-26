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
import java.util.Map;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.cost.ICostEstimate;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.utils.Util;

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
		Memorizer mem = new Memorizer(cs, sEst);
		for(CompressedSizeInfoColGroup g : inputColumns)
			mem.put(g);

		return coCodeBruteForce(inputColumns, cEst, mem);
	}

	private static List<CompressedSizeInfoColGroup> coCodeBruteForce(List<CompressedSizeInfoColGroup> inputColumns,
		ICostEstimate cEst, Memorizer mem) {

		List<ColIndexes> workset = new ArrayList<>(inputColumns.size());

		final boolean workloadCost = cEst instanceof ComputationCostEstimator;

		for(int i = 0; i < inputColumns.size(); i++)
			workset.add(new ColIndexes(inputColumns.get(i).getColumns()));
		// process merging iterations until no more change
		while(workset.size() > 1) {
			double changeInCost = 0;
			CompressedSizeInfoColGroup tmp = null;
			ColIndexes selected1 = null, selected2 = null;
			for(int i = 0; i < workset.size(); i++) {
				for(int j = i + 1; j < workset.size(); j++) {
					final ColIndexes c1 = workset.get(i);
					final ColIndexes c2 = workset.get(j);
					final double costC1 = cEst.getCostOfColumnGroup(mem.get(c1));
					final double costC2 = cEst.getCostOfColumnGroup(mem.get(c2));

					mem.incst1();

					// Pruning filter : skip dominated candidates
					// Since even if the entire size of one of the column lists is removed,
					// it still does not improve compression.
					// In the case of workload we relax the requirement for the filter.
					// if(-Math.min(costC1, costC2) > changeInCost)
					if(-Math.min(costC1, costC2) * (workloadCost ? 0.7 : 1) > changeInCost)
						continue;

					// Join the two column groups.
					// and Memorize the new join.
					final CompressedSizeInfoColGroup c1c2Inf = mem.getOrCreate(c1, c2);
					final double costC1C2 = cEst.getCostOfColumnGroup(c1c2Inf);

					final double newSizeChangeIfSelected = costC1C2 - costC1 - costC2;

					// Select the best join of either the currently selected
					// or keep the old one.
					if((tmp == null && newSizeChangeIfSelected < changeInCost) || tmp != null &&
						(newSizeChangeIfSelected < changeInCost || newSizeChangeIfSelected == changeInCost &&
							c1c2Inf.getColumns().length < tmp.getColumns().length)) {
						changeInCost = newSizeChangeIfSelected;
						tmp = c1c2Inf;
						selected1 = c1;
						selected2 = c2;
					}
				}
			}

			if(tmp != null) {
				workset.remove(selected1);
				workset.remove(selected2);
				mem.remove(selected1, selected2);
				workset.add(new ColIndexes(tmp.getColumns()));
			}
			else
				break;
		}
		if(LOG.isDebugEnabled())
			LOG.debug("Memorizer stats:" + mem.stats());
		mem.resetStats();

		List<CompressedSizeInfoColGroup> ret = new ArrayList<>(workset.size());

		for(ColIndexes w : workset)
			ret.add(mem.get(w));

		return ret;
	}

	protected static class Memorizer {
		private final CompressionSettings _cs;
		private final CompressedSizeEstimator _sEst;
		private final Map<ColIndexes, CompressedSizeInfoColGroup> mem;
		private int st1 = 0, st2 = 0, st3 = 0, st4 = 0;

		public Memorizer(CompressionSettings cs, CompressedSizeEstimator sEst) {
			_cs = cs;
			_sEst = sEst;
			mem = new HashMap<>();
		}

		public void put(CompressedSizeInfoColGroup g) {
			mem.put(new ColIndexes(g.getColumns()), g);
		}

		public CompressedSizeInfoColGroup get(ColIndexes c) {
			return mem.get(c);
		}

		public void remove(ColIndexes c1, ColIndexes c2) {
			mem.remove(c1);
			mem.remove(c2);
		}

		public CompressedSizeInfoColGroup getOrCreate(ColIndexes c1, ColIndexes c2) {
			final int[] c = Util.join(c1._indexes, c2._indexes);
			final ColIndexes cI = new ColIndexes(c);
			CompressedSizeInfoColGroup g = mem.get(cI);
			st2++;
			if(g == null) {
				final CompressedSizeInfoColGroup left = mem.get(c1);
				final CompressedSizeInfoColGroup right = mem.get(c2);
				final boolean leftConst = left.getBestCompressionType(_cs) == CompressionType.CONST &&
					left.getNumOffs() == 0;
				final boolean rightConst = right.getBestCompressionType(_cs) == CompressionType.CONST &&
					right.getNumOffs() == 0;
				if(leftConst)
					g = CompressedSizeInfoColGroup.addConstGroup(c, right, _cs.validCompressions);
				else if(rightConst)
					g = CompressedSizeInfoColGroup.addConstGroup(c, left, _cs.validCompressions);
				else {
					st3++;
					g = _sEst.estimateJoinCompressedSize(c, left, right);
				}

				if(leftConst || rightConst)
					st4++;

				mem.put(cI, g);
			}
			return g;
		}

		public void incst1() {
			st1++;
		}

		public String stats() {
			return st1 + " " + st2 + " " + st3 + " " + st4;
		}

		public void resetStats() {
			st1 = 0;
			st2 = 0;
			st3 = 0;
			st4 = 0;
		}

		@Override
		public String toString() {
			return mem.toString();
		}
	}

	private static class ColIndexes {
		final int[] _indexes;
		final int _hash;

		public ColIndexes(int[] indexes) {
			_indexes = indexes;
			_hash = Arrays.hashCode(_indexes);
		}

		@Override
		public int hashCode() {
			return _hash;
		}

		@Override
		public boolean equals(Object that) {
			ColIndexes thatGrp = (ColIndexes) that;
			return Arrays.equals(_indexes, thatGrp._indexes);
		}
	}
}
