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
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.cost.ICostEstimate;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.utils.Util;

/**
 * Column group partitioning with bin packing heuristic.
 */
public class CoCodeBinPacking extends AColumnCoCoder {

	private static final boolean FIRST_FIT_DEC = true;
	private static final int MAX_COL_FIRST_FIT = 16384;
	private static final int MAX_COL_PER_GROUP = 1024;

	private final Memorizer mem;

	/**
	 * Use a constant partition size independent of the number of rows in order to ensure constant compression speed
	 * independent of blocking. Higher values gives more CoCoding at the cost of longer compressionTimes.
	 * 
	 * VLDB 2018 Paper: Default 0.000032
	 */
	public static double BIN_CAPACITY = 0.000032;

	protected CoCodeBinPacking(CompressedSizeEstimator sizeEstimator, ICostEstimate costEstimator,
		CompressionSettings cs) {
		super(sizeEstimator, costEstimator, cs);
		mem = new Memorizer();
	}

	@Override
	protected CompressedSizeInfo coCodeColumns(CompressedSizeInfo colInfos, int k) {
		// establish memo table for extracted column groups

		List<CompressedSizeInfoColGroup> constantGroups = new ArrayList<>();
		List<CompressedSizeInfoColGroup> newGroups = new ArrayList<>();

		for(CompressedSizeInfoColGroup g : colInfos.getInfo()) {
			if(g.getBestCompressionType(_cs) == CompressionType.CONST)
				constantGroups.add(g);
			else {
				mem.put(g);
				newGroups.add(g);
			}
		}

		// make bins
		colInfos.setInfo(partitionColumns(newGroups));
		// Cocode compare all in bins
		getCoCodingGroupsBruteForce(colInfos, k);

		colInfos.getInfo().addAll(constantGroups);

		return colInfos;
	}

	/**
	 * Partition columns such that we make sure that the columns we analyze further actually have the potential of
	 * compressing together.
	 * 
	 * @param currentGroups The individual column groups
	 * @return The Partitioned column groups
	 */
	private List<CompressedSizeInfoColGroup> partitionColumns(List<CompressedSizeInfoColGroup> currentGroups) {

		List<CompressedSizeInfoColGroup> bins = new ArrayList<>();

		for(int i = 0; i < currentGroups.size(); i += MAX_COL_FIRST_FIT) {
			// extract subset of columns
			int iu = Math.min(i + MAX_COL_FIRST_FIT, currentGroups.size());
			List<CompressedSizeInfoColGroup> currentGroupsSubset = currentGroups.subList(i, iu);

			// sort items based on cardinality
			if(FIRST_FIT_DEC) {
				currentGroupsSubset.sort(new Comparator<CompressedSizeInfoColGroup>() {
					public int compare(CompressedSizeInfoColGroup a, CompressedSizeInfoColGroup b) {
						final double ac = a.getCardinalityRatio();
						final double bc = b.getCardinalityRatio();
						if(ac == bc)
							return 0;
						else if(ac > bc)
							return 1;
						else
							return -1;
					}
				});
			}
			// partition columns via bin packing
			bins.addAll(packFirstFit(currentGroupsSubset));
		}

		return bins;
	}

	/**
	 * Pack into the bins such that no bin exceed the Bin Capacity. This ensure that each bin only contains columns that
	 * are amenable to compression. And potentially CoCoding.
	 * 
	 * @param currentGroups Column groups in current batch
	 * @return The currently selected groups, packed together in bins.
	 */
	private List<CompressedSizeInfoColGroup> packFirstFit(List<CompressedSizeInfoColGroup> currentGroups) {
		List<CompressedSizeInfoColGroup> bins = new ArrayList<>();
		double[] binWeights = new double[16];

		for(int i = 0; i < currentGroups.size(); i++) {
			final CompressedSizeInfoColGroup c = currentGroups.get(i);
			// add to existing bin
			boolean assigned = false;
			for(int j = 0; j < bins.size(); j++) {
				double newBinWeight = binWeights[j] - c.getCardinalityRatio();
				if(newBinWeight >= 0 && bins.get(j).getColumns().length < MAX_COL_PER_GROUP - 1) {
					bins.set(j, joinWithoutAnalysis(Util.join(bins.get(j).getColumns(), c.getColumns()),bins.get(j), c));
					binWeights[j] = newBinWeight;
					assigned = true;
					break;
				}
			}

			// create new bin at end of list
			if(!assigned) {
				if(bins.size() == binWeights.length)
					binWeights = Arrays.copyOf(binWeights, 2 * binWeights.length);
				bins.add(c);
				binWeights[bins.size() - 1] = BIN_CAPACITY - c.getCardinalityRatio();
			}
		}

		return bins;
	}

	/**
	 * This methods verifies the coCoded bins actually are the best combinations within each individual Group based on
	 * the sample.
	 * 
	 * @param bins The bins constructed based on lightweight estimations
	 * @param k    The number of threads allowed to be used.
	 * @param est  The Estimator to be used.
	 * @return The cocoded columns
	 */
	private CompressedSizeInfo getCoCodingGroupsBruteForce(CompressedSizeInfo bins, int k) {

		List<CompressedSizeInfoColGroup> finalGroups = new ArrayList<>();
		// For each bin of columns that is allowed to potentially cocode.
		for(CompressedSizeInfoColGroup bin : bins.getInfo()) {
			final int len = bin.getColumns().length;
			if(len == 0)
				continue;
			else if(len == 1)
				// early termination
				finalGroups.add(bin);
			else
				finalGroups.addAll(coCodeBruteForce(bin));
		}

		bins.setInfo(finalGroups);
		return bins;
	}

	private List<CompressedSizeInfoColGroup> coCodeBruteForce(CompressedSizeInfoColGroup bin) {

		List<int[]> workset = new ArrayList<>(bin.getColumns().length);

		for(int i = 0; i < bin.getColumns().length; i++)
			workset.add(new int[] {bin.getColumns()[i]});

		// process merging iterations until no more change
		while(workset.size() > 1) {
			long changeInSize = 0;
			CompressedSizeInfoColGroup tmp = null;
			int[] selected1 = null, selected2 = null;
			for(int i = 0; i < workset.size(); i++) {
				for(int j = i + 1; j < workset.size(); j++) {
					final int[] c1 = workset.get(i);
					final int[] c2 = workset.get(j);
					final long sizeC1 = mem.get(c1).getMinSize();
					final long sizeC2 = mem.get(c2).getMinSize();

					mem.incst1();
					// pruning filter : skip dominated candidates
					// Since even if the entire size of one of the column lists is removed,
					// it still does not improve compression
					if(-Math.min(sizeC1, sizeC2) > changeInSize)
						continue;

					// Join the two column groups.
					// and Memorize the new join.
					final CompressedSizeInfoColGroup c1c2Inf = mem.getOrCreate(c1, c2);
					final long sizeC1C2 = c1c2Inf.getMinSize();

					long newSizeChangeIfSelected = sizeC1C2 - sizeC1 - sizeC2;
					// Select the best join of either the currently selected
					// or keep the old one.
					if((tmp == null && newSizeChangeIfSelected < changeInSize) || tmp != null &&
						(newSizeChangeIfSelected < changeInSize || newSizeChangeIfSelected == changeInSize &&
							c1c2Inf.getColumns().length < tmp.getColumns().length)) {
						changeInSize = newSizeChangeIfSelected;
						tmp = c1c2Inf;
						selected1 = c1;
						selected2 = c2;
					}
				}
			}

			if(tmp != null) {
				workset.remove(selected1);
				workset.remove(selected2);
				workset.add(tmp.getColumns());
			}
			else
				break;
		}

		LOG.debug(mem.stats());
		mem.resetStats();

		List<CompressedSizeInfoColGroup> ret = new ArrayList<>(workset.size());

		for(int[] w : workset)
			ret.add(mem.get(w));

		return ret;
	}

	protected class Memorizer {
		private final Map<ColIndexes, CompressedSizeInfoColGroup> mem;
		private int st1 = 0, st2 = 0, st3 = 0, st4 = 0;

		public Memorizer() {
			mem = new HashMap<>();
		}

		public void put(CompressedSizeInfoColGroup g) {
			mem.put(new ColIndexes(g.getColumns()), g);
		}

		public CompressedSizeInfoColGroup get(CompressedSizeInfoColGroup g) {
			return mem.get(new ColIndexes(g.getColumns()));
		}

		public CompressedSizeInfoColGroup get(int[] c) {
			return mem.get(new ColIndexes(c));
		}

		public CompressedSizeInfoColGroup getOrCreate(int[] c1, int[] c2) {
			final int[] c = Util.join(c1, c2);
			final ColIndexes cI = new ColIndexes(Util.join(c1, c2));
			CompressedSizeInfoColGroup g = mem.get(cI);
			st2++;
			if(g == null) {
				final CompressedSizeInfoColGroup left = mem.get(new ColIndexes(c1));
				final CompressedSizeInfoColGroup right = mem.get(new ColIndexes(c2));
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
					g = _sest.estimateJoinCompressedSize(c, left, right);
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

		public ColIndexes(int[] indexes) {
			_indexes = indexes;
		}

		@Override
		public int hashCode() {
			return Arrays.hashCode(_indexes);
		}

		@Override
		public boolean equals(Object that) {
			ColIndexes thatGrp = (ColIndexes) that;
			return Arrays.equals(_indexes, thatGrp._indexes);
		}
	}
}
