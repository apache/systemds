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

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.cost.ACostEstimate;
import org.apache.sysds.runtime.compress.estim.AComEst;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;

/**
 * Column group partitioning with bin packing heuristic.
 */
public class CoCodeBinPacking extends AColumnCoCoder {

	// private static final boolean FIRST_FIT_DEC = true;
	// private static final int MAX_COL_FIRST_FIT = 16384;
	// private static final int MAX_COL_PER_GROUP = 1024;

	// private final Memorizer mem;

	/**
	 * Use a constant partition size independent of the number of rows in order to ensure constant compression speed
	 * independent of blocking. Higher values gives more CoCoding at the cost of longer compressionTimes.
	 * 
	 * VLDB 2018 Paper: Default 0.000032
	 */
	// private static double BIN_CAPACITY = 0.000032;

	protected CoCodeBinPacking(AComEst sizeEstimator, ACostEstimate costEstimator,
		CompressionSettings cs) {
		super(sizeEstimator, costEstimator, cs);
		// mem = new Memorizer(sizeEstimator);
	}

	@Override
	protected CompressedSizeInfo coCodeColumns(CompressedSizeInfo colInfos, int k) {
		// establish memo table for extracted column groups
		
		throw new NotImplementedException("Not Implemented generic cost function in binPacking");
		// List<CompressedSizeInfoColGroup> constantGroups = new ArrayList<>();
		// List<CompressedSizeInfoColGroup> newGroups = new ArrayList<>();

		// for(CompressedSizeInfoColGroup g : colInfos.getInfo()) {
		// 	if(g.getBestCompressionType(_cs) == CompressionType.CONST)
		// 		constantGroups.add(g);
		// 	else {
		// 		mem.put(g);
		// 		newGroups.add(g);
		// 	}
		// }

		// make bins
		// colInfos.setInfo(partitionColumns(newGroups));
		// Cocode compare all in bins
		// getCoCodingGroupsBruteForce(colInfos, k);

		// colInfos.getInfo().addAll(constantGroups);


		// return colInfos;
	}

	/**
	 * Partition columns such that we make sure that the columns we analyze further actually have the potential of
	 * compressing together.
	 * 
	 * @param currentGroups The individual column groups
	 * @return The Partitioned column groups
	 */
	// private List<CompressedSizeInfoColGroup> partitionColumns(List<CompressedSizeInfoColGroup> currentGroups) {

	// 	List<CompressedSizeInfoColGroup> bins = new ArrayList<>();

	// 	for(int i = 0; i < currentGroups.size(); i += MAX_COL_FIRST_FIT) {
	// 		// extract subset of columns
	// 		int iu = Math.min(i + MAX_COL_FIRST_FIT, currentGroups.size());
	// 		List<CompressedSizeInfoColGroup> currentGroupsSubset = currentGroups.subList(i, iu);

	// 		// sort items based on cardinality
	// 		if(FIRST_FIT_DEC) {
	// 			currentGroupsSubset.sort(new Comparator<CompressedSizeInfoColGroup>() {
	// 				public int compare(CompressedSizeInfoColGroup a, CompressedSizeInfoColGroup b) {
	// 					final double ac = a.getCardinalityRatio();
	// 					final double bc = b.getCardinalityRatio();
	// 					if(ac == bc)
	// 						return 0;
	// 					else if(ac > bc)
	// 						return 1;
	// 					else
	// 						return -1;
	// 				}
	// 			});
	// 		}
	// 		// partition columns via bin packing
	// 		bins.addAll(packFirstFit(currentGroupsSubset));
	// 	}

	// 	return bins;
	// }

	/**
	 * Pack into the bins such that no bin exceed the Bin Capacity. This ensure that each bin only contains columns that
	 * are amenable to compression. And potentially CoCoding.
	 * 
	 * @param currentGroups Column groups in current batch
	 * @return The currently selected groups, packed together in bins.
	 */
	// private List<CompressedSizeInfoColGroup> packFirstFit(List<CompressedSizeInfoColGroup> currentGroups) {
	// 	List<CompressedSizeInfoColGroup> bins = new ArrayList<>();
	// 	double[] binWeights = new double[16];

	// 	for(int i = 0; i < currentGroups.size(); i++) {
	// 		final CompressedSizeInfoColGroup c = currentGroups.get(i);
	// 		// add to existing bin
	// 		boolean assigned = false;
	// 		for(int j = 0; j < bins.size(); j++) {
	// 			double newBinWeight = binWeights[j] - c.getCardinalityRatio();
	// 			if(newBinWeight >= 0 && bins.get(j).getColumns().length < MAX_COL_PER_GROUP - 1) {
	// 				bins.set(j, joinWithoutAnalysis(Util.combine(bins.get(j).getColumns(), c.getColumns()), bins.get(j), c));
	// 				binWeights[j] = newBinWeight;
	// 				assigned = true;
	// 				break;
	// 			}
	// 		}

	// 		// create new bin at end of list
	// 		if(!assigned) {
	// 			if(bins.size() == binWeights.length)
	// 				binWeights = Arrays.copyOf(binWeights, 2 * binWeights.length);
	// 			bins.add(c);
	// 			binWeights[bins.size() - 1] = BIN_CAPACITY - c.getCardinalityRatio();
	// 		}
	// 	}

	// 	return bins;
	// }

	/**
	 * This methods verifies the coCoded bins actually are the best combinations within each individual Group based on
	 * the sample.
	 * 
	 * @param bins The bins constructed based on lightweight estimations
	 * @param k    The number of threads allowed to be used.
	 * @param est  The Estimator to be used.
	 * @return The cocoded columns
	 */
	// private CompressedSizeInfo getCoCodingGroupsBruteForce(CompressedSizeInfo bins, int k) {

	// 	List<CompressedSizeInfoColGroup> finalGroups = new ArrayList<>();
	// 	// For each bin of columns that is allowed to potentially cocode.
	// 	for(CompressedSizeInfoColGroup bin : bins.getInfo()) {
	// 		final int len = bin.getColumns().length;
	// 		if(len == 0)
	// 			continue;
	// 		else if(len == 1)
	// 			// early termination
	// 			finalGroups.add(bin);
	// 		else
	// 			finalGroups.addAll(coCodeBruteForce(bin));
	// 	}

	// 	bins.setInfo(finalGroups);
	// 	return bins;
	// }

// 	private List<CompressedSizeInfoColGroup> coCodeBruteForce(CompressedSizeInfoColGroup bin) {

// 		List<ColIndexes> workSet = new ArrayList<>(bin.getColumns().length);

// 		for(int b : bin.getColumns())
// 			workSet.add(new ColIndexes(new int[] {b}));

// 		// process merging iterations until no more change
// 		while(workSet.size() > 1) {
// 			long changeInSize = 0;
// 			CompressedSizeInfoColGroup tmp = null;
// 			ColIndexes selected1 = null, selected2 = null;
// 			for(int i = 0; i < workSet.size(); i++) {
// 				for(int j = i + 1; j < workSet.size(); j++) {
// 					final ColIndexes c1 = workSet.get(i);
// 					final ColIndexes c2 = workSet.get(j);
// 					final long sizeC1 = mem.get(c1).getMinSize();
// 					final long sizeC2 = mem.get(c2).getMinSize();

// 					mem.incst1();
// 					// pruning filter : skip dominated candidates
// 					// Since even if the entire size of one of the column lists is removed,
// 					// it still does not improve compression
// 					if(-Math.min(sizeC1, sizeC2) > changeInSize)
// 						continue;

// 					// Join the two column groups.
// 					// and Memorize the new join.
// 					final CompressedSizeInfoColGroup c1c2Inf = mem.getOrCreate(c1, c2);
// 					final long sizeC1C2 = c1c2Inf.getMinSize();

// 					long newSizeChangeIfSelected = sizeC1C2 - sizeC1 - sizeC2;
// 					// Select the best join of either the currently selected
// 					// or keep the old one.
// 					if((tmp == null && newSizeChangeIfSelected < changeInSize) ||
// 						tmp != null && (newSizeChangeIfSelected < changeInSize || newSizeChangeIfSelected == changeInSize &&
// 							c1c2Inf.getColumns().length < tmp.getColumns().length)) {
// 						changeInSize = newSizeChangeIfSelected;
// 						tmp = c1c2Inf;
// 						selected1 = c1;
// 						selected2 = c2;
// 					}
// 				}
// 			}

// 			if(tmp != null) {
// 				workSet.remove(selected1);
// 				workSet.remove(selected2);
// 				workSet.add(new ColIndexes(tmp.getColumns()));
// 			}
// 			else
// 				break;
// 		}

// 		if(LOG.isDebugEnabled())
// 			LOG.debug("Memorizer stats:" + mem.stats());
// 		mem.resetStats();

// 		List<CompressedSizeInfoColGroup> ret = new ArrayList<>(workSet.size());

// 		for(ColIndexes w : workSet)
// 			ret.add(mem.get(w));

// 		return ret;
// 	}
}
