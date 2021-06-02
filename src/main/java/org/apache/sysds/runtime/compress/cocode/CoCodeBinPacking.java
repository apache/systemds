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
import java.util.List;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;

/**
 * Column group partitioning with bin packing heuristic.
 */
public class CoCodeBinPacking extends AColumnCoCoder {

	private static final boolean FIRST_FIT_DEC = true;
	private static final int MAX_COL_FIRST_FIT = 16384;
	private static final int MAX_COL_PER_GROUP = 1024;

	/**
	 * Use a constant partition size independent of the number of rows in order to ensure constant compression speed
	 * independent of blocking. Higher values gives more CoCoding at the cost of longer compressionTimes.
	 * 
	 * VLDB 2018 Paper: Default 0.000032
	 */
	public static double BIN_CAPACITY = 0.000032;

	protected CoCodeBinPacking(CompressedSizeEstimator sizeEstimator, CompressionSettings cs) {
		super(sizeEstimator, cs);
	}

	@Override
	protected CompressedSizeInfo coCodeColumns(CompressedSizeInfo colInfos, int k) {
		colInfos.setInfo(partitionColumns(colInfos.getInfo()));
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
					bins.set(j, joinWithoutAnalysis(bins.get(j), c));
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
}
