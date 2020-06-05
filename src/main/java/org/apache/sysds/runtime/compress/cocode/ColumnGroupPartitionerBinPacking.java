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
import java.util.stream.Collectors;

import org.apache.commons.lang.ArrayUtils;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.cocode.PlanningCoCoder.GroupableColInfo;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.util.SortUtils;

/**
 * Column group partitioning with bin packing heuristic.
 */
public class ColumnGroupPartitionerBinPacking extends ColumnGroupPartitioner {
	private static final boolean FIRST_FIT_DEC = true;
	private static final int MAX_COL_FIRST_FIT = 16384;
	private static final int MAX_COL_PER_GROUP = 1024;

	// we use a constant partition size (independent of the number of rows
	// in order to ensure constant compression speed independent of blocking)
	public static double BIN_CAPACITY = 0.000032; // higher values, more grouping

	@Override
	public List<int[]> partitionColumns(List<Integer> groupCols, HashMap<Integer, GroupableColInfo> groupColsInfo,
		CompressionSettings cs) {
		// obtain column weights
		int[] items = new int[groupCols.size()];
		double[] itemWeights = new double[groupCols.size()];
		for(int i = 0; i < groupCols.size(); i++) {
			int col = groupCols.get(i);
			items[i] = col;
			itemWeights[i] = groupColsInfo.get(col).cardRatio;
		}

		// run first fit heuristic over sequences of at most MAX_COL_FIRST_FIT
		// items to ensure robustness for matrices with many columns due to O(n^2)
		List<IntArrayList> bins = new ArrayList<>();
		for(int i = 0; i < items.length; i += MAX_COL_FIRST_FIT) {
			// extract sequence of items and item weights
			int iu = Math.min(i + MAX_COL_FIRST_FIT, items.length);
			int[] litems = Arrays.copyOfRange(items, i, iu);
			double[] litemWeights = Arrays.copyOfRange(itemWeights, i, iu);

			// sort items (first fit decreasing)
			if(FIRST_FIT_DEC) {
				SortUtils.sortByValue(0, litems.length, litemWeights, litems);
				ArrayUtils.reverse(litems);
				ArrayUtils.reverse(litemWeights);
			}

			// partition columns via bin packing
			bins.addAll(packFirstFit(litems, litemWeights));
		}

		// extract native int arrays for individual bins
		return bins.stream().map(b -> b.extractValues(true)).collect(Collectors.toList());
	}

	/**
	 * NOTE: upper bound is 17/10 OPT
	 * 
	 * @param items       the items in terms of columns
	 * @param itemWeights the weights of the items
	 * @return
	 */
	private static List<IntArrayList> packFirstFit(int[] items, double[] itemWeights) {
		List<IntArrayList> bins = new ArrayList<>();
		double[] binWeights = new double[16];

		for(int i = 0; i < items.length; i++) {
			// add to existing bin
			boolean assigned = false;
			for(int j = 0; j < bins.size(); j++) {
				double newBinWeight = binWeights[j] - itemWeights[i];
				if(newBinWeight >= 0 && bins.get(j).size() < MAX_COL_PER_GROUP - 1) {
					bins.get(j).appendValue(items[i]);
					binWeights[j] = newBinWeight;
					assigned = true;
					break;
				}
			}

			// create new bin at end of list
			if(!assigned) {
				if(bins.size() == binWeights.length)
					binWeights = Arrays.copyOf(binWeights, 2 * binWeights.length);
				bins.add(new IntArrayList(items[i]));
				binWeights[bins.size() - 1] = BIN_CAPACITY - itemWeights[i];
			}
		}

		return bins;
	}
}
