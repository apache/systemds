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

package org.apache.sysml.runtime.compress;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Used for the finding columns to co-code
 * 
 */
public class PlanningBinPacker 
{
	private final float _binWeight;
	private final List<Integer> _items;
	private final List<Float> _itemWeights;

	public PlanningBinPacker(float binWeight, List<Integer> items, List<Float> itemWeights) {
		_binWeight = binWeight;
		_items = items;
		_itemWeights = itemWeights;
	}

	/**
	 * NOTE: upper bound is 17/10 OPT
	 * 
	 * @return key: available space, value: list of the bins that have that free space
	 */
	public TreeMap<Float, List<List<Integer>>> packFirstFit() {
		return packFirstFit(_items, _itemWeights);
	}

	private TreeMap<Float, List<List<Integer>>> packFirstFit(List<Integer> items, List<Float> itemWeights) 
	{
		// when searching for a bin, the first bin in the list is used
		TreeMap<Float, List<List<Integer>>> bins = new TreeMap<Float, List<List<Integer>>>();
		// first bin
		bins.put(_binWeight, createBinList());
		int numItems = items.size();
		for (int i = 0; i < numItems; i++) {
			float itemWeight = itemWeights.get(i);
			Map.Entry<Float, List<List<Integer>>> entry = bins
					.ceilingEntry(itemWeight);
			if (entry == null) {
				// new bin
				float newBinWeight = _binWeight - itemWeight;
				List<List<Integer>> binList = bins.get(newBinWeight);
				if (binList == null) {
					bins.put(newBinWeight, createBinList(items.get(i)));
				} else {
					List<Integer> newBin = new ArrayList<Integer>();
					newBin.add(items.get(i));
					binList.add(newBin);
				}
			} else {
				// add to the first bin in the list
				List<Integer> assignedBin = entry.getValue().remove(0);
				assignedBin.add(items.get(i));
				if (entry.getValue().size() == 0)
					bins.remove(entry.getKey());
				float newBinWeight = entry.getKey() - itemWeight;
				List<List<Integer>> newBinsList = bins.get(newBinWeight);
				if (newBinsList == null) {
					// new bin
					bins.put(newBinWeight, createBinList(assignedBin));
				} else {
					newBinsList.add(assignedBin);
				}
			}
		}
		return bins;
	}

	private List<List<Integer>> createBinList() {
		List<List<Integer>> binList = new ArrayList<List<Integer>>();
		binList.add(new ArrayList<Integer>());
		return binList;
	}

	private List<List<Integer>> createBinList(int item) {
		List<List<Integer>> binList = new ArrayList<List<Integer>>();
		List<Integer> bin = new ArrayList<Integer>();
		binList.add(bin);
		bin.add(item);
		return binList;
	}

	private List<List<Integer>> createBinList(List<Integer> bin) {
		List<List<Integer>> binList = new ArrayList<List<Integer>>();
		binList.add(bin);
		return binList;
	}
}
