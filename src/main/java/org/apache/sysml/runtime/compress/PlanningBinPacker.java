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
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.commons.math3.random.RandomDataGenerator;

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

	/**
	 * shuffling the items to make some potential for having bins of different
	 * sizes when consecutive columns are of close cardinalities
	 * 
	 * @return key: available space, value: list of the bins that have that free
	 *         space
	 */
	public TreeMap<Float, List<List<Integer>>> packFirstFitShuffled() {
		RandomDataGenerator rnd = new RandomDataGenerator();
		int[] permutation = rnd.nextPermutation(_items.size(), _items.size());
		List<Integer> shuffledItems = new ArrayList<Integer>(_items.size());
		List<Float> shuffledWeights = new ArrayList<Float>(_items.size());
		for (int ix : permutation) {
			shuffledItems.add(_items.get(ix));
			shuffledWeights.add(_itemWeights.get(ix));
		}

		return packFirstFit(shuffledItems, shuffledWeights);
	}

	/**
	 * 
	 * @param items
	 * @param itemWeights
	 * @return
	 */
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

	/**
	 * NOTE: upper bound is 11/9 OPT + 6/9 (~1.22 OPT)
	 * 
	 * @return
	 */
	public TreeMap<Float, List<List<Integer>>> packFirstFitDescending() {
		// sort items descending based on their weights
		Integer[] indexes = new Integer[_items.size()];
		for (int i = 0; i < indexes.length; i++)
			indexes[i] = i;
		Arrays.sort(indexes, new Comparator<Integer>() {

			@Override
			public int compare(Integer o1, Integer o2) {
				return _itemWeights.get(o1).compareTo(_itemWeights.get(o2));
			}
		});
		List<Integer> sortedItems = new ArrayList<Integer>();
		List<Float> sortedItemWeights = new ArrayList<Float>();
		for (int i = indexes.length - 1; i >= 0; i--) {
			sortedItems.add(_items.get(i));
			sortedItemWeights.add(_itemWeights.get(i));
		}
		return packFirstFit(sortedItems, sortedItemWeights);
	}

	/**
	 * NOTE: upper bound is 71/60 OPT + 6/9 (~1.18 OPT)
	 * 
	 * @return
	 */
	public TreeMap<Float, List<List<Integer>>> packModifiedFirstFitDescending() {
		throw new UnsupportedOperationException("Not implemented yet!");
	}

	/**
	 * 
	 * @return
	 */
	private List<List<Integer>> createBinList() {
		List<List<Integer>> binList = new ArrayList<List<Integer>>();
		binList.add(new ArrayList<Integer>());
		return binList;
	}

	/**
	 * 
	 * @param item
	 * @return
	 */
	private List<List<Integer>> createBinList(int item) {
		List<List<Integer>> binList = new ArrayList<List<Integer>>();
		List<Integer> bin = new ArrayList<Integer>();
		binList.add(bin);
		bin.add(item);
		return binList;
	}

	/**
	 * 
	 * @param bin
	 * @return
	 */
	private List<List<Integer>> createBinList(List<Integer> bin) {
		List<List<Integer>> binList = new ArrayList<List<Integer>>();
		binList.add(bin);
		return binList;
	}
}
