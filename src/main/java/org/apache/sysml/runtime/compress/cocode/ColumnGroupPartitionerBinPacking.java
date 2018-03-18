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

package org.apache.sysml.runtime.compress.cocode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;
import org.apache.sysml.runtime.compress.cocode.PlanningCoCoder.GroupableColInfo;
import org.apache.sysml.runtime.util.SortUtils;

/**
 * Column group partitioning with bin packing heuristic.
 * 
 */
public class ColumnGroupPartitionerBinPacking extends ColumnGroupPartitioner
{
	private static final boolean FIRST_FIT_DEC = true;
	private static final int MAX_COL_PER_GROUP = Integer.MAX_VALUE;

	//we use a constant partition size (independent of the number of rows
	//in order to ensure constant compression speed independent of blocking)
	public static double BIN_CAPACITY = 0.000032; //higher values, more grouping
	
	@Override
	public List<List<Integer>> partitionColumns(List<Integer> groupCols, HashMap<Integer, GroupableColInfo> groupColsInfo) 
	{
		//obtain column weights
		int[] items = new int[groupCols.size()];
		double[] itemWeights = new double[groupCols.size()];
		for( int i=0; i<groupCols.size(); i++ ) {
			int col = groupCols.get(i);
			items[i] = col;
			itemWeights[i] = groupColsInfo.get(col).cardRatio;
		} 
		
		//sort items (first fit decreasing)
		if( FIRST_FIT_DEC ) {
			SortUtils.sortByValue(0, items.length, itemWeights, items);
			ArrayUtils.reverse(items);
			ArrayUtils.reverse(itemWeights);
		}
		
		//partition columns via bin packing
		return packFirstFit(items, itemWeights);
	}

	/**
	 * NOTE: upper bound is 17/10 OPT
	 * 
	 * @param items the items in terms of columns
	 * @param itemWeights the weights of the items
	 * @return
	 */
	private static List<List<Integer>> packFirstFit(int[] items, double[] itemWeights) 
	{
		List<List<Integer>> bins = new ArrayList<>();
		List<Double> binWeights = new ArrayList<>();
		
		for( int i = 0; i < items.length; i++ ) {
			//add to existing bin
			boolean assigned = false;
			for( int j = 0; j < bins.size(); j++ ) {
				double newBinWeight = binWeights.get(j)-itemWeights[i];
				if( newBinWeight >= 0 && bins.get(j).size() < MAX_COL_PER_GROUP-1 ){
					bins.get(j).add(items[i]);
					binWeights.set(j, newBinWeight);
					assigned = true; break;
				}
			}
			
			//create new bin at end of list
			if( !assigned ) {
				bins.add(new ArrayList<>(Arrays.asList(items[i])));
				binWeights.add(BIN_CAPACITY-itemWeights[i]);
			}
		}
		
		return bins;
	}
}
