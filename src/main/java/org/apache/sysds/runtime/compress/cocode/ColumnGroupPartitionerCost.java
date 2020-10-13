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
import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;
import java.util.Queue;
import java.util.TreeMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.cocode.PlanningCoCoder.GroupableColInfo;

/**
 * Column group partitioning with static number distinct elements heuristic
 */
public class ColumnGroupPartitionerCost extends ColumnGroupPartitioner {
	private static final Log LOG = LogFactory.getLog(ColumnGroupPartitionerCost.class.getName());
	/**
	 * This value specifies the maximum distinct count allowed int a coCoded group. Note that this value is the number
	 * of distinct rows not the total number of values. That value can be calculated by multiplying with the number of
	 * rows in the coCoded group.
	 */
	private static final int largestDistinct = 256;

	@Override
	public List<int[]> partitionColumns(List<Integer> groupCols, HashMap<Integer, GroupableColInfo> groupColsInfo,
		CompressionSettings cs) {

		TreeMap<Integer, Queue<Queue<Integer>>> distToColId = new TreeMap<>();
		for(Entry<Integer, GroupableColInfo> ent : groupColsInfo.entrySet()) {
			int distinct = ent.getValue().nrDistinct;
			if(distToColId.containsKey(distinct)) {
				Queue<Integer> cocodeGroup = new LinkedList<>();
				cocodeGroup.add(ent.getKey());
				distToColId.get(distinct).add(cocodeGroup);
			}
			else {
				Queue<Queue<Integer>> cocodeGroups = new LinkedList<>();
				Queue<Integer> cocodeGroup = new LinkedList<>();
				cocodeGroup.add(ent.getKey());
				cocodeGroups.add(cocodeGroup);
				distToColId.put(distinct, cocodeGroups);
			}
		}

		boolean change = false;
		while(distToColId.firstKey() < largestDistinct) {
			Entry<Integer, Queue<Queue<Integer>>> elm = distToColId.pollFirstEntry();
			if(elm.getValue().size() > 1) {
				int distinctCombinations = elm.getKey()>0 ? elm.getKey() : 1;
				Queue<Queue<Integer>> group = elm.getValue();
				int size = group.size();
				if(Math.pow(distinctCombinations, size) < largestDistinct) {
					Queue<Integer> t = elm.getValue().stream().reduce(new LinkedList<>(), (acc, e) -> {
						acc.addAll(e);
						return acc;
					});
					elm.getValue().clear();
					if(distToColId.containsKey((int) Math.pow(distinctCombinations, size))){
						distToColId.get((int) Math.pow(distinctCombinations, size)).add(t);
					}else{
						elm.getValue().add(t);
						distToColId.put((int) Math.pow(distinctCombinations, size), elm.getValue());
					}
					change = true;
				}
				else if(distinctCombinations * distinctCombinations < largestDistinct) {
					Queue<Integer> cols = elm.getValue().poll();
					cols.addAll(elm.getValue().poll());
					if(distToColId.containsKey(distinctCombinations * distinctCombinations)) {
						Queue<Queue<Integer>> p = distToColId.get(distinctCombinations * distinctCombinations);
						p.add(cols);
					}
					else {
						Queue<Queue<Integer>> n = new LinkedList<>();
						n.add(cols);
						distToColId.put(distinctCombinations * distinctCombinations, n);
					}
					if(elm.getValue().size() > 0) {
						distToColId.put(elm.getKey(), elm.getValue());
					}
					change = true;
				}
				else {
					change = false;
					distToColId.put(elm.getKey(), elm.getValue());
				}
			}
			else if(!distToColId.isEmpty()) {
				Entry<Integer, Queue<Queue<Integer>>> elm2 = distToColId.pollFirstEntry();
				int size1 = elm.getKey()>0 ? elm.getKey() : 1;
				int size2 = elm2.getKey()>0 ? elm2.getKey() : 1;
				if(size1 * size2 < largestDistinct) {
					Queue<Integer> cols = elm.getValue().poll();
					cols.addAll(elm2.getValue().poll());
					if(elm2.getKey() == size1 * size2){
						elm2.getValue().add(cols);
					}
					else if(distToColId.containsKey(size1 * size2)) {
						distToColId.get(size1 * size2).add(cols);
					}
					else {
						Queue<Queue<Integer>> n = new LinkedList<>();
						n.add(cols);
						distToColId.put(size1 * size2, n);
					}
					if(elm.getValue().size() > 0) {
						distToColId.put(elm.getKey(), elm.getValue());
					}
					if(elm2.getValue().size() > 0) {
						distToColId.put(elm2.getKey(), elm2.getValue());
					}
					change = true;
				}
				else {
					change = false;
					distToColId.put(elm.getKey(), elm.getValue());
					distToColId.put(elm2.getKey(), elm2.getValue());
				}
			}
			else {
				distToColId.put(elm.getKey(), elm.getValue());
				break;
			}
			if(!change)
				break;
		}
		List<int[]> ret = new ArrayList<>();

		for(Queue<Queue<Integer>> x : distToColId.values())
			for(Queue<Integer> y : x) {
				int[] g = new int[y.size()];
				int idx = 0;
				for(Integer id : y)
					g[idx++] = id;
				Arrays.sort(g);
				ret.add(g);
			}

		if(LOG.isDebugEnabled()){
			StringBuilder sb = new StringBuilder();
			for(int[] cg : ret)
				sb.append(Arrays.toString(cg));
			LOG.debug(sb.toString());
		}
		return ret;
	}
}
