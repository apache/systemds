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

package org.apache.sysds.lops.compile.linearization;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.Lop;

/**
 * A interface for the linearization algorithms that order the DAG nodes into a sequence of instructions to execute.
 * 
 * https://en.wikipedia.org/wiki/Linearizability#Linearization_points
 */
public interface ILinearize {
	public static Log LOG = LogFactory.getLog(ILinearize.class.getName());

	public enum DagLinearization {
		DEPTH_FIRST, BREADTH_FIRST, MIN_INTERMEDIATE
	}

	public static List<Lop> linearize(List<Lop> v) {
		try {
			DMLConfig dmlConfig = ConfigurationManager.getDMLConfig();
			DagLinearization linearization = DagLinearization
				.valueOf(dmlConfig.getTextValue(DMLConfig.DAG_LINEARIZATION).toUpperCase());

			switch(linearization) {
				case MIN_INTERMEDIATE:
					return doMinIntermediateSort(v);
				case BREADTH_FIRST:
					return doBreadthFirstSort(v);
				case DEPTH_FIRST:
				default:
					return depthFirst(v);
			}
		}
		catch(Exception e) {
			LOG.warn("Invalid or failed DAG_LINEARIZATION, fallback to DEPTH_FIRST ordering");
			return depthFirst(v);
		}
	}

	/**
	 * Sort lops depth-first
	 * 
	 * previously called doTopologicalSortTwoLevelOrder
	 * 
	 * @param v List of lops to sort
	 * @return Sorted list of lops
	 */
	private static List<Lop> depthFirst(List<Lop> v) {
		// partition nodes into leaf/inner nodes and dag root nodes,
		// + sort leaf/inner nodes by ID to force depth-first scheduling
		// + append root nodes in order of their original definition
		// (which also preserves the original order of prints)
		List<Lop> nodes = Stream
			.concat(v.stream().filter(l -> !l.getOutputs().isEmpty()).sorted(Comparator.comparing(l -> l.getID())),
				v.stream().filter(l -> l.getOutputs().isEmpty()))
			.collect(Collectors.toList());

		// NOTE: in contrast to hadoop execution modes, we avoid computing the transitive
		// closure here to ensure linear time complexity because its unnecessary for CP and Spark
		return nodes;
	}

	private static List<Lop> doBreadthFirstSort(List<Lop> v) {
		List<Lop> nodes = v.stream().sorted(Comparator.comparing(Lop::getLevel)).collect(Collectors.toList());

		return nodes;
	}

	/**
	 * Sort lops to execute them in an order that minimizes the memory requirements of intermediates
	 * 
	 * @param v List of lops to sort
	 * @return Sorted list of lops
	 */
	private static List<Lop> doMinIntermediateSort(List<Lop> v) {
		List<Lop> nodes = new ArrayList<>(v.size());
		// Get the lowest level in the tree to move upwards from
		List<Lop> lowestLevel = v.stream().filter(l -> l.getOutputs().isEmpty()).collect(Collectors.toList());

		// Traverse the tree bottom up, choose nodes with higher memory requirements, then reverse the list
		List<Lop> remaining = new LinkedList<>(v);
		sortRecursive(nodes, lowestLevel, remaining);

		// In some cases (function calls) some output lops are not in the list of nodes to be sorted.
		// With the next layer up having output lops, they are not added to the initial list of lops and are
		// subsequently never reached by the recursive sort.
		// We work around this issue by checking for remaining lops after the initial sort.
		while(!remaining.isEmpty()) {
			// Start with the lowest level lops, this time by level instead of no outputs
			int maxLevel = remaining.stream().mapToInt(Lop::getLevel).max().orElse(-1);
			List<Lop> lowestNodes = remaining.stream().filter(l -> l.getLevel() == maxLevel).collect(Collectors.toList());
			sortRecursive(nodes, lowestNodes, remaining);
		}

		// All lops were added bottom up, from highest to lowest memory consumption, now reverse this
		Collections.reverse(nodes);

		return nodes;
	}

	private static void sortRecursive(List<Lop> result, List<Lop> input, List<Lop> remaining) {
		// Sort input lops by memory estimate
		// Lowest level nodes (those with no outputs) receive a memory estimate of 0 to preserve order
		// This affects prints, writes, ...
		List<Map.Entry<Lop, Long>> memEst = input.stream().distinct().map(l -> new AbstractMap.SimpleEntry<>(l,
			l.getOutputs().isEmpty() ? 0 : OptimizerUtils.estimateSizeExactSparsity(l.getOutputParameters().getNumRows(),
				l.getOutputParameters().getNumCols(), l.getOutputParameters().getNnz())))
			.sorted(Comparator.comparing(e -> ((Map.Entry<Lop, Long>) e).getValue())).collect(Collectors.toList());

		// Start with the highest memory estimate because the entire list is reversed later
		Collections.reverse(memEst);
		for(Map.Entry<Lop, Long> e : memEst) {
			// Skip if the node is already in the result list
			// Skip if one of the lop's outputs is not in the result list yet (will be added once the output lop is
			// traversed), but only if any of the output lops is bound to be added to the result at a later stage
			if(result.contains(e.getKey()) || (!result.containsAll(e.getKey().getOutputs()) &&
				remaining.stream().anyMatch(l -> e.getKey().getOutputs().contains(l))))
				continue;
			result.add(e.getKey());
			remaining.remove(e.getKey());
			// Add input lops recursively
			sortRecursive(result, e.getKey().getInputs(), remaining);
		}
	}
}
