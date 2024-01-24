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

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.OperatorOrderingUtils;

public class PipelineAwareLinearize {

	// Minimum number of nodes in DAG for applying algorithm
	private final static int IGNORE_LIMIT = 0;

	// Relevant parameter for Step 3:	
	// (Force) merge for pipelines size of [0, ..., HARD_LIMIT]
	private final static int HARD_LIMIT = 4;
	// Merges two pipelines if p1.size() + p2.size() < UPPER_BOUND
	private final static int UPPER_BOUND = 10;

    /**
	 * Sort lops depth-first while assigning the nodes to pipelines
	 * 
	 * @param v List of lops to sort
	 * @return Sorted list of lops with set _pipelineID on the Lop Object
	 */
	public static List<Lop> pipelineDepthFirst(List<Lop> v) {

		// If size of DAG is smaller than IGNORE_LIMIT, give all nodes the same pipeline id
		if(v.size() <= IGNORE_LIMIT) {
			v.forEach(l -> l.setPipelineID(1));
			return ILinearize.depthFirst(v);
		}

		// Find all root nodes (starting points for the depth-first traversal)
		List<Lop> roots = v.stream()
			.filter(OperatorOrderingUtils::isLopRoot)
			.collect(Collectors.toList());

		// Initialize necessary data objects
		Integer pipelineId = 0;
		// Stores a resulting depth first sorted list of lops (same as in depthFirst())
		// Returned by this function
		ArrayList<Lop> opList = new ArrayList<>();
		// Stores the pipeline ids and the corresponding lops 
		// for further refinement of pipeline assignements
		Map<Integer, List<Lop>> pipelineMap = new HashMap<>();

		// Step 1: Depth-first assignment of pipeline ids to the roots
		for (Lop r : roots) {
			pipelineId = depthFirst(r, pipelineId, opList, pipelineMap) + 1;
		}
		DEVPrintDAG.asGraphviz("Step1", v);

		// Step 2: Merge pipelines with only one node to another (connected) pipeline
		PipelineAwareLinearize.mergeSingleNodePipelines(pipelineMap);
		DEVPrintDAG.asGraphviz("Step2", v);

		// Step 3: Merge small pipelines into bigger ones
		PipelineAwareLinearize.mergeSmallPipelines(pipelineMap);
		DEVPrintDAG.asGraphviz("Step3", v);

		// Reset the visited status of all nodes
		roots.forEach(Lop::resetVisitStatus);

		return opList;
	}

	// Step 1: Depth-first assignment of pipeline ids to the roots
	// Finds the branching out/in of Lops, that could be parallized 
	// (and with it assiging of different pipeline ids)
	private static int depthFirst(Lop root, int pipelineId, List<Lop> opList, Map<Integer, List<Lop>> pipelineMap) {

		// Abort if the node was already visited
		if (root.isVisited()) {
			return root.getPipelineID();
		}

		// Assign pipeline id to the node, given by the parent
		// Set the root node as visited
		root.setPipelineID(pipelineId);
		root.setVisited();

		// Add the root node to the pipeline list
		if(pipelineMap.containsKey(pipelineId)) {
			pipelineMap.get(pipelineId).add(root);
		} else {
			ArrayList<Lop> lopList = new ArrayList<>();
			lopList.add(root);
			pipelineMap.put(pipelineId, lopList);
		}

		// Children as inputs, as we are traversing the lops bottom up
		List<Lop> children = root.getInputs();
		// If root node has only one child, use the same pipeline id as root node
		if (children.size() == 1) {
			Lop child = children.get(0);
			// We need to find the max pipeline id of the child, because the child could branch out
			pipelineId = Math.max(pipelineId, depthFirst(child, pipelineId, opList, pipelineMap));
		} else {
			// Iteration over all children
			for (int i = 0; i < children.size(); i++) {
				Lop child = children.get(i);
				
				// If the child has only one output, or all outputs are the root node, use the same pipeline id as parent
				if(child.getOutputs().size() == 1 || 
				  (child.getOutputs().size() > 1 && child.getOutputs().stream().allMatch(o -> o == root))) {
					// No need for max, because the child can only have one output
					depthFirst(child, root.getPipelineID(), opList, pipelineMap);
				} else {
					// We need to find the max pipeline id of the child, because the child could branch out
					pipelineId = Math.max(pipelineId, depthFirst(child, pipelineId + 1, opList, pipelineMap));
				}
			}
		}

		opList.add(root);
		return pipelineId;
	}

	// Step 2: Merge pipelines with only one node to another (connected) pipeline
	// Return map by reference
	private static void mergeSingleNodePipelines(Map<Integer, List<Lop>> map) {

		Map<Integer, List<Lop>> pipelinesWithOneNode = map.entrySet().stream()
			.filter(e -> e.getValue().size() == 1)
			.collect(Collectors.toMap( e-> e.getKey(), e -> e.getValue()));
		
		if(pipelinesWithOneNode.size() == 0)
			return;

		pipelinesWithOneNode.entrySet().stream().forEach(e -> {
			Lop lop = e.getValue().get(0);
			
			// Merge to an existing output node
			if (lop.getOutputs().size() > 0) {
				lop.setPipelineID(lop.getOutputs().get(0).getPipelineID());
			// If no outputs are present, merge to an existing input node
			} else if (lop.getInputs().size() > 0) {
				lop.setPipelineID(lop.getInputs().get(0).getPipelineID());
			}
			// else (no inputs and no outputs): do nothing (unreachable node?)
			// Remove the pipeline from the list of pipelines
			if (lop.getOutputs().size() > 0 || lop.getInputs().size() > 0) {
				map.get(lop.getPipelineID()).add(lop);
				map.remove(e.getKey());
			}
		});
	}

	// Step 3: Merge small pipelines into bigger ones
	// Heuristic: Merge the smallest pipeline with the second smallest pipeline
	// We don't care about whether the pipelines are connected or not
	// This reduces the overhead as we avoid calculating the entire combinatorial problem space 
	// for finding an optimal solution.
	// An optimal solution could be defined as a solution that reduces unnecessary overhead from
	// too small pipelines (if executed in parallel, e.g., in a separate thread) 
	// and still find a maximum number of pipelines (for maximal parallelization)

	// A proposed way to achieve a balance between avoiding too small pipelines and maximizing the number of pipelines:
	// HARD_LIMIT: If the size of a pipeline is smaller than HARD_LIMIT, it will be merged with the next smallest pipeline.
	// UPPER_BOUND: If the combined size of the two smallest pipelines is less than UPPER_BOUND, merge them.
	// Return map by reference
	private static void mergeSmallPipelines(Map<Integer, List<Lop>> map) {

		// Needs to have atleast two pipelines
		if(map.size() < 2)
			return;

		// Sort the pipelines by size
		List<Map.Entry<Integer, Integer>> sortedPipelineSizes = getPipelinesSortedBySize(map);

		Map.Entry<Integer, Integer> sm0 = sortedPipelineSizes.get(0);
		Map.Entry<Integer, Integer> sm1 = sortedPipelineSizes.get(1);

		while((sm0 != null && sm1 != null) &&
  			  (sm0.getValue() < HARD_LIMIT ||
			   sm0.getValue() + sm1.getValue() < UPPER_BOUND)
		) {

			// Merge pipelines as they satifiy the conditions
			int mergeIntoId = sm1.getKey();
			map.get(sm0.getKey()).forEach(l -> l.setPipelineID(mergeIntoId));
			map.get(mergeIntoId).addAll(map.get(sm0.getKey()));
			map.remove(sm0.getKey());

			//Get new list of sizes from updated map!
			sortedPipelineSizes = getPipelinesSortedBySize(map);
			
			//Update sm0 and sm1, if possible
			if(sortedPipelineSizes.size() < 2) {
				sm0 = null;
				sm1 = null;
			}
			else {
				sm0 = sortedPipelineSizes.get(0);
				sm1 = sortedPipelineSizes.get(1);
			}
		}
	}

	private static List<Map.Entry<Integer, Integer>> getPipelinesSortedBySize(Map<Integer, List<Lop>> map) {
		return map.entrySet().stream()
				.sorted(Map.Entry.comparingByValue(Comparator.comparingInt(List::size)))
				.map(e -> Map.entry(e.getKey(), e.getValue().size()))
				.collect(Collectors.toList());
	}

}