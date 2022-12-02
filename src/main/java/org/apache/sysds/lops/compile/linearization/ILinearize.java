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
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.AggBinaryOp.SparkAggType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.CSVReBlock;
import org.apache.sysds.lops.CentralMoment;
import org.apache.sysds.lops.Checkpoint;
import org.apache.sysds.lops.CoVariance;
import org.apache.sysds.lops.DataGen;
import org.apache.sysds.lops.GroupedAggregate;
import org.apache.sysds.lops.GroupedAggregateM;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.MMTSJ;
import org.apache.sysds.lops.MMZip;
import org.apache.sysds.lops.MapMultChain;
import org.apache.sysds.lops.ParameterizedBuiltin;
import org.apache.sysds.lops.PickByCount;
import org.apache.sysds.lops.ReBlock;
import org.apache.sysds.lops.SpoofFused;
import org.apache.sysds.lops.UAggOuterChain;
import org.apache.sysds.lops.UnaryCP;

/**
 * A interface for the linearization algorithms that order the DAG nodes into a sequence of instructions to execute.
 *
 * https://en.wikipedia.org/wiki/Linearizability#Linearization_points
 */
public interface ILinearize {
	public static Log LOG = LogFactory.getLog(ILinearize.class.getName());

	public enum DagLinearization {
		DEPTH_FIRST, BREADTH_FIRST, MIN_INTERMEDIATE, MAX_PARALLELIZE
	}

	public static List<Lop> linearize(List<Lop> v) {
		try {
			DMLConfig dmlConfig = ConfigurationManager.getDMLConfig();
			DagLinearization linearization = ConfigurationManager.getLinearizationOrder();

			switch(linearization) {
				case MAX_PARALLELIZE:
					return doMaxParallelizeSort(v);
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
			LOG.warn("Invalid DAG_LINEARIZATION "+ConfigurationManager.getLinearizationOrder()+", fallback to DEPTH_FIRST ordering");
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

	// Place the Spark operation chains first (more expensive to less expensive),
	// followed by asynchronously triggering operators and CP chains.
	private static List<Lop> doMaxParallelizeSort(List<Lop> v)
	{
		List<Lop> final_v = null;
		if (v.stream().anyMatch(ILinearize::isSparkTriggeringOp)) {
			// Step 1: Collect the Spark roots and #Spark instructions in each subDAG
			Map<Long, Integer> sparkOpCount = new HashMap<>();
			List<Lop> roots = v.stream().filter(l -> l.getOutputs().isEmpty()).collect(Collectors.toList());
			List<Lop> sparkRoots = new ArrayList<>();
			roots.forEach(r -> collectSparkRoots(r, sparkOpCount, sparkRoots));

			// Step 2: Depth-first linearization. Place the Spark OPs first.
			// Sort the Spark roots based on number of Spark operators descending
			ArrayList<Lop> operatorList = new ArrayList<>();
			Lop[] sortedSPRoots = sparkRoots.toArray(new Lop[0]);
			Arrays.sort(sortedSPRoots, (l1, l2) -> sparkOpCount.get(l2.getID()) - sparkOpCount.get(l1.getID()));
			Arrays.stream(sortedSPRoots).forEach(r -> depthFirst(r, operatorList, sparkOpCount, true));

			// Step 3: Place the rest of the operators (CP). Sort the CP roots based on
			// #Spark operators in ascending order, i.e. execute the independent CP chains first
			roots.forEach(r -> depthFirst(r, operatorList, sparkOpCount, false));
			roots.forEach(Lop::resetVisitStatus);
			final_v = operatorList;
		}
		else
			// Fall back to depth if none of the operators returns results back to local
			final_v = depthFirst(v);

		// Step 4: Add Prefetch and Broadcast lops if necessary
		List<Lop> v_pf = ConfigurationManager.isPrefetchEnabled() ? addPrefetchLop(final_v) : final_v;
		List<Lop> v_bc = ConfigurationManager.isBroadcastEnabled() ? addBroadcastLop(v_pf) : v_pf;
		// TODO: Merge into a single traversal

		return v_bc;
	}

	// Gather the Spark operators which return intermediates to local (actions/single_block)
	// In addition count the number of Spark OPs underneath every Operator
	private static int collectSparkRoots(Lop root, Map<Long, Integer> sparkOpCount, List<Lop> sparkRoots) {
		if (sparkOpCount.containsKey(root.getID())) //visited before
			return sparkOpCount.get(root.getID());

		// Aggregate #Spark operators in the child DAGs
		int total = 0;
		for (Lop input : root.getInputs())
			total += collectSparkRoots(input, sparkOpCount, sparkRoots);

		// Check if this node is Spark
		total = root.isExecSpark() ? total + 1 : total;
		sparkOpCount.put(root.getID(), total);

		// Triggering point: Spark action/operator with all CP consumers
		if (isSparkTriggeringOp(root)) {
			sparkRoots.add(root);
			root.setAsynchronous(true); //candidate for async. execution
		}

		return total;
	}

	// Place the operators in a depth-first manner, but order
	// the DAGs based on number of Spark operators
	private static void depthFirst(Lop root, ArrayList<Lop> opList, Map<Long, Integer> sparkOpCount, boolean sparkFirst) {
		if (root.isVisited())
			return;

		if (root.getInputs().isEmpty()) {  //leaf node
			opList.add(root);
			root.setVisited();
			return;
		}
		// Sort the inputs based on number of Spark operators
		Lop[] sortedInputs = root.getInputs().toArray(new Lop[0]);
		if (sparkFirst) //to place the child DAG with more Spark OPs first
			Arrays.sort(sortedInputs, (l1, l2) -> sparkOpCount.get(l2.getID()) - sparkOpCount.get(l1.getID()));
		else //to place the child DAG with more CP OPs first
			Arrays.sort(sortedInputs, Comparator.comparingInt(l -> sparkOpCount.get(l.getID())));

		for (Lop input : sortedInputs)
			depthFirst(input, opList, sparkOpCount, sparkFirst);

		opList.add(root);
		root.setVisited();
	}

	private static boolean isSparkTriggeringOp(Lop lop) {
		return lop.isExecSpark() && (lop.getAggType() == SparkAggType.SINGLE_BLOCK
			|| lop.getDataType() == DataType.SCALAR || lop instanceof MapMultChain
			|| lop instanceof PickByCount || lop instanceof MMZip || lop instanceof CentralMoment
			|| lop instanceof CoVariance || lop instanceof MMTSJ || lop.isAllOutputsCP());
	}

	private static List<Lop> addPrefetchLop(List<Lop> nodes) {
		List<Lop> nodesWithPrefetch = new ArrayList<>();

		//Find the Spark nodes with all CP outputs
		for (Lop l : nodes) {
			nodesWithPrefetch.add(l);
			if (isPrefetchNeeded(l)) {
				List<Lop> oldOuts = new ArrayList<>(l.getOutputs());
				//Construct a Prefetch lop that takes this Spark node as a input
				UnaryCP prefetch = new UnaryCP(l, OpOp1.PREFETCH, l.getDataType(), l.getValueType(), ExecType.CP);
				prefetch.setAsynchronous(true);
				//Reset asynchronous flag for the input if already set (e.g. mapmm -> prefetch)
				l.setAsynchronous(false);
				for (Lop outCP : oldOuts) {
					//Rewire l -> outCP to l -> Prefetch -> outCP
					prefetch.addOutput(outCP);
					outCP.replaceInput(l, prefetch);
					l.removeOutput(outCP);
					//FIXME: Rewire _inputParams when needed (e.g. GroupedAggregate)
				}
				//Place it immediately after the Spark lop in the node list
				nodesWithPrefetch.add(prefetch);
			}
		}
		return nodesWithPrefetch;
	}

	private static List<Lop> addBroadcastLop(List<Lop> nodes) {
		List<Lop> nodesWithBroadcast = new ArrayList<>();

		for (Lop l : nodes) {
			nodesWithBroadcast.add(l);
			if (isBroadcastNeeded(l)) {
				List<Lop> oldOuts = new ArrayList<>(l.getOutputs());
				//Construct a Broadcast lop that takes this Spark node as an input
				UnaryCP bc = new UnaryCP(l, OpOp1.BROADCAST, l.getDataType(), l.getValueType(), ExecType.CP);
				bc.setAsynchronous(true);
				//FIXME: Wire Broadcast only with the necessary outputs
				for (Lop outCP : oldOuts) {
					//Rewire l -> outCP to l -> Broadcast -> outCP
					bc.addOutput(outCP);
					outCP.replaceInput(l, bc);
					l.removeOutput(outCP);
					//FIXME: Rewire _inputParams when needed (e.g. GroupedAggregate)
				}
				//Place it immediately after the Spark lop in the node list
				nodesWithBroadcast.add(bc);
			}
		}
		return nodesWithBroadcast;
	}

	private static List<Lop> addAsyncEagerCheckpointLop(List<Lop> nodes) {
		List<Lop> nodesWithCheckpoint = new ArrayList<>();
		 // Find the Spark action nodes
		for (Lop l : nodes) {
			if (isCheckpointNeeded(l)) {
				List<Lop> oldInputs = new ArrayList<>(l.getInputs());
				// Place a Checkpoint node just below this node (Spark action)
				for (Lop in : oldInputs) {
					if (in.getExecType() != ExecType.SPARK)
						continue;
					// Rewire in -> l to in -> Checkpoint -> l
					//UnaryCP checkpoint = new UnaryCP(in, OpOp1.TRIGREMOTE, in.getDataType(), in.getValueType(), ExecType.CP);
					Lop checkpoint = new Checkpoint(in, in.getDataType(), in.getValueType(),
						Checkpoint.getDefaultStorageLevelString(), true);
					checkpoint.addOutput(l);
					l.replaceInput(in, checkpoint);
					in.removeOutput(l);
					nodesWithCheckpoint.add(checkpoint);
				}
			}
			nodesWithCheckpoint.add(l);
		}
		return nodesWithCheckpoint;
	}

	private static boolean isPrefetchNeeded(Lop lop) {
		// Run Prefetch for a Spark instruction if the instruction is a Transformation
		// and the output is consumed by only CP instructions.
		boolean transformOP = lop.getExecType() == ExecType.SPARK && lop.getAggType() != SparkAggType.SINGLE_BLOCK
				// Always Action operations
				&& !(lop.getDataType() == DataType.SCALAR)
				&& !(lop instanceof MapMultChain) && !(lop instanceof PickByCount)
				&& !(lop instanceof MMZip) && !(lop instanceof CentralMoment)
				&& !(lop instanceof CoVariance)
				// Not qualified for prefetching
				&& !(lop instanceof Checkpoint) && !(lop instanceof ReBlock)
				&& !(lop instanceof CSVReBlock) && !(lop instanceof DataGen)
				// Cannot filter Transformation cases from Actions (FIXME)
				&& !(lop instanceof MMTSJ) && !(lop instanceof UAggOuterChain)
				&& !(lop instanceof ParameterizedBuiltin) && !(lop instanceof SpoofFused);

		//FIXME: Rewire _inputParams when needed (e.g. GroupedAggregate)
		boolean hasParameterizedOut = lop.getOutputs().stream()
				.anyMatch(out -> ((out instanceof ParameterizedBuiltin)
					|| (out instanceof GroupedAggregate)
					|| (out instanceof GroupedAggregateM)));
		//TODO: support non-matrix outputs
		return transformOP && !hasParameterizedOut
				&& lop.isAllOutputsCP() && lop.getDataType() == DataType.MATRIX;
	}

	private static boolean isBroadcastNeeded(Lop lop) {
		// Asynchronously broadcast a matrix if that is produced by a CP instruction,
		// and at least one Spark parent needs to broadcast this intermediate (eg. mapmm)
		boolean isBc = lop.getOutputs().stream()
				.anyMatch(out -> (out.getBroadcastInput() == lop));
		//TODO: Early broadcast objects that are bigger than a single block
		//return isCP && isBc && lop.getDataTypes() == DataType.Matrix;
		return isBc && lop.getDataType() == DataType.MATRIX;
	}

	private static boolean isCheckpointNeeded(Lop lop) {
		// Place checkpoint_e just before a Spark action (FIXME)
		boolean actionOP = lop.getExecType() == ExecType.SPARK
				&& ((lop.getAggType() == SparkAggType.SINGLE_BLOCK)
				// Always Action operations
				|| (lop.getDataType() == DataType.SCALAR)
				|| (lop instanceof MapMultChain) || (lop instanceof PickByCount)
				|| (lop instanceof MMZip) || (lop instanceof CentralMoment)
				|| (lop instanceof CoVariance) || (lop instanceof MMTSJ))
				// Not qualified for Checkpoint
				&& !(lop instanceof Checkpoint) && !(lop instanceof ReBlock)
				&& !(lop instanceof CSVReBlock)
				// Cannot filter Transformation cases from Actions (FIXME)
				&& !(lop instanceof UAggOuterChain)
				&& !(lop instanceof ParameterizedBuiltin) && !(lop instanceof SpoofFused);

		//FIXME: Rewire _inputParams when needed (e.g. GroupedAggregate)
		boolean hasParameterizedOut = lop.getOutputs().stream()
				.anyMatch(out -> ((out instanceof ParameterizedBuiltin)
					|| (out instanceof GroupedAggregate)
					|| (out instanceof GroupedAggregateM)));
		return actionOP && !hasParameterizedOut;
	}
}
