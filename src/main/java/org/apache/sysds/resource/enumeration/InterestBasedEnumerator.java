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

package org.apache.sysds.resource.enumeration;

import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.runtime.controlprogram.*;
import org.apache.sysds.resource.enumeration.EnumerationUtils.InstanceSearchSpace;

import java.util.*;
import java.util.stream.Collectors;

public class InterestBasedEnumerator extends Enumerator {
	public final static long MINIMUM_RELEVANT_MEM_ESTIMATE = 2L * 1024 * 1024 * 1024; // 2GB
	// different instance families can have slightly different memory characteristics (e.g. EC2 Graviton (arm) instances)
	// and using memory delta allows not ignoring such instances
	// TODO: enable usage of memory delta when FLOPS and bandwidth characteristics added to cost estimator
	public final static boolean USE_MEMORY_DELTA = false;
	public final static double MEMORY_DELTA_FRACTION = 0.05; // 5%
	public final static double BROADCAST_MEMORY_FACTOR = 0.21; // fraction of the minimum memory fraction for storage
	// marks if memory estimates should be used at deciding
	// for the search space of the instance for the driver nodes
	private final boolean fitDriverMemory;
	// marks if memory estimates should be used at deciding
	// for the search space of the instance for the executor nodes
	private final boolean fitBroadcastMemory;
	// marks if the estimation of the range of number of executors
	// for consideration should exclude single node execution mode
	// if any of the estimates cannot fit in the driver's memory
	private final boolean checkSingleNodeExecution;
	// marks if the estimated output size should be
	// considered as interesting point at deciding the
	// number of executors - checkpoint storage level
	// TODO: reconsider name
	private final boolean fitCheckpointMemory;
	// largest full memory estimate (scaled)
	private long largestMemoryEstimateCP;
	// ordered set ot output memory estimates (scaled)
	private TreeSet<Long> memoryEstimatesSpark;
	public InterestBasedEnumerator(
			Builder builder,
			boolean fitDriverMemory,
			boolean fitBroadcastMemory,
			boolean checkSingleNodeExecution,
			boolean fitCheckpointMemory
	) {
		super(builder);
		this.fitDriverMemory = fitDriverMemory;
		this.fitBroadcastMemory = fitBroadcastMemory;
		this.checkSingleNodeExecution = checkSingleNodeExecution;
		this.fitCheckpointMemory = fitCheckpointMemory;
	}

	@Override
	public void preprocessing() {
		InstanceSearchSpace fullSearchSpace = new InstanceSearchSpace();
		fullSearchSpace.initSpace(instances);

		if (fitDriverMemory) {
			// get full memory estimates and scale according ot the driver memory factor
			TreeSet<Long> memoryEstimatesForDriver = getMemoryEstimates(program, false, OptimizerUtils.MEM_UTIL_FACTOR);
			setInstanceSpace(fullSearchSpace, driverSpace, memoryEstimatesForDriver);
			if (checkSingleNodeExecution) {
				largestMemoryEstimateCP = !memoryEstimatesForDriver.isEmpty()? memoryEstimatesForDriver.last() : -1;
			}
		}

		if (fitBroadcastMemory) {
			// get output memory estimates and scaled according the broadcast memory factor
			// for executors' memory search space and driver memory factor for driver's memory search space
			TreeSet<Long> memoryEstimatesOutputSpark = getMemoryEstimates(program, true, BROADCAST_MEMORY_FACTOR);
			// avoid calling getMemoryEstimates with different factor but rescale: output should fit twice in the CP memory
			TreeSet<Long> memoryEstimatesOutputCP = memoryEstimatesOutputSpark.stream()
					.map(mem -> 2 * (long) (mem * BROADCAST_MEMORY_FACTOR / OptimizerUtils.MEM_UTIL_FACTOR))
					.collect(Collectors.toCollection(TreeSet::new));
			setInstanceSpace(fullSearchSpace, driverSpace, memoryEstimatesOutputCP);
			setInstanceSpace(fullSearchSpace, executorSpace, memoryEstimatesOutputSpark);
			if (checkSingleNodeExecution) {
				largestMemoryEstimateCP = !memoryEstimatesOutputCP.isEmpty()? memoryEstimatesOutputCP.last() : -1;
			}
			if (fitCheckpointMemory) {
				memoryEstimatesSpark = memoryEstimatesOutputSpark;
			}
		} else {
			executorSpace.putAll(fullSearchSpace);
			if (fitCheckpointMemory) {
				memoryEstimatesSpark = getMemoryEstimates(program, true, BROADCAST_MEMORY_FACTOR);
			}
		}

		if (!fitDriverMemory && !fitBroadcastMemory) {
			driverSpace.putAll(fullSearchSpace);
			if (checkSingleNodeExecution) {
				TreeSet<Long> memoryEstimatesForDriver = getMemoryEstimates(program, false, OptimizerUtils.MEM_UTIL_FACTOR);
				largestMemoryEstimateCP = !memoryEstimatesForDriver.isEmpty()? memoryEstimatesForDriver.last() : -1;
			}
		}
	}

	@Override
	public boolean evaluateSingleNodeExecution(long driverMemory) {
		// Checking if single node execution should be excluded is optional.
		if (checkSingleNodeExecution && minExecutors == 0 && largestMemoryEstimateCP > 0) {
			return largestMemoryEstimateCP <= driverMemory;
		}
		return minExecutors == 0;
	}

	@Override
	public ArrayList<Integer> estimateRangeExecutors(long executorMemory, int executorCores) {
		// consider the maximum level of parallelism and
		// based on the initiated flags decides on the following methods
		// for enumeration of the number of executors:
		// 1. Such a number that leads to combined distributed memory
		//	close to the output size of the HOPs
		// 3. Enumerating all options with the established range
		int min = Math.max(1, minExecutors);
		int max = Math.min(maxExecutors, (MAX_LEVEL_PARALLELISM / executorCores));

		ArrayList<Integer> result;
		if (fitCheckpointMemory) {
			result = new ArrayList<>(memoryEstimatesSpark.size() + 1);
			int previousNumber = -1;
			for (long estimate: memoryEstimatesSpark) {
				// the ratio is just an intermediate for the new enumerated number of executors
				double ratio = (double) estimate / executorMemory;
				int currentNumber = (int) Math.max(1, Math.floor(ratio));
				if (currentNumber < min || currentNumber == previousNumber) {

					continue;
				}
				if (currentNumber <= max) {
					result.add(currentNumber);
					previousNumber = currentNumber;
				} else {
					break;
				}
			}
			// add a number that allow also the largest checkpoint to be done in memory
			if (previousNumber < 0) {
				// always append at least one value to allow evaluating Spark execution
				result.add(min);
			} else if (previousNumber < max) {
				result.add(previousNumber + 1);
			}
		} else { // enumerate all options within the min-max range
			result = new ArrayList<>((max - min) + 1);
			for (int n = min; n <= max; n++) {
				result.add(n);
			}
		}
		return result;
	}

	// Static helper methods -------------------------------------------------------------------------------------------
	private static void setInstanceSpace(InstanceSearchSpace inputSpace, InstanceSearchSpace outputSpace, TreeSet<Long> memoryEstimates) {
		TreeSet<Long> memoryPoints = getMemoryPoints(memoryEstimates, inputSpace.keySet());
		for (long memory: memoryPoints) {
			outputSpace.put(memory, inputSpace.get(memory));
		}
		// in case no large enough memory estimates exist set the instances with minimal memory
		if (outputSpace.isEmpty()) {
			long minMemory = inputSpace.firstKey();
			outputSpace.put(minMemory, inputSpace.get(minMemory));
		}
	}

	/**
	 * @param availableMemory should be always a sorted set;
	 * this is always the case for the result of {@code keySet()} called on {@code TreeMap}
	 */
	private static TreeSet<Long> getMemoryPoints(TreeSet<Long> estimates, Set<Long> availableMemory) {
		// use tree set to avoid adding duplicates and ensure ascending order
		TreeSet<Long> result = new TreeSet<>();
		// assumed ascending order
		List<Long> relevantPoints = new ArrayList<>(availableMemory);
		for (long estimate: estimates) {
			if (availableMemory.isEmpty()) {
				break;
			}
			// divide list on larger and smaller by partitioning - partitioning preserve the order
			Map<Boolean, List<Long>> divided = relevantPoints.stream()
					.collect(Collectors.partitioningBy(n -> n < estimate));
			// get the points smaller than the current memory estimate
			List<Long> smallerPoints = divided.get(true);
			long largestOfTheSmaller = smallerPoints.isEmpty() ? -1 : smallerPoints.get(smallerPoints.size() - 1);
			// reduce the list of relevant points - equal or larger than the estimate
			relevantPoints = divided.get(false);
			// get points greater or equal than the current memory estimate
			long smallestOfTheLarger = relevantPoints.isEmpty()? -1 : relevantPoints.get(0);

			if (USE_MEMORY_DELTA) {
				// Delta memory of 5% of the node's memory allows not ignoring
				// memory points with potentially equivalent values but not exactly the same values.
				// This is the case for example in AWS for instances of the same type but with
				// different additional capabilities: m5.xlarge (16GB) vs m5n.xlarge (15.25GB).
				// Get points smaller than the current memory estimate within the memory delta
				long memoryDelta = Math.round(estimate * MEMORY_DELTA_FRACTION);
				for (long point : smallerPoints) {
					if (point >= (largestOfTheSmaller - memoryDelta)) {
						result.add(point);
					}
				}
				for (long point : relevantPoints) {
					if (point <= (smallestOfTheLarger + memoryDelta)) {
						result.add(point);
					} else {
						break;
					}
				}
			} else {
				if (largestOfTheSmaller > 0) {
					result.add(largestOfTheSmaller);
				}
				if (smallestOfTheLarger > 0) {
					result.add(smallestOfTheLarger);
				}
			}
		}
		return result;
	}

	/**
	 * Extracts the memory estimates which original size is larger than {@code MINIMUM_RELEVANT_MEM_ESTIMATE}
	 *
	 * @param currentProgram program for extracting the memory estimates from
	 * @param outputOnly {@code true} - output estimate only;
	 *				   {@code false} - sum of input, intermediate and output estimates
	 * @param memoryFactor factor for reverse scaling the estimates to avoid
	 *					 scaling the search space parameters representing the nodes' memory budget
	 * @return memory estimates in ascending order ensured by the {@code TreeSet} data structure
	 */
	public static TreeSet<Long> getMemoryEstimates(Program currentProgram, boolean outputOnly, double memoryFactor) {
		TreeSet<Long> estimates = new TreeSet<>();
		getMemoryEstimates(currentProgram.getProgramBlocks(), estimates, outputOnly);
		return estimates.stream()
				.filter(mem -> mem > MINIMUM_RELEVANT_MEM_ESTIMATE)
				.map(mem -> (long) (mem / memoryFactor))
				.collect(Collectors.toCollection(TreeSet::new));
	}

	private static void getMemoryEstimates(ArrayList<ProgramBlock> pbs, TreeSet<Long> mem, boolean outputOnly) {
		for( ProgramBlock pb : pbs ) {
			getMemoryEstimates(pb, mem, outputOnly);
		}
	}

	private static void getMemoryEstimates(ProgramBlock pb, TreeSet<Long> mem, boolean outputOnly) {
		if (pb instanceof FunctionProgramBlock)
		{
			FunctionProgramBlock fpb = (FunctionProgramBlock)pb;
			getMemoryEstimates(fpb.getChildBlocks(), mem, outputOnly);
		}
		else if (pb instanceof WhileProgramBlock)
		{
			WhileProgramBlock fpb = (WhileProgramBlock)pb;
			getMemoryEstimates(fpb.getChildBlocks(), mem, outputOnly);
		}
		else if (pb instanceof IfProgramBlock)
		{
			IfProgramBlock fpb = (IfProgramBlock)pb;
			getMemoryEstimates(fpb.getChildBlocksIfBody(), mem, outputOnly);
			getMemoryEstimates(fpb.getChildBlocksElseBody(), mem, outputOnly);
		}
		else if (pb instanceof ForProgramBlock) // including parfor
		{
			ForProgramBlock fpb = (ForProgramBlock)pb;
			getMemoryEstimates(fpb.getChildBlocks(), mem, outputOnly);
		}
		else
		{
			StatementBlock sb = pb.getStatementBlock();
			if( sb != null && sb.getHops() != null ){
				Hop.resetVisitStatus(sb.getHops());
				for( Hop hop : sb.getHops() )
					getMemoryEstimates(hop, mem, outputOnly);
			}
		}
	}

	private static void getMemoryEstimates(Hop hop, TreeSet<Long> mem, boolean outputOnly)
	{
		if( hop.isVisited() )
			return;
		//process children
		for(Hop hi : hop.getInput())
			getMemoryEstimates(hi, mem, outputOnly);

		if (outputOnly) {
			long estimate = (long) hop.getOutputMemEstimate(0);
			if (estimate > 0)
				mem.add(estimate);
		} else {
			mem.add((long) hop.getMemEstimate());
		}
		hop.setVisited();
	}
}
