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

import org.apache.hadoop.util.Lists;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.resource.CloudInstance;
import org.apache.sysds.resource.ResourceCompiler;
import org.apache.sysds.runtime.controlprogram.BasicProgramBlock;
import org.apache.sysds.runtime.controlprogram.ForProgramBlock;
import org.apache.sysds.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysds.runtime.controlprogram.IfProgramBlock;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.ProgramBlock;
import org.apache.sysds.runtime.controlprogram.WhileProgramBlock;
import org.apache.sysds.runtime.instructions.Instruction;

import java.util.HashMap;
import java.util.TreeMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.List;
import java.util.LinkedList;
import java.util.ArrayList;

public class PruneBasedEnumerator extends Enumerator {
	long insufficientSingleNodeMemory;
	long singleNodeOnlyMemory;
	HashMap<Long, Integer> maxExecutorsPerInstanceMap;

	public PruneBasedEnumerator(Builder builder) {
		super(builder);
		insufficientSingleNodeMemory = -1;
		singleNodeOnlyMemory = Long.MAX_VALUE;
		maxExecutorsPerInstanceMap = new HashMap<>();
	}

	@Override
	public void preprocessing() {
		driverSpace.initSpace(instances);
		executorSpace.initSpace(instances);
		for (Map.Entry<Long, TreeMap<Integer, LinkedList<CloudInstance>>> eMemoryEntry: executorSpace.entrySet()) {
			for (Integer eCores: eMemoryEntry.getValue().keySet()) {
				long combinationHash = combineHash(eMemoryEntry.getKey(), eCores);
				maxExecutorsPerInstanceMap.put(combinationHash, maxExecutors);
			}
		}
	}

	@Override
	public void processing() {
		long driverMemory, executorMemory;
		int driverCores, executorCores;
		EnumerationUtils.ConfigurationPoint configurationPoint;

		for (Entry<Long, TreeMap<Integer, LinkedList<CloudInstance>>> dMemoryEntry: driverSpace.entrySet()) {
			driverMemory = dMemoryEntry.getKey();
			// loop over the search space to enumerate the driver configurations
			for (Entry<Integer, LinkedList<CloudInstance>> dCoresEntry: dMemoryEntry.getValue().entrySet()) {
				driverCores = dCoresEntry.getKey();
				// single node execution mode
				if (evaluateSingleNodeExecution(driverMemory, driverCores)) {
					ResourceCompiler.setSingleNodeResourceConfigs(driverMemory, driverCores);
					program = ResourceCompiler.doFullRecompilation(program);
					// no need of recompilation for single nodes with identical memory budget and #v. cores
					for (CloudInstance dInstance: dCoresEntry.getValue()) {
						// iterate over all driver nodes with the currently evaluated memory and #cores values
						configurationPoint = new EnumerationUtils.ConfigurationPoint(dInstance);
						double[] newEstimates = getCostEstimate(configurationPoint);
						if (isInvalidConfiguration(newEstimates)) {
							// mark the current CP memory budget as insufficient for single node execution
							insufficientSingleNodeMemory = driverMemory;
							break;
						}
						updateOptimalSolution(newEstimates[0], newEstimates[1], configurationPoint);
					}
				}
				if (driverMemory >= singleNodeOnlyMemory) continue;
				// enumeration for distributed execution
				for (Entry<Long, TreeMap<Integer, LinkedList<CloudInstance>>> eMemoryEntry: executorSpace.entrySet()) {
					if (driverMemory >= singleNodeOnlyMemory) continue;
					executorMemory = eMemoryEntry.getKey();
					// loop over the search space to enumerate the executor configurations
					for (Entry<Integer, LinkedList<CloudInstance>> eCoresEntry: eMemoryEntry.getValue().entrySet()) {
						if (driverMemory >= singleNodeOnlyMemory) continue;
						executorCores = eCoresEntry.getKey();
						List<Integer> numberExecutorsSet = estimateRangeExecutors(driverCores, eMemoryEntry.getKey(), eCoresEntry.getKey());
						// variables for tracking the best possible number of executors for each executor instance type
						double localBestCostScore = Double.MAX_VALUE;
						int newLocalBestNumberExecutors = -1;
						// for Spark execution mode
						for (int numberExecutors: numberExecutorsSet) {
							try {
								ResourceCompiler.setSparkClusterResourceConfigs(
										driverMemory,
										driverCores,
										numberExecutors,
										executorMemory,
										executorCores
								);
							} catch (IllegalArgumentException e) {
								// insufficient driver memory detected
								break;
							}
							program = ResourceCompiler.doFullRecompilation(program);
							if (!hasSparkInstructions(program)) {
								// mark the current CP memory budget as dominant for the global optimal solution
								// -> higher CP memory could not introduce Spark operations
								singleNodeOnlyMemory = driverMemory;
								break;
							}
							// no need of recompilation for a cluster with identical #executors and
							// with identical memory and #v. cores for driver and executor nodes
							for (CloudInstance dInstance: dCoresEntry.getValue()) {
								// iterate over all driver nodes with the currently evaluated memory and #cores values
								for (CloudInstance eInstance: eCoresEntry.getValue()) {
									// iterate over all executor nodes for the evaluated cluster size
									// with the currently evaluated memory and #cores values
									configurationPoint = new EnumerationUtils.ConfigurationPoint(
											dInstance,
											eInstance,
											numberExecutors
									);
									double[] newEstimates = getCostEstimate(configurationPoint);
									updateOptimalSolution(
											newEstimates[0],
											newEstimates[1],
											configurationPoint
									);
									// now checking for cost improvements regarding the current executor instance type
									// this is not in all cases part of the optimal solution because other
									// solutions with 0 executors or executors of other instance type could be better
									if (optStrategy == OptimizationStrategy.MinCosts) {
										double optimalScore = linearScoringFunction(newEstimates[0], newEstimates[1]);
										if (localBestCostScore > optimalScore) {
											localBestCostScore = optimalScore;
											newLocalBestNumberExecutors = configurationPoint.numberExecutors;
										}
									} else if (optStrategy == OptimizationStrategy.MinTime) {
										if (localBestCostScore > newEstimates[0]) {
											// do not check for max. price here
											localBestCostScore = newEstimates[0];
											newLocalBestNumberExecutors = configurationPoint.numberExecutors;
										}
									} else { // minPrice
										if (localBestCostScore > newEstimates[1]) {
											// do not check for max. time here
											localBestCostScore = newEstimates[1];
											newLocalBestNumberExecutors = configurationPoint.numberExecutors;
										}
									}
								}
							}
						}
						// evaluate the local best number of executors to avoid evaluating solutions with
						// more executors in the next iterations
						if (localBestCostScore < Double.MAX_VALUE && newLocalBestNumberExecutors > 0) {
							long combinationHash = combineHash(executorMemory, executorCores);
							maxExecutorsPerInstanceMap.put(combinationHash, newLocalBestNumberExecutors);
						}
					}
				}
			}
		}
	}

	@Override
	public boolean evaluateSingleNodeExecution(long driverMemory, int cores) {
		if (cores > CPU_QUOTA || minExecutors > 0) return false;
		return insufficientSingleNodeMemory != driverMemory;
	}

	@Override
	public ArrayList<Integer> estimateRangeExecutors(int driverCores, long executorMemory, int executorCores) {
		// consider the cpu quota (limit) for cloud instances and the 'local' best number of executors
		// to decide for the range of number of executors to be evaluated next
		int maxAchievableLevelOfParallelism  = CPU_QUOTA - driverCores;
		int currentMax = Math.min(maxExecutors, maxAchievableLevelOfParallelism / executorCores);
		long combinationHash = combineHash(executorMemory, executorCores);
		int maxExecutorsToConsider = maxExecutorsPerInstanceMap.get(combinationHash);
		currentMax = Math.min(currentMax, maxExecutorsToConsider);
		ArrayList<Integer> result = new ArrayList<>();
		for (int i = 1; i <= currentMax; i++) {
			result.add(i);
		}
		return result;
	}

	// Helpers ---------------------------------------------------------------------------------------------------------

	/**
	 * Ensures unique mapping for a combination of node memory and number of
	 * executor cores due to the discrete nature of the node memory given in bytes.
	 * The smallest margin for cloud instances would be around 500MB ~ 500*1024^2 bytes,
	 * what is by far larger than the maximum number of executors core physically possible.
	 *
	 * @param executorMemory node memory in bytes
	 * @param cores number virtual cores (physical threads) for the node
	 * @return hash value
	 */
	public static long combineHash(long executorMemory, int cores) {
		return executorMemory + cores;
	}


	public static boolean isInvalidConfiguration(double[] estimates) {
		return estimates[0] == Double.MAX_VALUE && estimates[1] == Double.MAX_VALUE;
	}

	/**
	 * Checks for Spark instruction in the given runtime program.
	 * It excludes from the check instructions for reblock operations
	 * and for caching since these are not always remove from the runtime
	 * program even their outputs are never used and ignored at execution.
	 *
	 * @param program runtime program
	 * @return boolean to mark if
	 * the execution would execute any Spark operation
	 */
	public static boolean hasSparkInstructions(Program program) {
		boolean hasSparkInst;
		Map<String, FunctionProgramBlock> funcMap = program.getFunctionProgramBlocks();
		if( funcMap != null && !funcMap.isEmpty() )
		{
			for( Map.Entry<String, FunctionProgramBlock> e : funcMap.entrySet() ) {
				String fkey = e.getKey();
				FunctionProgramBlock fpb = e.getValue();
				for(ProgramBlock pb : fpb.getChildBlocks()) {
					hasSparkInst = hasSparkInstructions(pb);
					if (hasSparkInst) return true;
				}
				if(program.containsFunctionProgramBlock(fkey, false) ) {
					FunctionProgramBlock fpb2 = program.getFunctionProgramBlock(fkey, false);
					for(ProgramBlock pb : fpb2.getChildBlocks()) {
						hasSparkInst = hasSparkInstructions(pb);
						if (hasSparkInst) return true;
					}
				}
			}
		}

		for(ProgramBlock pb : program.getProgramBlocks()) {
			hasSparkInst = hasSparkInstructions(pb);
			if (hasSparkInst) return true;
		}
		return false;
	}

	private static boolean hasSparkInstructions(ProgramBlock pb) {
		boolean hasSparkInst;
		if (pb instanceof FunctionProgramBlock ) {
			FunctionProgramBlock fpb = (FunctionProgramBlock)pb;
			for(ProgramBlock pbc : fpb.getChildBlocks()) {
				hasSparkInst = hasSparkInstructions(pbc);
				if (hasSparkInst) return true;
			}
		}
		else if (pb instanceof WhileProgramBlock) {
			WhileProgramBlock wpb = (WhileProgramBlock) pb;
			hasSparkInst = hasSparkInstructions(wpb.getPredicate());
			if (hasSparkInst) return true;
			for(ProgramBlock pbc : wpb.getChildBlocks()) {
				hasSparkInst = hasSparkInstructions(pbc);
				if (hasSparkInst) return true;
			}
			if(wpb.getExitInstruction() != null) {
				hasSparkInst = hasSparkInstructions(Lists.newArrayList(wpb.getExitInstruction()));
				return hasSparkInst;
			}
		}
		else if (pb instanceof IfProgramBlock) {
			IfProgramBlock ipb = (IfProgramBlock) pb;
			hasSparkInst = hasSparkInstructions(ipb.getPredicate());
			if (hasSparkInst) return true;
			for(ProgramBlock pbc : ipb.getChildBlocksIfBody()) {
				hasSparkInst = hasSparkInstructions(pbc);
				if (hasSparkInst) return true;
			}
			if(!ipb.getChildBlocksElseBody().isEmpty()) {
				for(ProgramBlock pbc : ipb.getChildBlocksElseBody()) {
					hasSparkInst = hasSparkInstructions(pbc);
					if (hasSparkInst) return true;
				}
			}
			if(ipb.getExitInstruction() != null) {
				hasSparkInst = hasSparkInstructions(Lists.newArrayList(ipb.getExitInstruction()));
				return hasSparkInst;
			}
		}
		else if (pb instanceof ForProgramBlock) { // for and parfor loops
			ForProgramBlock fpb = (ForProgramBlock) pb;
			hasSparkInst = hasSparkInstructions(fpb.getFromInstructions());
			if (hasSparkInst) return true;
			hasSparkInst = hasSparkInstructions(fpb.getToInstructions());
			if (hasSparkInst) return true;
			hasSparkInst = hasSparkInstructions(fpb.getIncrementInstructions());
			if (hasSparkInst) return true;
			for(ProgramBlock pbc : fpb.getChildBlocks()) {
				hasSparkInst = hasSparkInstructions(pbc);
				if (hasSparkInst) return true;
			}
			if (fpb.getExitInstruction() != null) {
				hasSparkInst = hasSparkInstructions(Lists.newArrayList(fpb.getExitInstruction()));
				return hasSparkInst;
			}
		}
		else if( pb instanceof BasicProgramBlock ) {
			BasicProgramBlock bpb = (BasicProgramBlock) pb;
			hasSparkInst = hasSparkInstructions(bpb.getInstructions());
			return hasSparkInst;
		}
		return false;
	}

	private static boolean hasSparkInstructions(List<Instruction> instructions) {
		for (Instruction inst : instructions) {
			Instruction.IType iType = inst.getType();
			if (iType.equals(Instruction.IType.SPARK)) {
				String opcode = inst.getOpcode();
				if (!(opcode.contains(Opcodes.RBLK.toString()) || opcode.contains("chkpoint"))) {
					// reblock and checkpoint instructions may occur in a program
					// compiled for hybrid execution mode but without effective Spark instruction
					return true;
				}
			}
		}
		return false;
	}
}
