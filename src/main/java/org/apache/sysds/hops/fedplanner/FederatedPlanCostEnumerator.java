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

package org.apache.sysds.hops.fedplanner;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Objects;
import java.util.LinkedHashMap;

import org.apache.commons.lang3.tuple.Pair;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.fedplanner.FederatedMemoTable.HopCommon;
import org.apache.sysds.hops.fedplanner.FederatedMemoTable.ConflictedFedPlanVariants;
import org.apache.sysds.hops.fedplanner.FederatedMemoTable.FedPlan;
import org.apache.sysds.hops.fedplanner.FederatedMemoTable.FedPlanVariants;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction.FederatedOutput;

/**
 * Enumerates and evaluates all possible federated execution plans for a given Hop DAG.
 * Works with FederatedMemoTable to store plan variants and FederatedPlanCostEstimator
 * to compute their costs.
 */
public class FederatedPlanCostEnumerator {
	/**
	 * Entry point for federated plan enumeration. This method creates a memo table
	 * and returns the minimum cost plan for the entire Directed Acyclic Graph (DAG).
	 * It also resolves conflicts where FedPlans have different FederatedOutput types.
	 * 
	 * @param rootHop The root Hop node from which to start the plan enumeration.
	 * @param printTree A boolean flag indicating whether to print the federated plan tree.
	 * @return The optimal FedPlan with the minimum cost for the entire DAG.
	 */
	public static FedPlan enumerateOptimalFederatedPlanCost(Hop rootHop, boolean printTree) {
		Set<Long> visited = new HashSet<>();
		Map<Long, List<ConflictMergeResolveInfo>> conflictMergeResolveMap = new HashMap<>();
		Map<Long, List<Long>> resolveMap = new HashMap<>();
		detectPossibleConflicts(rootHop, visited, conflictMergeResolveMap, resolveMap);

		// Create new memo table to store all plan variants
		FederatedMemoTable memoTable = new FederatedMemoTable();
		// Recursively enumerate all possible plans
		enumerateFederatedPlanCost(rootHop, memoTable, conflictMergeResolveMap);

		// Return the minimum cost plan for the root node
		FedPlan optimalPlan = getMinCostRootFedPlan(rootHop.getHopID(), memoTable);

		// Detect conflicts in the federated plans where different FedPlans have different FederatedOutput types
		// double additionalTotalCost = detectAndResolveConflictFedPlan(optimalPlan, memoTable);

		// Optionally print the federated plan tree if requested
		// if (printTree) FederatedMemoTablePrinter.printFedPlanTree(optimalPlan, memoTable, additionalTotalCost);

		return optimalPlan;
	}

	public static void detectPossibleConflicts(Hop hop, Set<Long> visited, Map<Long, List<ConflictMergeResolveInfo>> conflictMergeResolveMap, Map<Long, List<Long>> resolveMap) {
		for (Hop inputHop : hop.getInput()) {
			if (visited.contains(hop.getHopID()))
				return;

			visited.add(hop.getHopID());

			if (inputHop.getParent().size() > 1)
				findMergeResolvePaths(inputHop, conflictMergeResolveMap);

			detectPossibleConflicts(inputHop, visited, conflictMergeResolveMap);
		}
	}

    /**
     * Identifies and marks conflicts and merge points in a Hop DAG starting from a conflicted Hop.
     * A conflicted Hop is one that has multiple parent nodes, indicating potential execution path conflicts.
     *
     * The algorithm performs a breadth-first search (BFS) through the DAG to:
     * 1. Start from a conflicted hop (one with multiple parents)
     * 2. Traverse upward through parent nodes using BFS
     * 3. Track merge points where execution paths converge
     * 4. Mark nodes as resolved when all required merges are found
     * 5. Track the count of merged hops at each merge point
     *
     * @param conflictedHop The Hop node with multiple parents that initiates the conflict detection
     * @param conflictMergeResolveMap Map storing conflict and merge information for each Hop ID
     */
    private static void findMergeResolvePaths(Hop conflictedHop, Map<Long, List<ConflictMergeResolveInfo>> conflictMergeResolveMap, Map<Long, ResolveInfo> resolveMap) {
        // Initialize counter for remaining merges needed (parents - 1 since we need n-1 merges for n paths)
		long conflictedHopID = conflictedHop.getHopID();
		int leftMergeCount = conflictedHop.getParent().size() - 1;
		boolean isConverged = true;

		Set<Long> visited = new HashSet<>();
        Queue<Pair<Hop, SplitInfo>> BFSqueue = new LinkedList<>();

		long convergeHopID = -1;
		List<Hop> topResolveHops = new ArrayList<>();
		List<Long> topResolveHopIDs = new ArrayList<>();

		Map<Long, SplitInfo> splitPointMap = new HashMap<>();
		Set<Long> mergeHopIDs = new HashSet<>();
		Set<Long> splitHopIDs = new HashSet<>();

		// 여러 개의 부모 집합을 추가하는 경우
		for (Hop parentHop : conflictedHop.getParent()) {
			SplitInfo splitInfo = new SplitInfo(parentHop);
			BFSqueue.offer(Pair.of(parentHop, splitInfo));
			splitPointMap.put(parentHop.getHopID(), splitInfo);
		}

		// 의문점 1. 모든 hop을 다 거치는가?
		// 의문점 2. resolve Point 너머도 진행되지는 않았는가? 진행되었다면 지워야 한다.

        // Start BFS traversal through the DAG
        while (!BFSqueue.isEmpty() || leftMergeCount > 0) {
            Pair<Hop, SplitInfo> current = BFSqueue.poll();
			Hop currentHop = current.getKey();
			SplitInfo splitInfo = current.getValue();
            int numOfParent = currentHop.getParent().size();

            if (numOfParent == 0) {
				isConverged = false;
				leftMergeCount--;
				updateConflictResolveType(conflictMergeResolveMap, currentHop.getHopID(), conflictedHopID, false, false, ResolvedType.TOP);
				topResolveHopIDs.add(currentHop.getHopID());
				topResolveHops.add(currentHop);
				continue;
            }

            // For nodes with multiple parents, update the merge count
            // Each additional parent represents another path that needs to be merged
			boolean isSplited = false;
            if (numOfParent > 1){
				isSplited = true;
                leftMergeCount += numOfParent - 1;
			}

            // Process all parent nodes of the current node
            for (Hop parentHop : currentHop.getParent()) {
                long parentHopID = parentHop.getHopID();

				if (isSplited) {
					splitHopIDs.add(parentHopID);
				}

                // Handle potential merge points (nodes with multiple inputs)
                if (parentHop.getInput().size() > 1) {
                    // If node was previously visited, update merge information
                    if (visited.contains(parentHopID)) {
                        leftMergeCount--;
						mergeHopIDs.add(parentHopID);

						if (leftMergeCount == 0 && isConverged){
							updateConflictResolveType(conflictMergeResolveMap, parentHopID, conflictedHopID, true, isSplited, ResolvedType.RESOLVE);
							convergeHopID = parentHopID;
						} else {
							updateConflictResolveType(conflictMergeResolveMap, parentHopID, conflictedHopID, true, isSplited, ResolvedType.INNER_PATH);
						}
                    } else {
                        // First visit to this node - initialize tracking information
                        visited.add(parentHopID);
                        BFSqueue.offer(parentHop);
						addConflictResolveType(conflictMergeResolveMap, parentHopID, conflictedHopID, false, isSplited, ResolvedType.INNER_PATH);
                    }
                } else {
                    // Handle nodes with single input
                    // No need to track visit count as these aren't merge points
                    BFSqueue.offer(parentHop);
					addConflictResolveType(conflictMergeResolveMap, parentHopID, conflictedHopID, false, isSplited, ResolvedType.INNER_PATH);
                }
            }
        }
		
		ResolveInfo resolveInfo;
		
		if (isConverged) {
			resolveInfo = new ResolveInfo(conflictedHopID, convergeHopID, null, null);
		} else {
			for (Hop topHop : topResolveHops) {
				boolean isfound = false;

				while (!isfound) {
					// 공통점 1: 자신의 부모에서 더 이상 merge가 발생하지 않음
					// 공통점 2: 자식이 자식들이 split하였다면, 반드시 merge 되어야 함.
					// 차이점 1: last-merge는 자신이 merge하나, first-split은 자신이 merge하지 않음.
					// 차이점 2: last-merge는 자식이 split하지 않아도 되나, first-split은 자식이 반드시 split해야 함.

					for (Hop childHop : topHop.getInput()) {
						// Todo: 여기부터 하자.
						// visited, merge인지, split인지, split되면 merge 되었는지...
						// bfs queues는 hop과 hop의 split point들을 가지고 다님.
						// merge가 되면 마지막 split point를 지우고, 차례대로 지움.

						if (!visited.contains(childHop.getHopID()))
							continue;
						

						if (mergeHopIDs.contains(childHop.getHopID()) && childHop.getParent().size() == 1) {
							isfound = true;
							updateConflictResolveType(conflictMergeResolveMap, childHop.getHopID(), conflictedHopID, true, false, ResolvedType.FIRST_SPLIT_LAST_MERGE);
						}

						if (mergeHopIDs.contains(childHop.getHopID()) && childHop.getParent().size() > 1) {
							for (Hop childParentHop : childHop.getParent()) {
								if (childParentHop == topHop)
									continue;
								
								if (childParentHop is Merged)

							}
						}

						if ()

							if (childHop.getParent().size() == 1) {
						if (mergeHopIDs.contains(childHop.getHopID())) {
							if (childHop.getParent().size() == 1) {
								isfound = true;	
								updateConflictResolveType(conflictMergeResolveMap, childHop.getHopID(), conflictedHopID, true, false, ResolvedType.FIRST_SPLIT_LAST_MERGE);	
							} else{
								
							}
							
						}

						if (splitHopIDs.contains(childHop.getHopID())) {
							
						}
					}
				}
			}


							// // childHop이 merge혹은 initial parent일 때까지 내려가야함.
							// if (childInfo.isMerged() || initialParentHopIDs.contains(childHop.getHopID())) {
							// 	// 1. single-parent이면, child가 last-merge 혹은 first-split임
							// 	if (childHop.getParent().size() == 1) {
							// 		isfound = true;	
							// 		updateConflictResolveType(conflictMergeResolveMap, childHop.getHopID(), conflictedHopID, true, false, ResolvedType.FIRST_SPLIT_LAST_MERGE);	
							// 	} else {
							// 		ResolvedType resolvedType = conflictMergeResolveMap.get(childHop.getHopID()).stream()	
							// 			.filter(resolveInfo -> resolveInfo.conflictedHopID == conflictedHopID)
							// 			.findFirst()
							// 			.get()
							// 			.getResolvedType();
									
							// 		if (resolvedType != ResolvedType.INNER_PATH && resolvedType != ResolvedType.OUTER_PATH) {
							// 			isfound = true;
							// 			updateConflictResolveType(conflictMergeResolveMap, childHop.getHopID(), conflictedHopID, true, false, resolvedType);
							// 		}

							// 		for (Hop parentHop : childHop.getParent()) {
							// 			// childHop의 다른 parent가 merge되었는지 확인해야함.
							// 			// merge한 hop을 기억해야함
							// 			// split한 hop이면 더해졌을 수도 있으니 그것도 문제임
							// 			// path에서 split 포인트를 기억하고 있어야 하나?
							// 			// 나중에 모았다가 진행해야 하는 듯.
							// 			// left merge count가 줄어드는 건 맞으니까.
							// 			// 서로 엉킬수도 있나?
							// 		}
							// 		// 2. multi-parent이면, child가 first-split임.
							// 		// 2-1: 다른 parent가 모두 merge하지 않으면, childHop은 last-merge임
							// 		// 2-2: 다른 parent가 하나라도 merge하면, currentHop이 first-split임.
							// 	}
							// 	// end case decision
							// 	break;
							// } else {
							// 	currentHop = childHop;
							// 	updateConflictResolveType(conflictMergeResolveMap, childHop.getHopID(), conflictedHopID, false, false, ResolvedType.OUTER_PATH);
							// }
			}
			resolveInfo = new ResolveInfo(conflictedHopID, convergeHopID, topResolveHopIDs, firstSplitLastMergeHopIDs);
		}
		resolveMap.put(conflictedHopID, resolveInfo);
    }

	public static class SplitInfo {
		private Hop hopRef;
		private int numOfParents;
		private Set<Long> mergeParentHopIDs;

		public SplitInfo(Hop hopRef) {
			this.hopRef = hopRef;
			this.numOfParents = hopRef.getParent().size();
			this.mergeParentHopIDs = new HashSet<>();
		}
	}

	private static void updateConflictResolveType(Map<Long, List<ConflictMergeResolveInfo>> conflictMergeResolveMap, long currentHopID, long conflictedHopID, boolean isMerged, boolean isSplited, ResolvedType resolvedType) {
		List<ConflictMergeResolveInfo> mergeInfoList = conflictMergeResolveMap.get(currentHopID);
		mergeInfoList.stream()
			.filter(info -> info.conflictedHopID == conflictedHopID)
			.forEach(info -> {
				info.isMerged |= isMerged;
				info.isSplited |= isSplited;
				info.resolvedType = resolvedType;
			});
	}

	private static void addConflictResolveType(Map<Long, List<ConflictMergeResolveInfo>> conflictMergeResolveMap, 
		long currentHopID, long conflictedHopID, boolean isMerged, boolean isSplited, ResolvedType resolvedType) {
		conflictMergeResolveMap.putIfAbsent(currentHopID, new ArrayList<>());
		conflictMergeResolveMap.get(currentHopID).add(new ConflictMergeResolveInfo(conflictedHopID, isMerged, isSplited, resolvedType));
	}

	public static class ResolveInfo {
		private long conflictHopID;
		private long convergeHopID;
		private List<Long> topResolveHopIDs;
		private List<Long> firstSplitLastMergeHopIDs;

		public ResolveInfo(long conflictHopID, long convergeHopID, List<Long> topResolveHopIDs, List<Long> firstSplitLastMergeHopIDs) {
			this.conflictHopID = conflictHopID;
			this.convergeHopID = convergeHopID;
			this.topResolveHopIDs = topResolveHopIDs;
			this.firstSplitLastMergeHopIDs = firstSplitLastMergeHopIDs;
		}
	}

	

	/**
	 * Recursively enumerates all possible federated execution plans for a Hop DAG.
	 * For each node:
	 * 1. First processes all input nodes recursively if not already processed
	 * 2. Generates all possible combinations of federation types (FOUT/LOUT) for inputs
	 * 3. Creates and evaluates both FOUT and LOUT variants for current node with each input combination
	 * 
	 * The enumeration uses a bottom-up approach where:
	 * - Each input combination is represented by a binary number (i)
	 * - Bit j in i determines whether input j is FOUT (1) or LOUT (0)
	 * - Total number of combinations is 2^numInputs
	 * 
	 * @param hop ?
	 * @param memoTable ?
	 */
	private static void enumerateFederatedPlanCost(Hop hop, FederatedMemoTable memoTable, 
		Map<Long, List<ConflictMergeResolveInfo>> conflictMergeResolveMap) {

		// Process all input nodes first if not already in memo table
		for (Hop inputHop : hop.getInput()) {
			if (!memoTable.contains(inputHop.getHopID(), FederatedOutput.FOUT) 
				&& !memoTable.contains(inputHop.getHopID(), FederatedOutput.LOUT)) {
					enumerateFederatedPlanCost(inputHop, memoTable, conflictMergeResolveMap);
			}
		}
		long hopID = hop.getHopID();
		HopCommon hopCommon = new HopCommon(hop);
		FederatedPlanCostEstimator.computeHopCost(hopCommon);

		int numInputs = hop.getInput().size();
		double selfCost = hopCommon.getSelfCost();

		// Todo: (구현) conflict hop의 initial parent 처리
		// Todo: (구현) resolve point 위에서 처리 (resolve, first-split & last-merge, top-level)
		
		if (!conflictMergeResolveMap.containsKey(hopID)){
			FedPlanVariants LOutFedPlanVariants = new FedPlanVariants(hopCommon, FederatedOutput.LOUT);
			FedPlanVariants FOutFedPlanVariants = new FedPlanVariants(hopCommon, FederatedOutput.FOUT);

			// # of child, LOUT/FOUT of child
			double[][] childCumulativeCost = new double[numInputs][2];
			// # of child
			double[] childForwardingCost = new double[numInputs];
			
			FederatedPlanCostEstimator.getChildCosts(hopCommon, memoTable, childCumulativeCost, childForwardingCost);

			for (int i = 0; i < (1 << numInputs); i++) {
				List<Pair<Long, FederatedOutput>> planChilds = new ArrayList<>();
				double lOutCumulativeCost = selfCost;
				double fOutCumulativeCost = selfCost;
	
				// For each input, determine if it should be FOUT or LOUT based on bit j in i
				for (int j = 0; j < numInputs; j++) {
					Hop inputHop = hop.getInput().get(j);
					final int bit = (i & (1 << j)) != 0 ? 1 : 0; // bit 값 계산 (FOUT/LOUT 결정)
					final FederatedOutput childType = (bit == 1) ? FederatedOutput.FOUT : FederatedOutput.LOUT;
					planChilds.add(Pair.of(inputHop.getHopID(), childType));

					lOutCumulativeCost += childCumulativeCost[j][bit];
					fOutCumulativeCost += childCumulativeCost[j][bit];
					// 비트 기반 산술 연산을 사용하여 전달 비용 추가
					fOutCumulativeCost += childForwardingCost[j] * (1 - bit); // bit == 0일 때 활성화
					lOutCumulativeCost += childForwardingCost[j] * bit; // bit == 1일 때 활성화
				}
				LOutFedPlanVariants.addFedPlan(new FedPlan(lOutCumulativeCost, LOutFedPlanVariants, planChilds));
				FOutFedPlanVariants.addFedPlan(new FedPlan(fOutCumulativeCost, FOutFedPlanVariants, planChilds));
			}
			LOutFedPlanVariants.pruneFedPlans();
			FOutFedPlanVariants.pruneFedPlans();

			memoTable.addFedPlanVariants(hopID, FederatedOutput.LOUT, LOutFedPlanVariants);
			memoTable.addFedPlanVariants(hopID, FederatedOutput.FOUT, FOutFedPlanVariants);
		} else {
			List<ConflictMergeResolveInfo> conflictMergeResolveInfos = conflictMergeResolveMap.get(hopID);
			conflictMergeResolveInfos.sort(Comparator.comparingLong(ConflictMergeResolveInfo::getConflictedHopID));

			ConflictedFedPlanVariants LOutFedPlanVariants = new ConflictedFedPlanVariants(hopCommon, FederatedOutput.LOUT, conflictMergeResolveInfos);
			ConflictedFedPlanVariants FOutFedPlanVariants = new ConflictedFedPlanVariants(hopCommon, FederatedOutput.FOUT, conflictMergeResolveInfos);
			
			int numOfConflictCombinations = 1 << conflictMergeResolveInfos.size();
			double mergeCost = FederatedPlanCostEstimator.computeMergeCost(conflictMergeResolveInfos, memoTable);
			selfCost += mergeCost;

			// 2^(# of conflicts), # of childs, LOUT/FOUT of child
			double[][][] childCumulativeCost = new double[numOfConflictCombinations][numInputs][2];
			int[][][] childForwardingBitMap = new int[numOfConflictCombinations][numInputs][2];
			double[] childForwardingCost = new double[numInputs]; // # of childs
	
			FederatedPlanCostEstimator.getConflictedChildCosts(hopCommon, memoTable, conflictMergeResolveInfos, childCumulativeCost, childForwardingBitMap, childForwardingCost);
			
			for (int i = 0; i < (1 << numInputs); i++) {
				List<Pair<Long, FederatedOutput>> planChilds = new ArrayList<>();
				
				for (int j = 0; j < numOfConflictCombinations; j++) {
					LOutFedPlanVariants.cumulativeCost[j][i] = selfCost;
					FOutFedPlanVariants.cumulativeCost[j][i] = selfCost;
				}
	
				for (int j = 0; j < numInputs; j++) {
					Hop inputHop = hop.getInput().get(j);
					
					final int bit = (i & (1 << j)) != 0 ? 1 : 0; // bit 값 계산 (FOUT/LOUT 결정)
					final FederatedOutput childType = (bit == 1) ? FederatedOutput.FOUT : FederatedOutput.LOUT;
					planChilds.add(Pair.of(inputHop.getHopID(), childType));
					
					for (int k = 0; k < numOfConflictCombinations; k++) {
						// 비트 기반 인덱스를 사용하여 누적 비용 업데이트
						LOutFedPlanVariants.cumulativeCost[k][i] += childCumulativeCost[k][j][bit];
						FOutFedPlanVariants.cumulativeCost[k][i] += childCumulativeCost[k][j][bit];
						
						// 비트 기반 산술 연산을 사용하여 전달 비용 추가
						FOutFedPlanVariants.cumulativeCost[k][i] += childForwardingCost[j] * (1 - bit); // bit == 0일 때 활성화
						LOutFedPlanVariants.cumulativeCost[k][i] += childForwardingCost[j] * bit; // bit == 1일 때 활성화
						
						if (mergeCost != 0) {
							FederatedPlanCostEstimator.computeForwardingMergeCost(LOutFedPlanVariants.forwardingBitMap[k][i], 
									childForwardingBitMap[k][j][bit], conflictMergeResolveInfos, memoTable);
						}

						LOutFedPlanVariants.forwardingBitMap[k][i] |= childForwardingBitMap[k][j][bit];
						FOutFedPlanVariants.forwardingBitMap[k][i] |= childForwardingBitMap[k][j][bit];
					}
				}
				LOutFedPlanVariants.addFedPlan(new FedPlan(0, LOutFedPlanVariants, planChilds));
				FOutFedPlanVariants.addFedPlan(new FedPlan(0, FOutFedPlanVariants, planChilds));
			}
			LOutFedPlanVariants.pruneConflictedFedPlans();
			FOutFedPlanVariants.pruneConflictedFedPlans();
			
			memoTable.addFedPlanVariants(hopID, FederatedOutput.LOUT, LOutFedPlanVariants);
			memoTable.addFedPlanVariants(hopID, FederatedOutput.FOUT, FOutFedPlanVariants);
		}
	}

	/**
	 * Returns the minimum cost plan for the root Hop, comparing both FOUT and LOUT variants.
	 * Used to select the final execution plan after enumeration.
	 * 
	 * @param HopID ?
	 * @param memoTable ?
	 * @return ?
	 */
	private static FedPlan getMinCostRootFedPlan(long HopID, FederatedMemoTable memoTable) {
		FedPlan lOutFedPlan = memoTable.getFedPlanAfterPrune(HopID, FederatedOutput.LOUT);
		FedPlan fOutFedPlan = memoTable.getFedPlanAfterPrune(HopID, FederatedOutput.FOUT);

		if (lOutFedPlan.getCumulativeCost() < fOutFedPlan.getCumulativeCost()){
			return lOutFedPlan;
		} else{
			return fOutFedPlan;
		}
	}

	/**
	 * Detects and resolves conflicts in federated plans starting from the root plan.
	 * This function performs a breadth-first search (BFS) to traverse the federated plan tree.
	 * It identifies conflicts where the same plan ID has different federated output types.
	 * For each conflict, it records the plan ID and its conflicting parent plans.
	 * The function ensures that each plan ID is associated with a consistent federated output type
	 * by resolving these conflicts iteratively.
	 *
	 * The process involves:
	 * - Using a map to track conflicts, associating each plan ID with its federated output type
	 *   and a list of parent plans.
	 * - Storing detected conflicts in a linked map, each entry containing a plan ID and its
	 *   conflicting parent plans.
	 * - Performing BFS traversal starting from the root plan, checking each child plan for conflicts.
	 * - If a conflict is detected (i.e., a plan ID has different output types), the conflicting plan
	 *   is removed from the BFS queue and added to the conflict map to prevent duplicate calculations.
	 * - Resolving conflicts by ensuring a consistent federated output type across the plan.
	 * - Re-running BFS with resolved conflicts to ensure all inconsistencies are addressed.
	 *
	 * @param rootPlan The root federated plan from which to start the conflict detection.
	 * @param memoTable The memoization table used to retrieve pruned federated plans.
	 * @return The cumulative additional cost for resolving conflicts.
	 */
	private static double detectAndResolveConflictFedPlan(FedPlan rootPlan, FederatedMemoTable memoTable) {
		// Map to track conflicts: maps a plan ID to its federated output type and list of parent plans
		Map<Long, Pair<FederatedOutput, List<FedPlan>>> conflictCheckMap = new HashMap<>();

		// LinkedMap to store detected conflicts, each with a plan ID and its conflicting parent plans
		LinkedHashMap<Long, List<FedPlan>> conflictLinkedMap = new LinkedHashMap<>();

		// LinkedMap for BFS traversal starting from the root plan (Do not use value (boolean))
		LinkedHashMap<FedPlan, Boolean> bfsLinkedMap = new LinkedHashMap<>();
		bfsLinkedMap.put(rootPlan, true);

		// Array to store cumulative additional cost for resolving conflicts
		double[] cumulativeAdditionalCost = new double[]{0.0};

		while (!bfsLinkedMap.isEmpty()) {
			// Perform BFS to detect conflicts in federated plans
			while (!bfsLinkedMap.isEmpty()) {
				FedPlan currentPlan = bfsLinkedMap.keySet().iterator().next();
				bfsLinkedMap.remove(currentPlan);

				// Iterate over each child plan of the current plan
				for (Pair<Long, FederatedOutput> childPlanPair : currentPlan.getChildFedPlans()) {
					FedPlan childFedPlan = memoTable.getFedPlanAfterPrune(childPlanPair);

					// Check if the child plan ID is already visited
					if (conflictCheckMap.containsKey(childPlanPair.getLeft())) {
						// Retrieve the existing conflict pair for the child plan
						Pair<FederatedOutput, List<FedPlan>> conflictChildPlanPair = conflictCheckMap.get(childPlanPair.getLeft());
						// Add the current plan to the list of parent plans
						conflictChildPlanPair.getRight().add(currentPlan);

						// If the federated output type differs, a conflict is detected
						if (conflictChildPlanPair.getLeft() != childPlanPair.getRight()) {
							// If this is the first detection, remove conflictChildFedPlan from the BFS queue and add it to the conflict linked map (queue)
							// If the existing FedPlan is not removed from the bfsqueue or both actions are performed, duplicate calculations for the same FedPlan and its children occur
							if (!conflictLinkedMap.containsKey(childPlanPair.getLeft())) {
								conflictLinkedMap.put(childPlanPair.getLeft(), conflictChildPlanPair.getRight());
								bfsLinkedMap.remove(childFedPlan);
							}
						}
					} else {
						// If no conflict exists, create a new entry in the conflict check map
						List<FedPlan> parentFedPlanList = new ArrayList<>();
						parentFedPlanList.add(currentPlan);

						// Map the child plan ID to its output type and list of parent plans
						conflictCheckMap.put(childPlanPair.getLeft(), new ImmutablePair<>(childPlanPair.getRight(), parentFedPlanList));
						// Add the child plan to the BFS queue
						bfsLinkedMap.put(childFedPlan, true);
					}
				}
			}
			// Resolve these conflicts to ensure a consistent federated output type across the plan
			// Re-run BFS with resolved conflicts
			bfsLinkedMap = FederatedPlanCostEstimator.resolveConflictFedPlan(memoTable, conflictLinkedMap, cumulativeAdditionalCost);
			conflictLinkedMap.clear();
		}

		// Return the cumulative additional cost for resolving conflicts
		return cumulativeAdditionalCost[0];
	}

	/**
	 * Data structure to store conflict and merge information for a specific Hop.
	 * This class maintains the state of conflict resolution and merge operations
	 * for a given Hop in the execution plan.
	 */
	public static class ConflictMergeResolveInfo {
		private long conflictedHopID;    		// ID of the Hop that originated the conflict
		private boolean isMerged;
		private boolean isSplited;
		private ResolvedType resolvedType;

		public ConflictMergeResolveInfo(long conflictedHopID, boolean isMerged, boolean isSplited, ResolvedType resolvedType) {
			this.conflictedHopID = conflictedHopID;
			this.isMerged = isMerged;
			this.isSplited = isSplited;
			this.resolvedType = resolvedType;
		}

		public long getConflictedHopID() {
			return conflictedHopID;
		}

		public boolean isMerged() {
			return isMerged;
		}

		public boolean isSplited() {
			return isSplited;
		}

		public ResolvedType getResolvedType() {
			return resolvedType;
		}
	}

	public static enum ResolvedType {
		INNER_PATH,
		OUTER_PATH,
		FIRST_SPLIT_LAST_MERGE,    // 첫 분기점 또는 마지막 		
		RESOLVE,       // 해결 지점
		TOP            // 최상위 지점
	};
}
