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
import java.util.Queue;
import java.util.LinkedList;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.sysds.hops.Hop;
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
	public static FedPlan enumerateFederatedPlanCost(Hop rootHop, boolean printTree) {
		// Create new memo table to store all plan variants
		FederatedMemoTable memoTable = new FederatedMemoTable();

		// Recursively enumerate all possible plans
		enumerateFederatedPlanCost(rootHop, memoTable);

		// Return the minimum cost plan for the root node
		FedPlan optimalPlan = getMinCostRootFedPlan(rootHop.getHopID(), memoTable);
		memoTable.pruneMemoTable();

		// Detect conflicts in the federated plans where different FedPlans have different FederatedOutput types
		List<Pair<Long, List<FedPlan>>> conflictFedPlanList = detectConflictFedPlan(optimalPlan, memoTable);
		
		// Resolve these conflicts to ensure a consistent federated output type across the plan
		FederatedPlanCostEstimator.resolveConflictFedPlan(optimalPlan, memoTable, conflictFedPlanList);

		// Optionally print the federated plan tree if requested
		if (printTree) memoTable.printFedPlanTree(optimalPlan);

		return optimalPlan;
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
	private static void enumerateFederatedPlanCost(Hop hop, FederatedMemoTable memoTable) {
		int numInputs = hop.getInput().size();

		// Process all input nodes first if not already in memo table
		for (Hop inputHop : hop.getInput()) {
			if (!memoTable.contains(inputHop.getHopID(), FederatedOutput.FOUT) 
				&& !memoTable.contains(inputHop.getHopID(), FederatedOutput.LOUT)) {
					enumerateFederatedPlanCost(inputHop, memoTable);
			}
		}

		// Generate all possible input combinations using binary representation
		// i represents a specific combination of FOUT/LOUT for inputs
		for (int i = 0; i < (1 << numInputs); i++) {
			List<Pair<Long, FederatedOutput>> planChilds = new ArrayList<>(); 

			// For each input, determine if it should be FOUT or LOUT based on bit j in i
			for (int j = 0; j < numInputs; j++) {
				Hop inputHop = hop.getInput().get(j);
				// If bit j is set (1), use FOUT; otherwise use LOUT
				FederatedOutput childType = ((i & (1 << j)) != 0) ?
					FederatedOutput.FOUT : FederatedOutput.LOUT;
				planChilds.add(Pair.of(inputHop.getHopID(), childType));
			}
			
			// Create and evaluate FOUT variant for current input combination
			FedPlan fOutPlan = memoTable.addFedPlan(hop, FederatedOutput.FOUT, planChilds);
			FederatedPlanCostEstimator.computeFederatedPlanCost(fOutPlan, memoTable);

			// Create and evaluate LOUT variant for current input combination
			FedPlan lOutPlan = memoTable.addFedPlan(hop, FederatedOutput.LOUT, planChilds);
			FederatedPlanCostEstimator.computeFederatedPlanCost(lOutPlan, memoTable);
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
		FedPlanVariants fOutFedPlanVariants = memoTable.getFedPlanVariants(HopID, FederatedOutput.FOUT);
		FedPlanVariants lOutFedPlanVariants = memoTable.getFedPlanVariants(HopID, FederatedOutput.LOUT);

		FedPlan minFOutFedPlan = fOutFedPlanVariants._fedPlanVariants.stream()
									.min(Comparator.comparingDouble(FedPlan::getTotalCost))
									.orElse(null);
		FedPlan minlOutFedPlan = lOutFedPlanVariants._fedPlanVariants.stream()
									.min(Comparator.comparingDouble(FedPlan::getTotalCost))
									.orElse(null);

		if (Objects.requireNonNull(minFOutFedPlan).getTotalCost()
				< Objects.requireNonNull(minlOutFedPlan).getTotalCost()) {
			return minFOutFedPlan;
		}
		return minlOutFedPlan;
	}

	/**
	 * Detects conflicts in federated plans starting from the root plan.
	 * This function performs a breadth-first search (BFS) to traverse the federated plan tree.
	 * It identifies conflicts where the same plan ID has different federated output types
	 * and returns a list of such conflicts, each represented by a plan ID and its conflicting parent plans.
	 *
	 * @param rootPlan The root federated plan from which to start the conflict detection.
	 * @param memoTable The memoization table used to retrieve pruned federated plans.
	 * @return A list of pairs, each containing a plan ID and a list of parent plans that have conflicting federated outputs.
	 */
	private static List<Pair<Long, List<FedPlan>>> detectConflictFedPlan(FedPlan rootPlan, FederatedMemoTable memoTable) {
		// Map to track conflicts: maps a plan ID to its federated output type and list of parent plans
		Map<Long, Pair<FederatedOutput, List<FedPlan>>> conflictCheckMap = new HashMap<>();
		// List to store detected conflicts, each with a plan ID and its conflicting parent plans
		List<Pair<Long, List<FedPlan>>> conflictFedPlanList = new ArrayList<>();

		// Queue for BFS traversal starting from the root plan
		Queue<FedPlan> bfsQueue = new LinkedList<>();
		bfsQueue.add(rootPlan);

		// Perform BFS to detect conflicts in federated plans
		while (!bfsQueue.isEmpty()) {
			FedPlan currentPlan = bfsQueue.poll();

			// Iterate over each child plan of the current plan
			for (Pair<Long, FederatedOutput> childPlanPair : currentPlan.getChildFedPlans()) {
				FedPlan childFedPlan = memoTable.getFedPlanAfterPrune(childPlanPair.getLeft(), childPlanPair.getRight());

				// Check if the child plan ID is already in the conflict check map
				if (conflictCheckMap.containsKey(childPlanPair.getLeft())) {
					// Retrieve the existing conflict pair for the child plan
					Pair<FederatedOutput, List<FedPlan>> conflictFedPlanPair = conflictCheckMap.get(childPlanPair.getLeft());
					// Add the current plan to the list of parent plans
					conflictFedPlanPair.getRight().add(currentPlan);

					// If the federated output type differs, a conflict is detected
					if (conflictFedPlanPair.getLeft() != childPlanPair.getRight()) {
						// Add the conflict to the conflict list
						conflictFedPlanList.add(new ImmutablePair<>(childPlanPair.getLeft(), conflictFedPlanPair.getRight()));
						// Add the child plan to the BFS queue for further exploration
						// Todo: Unsure whether to skip or continue traversal when encountering the same Hop ID with different FederatedOutput types
						bfsQueue.add(childFedPlan);
					}
				} else {
					// If no conflict exists, create a new entry in the conflict check map
					List<FedPlan> parentFedPlanList = new ArrayList<>();
					parentFedPlanList.add(currentPlan);

					// Map the child plan ID to its output type and list of parent plans
					conflictCheckMap.put(childPlanPair.getLeft(), new ImmutablePair<>(childPlanPair.getRight(), parentFedPlanList));
					// Add the child plan to the BFS queue
					bfsQueue.add(childFedPlan);
				}
			}
		}

		// Return the list of detected conflicts
		return conflictFedPlanList;
	}
}
