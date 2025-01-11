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
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.cost.ComputeCost;
import org.apache.sysds.hops.fedplanner.FederatedMemoTable.FedPlan;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction.FederatedOutput;

import java.util.LinkedHashMap;
import java.util.NoSuchElementException;
import java.util.List;
import java.util.Map;

/**
 * Cost estimator for federated execution plans.
 * Calculates computation, memory access, and network transfer costs for federated operations.
 * Works in conjunction with FederatedMemoTable to evaluate different execution plan variants.
 */
public class FederatedPlanCostEstimator {
	// Default value is used as a reasonable estimate since we only need
	// to compare relative costs between different federated plans
	// Memory bandwidth for local computations (25 GB/s)
	private static final double DEFAULT_MBS_MEMORY_BANDWIDTH = 25000.0;
	// Network bandwidth for data transfers between federated sites (1 Gbps)
	private static final double DEFAULT_MBS_NETWORK_BANDWIDTH = 125.0;

	/**
	 * Computes total cost of federated plan by:
	 * 1. Computing current node cost (if not cached)
	 * 2. Adding minimum-cost child plans
	 * 3. Including network transfer costs when needed
	 *
	 * @param currentPlan Plan to compute cost for
	 * @param memoTable Table containing all plan variants
	 */
	public static void computeFederatedPlanCost(FedPlan currentPlan, FederatedMemoTable memoTable) {
		double totalCost;
		Hop currentHop = currentPlan.getHopRef();

		// Step 1: Calculate current node costs if not already computed
		if (currentPlan.getSelfCost() == 0) {
			// Compute cost for current node (computation + memory access)
			totalCost = computeCurrentCost(currentHop);
			currentPlan.setSelfCost(totalCost);
			// Calculate potential network transfer cost if federation type changes
			currentPlan.setNetTransferCost(computeHopNetworkAccessCost(currentHop.getOutputMemEstimate()));
		} else {
			totalCost = currentPlan.getSelfCost();
		}
		
		// Step 2: Process each child plan and add their costs
		for (Pair<Long, FederatedOutput> childPlanPair : currentPlan.getChildFedPlans()) {
			// Find minimum cost child plan considering federation type compatibility
			// Note: This approach might lead to suboptimal or wrong solutions when a child has multiple parents
			// because we're selecting child plans independently for each parent
			FedPlan planRef = memoTable.getMinCostFedPlan(childPlanPair);

			// Add child plan cost (includes network transfer cost if federation types differ)
			totalCost += planRef.getTotalCost() + planRef.getCondNetTransferCost(currentPlan.getFedOutType());
		}
		
		// Step 3: Set final cumulative cost including current node
		currentPlan.setTotalCost(totalCost);
	}

	/**
	 * Resolves conflicts in federated plans where different plans have different FederatedOutput types.
	 * This function traverses the list of conflicting plans in reverse order to ensure that conflicts
	 * are resolved from the bottom-up, allowing for consistent federated output types across the plan.
	 * It calculates additional costs for each potential resolution and updates the cumulative additional cost.
	 *
	 * @param memoTable The FederatedMemoTable containing all federated plan variants.
	 * @param conflictFedPlanLinkedMap A map of plan IDs to lists of parent plans with conflicting federated outputs.
	 * @param cumulativeAdditionalCost An array to store the cumulative additional cost incurred by resolving conflicts.
	 * @return A LinkedHashMap of resolved federated plans, marked with a boolean indicating resolution status.
	 */
	public static LinkedHashMap<FedPlan, Boolean> resolveConflictFedPlan(FederatedMemoTable memoTable, LinkedHashMap<Long, List<FedPlan>> conflictFedPlanLinkedMap, double[] cumulativeAdditionalCost) {
		// LinkedHashMap to store resolved federated plans for BFS traversal.
		LinkedHashMap<FedPlan, Boolean> resolvedFedPlanLinkedMap = new LinkedHashMap<>();

		// Traverse the conflictFedPlanList in reverse order after BFS to resolve conflicts
		for (Map.Entry<Long, List<FedPlan>> conflictFedPlanPair : conflictFedPlanLinkedMap.entrySet()) {
			long conflictHopID = conflictFedPlanPair.getKey();
			List<FedPlan> conflictParentFedPlans = conflictFedPlanPair.getValue();

			// Retrieve the conflicting federated plans for LOUT and FOUT types
			FedPlan confilctLOutFedPlan = memoTable.getFedPlanAfterPrune(conflictHopID, FederatedOutput.LOUT);
			FedPlan confilctFOutFedPlan = memoTable.getFedPlanAfterPrune(conflictHopID, FederatedOutput.FOUT);

			// Variables to store additional costs for LOUT and FOUT types
			double lOutAdditionalCost = 0;
			double fOutAdditionalCost = 0;

			// Flags to check if the plan involves network transfer
			// Network transfer cost is calculated only once, even if it occurs multiple times
			boolean isLOutNetTransfer = false;
			boolean isFOutNetTransfer = false; 

			// Determine the optimal federated output type based on the calculated costs
			FederatedOutput optimalFedOutType;

			// Iterate over each parent federated plan in the current conflict pair
			for (FedPlan conflictParentFedPlan : conflictParentFedPlans) {
				// Find the calculated FedOutType of the child plan
				Pair<Long, FederatedOutput> cacluatedConflictPlanPair = conflictParentFedPlan.getChildFedPlans().stream()
					.filter(pair -> pair.getLeft().equals(conflictHopID))
					.findFirst()
					.orElseThrow(() -> new NoSuchElementException("No matching pair found for ID: " + conflictHopID));
							
				// CASE 1. Calculated LOUT / Parent LOUT / Current LOUT: Total cost remains unchanged.
				// CASE 2. Calculated LOUT / Parent FOUT / Current LOUT: Total cost remains unchanged, subtract net cost, add net cost later.
				// CASE 3. Calculated FOUT / Parent LOUT / Current LOUT: Change total cost, subtract net cost.
				// CASE 4. Calculated FOUT / Parent FOUT / Current LOUT: Change total cost, add net cost later.
				// CASE 5. Calculated LOUT / Parent LOUT / Current FOUT: Change total cost, add net cost later.
				// CASE 6. Calculated LOUT / Parent FOUT / Current FOUT: Change total cost, subtract net cost.
				// CASE 7. Calculated FOUT / Parent LOUT / Current FOUT: Total cost remains unchanged, subtract net cost, add net cost later.
				// CASE 8. Calculated FOUT / Parent FOUT / Current FOUT: Total cost remains unchanged.
				
				// Adjust LOUT, FOUT costs based on the calculated plan's output type
				if (cacluatedConflictPlanPair.getRight() == FederatedOutput.LOUT) {
					// When changing from calculated LOUT to current FOUT, subtract the existing LOUT total cost and add the FOUT total cost
					// When maintaining calculated LOUT to current LOUT, the total cost remains unchanged.
					fOutAdditionalCost += confilctFOutFedPlan.getTotalCost() - confilctLOutFedPlan.getTotalCost();

					if (conflictParentFedPlan.getFedOutType() == FederatedOutput.LOUT) {
						// (CASE 1) Previously, calculated was LOUT and parent was LOUT, so no network transfer cost occurred
						// (CASE 5) If changing from calculated LOUT to current FOUT, network transfer cost occurs, but calculated later
						isFOutNetTransfer = true;
					} else {
						// Previously, calculated was LOUT and parent was FOUT, so network transfer cost occurred
                    	// (CASE 2) If maintaining calculated LOUT to current LOUT, subtract existing network transfer cost and calculate later
						isLOutNetTransfer = true;
						lOutAdditionalCost -= confilctLOutFedPlan.getNetTransferCost();

						// (CASE 6) If changing from calculated LOUT to current FOUT, no network transfer cost occurs, so subtract it
						fOutAdditionalCost -= confilctLOutFedPlan.getNetTransferCost();
					}
				} else {
					lOutAdditionalCost += confilctLOutFedPlan.getTotalCost() - confilctFOutFedPlan.getTotalCost();

					if (conflictParentFedPlan.getFedOutType() == FederatedOutput.FOUT) {
						isLOutNetTransfer = true;
					} else {
						isFOutNetTransfer = true;
						lOutAdditionalCost -= confilctLOutFedPlan.getNetTransferCost();
						fOutAdditionalCost -= confilctLOutFedPlan.getNetTransferCost();
					}
				}
			}

			// Add network transfer costs if applicable
			if (isLOutNetTransfer) {
				lOutAdditionalCost += confilctLOutFedPlan.getNetTransferCost();
			}
			if (isFOutNetTransfer) {
				fOutAdditionalCost += confilctFOutFedPlan.getNetTransferCost();
			}

			// Determine the optimal federated output type based on the calculated costs
			if (lOutAdditionalCost <= fOutAdditionalCost) {
				optimalFedOutType = FederatedOutput.LOUT;
				cumulativeAdditionalCost[0] += lOutAdditionalCost;
				resolvedFedPlanLinkedMap.put(confilctLOutFedPlan, true);
			} else {
				optimalFedOutType = FederatedOutput.FOUT;
				cumulativeAdditionalCost[0] += fOutAdditionalCost;
				resolvedFedPlanLinkedMap.put(confilctFOutFedPlan, true);
			}    

			// Update only the optimal federated output type, not the cost itself or recursively
			for (FedPlan conflictParentFedPlan : conflictParentFedPlans) {
				for (Pair<Long, FederatedOutput> childPlanPair : conflictParentFedPlan.getChildFedPlans()) {
					if (childPlanPair.getLeft() == conflictHopID && childPlanPair.getRight() != optimalFedOutType) {
						int index = conflictParentFedPlan.getChildFedPlans().indexOf(childPlanPair);
						conflictParentFedPlan.getChildFedPlans().set(index, 
							Pair.of(childPlanPair.getLeft(), optimalFedOutType));
						break;
					}
				}
			}
		}
		return resolvedFedPlanLinkedMap;
	}
	
	/**
	 * Computes the cost for the current Hop node.
	 * 
	 * @param currentHop The Hop node whose cost needs to be computed
	 * @return The total cost for the current node's operation
	 */
	private static double computeCurrentCost(Hop currentHop){
		double computeCost = ComputeCost.getHOPComputeCost(currentHop);
		double inputAccessCost = computeHopMemoryAccessCost(currentHop.getInputMemEstimate());
		double ouputAccessCost = computeHopMemoryAccessCost(currentHop.getOutputMemEstimate());
		
		// Compute total cost assuming:
		// 1. Computation and input access can be overlapped (hence taking max)
		// 2. Output access must wait for both to complete (hence adding)
		return Math.max(computeCost, inputAccessCost) + ouputAccessCost;
	}

	/**
	 * Calculates the memory access cost based on data size and memory bandwidth.
	 * 
	 * @param memSize Size of data to be accessed (in bytes)
	 * @return Time cost for memory access (in seconds)
	 */
	private static double computeHopMemoryAccessCost(double memSize) {
		return memSize / (1024*1024) / DEFAULT_MBS_MEMORY_BANDWIDTH;
	}

	/**
	 * Calculates the network transfer cost based on data size and network bandwidth.
	 * Used when federation status changes between parent and child plans.
	 * 
	 * @param memSize Size of data to be transferred (in bytes)
	 * @return Time cost for network transfer (in seconds)
	 */
	private static double computeHopNetworkAccessCost(double memSize) {
		return memSize / (1024*1024) / DEFAULT_MBS_NETWORK_BANDWIDTH;
	}
}
