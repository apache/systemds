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

import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.cost.ComputeCost;
import org.apache.sysds.hops.fedplanner.FederatedMemoTable.FedPlan;
import org.apache.sysds.hops.fedplanner.FederatedMemoTable.HopCommon;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction.FederatedOutput;


/**
 * Cost estimator for federated execution plans.
 * Calculates computation, memory access, and network transfer costs for
 * federated operations.
 * Works in conjunction with FederatedMemoTable to evaluate different execution
 * plan variants.
 */
public class FederatedPlanCostEstimator {
	// Default value is used as a reasonable estimate since we only need
	// to compare relative costs between different federated plans
	// Memory bandwidth for local computations (25 GB/s)
	private static final double DEFAULT_MBS_MEMORY_BANDWIDTH = 25000.0;
	// Network bandwidth for data transfers between federated sites (1 Gbps)
	private static final double DEFAULT_MBS_NETWORK_BANDWIDTH = 125.0;
	private static final double DEFAULT_MBS_NETWORK_LATENCY = 0.001;

	// Retrieves the cumulative and forwarding costs of the child hops and stores
	// them in arrays
	public static void getChildCosts(HopCommon hopCommon, FederatedMemoTable memoTable, List<Hop> inputHops,
			double[][] childCumulativeCost, double[] childForwardingCost, List<Hop> lOUTOnlyinputHops,
			List<Double> lOUTOnlychildCumulativeCost, List<Double> lOUTOnlychildForwardingCost,
			List<Hop> fOUTOnlyinputHops, List<Double> fOUTOnlychildCumulativeCost,
			List<Double> fOUTOnlychildForwardingCost) {

		Iterator<Hop> iterator = inputHops.iterator();
		int currentIndex = 0;

		while (iterator.hasNext()) {
			Hop childHop = iterator.next();
			long childHopID = childHop.getHopID();

			FedPlan childFOutFedPlan = memoTable.getFedPlanAfterPrune(childHopID, FederatedOutput.FOUT);
			if (childFOutFedPlan == null) {
				lOUTOnlyinputHops.add(childHop);
				iterator.remove();
				continue;
			}

			FedPlan childLOutFedPlan = memoTable.getFedPlanAfterPrune(childHopID, FederatedOutput.LOUT);
			if (childLOutFedPlan == null) {
				fOUTOnlyinputHops.add(childHop);
				iterator.remove();
				continue;
			}

			childCumulativeCost[currentIndex][0] = childLOutFedPlan.getCumulativeCostPerParents();
			childCumulativeCost[currentIndex][1] = childFOutFedPlan.getCumulativeCostPerParents();
			childForwardingCost[currentIndex] = hopCommon.getChildForwardingWeight(childLOutFedPlan.getLoopContext())
					* childLOutFedPlan.getForwardingCostPerParents();
			currentIndex++;
		}

		for (int i = 0; i < lOUTOnlyinputHops.size(); i++) {
			Hop childHop = lOUTOnlyinputHops.get(i);
			long childHopID = childHop.getHopID();

			FedPlan childLOutFedPlan = memoTable.getFedPlanAfterPrune(childHopID, FederatedOutput.LOUT);
			
			if (childLOutFedPlan == null) {
				throw new RuntimeException("childLOutFedPlan is null for hopID: " + childHopID + " (see details above)");
			}
			lOUTOnlychildCumulativeCost.add(childLOutFedPlan.getCumulativeCostPerParents());
			lOUTOnlychildForwardingCost.add(hopCommon.getChildForwardingWeight(childLOutFedPlan.getLoopContext())
					* childLOutFedPlan.getForwardingCostPerParents());
		}

		for (int i = 0; i < fOUTOnlyinputHops.size(); i++) {
			Hop childHop = fOUTOnlyinputHops.get(i);
			long childHopID = childHop.getHopID();

			FedPlan childFOutFedPlan = memoTable.getFedPlanAfterPrune(childHopID, FederatedOutput.FOUT);

			if (childFOutFedPlan == null) {
				throw new RuntimeException("childFOutFedPlan is null for hopID: " + childHopID + " (see details above)");
			}
			fOUTOnlychildCumulativeCost.add(childFOutFedPlan.getCumulativeCostPerParents());
			fOUTOnlychildForwardingCost.add(hopCommon.getChildForwardingWeight(childFOutFedPlan.getLoopContext())
					* childFOutFedPlan.getForwardingCostPerParents());
		}
	}

	/**
	 * Computes the cost associated with a given Hop node.
	 * This method calculates both the self cost and the forwarding cost for the
	 * Hop,
	 * taking into account its type and the number of parent nodes.
	 *
	 * @param hopCommon The HopCommon object containing the Hop and its properties.
	 * @return The self cost of the Hop.
	 */
	public static double computeHopCost(HopCommon hopCommon) {
		// TWrite and TRead are meta-data operations, hence selfCost is zero
		if (hopCommon.hopRef instanceof DataOp) {
			if (((DataOp) hopCommon.hopRef).getOp() == Types.OpOpData.TRANSIENTWRITE) {
				hopCommon.setSelfCost(0);
				// Since TWrite and TRead have the same FedOutType, forwarding cost is zero
				hopCommon.setForwardingCost(0);
				return 0;
			} else if (((DataOp) hopCommon.hopRef).getOp() == Types.OpOpData.TRANSIENTREAD) {
				hopCommon.setSelfCost(0);
				// TRead may have a different FedOutType from its parent, so calculate
				// forwarding cost
				hopCommon.setForwardingCost(computeHopForwardingCost(hopCommon.hopRef.getOutputMemEstimate()));
				return 0;
			}
		}

		double selfCost = hopCommon.getComputeWeight() * computeSelfCost(hopCommon.hopRef);
		double forwardingCost = computeHopForwardingCost(hopCommon.hopRef.getOutputMemEstimate());

		hopCommon.setSelfCost(selfCost);
		hopCommon.setForwardingCost(forwardingCost);

		return selfCost;
	}

	/**
	 * Computes the cost for the current Hop node.
	 *
	 * @param currentHop The Hop node whose cost needs to be computed
	 * @return The total cost for the current node's operation
	 */
	private static double computeSelfCost(Hop currentHop) {
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
		return memSize / (1024 * 1024) / DEFAULT_MBS_MEMORY_BANDWIDTH;
	}

	/**
	 * Calculates the network transfer cost based on data size and network
	 * bandwidth.
	 * Used when federation status changes between parent and child plans.
	 *
	 * @param memSize Size of data to be transferred (in bytes)
	 * @return Time cost for network transfer (in seconds)
	 */
	private static double computeHopForwardingCost(double memSize) {
		return DEFAULT_MBS_NETWORK_LATENCY + (memSize / (1024 * 1024) / DEFAULT_MBS_NETWORK_BANDWIDTH);
	}

	/**
	 * Resolves conflicts in federated plans where different plans have different
	 * FederatedOutput types.
	 * This function traverses the list of conflicting plans in reverse order to
	 * ensure that conflicts
	 * are resolved from the bottom-up, allowing for consistent federated output
	 * types across the plan.
	 * It calculates additional costs for each potential resolution and updates the
	 * cumulative additional cost.
	 *
	 * @param memoTable                The FederatedMemoTable containing all
	 *                                 federated plan variants.
	 * @param conflictFedPlanLinkedMap A map of plan IDs to lists of parent plans
	 *                                 with conflicting federated outputs.
	 * @param cumulativeAdditionalCost An array to store the cumulative additional
	 *                                 cost incurred by resolving conflicts.
	 * @return A LinkedHashMap of resolved federated plans, marked with a boolean
	 *         indicating resolution status.
	 */
	public static LinkedHashMap<FedPlan, Boolean> resolveConflictFedPlan(FederatedMemoTable memoTable,
			LinkedHashMap<Long, List<FedPlan>> conflictFedPlanLinkedMap, double[] cumulativeAdditionalCost) {
		// LinkedHashMap to store resolved federated plans for BFS traversal.
		LinkedHashMap<FedPlan, Boolean> resolvedFedPlanLinkedMap = new LinkedHashMap<>();

		// Traverse the conflictFedPlanList in reverse order after BFS to resolve
		// conflicts
		for (Map.Entry<Long, List<FedPlan>> conflictFedPlanPair : conflictFedPlanLinkedMap.entrySet()) {
			long conflictHopID = conflictFedPlanPair.getKey();
			List<FedPlan> conflictParentFedPlans = conflictFedPlanPair.getValue();

			// Retrieve the conflicting federated plans for LOUT and FOUT types
			FedPlan confilctLOutFedPlan = memoTable.getFedPlanAfterPrune(conflictHopID, FederatedOutput.LOUT);
			FedPlan confilctFOutFedPlan = memoTable.getFedPlanAfterPrune(conflictHopID, FederatedOutput.FOUT);

			if (confilctLOutFedPlan == null || confilctFOutFedPlan == null) {
				// Todo: Handle Error
				FederatedPlannerLogger.logConflictResolutionError(conflictHopID, confilctLOutFedPlan, "Resolve Conflict");
				continue;
			}

			// Variables to store additional costs for LOUT and FOUT types
			double lOutAdditionalCost = 0;
			double fOutAdditionalCost = 0;

			// Flags to check if the plan involves network transfer
			// Network transfer cost is calculated only once, even if it occurs multiple
			// times
			boolean isLOutForwarding = false;
			boolean isFOutForwarding = false;

			// Determine the optimal federated output type based on the calculated costs
			FederatedOutput optimalFedOutType;

			// Iterate over each parent federated plan in the current conflict pair
			for (FedPlan conflictParentFedPlan : conflictParentFedPlans) {
				// Find the calculated FedOutType of the child plan
				Pair<Long, FederatedOutput> cacluatedConflictPlanPair = conflictParentFedPlan.getChildFedPlans()
						.stream()
						.filter(pair -> pair.getLeft().equals(conflictHopID))
						.findFirst()
						.orElseThrow(
								() -> new NoSuchElementException("No matching pair found for ID: " + conflictHopID));

				// CASE 1. Calculated LOUT / Parent LOUT / Current LOUT: Total cost remains
				// unchanged.
				// CASE 2. Calculated LOUT / Parent FOUT / Current LOUT: Total cost remains
				// unchanged, subtract net cost, add net cost later.
				// CASE 3. Calculated FOUT / Parent LOUT / Current LOUT: Change total cost,
				// subtract net cost.
				// CASE 4. Calculated FOUT / Parent FOUT / Current LOUT: Change total cost, add
				// net cost later.
				// CASE 5. Calculated LOUT / Parent LOUT / Current FOUT: Change total cost, add
				// net cost later.
				// CASE 6. Calculated LOUT / Parent FOUT / Current FOUT: Change total cost,
				// subtract net cost.
				// CASE 7. Calculated FOUT / Parent LOUT / Current FOUT: Total cost remains
				// unchanged, subtract net cost, add net cost later.
				// CASE 8. Calculated FOUT / Parent FOUT / Current FOUT: Total cost remains
				// unchanged.

				// Adjust LOUT, FOUT costs based on the calculated plan's output type
				if (cacluatedConflictPlanPair.getRight() == FederatedOutput.LOUT) {
					// When changing from calculated LOUT to current FOUT, subtract the existing
					// LOUT total cost and add the FOUT total cost
					// When maintaining calculated LOUT to current LOUT, the total cost remains
					// unchanged.
					fOutAdditionalCost += confilctFOutFedPlan.getCumulativeCostPerParents()
							- confilctLOutFedPlan.getCumulativeCostPerParents();

					if (conflictParentFedPlan.getFedOutType() == FederatedOutput.LOUT) {
						// (CASE 1) Previously, calculated was LOUT and parent was LOUT, so no network
						// transfer cost occurred
						// (CASE 5) If changing from calculated LOUT to current FOUT, network transfer
						// cost occurs, but calculated later
						isFOutForwarding = true;
					} else {
						// Previously, calculated was LOUT and parent was FOUT, so network transfer cost
						// occurred
						// (CASE 2) If maintaining calculated LOUT to current LOUT, subtract existing
						// network transfer cost and calculate later
						isLOutForwarding = true;
						lOutAdditionalCost -= confilctLOutFedPlan.getForwardingCostPerParents();

						// (CASE 6) If changing from calculated LOUT to current FOUT, no network
						// transfer cost occurs, so subtract it
						fOutAdditionalCost -= confilctLOutFedPlan.getForwardingCostPerParents();
					}
				} else {
					lOutAdditionalCost += confilctLOutFedPlan.getCumulativeCostPerParents()
							- confilctFOutFedPlan.getCumulativeCostPerParents();

					if (conflictParentFedPlan.getFedOutType() == FederatedOutput.FOUT) {
						isLOutForwarding = true;
					} else {
						isFOutForwarding = true;
						lOutAdditionalCost -= conflictParentFedPlan
								.getChildForwardingWeight(confilctLOutFedPlan.getLoopContext())
								* confilctLOutFedPlan.getForwardingCostPerParents();
						fOutAdditionalCost -= conflictParentFedPlan
								.getChildForwardingWeight(confilctLOutFedPlan.getLoopContext())
								* confilctLOutFedPlan.getForwardingCostPerParents();
					}
				}
			}

			// Add network transfer costs if applicable
			if (isLOutForwarding) {
				lOutAdditionalCost += confilctLOutFedPlan.getForwardingCost();
			}
			if (isFOutForwarding) {
				fOutAdditionalCost += confilctFOutFedPlan.getForwardingCost();
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

			// Update only the optimal federated output type, not the cost itself or
			// recursively
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
}
