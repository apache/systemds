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
import org.apache.sysds.hops.fedplanner.FederatedMemoTable.FedPlanVariants;
import org.apache.sysds.hops.fedplanner.FederatedMemoTable.HopCommon;
import org.apache.sysds.hops.fedplanner.FederatedPlanCostEnumerator.ConflictMergeResolveInfo;
import org.apache.sysds.hops.fedplanner.FederatedMemoTable.ConflictedFedPlanVariants;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction.FederatedOutput;

import java.util.ArrayList;
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

	public static void getChildCosts(HopCommon hopCommon, FederatedMemoTable memoTable, double[][] childCumulativeCost, double[] childForwardingCost) {
		List<Hop> inputHops = hopCommon.hopRef.getInput();
		
		for (int i = 0; i < inputHops.size(); i++) {
			long childHopID = inputHops.get(i).getHopID();
			
			FedPlan childLOutFedPlan = memoTable.getFedPlanAfterPrune(childHopID, FederatedOutput.LOUT);
			FedPlan childFOutFedPlan = memoTable.getFedPlanAfterPrune(childHopID, FederatedOutput.FOUT);

			childCumulativeCost[i][0] = childLOutFedPlan.getCumulativeCost();
			childCumulativeCost[i][1] = childFOutFedPlan.getCumulativeCost();
			childForwardingCost[i] = childLOutFedPlan.getForwardingCost();
		}
	}

	public static void getConflictedChildCosts(HopCommon hopCommon, FederatedMemoTable memoTable, List<ConflictMergeResolveInfo> conflictMergeResolveInfos, 
												double[][][] childCumulativeCost, int[][][] childForwardingBitMap, double[] childForwardingCost) {
		List<Hop> inputHops = hopCommon.hopRef.getInput();
		int numConflictCombinations = 1 << conflictMergeResolveInfos.size();

		for (int i = 0; i < inputHops.size(); i++) {
			long childHopID = inputHops.get(i).getHopID();
			
			FedPlanVariants childLOutVariants = memoTable.getFedPlanVariants(childHopID, FederatedOutput.LOUT);
			FedPlanVariants childFOutVariants = memoTable.getFedPlanVariants(childHopID, FederatedOutput.FOUT);
			
			childForwardingCost[i] = childLOutVariants.getForwardingCost();

			if (childLOutVariants instanceof ConflictedFedPlanVariants) {
				FedPlan childLOutFedPlan = childLOutVariants.getFedPlanVariants().get(0);
				FedPlan childFOutFedPlan = childFOutVariants.getFedPlanVariants().get(0);

				for (int j = 0; j < numConflictCombinations; j++) {
					childCumulativeCost[j][i][0] = childLOutFedPlan.getCumulativeCost();
					childCumulativeCost[j][i][1] = childFOutFedPlan.getCumulativeCost();
				}	
			}
			else {
				ConflictedFedPlanVariants conflictedChildLOutVariants = (ConflictedFedPlanVariants) childLOutVariants;
				ConflictedFedPlanVariants conflictedChildFOutVariants = (ConflictedFedPlanVariants) childFOutVariants;

				computeConflictedChildCosts(conflictMergeResolveInfos, conflictedChildLOutVariants, childCumulativeCost, childForwardingBitMap, i, 0);
				computeConflictedChildCosts(conflictMergeResolveInfos, conflictedChildFOutVariants, childCumulativeCost, childForwardingBitMap, i, 1);
			}
		}
	}

	private static void computeConflictedChildCosts(List<ConflictMergeResolveInfo> conflictInfos, ConflictedFedPlanVariants conflictedChildVariants, 
										double[][][] childCumulativeCost, int[][][] childForwardingBitMap, int childIdx, int fedOutTypeIdx){
		int i = 0, j = 0;
		int pLen = conflictInfos.size();
		int cLen = conflictedChildVariants.conflictInfos.size();
		int numConflictCombinations = 1 << conflictInfos.size();

		// Step 1: 공통 제약 조건과 비공통 자식 위치 계산
		List<CommonConstraint> common = new ArrayList<>();
		List<Integer> nonCommonChildPos = new ArrayList<>();

		while (i < pLen && j < cLen) {
			long pHopID = conflictInfos.get(i).getConflictedHopID();
			long cHopID = conflictedChildVariants.conflictInfos.get(j).getConflictedHopID();

			if (pHopID == cHopID) {
				int pBitPos = pLen - 1 - i;
				int cBitPos = cLen - 1 - j;
				common.add(new CommonConstraint(pHopID, pBitPos, cBitPos));
				i++;
				j++;
			} else if (pHopID < cHopID) {
				i++;
			} else {
				int cBitPos = cLen - 1 - j;
				nonCommonChildPos.add(cBitPos);
				j++;
			}
		}

		int restNumBits = nonCommonChildPos.size();
		for (int parentIdx = 0; parentIdx < numConflictCombinations; parentIdx++) {
			// 공통 제약 조건을 기반으로 baseChildIdx 계산
			int baseChildIdx = 0;
			for (CommonConstraint cc : common) {
				int bit = (parentIdx >> cc.pBitPos) & 1;
				baseChildIdx |= (bit << cc.cBitPos);
			}

			// 최소 비용을 가진 자식 인덱스 찾기
			double minChildCost = Double.MAX_VALUE;
			int minChildIdx = -1;
			for (int restValue = 0; restValue < (1 << restNumBits); restValue++) {
				int temp = 0;
				for (int bitIdx = 0; bitIdx < restNumBits; bitIdx++) {
					if (((restValue >> bitIdx) & 1) == 1) {
						temp |= (1 << nonCommonChildPos.get(bitIdx));
					}
				}
				int tempChildIdx = baseChildIdx | temp;
				if (conflictedChildVariants.cumulativeCost[tempChildIdx][0] < minChildCost) {
					minChildCost = conflictedChildVariants.cumulativeCost[tempChildIdx][0];
					minChildIdx = tempChildIdx;
				}
			}

			// 자식의 isForwardBitMap을 부모의 비트 위치로 변환
			int childForwardBitMap = conflictedChildVariants.forwardingBitMap[minChildIdx][0];
			int convertedBitmask = 0;
			for (CommonConstraint cc : common) {
				int childBit = (childForwardBitMap >> cc.cBitPos) & 1;
				if (childBit == 1) {
					convertedBitmask |= (1 << cc.pBitPos);
				}
			}

			childCumulativeCost[parentIdx][childIdx][fedOutTypeIdx] = minChildCost;
			childForwardingBitMap[parentIdx][childIdx][fedOutTypeIdx] = convertedBitmask;
		}
	} 

	// Todo: (최적화) 추후에 MemoTable retrieve 하지 않게 최적화 가능
	public static double computeForwardingMergeCost(int parentBitmask, int childBitmask, List<ConflictMergeResolveInfo> conflictInfos, FederatedMemoTable memoTable){
		int overlappingBits = parentBitmask & childBitmask;
		double overlappingForwardingCost = 0.0;

		int pLen = conflictInfos.size();
		for (int b = 0; b < pLen; b++) {
			int bitPos = pLen - 1 - b;
			if ((overlappingBits & (1 << bitPos)) != 0) {
				overlappingForwardingCost += memoTable.getFedPlanVariants(conflictInfos.get(b).getConflictedHopID(), FederatedOutput.LOUT).getForwardingCost();
			}
		}
		
		return overlappingForwardingCost;
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
					fOutAdditionalCost += confilctFOutFedPlan.getCumulativeCost() - confilctLOutFedPlan.getCumulativeCost();

					if (conflictParentFedPlan.getFedOutType() == FederatedOutput.LOUT) {
						// (CASE 1) Previously, calculated was LOUT and parent was LOUT, so no network transfer cost occurred
						// (CASE 5) If changing from calculated LOUT to current FOUT, network transfer cost occurs, but calculated later
						isFOutNetTransfer = true;
					} else {
						// Previously, calculated was LOUT and parent was FOUT, so network transfer cost occurred
                    	// (CASE 2) If maintaining calculated LOUT to current LOUT, subtract existing network transfer cost and calculate later
						isLOutNetTransfer = true;
						lOutAdditionalCost -= confilctLOutFedPlan.getForwardingCost();

						// (CASE 6) If changing from calculated LOUT to current FOUT, no network transfer cost occurs, so subtract it
						fOutAdditionalCost -= confilctLOutFedPlan.getForwardingCost();
					}
				} else {
					lOutAdditionalCost += confilctLOutFedPlan.getCumulativeCost() - confilctFOutFedPlan.getCumulativeCost();

					if (conflictParentFedPlan.getFedOutType() == FederatedOutput.FOUT) {
						isLOutNetTransfer = true;
					} else {
						isFOutNetTransfer = true;
						lOutAdditionalCost -= confilctLOutFedPlan.getForwardingCost();
						fOutAdditionalCost -= confilctLOutFedPlan.getForwardingCost();
					}
				}
			}

			// Add network transfer costs if applicable
			if (isLOutNetTransfer) {
				lOutAdditionalCost += confilctLOutFedPlan.getForwardingCost();
			}
			if (isFOutNetTransfer) {
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

	// Todo: (구현) forwarding bitmap을 본 뒤, merge cost 일일히 type에 따라 계산해야함.
	public static double computeMergeCost(List<ConflictMergeResolveInfo> conflictMergeResolveInfos, FederatedMemoTable memoTable){
		double mergeCost = 0;

		for (ConflictMergeResolveInfo conflictInfo: conflictMergeResolveInfos){
			int numOfMergedHops = conflictInfo.getNumOfMergedHops();
			
			if (numOfMergedHops != 0){
				double selfCost = memoTable.getFedPlanVariants(conflictInfo.getConflictedHopID(), FederatedOutput.LOUT).getSelfCost();
				mergeCost += selfCost * numOfMergedHops;
			}
		}

		return mergeCost;
	}

	public static void computeHopCost(HopCommon hopCommon){
		Hop hop = hopCommon.hopRef;
		hopCommon.setSelfCost(computeSelfCost(hop));
		hopCommon.setForwardingCost(computeHopForwardingCost(hop.getOutputMemEstimate()));
	}

	/**
	 * Computes the cost for the current Hop node.
	 * 
	 * @param currentHop The Hop node whose cost needs to be computed
	 * @return The total cost for the current node's operation
	 */
	private static double computeSelfCost(Hop currentHop){
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
	private static double computeHopForwardingCost(double memSize) {
		return memSize / (1024*1024) / DEFAULT_MBS_NETWORK_BANDWIDTH;
	}

	public static class CommonConstraint {
		long name;
		int pBitPos;
		int cBitPos;

		CommonConstraint(long name, int pBitPos, int cBitPos) {
			this.name = name;
			this.pBitPos = pBitPos;
			this.cBitPos = cBitPos;
		}
	}
}
