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

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.Arrays;
import java.util.Collections;
import org.apache.sysds.hops.Hop;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction.FederatedOutput;
import org.apache.sysds.hops.fedplanner.FederatedPlanCostEnumerator.ConflictMergeResolveInfo;
import org.apache.sysds.hops.fedplanner.FederatedPlanCostEnumerator.ResolvedType;

/**
 * A Memoization Table for managing federated plans (FedPlan) based on combinations of Hops and fedOutTypes.
 * This table stores and manages different execution plan variants for each Hop and fedOutType combination,
 * facilitating the optimization of federated execution plans.
 */
public class FederatedMemoTable {
	// Maps Hop ID and fedOutType pairs to their plan variants
	private final Map<Pair<Long, FederatedOutput>, FedPlanVariants> hopMemoTable = new HashMap<>();

	public void addFedPlanVariants(long hopID, FederatedOutput fedOutType, FedPlanVariants fedPlanVariants) {
		hopMemoTable.put(new ImmutablePair<>(hopID, fedOutType), fedPlanVariants);
	}

	public FedPlanVariants getFedPlanVariants(Pair<Long, FederatedOutput> fedPlanPair) {
		return hopMemoTable.get(fedPlanPair);
	}

	public FedPlanVariants getFedPlanVariants(long hopID, FederatedOutput fedOutType) {
		return hopMemoTable.get(new ImmutablePair<>(hopID, fedOutType));
	}

	public FedPlan getFedPlanAfterPrune(long hopID, FederatedOutput fedOutType) {
		FedPlanVariants fedPlanVariantList = hopMemoTable.get(new ImmutablePair<>(hopID, fedOutType));
		return fedPlanVariantList._fedPlanVariants.get(0);
	}

	public FedPlan getFedPlanAfterPrune(Pair<Long, FederatedOutput> fedPlanPair) {
		FedPlanVariants fedPlanVariantList = hopMemoTable.get(fedPlanPair);
		return fedPlanVariantList._fedPlanVariants.get(0);
	}

	public boolean contains(long hopID, FederatedOutput fedOutType) {
		return hopMemoTable.containsKey(new ImmutablePair<>(hopID, fedOutType));
	}

	public static class ConflictedFedPlanVariants extends FedPlanVariants {
		public List<ConflictMergeResolveInfo> conflictInfos;
		protected int numConflictCombinations;
		// 2^(# of conflicts), 2^(# of childs)
		protected double[][] cumulativeCost;
		protected int[][] forwardingBitMap;

		// bitset array (java class) >> arbitary length >>

		public ConflictedFedPlanVariants(HopCommon hopCommon, FederatedOutput fedOutType, 
				List<ConflictMergeResolveInfo> conflictMergeResolveInfos) {
			super(hopCommon, fedOutType);
			this.conflictInfos = conflictMergeResolveInfos;
			this.numConflictCombinations = 1 << this.conflictInfos.size();
			this.cumulativeCost = new double[this.numConflictCombinations][this._fedPlanVariants.size()];
			this.forwardingBitMap = new int[this.numConflictCombinations][this._fedPlanVariants.size()];
			// Initialize isForwardBitMap to 0
			for (int i = 0; i < this.numConflictCombinations; i++) {
				Arrays.fill(this.cumulativeCost[i], 0);
				Arrays.fill(this.forwardingBitMap[i], 0);
			}
		}
		
		// Todo: (최적화) java bitset 사용하여, 다수의 conflict 처리할 수 있도록 해야 함.
		// Todo: (구현) 만약 resolve point (converge, first-split & last-merge) child로 내려가면서 recursive하게 prune 해야 함. (이때, parents의 LOUT/FOUT의 Optimal Plan을 동시에 고려해야함)
		public void pruneConflictedFedPlans() {
			// Step 1: Initialize prunedCost and prunedIsForwardingBitMap with minimal values per combination
			double[][] prunedCost = new double[this.numConflictCombinations][1];
			int[][] prunedIsForwardingBitMap = new int[this.numConflictCombinations][1];
			List<FedPlan> prunedFedPlanVariants = new ArrayList<>();

			for (int i = 0; i < this.numConflictCombinations; i++) {
				double minCost = Double.MAX_VALUE;
				int minIndex = -1;
				for (int j = 0; j < _fedPlanVariants.size(); j++) {
					if (cumulativeCost[i][j] < minCost) {
						minCost = cumulativeCost[i][j];
						minIndex = j;
					}
				}
				prunedCost[i][0] = minCost;
				prunedIsForwardingBitMap[i][0] = (minIndex != -1) ? forwardingBitMap[i][minIndex] : 0;
				prunedFedPlanVariants.add(_fedPlanVariants.get(minIndex));
			}
			
			this.cumulativeCost = prunedCost;
			this.forwardingBitMap = prunedIsForwardingBitMap;
			this._fedPlanVariants = prunedFedPlanVariants;

			// Step 2: Collect resolved conflict bit positions
			List<Integer> resolvedBits = new ArrayList<>();
			for (int i = 0; i < conflictInfos.size(); i++) {
				ConflictMergeResolveInfo info = conflictInfos.get(i);
				if (info.getResolvedType() == ResolvedType.RESOLVE) {
					resolvedBits.add(i); // Assuming index corresponds to bit position
				}
			}
			
			int resolvedBitsSize = resolvedBits.size();

			// CASE 1: if not resolved, return
			if (resolvedBitsSize == 0){
				return;
			}

			// CASE 2: if all resolved, transform to FedPlanVariants
			if (resolvedBits.size() == conflictInfos.size()){
				double minCost = Double.MAX_VALUE;
				int minCostIdx = -1;

				for (int i = 0; i < this.numConflictCombinations; i++) {
					if (cumulativeCost[i][0] < minCost) {
						minCost = cumulativeCost[i][0];
						minCostIdx = i;
					}
				}
				
				FedPlan finalFedPlan = this.getFedPlanVariants().get(minCostIdx);
				finalFedPlan.setCumulativeCost(minCost);
				this._fedPlanVariants.clear();
				this._fedPlanVariants.add(finalFedPlan);

				this.conflictInfos = null;
				this.cumulativeCost = null;
				this.forwardingBitMap = null;
				this.numConflictCombinations = 0;

				return;
			}

			// CASE 3: if some resolved, some not, merge them
			int mask = 0;
			for (int bit : resolvedBits) {
				mask |= (1 << bit);
			}
			mask = ~mask;
	
			List<Integer> unresolvedBits = new ArrayList<>();
			for (int bit = 0; bit < conflictInfos.size(); bit++) {
				if (!resolvedBits.contains(bit)) {
					unresolvedBits.add(bit);
				}
			}
			Collections.sort(unresolvedBits); // Ensure consistent ordering
	
			// Create newConflictInfos with unresolved conflicts
			List<ConflictMergeResolveInfo> newConflictInfos = new ArrayList<>();
			for (int bit : unresolvedBits) {
				newConflictInfos.add(conflictInfos.get(bit));
			}

			// Step 4: Group combinations by their base (ignoring resolved bits)
			Map<Integer, List<Integer>> groups = new HashMap<>();
			for (int i = 0; i < this.numConflictCombinations; i++) {
				int base = i & mask;
				groups.computeIfAbsent(base, k -> new ArrayList<>()).add(i);
			}
	
			// Step 5: Merge groups and create new arrays with reduced size
			int newSize = 1 << unresolvedBits.size();
			double[][] newPrunedCost = new double[newSize][1];
			int[][] newPrunedBitMap = new int[newSize][1];
			List<FedPlan> newPrunedFedPlanVariants = new ArrayList<>(newSize);
			Arrays.fill(newPrunedCost, Double.MAX_VALUE);
	
			for (Map.Entry<Integer, List<Integer>> entry : groups.entrySet()) {
				int base = entry.getKey();
				List<Integer> group = entry.getValue();
	
				// Find minimal cost and bitmap in the group
				double minGroupCost = Double.MAX_VALUE;
				int minBitmap = 0;
				int minIdx = -1;

				for (int comb : group) {
					if (cumulativeCost[comb][0] < minGroupCost) {
						minGroupCost = cumulativeCost[comb][0];
						minBitmap = forwardingBitMap[comb][0];
						minIdx = comb;
					}
				}
	
				// Compute new index based on unresolved bits
				int newIndex = 0;
				for (int i = 0; i < unresolvedBits.size(); i++) {
					int bitPos = unresolvedBits.get(i);
					if ((base & (1 << bitPos)) != 0) {
						newIndex |= (1 << i); // Set the i-th bit in newIndex
					}
				}
	
				// Update newPruned arrays
				if (newIndex < newSize) {
					newPrunedCost[newIndex][0] = minGroupCost;
					newPrunedBitMap[newIndex][0] = minBitmap;
					newPrunedFedPlanVariants.add(newIndex, _fedPlanVariants.get(minIdx));
				}
			}
	
			// Replace the pruned arrays with the merged results and update size
			this.conflictInfos = newConflictInfos;
			this.cumulativeCost = newPrunedCost;
			this.forwardingBitMap = newPrunedBitMap;
			this.numConflictCombinations = newSize; // Update to the new reduced size
		}
	}

	/**
	 * Represents a collection of federated execution plan variants for a specific Hop and FederatedOutput.
	 * This class contains cost information and references to the associated plans.
	 * It uses HopCommon to store common properties and costs related to the Hop.
	 */
	public static class FedPlanVariants {
		protected HopCommon hopCommon;      // Common properties and costs for the Hop
		private final FederatedOutput fedOutType;  // Output type (FOUT/LOUT)
		protected List<FedPlan> _fedPlanVariants;  // List of plan variants

		public FedPlanVariants(HopCommon hopCommon, FederatedOutput fedOutType) {
			this.hopCommon = hopCommon;
			this.fedOutType = fedOutType;
			this._fedPlanVariants = new ArrayList<>();
		}

		public boolean isEmpty() {return _fedPlanVariants.isEmpty();}
		public void addFedPlan(FedPlan fedPlan) {_fedPlanVariants.add(fedPlan);}
		public List<FedPlan> getFedPlanVariants() {return _fedPlanVariants;}
		public FederatedOutput getFedOutType() {return fedOutType;}
		public double getSelfCost() {return hopCommon.getSelfCost();}
		public double getForwardingCost() {return hopCommon.getForwardingCost();}

		public void pruneFedPlans() {
			if (_fedPlanVariants.size() > 1) {
				// Find the FedPlan with the minimum cost
				FedPlan minCostPlan = _fedPlanVariants.stream()
						.min(Comparator.comparingDouble(FedPlan::getCumulativeCost))
						.orElse(null);

				// Retain only the minimum cost plan
				_fedPlanVariants.clear();
				_fedPlanVariants.add(minCostPlan);
			}
		}
	}

	/**
	 * Represents a single federated execution plan with its associated costs and dependencies.
	 * This class contains:
	 * 1. selfCost: Cost of current hop (compute + input/output memory access)
	 * 2. cumulativeCost: Cumulative cost including this plan and all child plans
	 * 3. netTransferCost: Network transfer cost for this plan to parent plan.
	 * 
	 * FedPlan is linked to FedPlanVariants, which in turn uses HopCommon to manage common properties and costs.
	 */
	public static class FedPlan {
		private double cumulativeCost;                  // Total cost including child plans
		private final FedPlanVariants fedPlanVariants;  // Reference to variant list
		private final List<Pair<Long, FederatedOutput>> childFedPlans;  // Child plan references

		public FedPlan(double cumulativeCost, FedPlanVariants fedPlanVariants, List<Pair<Long, FederatedOutput>> childFedPlans) {
			this.cumulativeCost = cumulativeCost;
			this.fedPlanVariants = fedPlanVariants;
			this.childFedPlans = childFedPlans;			
		}

		public Hop getHopRef() {return fedPlanVariants.hopCommon.getHopRef();}
		public long getHopID() {return fedPlanVariants.hopCommon.getHopRef().getHopID();}
		public FederatedOutput getFedOutType() {return fedPlanVariants.getFedOutType();}
		public double getCumulativeCost() {return cumulativeCost;}
		public double getSelfCost() {return fedPlanVariants.hopCommon.getSelfCost();}
		public double getForwardingCost() {return fedPlanVariants.hopCommon.getForwardingCost();}
		public List<Pair<Long, FederatedOutput>> getChildFedPlans() {return childFedPlans;}

		public void setCumulativeCost(double cumulativeCost) {this.cumulativeCost = cumulativeCost;}
	}

	/**
	 * Represents common properties and costs associated with a Hop.
	 * This class holds a reference to the Hop and tracks its execution and network transfer costs.
	 */
	public static class HopCommon {
		protected final Hop hopRef;
		protected double selfCost;
		protected double forwardingCost;

		public HopCommon(Hop hopRef) {
			this.hopRef = hopRef;
			this.selfCost = 0;
			this.forwardingCost = 0;
		}

		public Hop getHopRef() {return hopRef;}
		public double getSelfCost() {return selfCost;}
		public double getForwardingCost() {return forwardingCost;}

		public void setSelfCost(double selfCost) {this.selfCost = selfCost;}
		public void setForwardingCost(double forwardingCost) {this.forwardingCost = forwardingCost;}
	}
}
