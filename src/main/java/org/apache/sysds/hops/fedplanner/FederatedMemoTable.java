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

import org.apache.sysds.hops.Hop;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction.FederatedOutput;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;

/**
 * A Memoization Table for managing federated plans (FedPlan) based on combinations of Hops and fedOutTypes.
 * This table stores and manages different execution plan variants for each Hop and fedOutType combination,
 * facilitating the optimization of federated execution plans.
 */
public class FederatedMemoTable {
	// Maps Hop ID and fedOutType pairs to their plan variants
	private final Map<Pair<Long, FederatedOutput>, FedPlanVariants> hopMemoTable = new HashMap<>();

	/**
	 * Adds a new federated plan to the memo table.
	 * Creates a new variant list if none exists for the given Hop and fedOutType.
	 *
	 * @param hop		 The Hop node
	 * @param fedOutType  The federated output type
	 * @param planChilds  List of child plan references
	 * @return		   The newly created FedPlan
	 */
	public FedPlan addFedPlan(Hop hop, FederatedOutput fedOutType, List<Pair<Long, FederatedOutput>> planChilds) {
		long hopID = hop.getHopID();
		FedPlanVariants fedPlanVariantList;

		if (contains(hopID, fedOutType)) {
			fedPlanVariantList = hopMemoTable.get(new ImmutablePair<>(hopID, fedOutType));
		} else {
			fedPlanVariantList = new FedPlanVariants(hop, fedOutType);
			hopMemoTable.put(new ImmutablePair<>(hopID, fedOutType), fedPlanVariantList);
		}

		FedPlan newPlan = new FedPlan(planChilds, fedPlanVariantList);
		fedPlanVariantList.addFedPlan(newPlan);

		return newPlan;
	}

	/**
	 * Retrieves the minimum cost child plan considering the parent's output type.
	 * The cost is calculated using getParentViewCost to account for potential type mismatches.
	 * 
	 * @param fedPlanPair ???
	 * @return min cost fed plan
	 */
	public FedPlan getMinCostFedPlan(Pair<Long, FederatedOutput> fedPlanPair) {
		FedPlanVariants fedPlanVariantList = hopMemoTable.get(fedPlanPair);
		return fedPlanVariantList._fedPlanVariants.stream()
				.min(Comparator.comparingDouble(FedPlan::getTotalCost))
				.orElse(null);
	}

	public FedPlanVariants getFedPlanVariants(long hopID, FederatedOutput fedOutType) {
		return hopMemoTable.get(new ImmutablePair<>(hopID, fedOutType));
	}

	public FedPlanVariants getFedPlanVariants(Pair<Long, FederatedOutput> fedPlanPair) {
		return hopMemoTable.get(fedPlanPair);
	}

	public FedPlan getFedPlanAfterPrune(long hopID, FederatedOutput fedOutType) {
		// Todo: Consider whether to verify if pruning has been performed
		FedPlanVariants fedPlanVariantList = hopMemoTable.get(new ImmutablePair<>(hopID, fedOutType));
		return fedPlanVariantList._fedPlanVariants.get(0);
	}

	public FedPlan getFedPlanAfterPrune(Pair<Long, FederatedOutput> fedPlanPair) {
		// Todo: Consider whether to verify if pruning has been performed
		FedPlanVariants fedPlanVariantList = hopMemoTable.get(fedPlanPair);
		return fedPlanVariantList._fedPlanVariants.get(0);
	}

	/**
	 * Checks if the memo table contains an entry for a given Hop and fedOutType.
	 *
	 * @param hopID   The Hop ID.
	 * @param fedOutType The associated fedOutType.
	 * @return True if the entry exists, false otherwise.
	 */
	public boolean contains(long hopID, FederatedOutput fedOutType) {
		return hopMemoTable.containsKey(new ImmutablePair<>(hopID, fedOutType));
	}

	/**
	 * Prunes the specified entry in the memo table, retaining only the minimum-cost
	 * FedPlan for the given Hop ID and federated output type.
	 *
	 * @param hopID The ID of the Hop to prune
	 * @param federatedOutput The federated output type associated with the Hop
	 */
	public void pruneFedPlan(long hopID, FederatedOutput federatedOutput) {
		hopMemoTable.get(new ImmutablePair<>(hopID, federatedOutput)).prune();
	}

	/**
	 * Represents common properties and costs associated with a Hop.
	 * This class holds a reference to the Hop and tracks its execution and network transfer costs.
	 */
	public static class HopCommon {
		protected final Hop hopRef;         // Reference to the associated Hop
		protected double selfCost;          // Current execution cost (compute + memory access)
		protected double netTransferCost;   // Network transfer cost

		protected HopCommon(Hop hopRef) {
			this.hopRef = hopRef;
			this.selfCost = 0;
			this.netTransferCost = 0;
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

		public FedPlanVariants(Hop hopRef, FederatedOutput fedOutType) {
			this.hopCommon = new HopCommon(hopRef);
			this.fedOutType = fedOutType;
			this._fedPlanVariants = new ArrayList<>();
		}

		public void addFedPlan(FedPlan fedPlan) {_fedPlanVariants.add(fedPlan);}
		public List<FedPlan> getFedPlanVariants() {return _fedPlanVariants;}
		public boolean isEmpty() {return _fedPlanVariants.isEmpty();}

		public void prune() {
			if (_fedPlanVariants.size() > 1) {
				// Find the FedPlan with the minimum cost
				FedPlan minCostPlan = _fedPlanVariants.stream()
						.min(Comparator.comparingDouble(FedPlan::getTotalCost))
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
	 * 2. totalCost: Cumulative cost including this plan and all child plans
	 * 3. netTransferCost: Network transfer cost for this plan to parent plan.
	 * 
	 * FedPlan is linked to FedPlanVariants, which in turn uses HopCommon to manage common properties and costs.
	 */
	public static class FedPlan {
		private double totalCost;                  // Total cost including child plans
		private final FedPlanVariants fedPlanVariants;  // Reference to variant list
		private final List<Pair<Long, FederatedOutput>> childFedPlans;  // Child plan references

		public FedPlan(List<Pair<Long, FederatedOutput>> childFedPlans, FedPlanVariants fedPlanVariants) {
			this.totalCost = 0;
			this.childFedPlans = childFedPlans;
			this.fedPlanVariants = fedPlanVariants;
		}

		public void setTotalCost(double totalCost) {this.totalCost = totalCost;}
		public void setSelfCost(double selfCost) {fedPlanVariants.hopCommon.selfCost = selfCost;}
		public void setNetTransferCost(double netTransferCost) {fedPlanVariants.hopCommon.netTransferCost = netTransferCost;}
		
		public Hop getHopRef() {return fedPlanVariants.hopCommon.hopRef;}
		public long getHopID() {return fedPlanVariants.hopCommon.hopRef.getHopID();}
		public FederatedOutput getFedOutType() {return fedPlanVariants.fedOutType;}
		public double getTotalCost() {return totalCost;}
		public double getSelfCost() {return fedPlanVariants.hopCommon.selfCost;}
		public double getNetTransferCost() {return fedPlanVariants.hopCommon.netTransferCost;}
		public List<Pair<Long, FederatedOutput>> getChildFedPlans() {return childFedPlans;}

		/**
		 * Calculates the conditional network transfer cost based on output type compatibility.
		 * Returns 0 if output types match, otherwise returns the network transfer cost.
		 * @param parentFedOutType The federated output type of the parent plan.
		 * @return The conditional network transfer cost.
		 */
		public double getCondNetTransferCost(FederatedOutput parentFedOutType) {
			if (parentFedOutType == getFedOutType()) return 0;
			return fedPlanVariants.hopCommon.netTransferCost;
		}
	}
}
