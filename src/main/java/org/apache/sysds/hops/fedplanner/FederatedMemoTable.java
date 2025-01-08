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
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction.FederatedOutput;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashSet;
import java.util.Set;

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
	 * @param childHopID ?
	 * @param childFedOutType ?
	 * @return ?
	 */
	public FedPlan getMinCostFedPlan(long hopID, FederatedOutput fedOutType) {
		FedPlanVariants fedPlanVariantList = hopMemoTable.get(new ImmutablePair<>(hopID, fedOutType));
		return fedPlanVariantList._fedPlanVariants.stream()
				.min(Comparator.comparingDouble(FedPlan::getTotalCost))
				.orElse(null);
	}

	public FedPlanVariants getFedPlanVariants(long hopID, FederatedOutput fedOutType) {
		return hopMemoTable.get(new ImmutablePair<>(hopID, fedOutType));
	}

	public FedPlan getFedPlanAfterPrune(long hopID, FederatedOutput fedOutType) {
		// Todo: Consider whether to verify if pruning has been performed
		FedPlanVariants fedPlanVariantList = hopMemoTable.get(new ImmutablePair<>(hopID, fedOutType));
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
	 * Prunes all entries in the memo table, retaining only the minimum-cost
	 * FedPlan for each entry.
	 */
	public void pruneMemoTable() {
		for (Map.Entry<Pair<Long, FederatedOutput>, FedPlanVariants> entry : hopMemoTable.entrySet()) {
			List<FedPlan> fedPlanList = entry.getValue().getFedPlanVariants();
			if (fedPlanList.size() > 1) {
				// Find the FedPlan with the minimum cost
				FedPlan minCostPlan = fedPlanList.stream()
							.min(Comparator.comparingDouble(FedPlan::getTotalCost))
						.orElse(null);

				// Retain only the minimum cost plan
				fedPlanList.clear();
				fedPlanList.add(minCostPlan);
			}
		}
	}

	// Todo: Separate print functions from FederatedMemoTable
	/**
	 * Recursively prints a tree representation of the DAG starting from the given root FedPlan.
	 * Includes information about hopID, fedOutType, TotalCost, SelfCost, and NetCost for each node.
	 *
	 * @param rootFedPlan The starting point FedPlan to print
	 */
	public void printFedPlanTree(FedPlan rootFedPlan) {
		Set<FedPlan> visited = new HashSet<>();
		printFedPlanTreeRecursive(rootFedPlan, visited, 0, true);
	}

	/**
	 * Helper method to recursively print the FedPlan tree.
	 *
	 * @param plan  The current FedPlan to print
	 * @param visited Set to keep track of visited FedPlans (prevents cycles)
	 * @param depth   The current depth level for indentation
	 * @param isLast  Whether this node is the last child of its parent
	 */
	private void printFedPlanTreeRecursive(FedPlan plan, Set<FedPlan> visited, int depth, boolean isLast) {
		if (plan == null || visited.contains(plan)) {
			return;
		}

		visited.add(plan);

		Hop hop = plan.getHopRef();
		StringBuilder sb = new StringBuilder();

		// Add FedPlan information
		sb.append(String.format("(%d) ", plan.getHopRef().getHopID()))
				.append(plan.getHopRef().getOpString())
				.append(" [")
				.append(plan.getFedOutType())
				.append("]");

		StringBuilder childs = new StringBuilder();
		childs.append(" (");
		boolean childAdded = false;
		for( Hop input : hop.getInput()){
			childs.append(childAdded?",":"");
			childs.append(input.getHopID());
			childAdded = true;
		}
		childs.append(")");
		if( childAdded )
			sb.append(childs.toString());
		 
		 
		sb.append(String.format(" {Total: %.1f, Self: %.1f, Net: %.1f}",
				plan.getTotalCost(),
				plan.getSelfCost(),
				plan.getNetTransferCost()));

		// Add matrix characteristics
		sb.append(" [")
			.append(hop.getDim1()).append(", ")
			.append(hop.getDim2()).append(", ")
			.append(hop.getBlocksize()).append(", ")
			.append(hop.getNnz());

		if (hop.getUpdateType().isInPlace()) {
			sb.append(", ").append(hop.getUpdateType().toString().toLowerCase());
		}
		sb.append("]");

		// Add memory estimates
		sb.append(" [")
			.append(OptimizerUtils.toMB(hop.getInputMemEstimate())).append(", ")
			.append(OptimizerUtils.toMB(hop.getIntermediateMemEstimate())).append(", ")
			.append(OptimizerUtils.toMB(hop.getOutputMemEstimate())).append(" -> ")
			.append(OptimizerUtils.toMB(hop.getMemEstimate())).append("MB]");

		// Add reblock and checkpoint requirements
		if (hop.requiresReblock() && hop.requiresCheckpoint()) {
			sb.append(" [rblk, chkpt]");
		} else if (hop.requiresReblock()) {
			sb.append(" [rblk]");
		} else if (hop.requiresCheckpoint()) {
			sb.append(" [chkpt]");
		}

		// Add execution type
		if (hop.getExecType() != null) {
			sb.append(", ").append(hop.getExecType());
		}

		System.out.println(sb);

		// Process child nodes
		List<Pair<Long, FederatedOutput>> childRefs = plan.getChildFedPlans();
		for (int i = 0; i < childRefs.size(); i++) {
			Pair<Long, FederatedOutput> childRef = childRefs.get(i);
			FedPlanVariants childVariants = getFedPlanVariants(childRef.getLeft(), childRef.getRight());
			if (childVariants == null || childVariants.getFedPlanVariants().isEmpty())
				continue;

			boolean isLastChild = (i == childRefs.size() - 1);
			for (FedPlan childPlan : childVariants.getFedPlanVariants()) {
				printFedPlanTreeRecursive(childPlan, visited, depth + 1, isLastChild);
			}
		}
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
