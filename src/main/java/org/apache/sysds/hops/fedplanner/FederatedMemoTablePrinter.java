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
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.fedplanner.FederatedMemoTable.FedPlan;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction.FederatedOutput;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class FederatedMemoTablePrinter {
	/**
	 * Recursively prints a tree representation of the DAG starting from the given root FedPlan.
	 * Includes information about hopID, fedOutType, TotalCost, SelfCost, and NetCost for each node.
	 * Additionally, prints the additional total cost once at the beginning.
	 *
	 * @param rootFedPlan The starting point FedPlan to print
	 * @param rootHopStatSet ???
	 * @param memoTable The memoization table containing FedPlan variants
	 * @param additionalTotalCost The additional cost to be printed once
	 */
	public static void printFedPlanTree(FederatedMemoTable.FedPlan rootFedPlan, Set<Hop> rootHopStatSet,
										FederatedMemoTable memoTable, double additionalTotalCost) {
		System.out.println("Additional Cost: " + additionalTotalCost);
		Set<Long> visited = new HashSet<>();
		printFedPlanTreeRecursive(rootFedPlan, memoTable, visited, 0);
		
		for (Hop hop : rootHopStatSet) {
			FedPlan plan = memoTable.getFedPlanAfterPrune(hop.getHopID(), FederatedOutput.LOUT);
			printNotReferencedFedPlanRecursive(plan, memoTable, visited, 1);
		}
	}

		/**
	 * Helper method to recursively print the FedPlan tree.
	 *
	 * @param plan  The current FedPlan to print
	 * @param visited Set to keep track of visited FedPlans (prevents cycles)
	 * @param depth   The current depth level for indentation
	 */
	private static void printNotReferencedFedPlanRecursive(FederatedMemoTable.FedPlan plan, FederatedMemoTable memoTable,
										   Set<Long> visited, int depth) {
		long hopID = plan.getHopRef().getHopID();

		if (visited.contains(hopID)) {
			return;
		}

		visited.add(hopID);
		printFedPlan(plan, depth, true);

		// Process child nodes
		List<Pair<Long, FEDInstruction.FederatedOutput>> childFedPlanPairs = plan.getChildFedPlans();
		for (int i = 0; i < childFedPlanPairs.size(); i++) {
			Pair<Long, FEDInstruction.FederatedOutput> childFedPlanPair = childFedPlanPairs.get(i);
			FederatedMemoTable.FedPlanVariants childVariants = memoTable.getFedPlanVariants(childFedPlanPair);
			if (childVariants == null || childVariants.isEmpty())
				continue;

			for (FederatedMemoTable.FedPlan childPlan : childVariants.getFedPlanVariants()) {
				printNotReferencedFedPlanRecursive(childPlan, memoTable, visited, depth + 1);
			}
		}
	}

	/**
	 * Helper method to recursively print the FedPlan tree.
	 *
	 * @param plan  The current FedPlan to print
	 * @param visited Set to keep track of visited FedPlans (prevents cycles)
	 * @param depth   The current depth level for indentation
	 */
	private static void printFedPlanTreeRecursive(FederatedMemoTable.FedPlan plan, FederatedMemoTable memoTable,
										   Set<Long> visited, int depth) {
		long hopID = 0;

		if (depth == 0) {
			hopID = -1;
		} else {
			hopID = plan.getHopRef().getHopID();
		}

		if (visited.contains(hopID)) {
			return;
		}

		visited.add(hopID);
		printFedPlan(plan, depth, false);

		// Process child nodes
		List<Pair<Long, FEDInstruction.FederatedOutput>> childFedPlanPairs = plan.getChildFedPlans();
		for (int i = 0; i < childFedPlanPairs.size(); i++) {
			Pair<Long, FEDInstruction.FederatedOutput> childFedPlanPair = childFedPlanPairs.get(i);
			FederatedMemoTable.FedPlanVariants childVariants = memoTable.getFedPlanVariants(childFedPlanPair);
			if (childVariants == null || childVariants.isEmpty())
				continue;

			for (FederatedMemoTable.FedPlan childPlan : childVariants.getFedPlanVariants()) {
				printFedPlanTreeRecursive(childPlan, memoTable, visited, depth + 1);
			}
		}
	}

	private static void printFedPlan(FederatedMemoTable.FedPlan plan, int depth, boolean isNotReferenced) {
		StringBuilder sb = new StringBuilder();
		Hop hop = null;

		if (depth == 0){
			sb.append("(R) ROOT [Root]");
		} else {
			hop = plan.getHopRef();
			// Add FedPlan information
			sb.append(String.format("(%d) ", hop.getHopID()))
					.append(hop.getOpString())
					.append(" [");

			if (isNotReferenced) {
				sb.append("NRef");
			} else{
				sb.append(plan.getFedOutType());
			}
			sb.append("]");
		}

		StringBuilder childs = new StringBuilder();
		childs.append(" (");

		boolean childAdded = false;
		for (Pair<Long, FederatedOutput> childPair : plan.getChildFedPlans()){
			childs.append(childAdded?",":"");
			childs.append(childPair.getLeft());
			childAdded = true;
		}
		
		childs.append(")");

		if( childAdded )
			sb.append(childs.toString());

		if (depth == 0){
			sb.append(String.format(" {Total: %.1f}", plan.getCumulativeCost()));
			System.out.println(sb);
			return;
		}

		sb.append(String.format(" {Total: %.1f, Self: %.1f, Net: %.1f, Weight: %.1f}",
				plan.getCumulativeCost(),
				plan.getSelfCost(),
				plan.getForwardingCost(),
				plan.getWeight()));

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
	}
}

