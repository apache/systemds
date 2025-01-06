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
		double totalCost = 0;
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
		for (Pair<Long, FederatedOutput> planRefMeta : currentPlan.getChildFedPlans()) {
			// Find minimum cost child plan considering federation type compatibility
			// Note: This approach might lead to suboptimal or wrong solutions when a child has multiple parents
			// because we're selecting child plans independently for each parent
			FedPlan planRef = memoTable.getMinCostChildFedPlan(planRefMeta.getLeft(), planRefMeta.getRight());

			// Add child plan cost (includes network transfer cost if federation types differ)
			totalCost += planRef.getTotalCost() + planRef.getCondNetTransferCost(currentPlan.getFedOutType());
		}
		
		// Step 3: Set final cumulative cost including current node
		currentPlan.setTotalCost(totalCost);
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
