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
import java.util.Comparator;
import java.util.Objects;

import org.apache.commons.lang3.tuple.Pair;
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
	 * Entry point for federated plan enumeration. Creates a memo table and returns
	 * the minimum cost plan for the entire DAG.
	 * 
	 * @param rootHop ?
	 * @param printTree ?
	 * @return ?
	 */
	public static FedPlan enumerateFederatedPlanCost(Hop rootHop, boolean printTree) {
		// Create new memo table to store all plan variants
		FederatedMemoTable memoTable = new FederatedMemoTable();

		// Recursively enumerate all possible plans
		enumerateFederatedPlanCost(rootHop, memoTable);

		// Return the minimum cost plan for the root node
		FedPlan optimalPlan = getMinCostRootFedPlan(rootHop.getHopID(), memoTable);
		memoTable.pruneMemoTable();
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
}
