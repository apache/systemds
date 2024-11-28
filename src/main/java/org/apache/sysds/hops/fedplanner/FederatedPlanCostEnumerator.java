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
     */
    public static FedPlan enumerateFederatedPlanCost(Hop rootHop) {              
        // Create new memo table to store all plan variants
        FederatedMemoTable memoTable = new FederatedMemoTable();

        // Recursively enumerate all possible plans
        enumerateFederatedPlanCost(rootHop, memoTable);

        // Return the minimum cost plan for the root node

        return getMinCostRootFedPlan(rootHop.getHopID(), memoTable);
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
     */
    private static FedPlan getMinCostRootFedPlan(long HopID, FederatedMemoTable memoTable) {
        FedPlanVariants fOutFedPlanVariantList = memoTable.getFedPlanVariantList(HopID, FederatedOutput.FOUT);
        FedPlanVariants lOutFedPlanVariantList = memoTable.getFedPlanVariantList(HopID, FederatedOutput.LOUT);

        FedPlan minFOutFedPlan = fOutFedPlanVariantList._fedPlanVariants.stream()
                                    .min(Comparator.comparingDouble(FedPlan::getCumulativeCost))
                                    .orElse(null);
        FedPlan minlOutFedPlan = lOutFedPlanVariantList._fedPlanVariants.stream()
                                    .min(Comparator.comparingDouble(FedPlan::getCumulativeCost))
                                    .orElse(null);

        if (Objects.requireNonNull(minFOutFedPlan).getCumulativeCost()
                < Objects.requireNonNull(minlOutFedPlan).getCumulativeCost()) {
            return minFOutFedPlan;
        }
        return minlOutFedPlan;
    }

}