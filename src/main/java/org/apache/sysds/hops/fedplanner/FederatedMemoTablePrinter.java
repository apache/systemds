package org.apache.sysds.hops.fedplanner;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction;

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
     * @param memoTable The memoization table containing FedPlan variants
     * @param additionalTotalCost The additional cost to be printed once
     */
    public static void printFedPlanTree(FederatedMemoTable.FedPlan rootFedPlan, FederatedMemoTable memoTable,
                                        double additionalTotalCost) {
        System.out.println("Additional Cost: " + additionalTotalCost);
        Set<FederatedMemoTable.FedPlan> visited = new HashSet<>();
        printFedPlanTreeRecursive(rootFedPlan, memoTable, visited, 0);
    }

    /**
     * Helper method to recursively print the FedPlan tree.
     *
     * @param plan  The current FedPlan to print
     * @param visited Set to keep track of visited FedPlans (prevents cycles)
     * @param depth   The current depth level for indentation
     */
    private static void printFedPlanTreeRecursive(FederatedMemoTable.FedPlan plan, FederatedMemoTable memoTable,
                                           Set<FederatedMemoTable.FedPlan> visited, int depth) {
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
                plan.setForwardingCost()));

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
}
