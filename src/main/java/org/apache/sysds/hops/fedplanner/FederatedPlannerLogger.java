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
import org.apache.sysds.hops.fedplanner.FTypes.Privacy;
import org.apache.sysds.hops.fedplanner.FTypes.FType;
import org.apache.sysds.hops.fedplanner.FederatedMemoTable.FedPlan;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction.FederatedOutput;
import org.apache.commons.lang3.tuple.Pair;
import java.util.HashSet;
import java.util.Map;
import java.util.List;
import java.util.Set;

/**
 * Unified utility class for logging federated planner information.
 * Provides methods to log hop details including privacy constraints and FType information,
 * as well as methods to print detailed FederatedMemoTable tree structures and cost analysis.
 * This class integrates the functionality of the former FederatedMemoTablePrinter.
 */
public class FederatedPlannerLogger {
    
    /**
     * Logs hop information including name, hop ID, child hop IDs, privacy constraint, and ftype
     * @param hop The hop to log information for
     * @param privacyConstraintMap Map containing privacy constraints for hops
     * @param fTypeMap Map containing FType information for hops
     * @param logPrefix Prefix string to identify the log source
     */
    public static void logHopInfo(Hop hop, Map<Long, Privacy> privacyConstraintMap, 
                                  Map<Long, FType> fTypeMap, String logPrefix) {
        StringBuilder childIds = new StringBuilder();
        if (hop.getInput() != null && !hop.getInput().isEmpty()) {
            for (int i = 0; i < hop.getInput().size(); i++) {
                if (i > 0) childIds.append(",");
                childIds.append(hop.getInput().get(i).getHopID());
            }
        } else {
            childIds.append("none");
        }
        
        Privacy privacyConstraint = privacyConstraintMap.get(hop.getHopID());
        FType ftype = fTypeMap.get(hop.getHopID());
        
        // Get hop type and opcode information
        String hopType = hop.getClass().getSimpleName();
        String opCode = hop.getOpString();
        
        System.out.println("[" + logPrefix + "] (ID:" + hop.getHopID() + " Name:" + hop.getName() + 
                          ") Type:" + hopType + " OpCode:" + opCode + 
                          " ChildIDs:(" + childIds.toString() + ") Privacy:" + 
                          (privacyConstraint != null ? privacyConstraint : "null") + 
                          " FType:" + (ftype != null ? ftype : "null"));
    }
    
    /**
     * Logs basic hop information without privacy and FType details
     * @param hop The hop to log information for
     * @param logPrefix Prefix string to identify the log source
     */
    public static void logBasicHopInfo(Hop hop, String logPrefix) {
        StringBuilder childIds = new StringBuilder();
        if (hop.getInput() != null && !hop.getInput().isEmpty()) {
            for (int i = 0; i < hop.getInput().size(); i++) {
                if (i > 0) childIds.append(",");
                childIds.append(hop.getInput().get(i).getHopID());
            }
        } else {
            childIds.append("none");
        }
        
        String hopType = hop.getClass().getSimpleName();
        String opCode = hop.getOpString();
        
        System.out.println("[" + logPrefix + "] (ID:" + hop.getHopID() + " Name:" + hop.getName() + 
                          ") Type:" + hopType + " OpCode:" + opCode + 
                          " ChildIDs:(" + childIds.toString() + ")");
    }
    
    /**
     * Logs detailed hop information with dimension and data type
     * @param hop The hop to log information for
     * @param privacyConstraintMap Map containing privacy constraints for hops
     * @param fTypeMap Map containing FType information for hops
     * @param logPrefix Prefix string to identify the log source
     */
    public static void logDetailedHopInfo(Hop hop, Map<Long, Privacy> privacyConstraintMap, 
                                         Map<Long, FType> fTypeMap, String logPrefix) {
        StringBuilder childIds = new StringBuilder();
        if (hop.getInput() != null && !hop.getInput().isEmpty()) {
            for (int i = 0; i < hop.getInput().size(); i++) {
                if (i > 0) childIds.append(",");
                childIds.append(hop.getInput().get(i).getHopID());
            }
        } else {
            childIds.append("none");
        }
        
        Privacy privacyConstraint = privacyConstraintMap.get(hop.getHopID());
        FType ftype = fTypeMap.get(hop.getHopID());
        
        String hopType = hop.getClass().getSimpleName();
        String opCode = hop.getOpString();
        String dataType = hop.getDataType().toString();
        String dimensions = "[" + hop.getDim1() + "x" + hop.getDim2() + "]";
        
        System.out.println("[" + logPrefix + "] (ID:" + hop.getHopID() + " Name:" + hop.getName() + 
                          ") Type:" + hopType + " OpCode:" + opCode + " DataType:" + dataType + 
                          " Dims:" + dimensions + " ChildIDs:(" + childIds.toString() + ") Privacy:" + 
                          (privacyConstraint != null ? privacyConstraint : "null") + 
                          " FType:" + (ftype != null ? ftype : "null"));
    }
    
    /**
     * Logs error information for null fed plan scenarios
     * @param hopID The hop ID that caused the error
     * @param logPrefix Prefix string to identify the log source
     */
    public static void logNullFedPlanError(long hopID, String logPrefix) {
        System.err.println("[" + logPrefix + "] childFedPlan is null for hopID: " + hopID);
    }
    
    /**
     * Logs detailed error information for conflict resolution scenarios
     * @param hopID The hop ID that caused the error
     * @param fedPlan The federated plan with error details
     * @param logPrefix Prefix string to identify the log source
     */
    public static void logConflictResolutionError(long hopID, Object fedPlan, String logPrefix) {
        System.err.println("[" + logPrefix + "] confilctLOutFedPlan or confilctFOutFedPlan is null for hopID: " + hopID);
        System.err.println("  Child Hop Details:");
        if (fedPlan != null) {
            // Note: This assumes fedPlan has a getHopRef() method
            // In actual implementation, you might need to cast or handle differently
            System.err.println("    - Class: N/A");
            System.err.println("    - Name: N/A");
            System.err.println("    - OpString: N/A");
            System.err.println("    - HopID: " + hopID);
        }
    }
    
    /**
     * Logs detailed hop error information with complete hop details
     * @param hop The hop that caused the error
     * @param logPrefix Prefix string to identify the log source
     * @param additionalMessage Additional error message
     */
    public static void logHopErrorDetails(Hop hop, String logPrefix, String additionalMessage) {
        System.err.println("[" + logPrefix + "] " + additionalMessage);
        System.err.println("  Child Hop Details:");
        System.err.println("    - Class: " + hop.getClass().getSimpleName());
        System.err.println("    - Name: " + (hop.getName() != null ? hop.getName() : "null"));
        System.err.println("    - OpString: " + hop.getOpString());
        System.err.println("    - HopID: " + hop.getHopID());
    }
    
    /**
     * Logs detailed null child plan debugging information
     * @param childFedPlanPair The child federated plan pair that is null
     * @param optimalPlan The current optimal plan (parent)
     * @param memoTable The memo table for lookups
     */
    public static void logNullChildPlanDebug(Pair<Long, FederatedOutput> childFedPlanPair, 
                                           FedPlan optimalPlan, 
                                           org.apache.sysds.hops.fedplanner.FederatedMemoTable memoTable) {
        FederatedOutput alternativeFedType = (childFedPlanPair.getRight() == FederatedOutput.LOUT) ? 
                                           FederatedOutput.FOUT : FederatedOutput.LOUT;
        FedPlan alternativeChildPlan = memoTable.getFedPlanAfterPrune(childFedPlanPair.getLeft(), alternativeFedType);
        
        // Get child hop info
        Hop childHop = null;
        String childInfo = "UNKNOWN";
        if (alternativeChildPlan != null) {
            childHop = alternativeChildPlan.getHopRef();
            // Check if required fed type plan exists
            String requiredExists = memoTable.getFedPlanAfterPrune(childFedPlanPair.getLeft(), childFedPlanPair.getRight()) != null ? "O" : "X";
            // Check if alternative fed type plan exists  
            String altExists = alternativeChildPlan != null ? "O" : "X";
            
            childInfo = String.format("ID:%d|Name:%s|Op:%s|RequiredFedType:%s(%s)|AltFedType:%s(%s)", 
                childHop.getHopID(),
                childHop.getName() != null ? childHop.getName() : "null",
                childHop.getOpString(),
                childFedPlanPair.getRight(),
                requiredExists,
                alternativeFedType,
                altExists);
        }
        
        // Current parent hop info
        String currentParentInfo = String.format("ID:%d|Name:%s|Op:%s|FedType:%s|RequiredChild:%s", 
            optimalPlan.getHopID(),
            optimalPlan.getHopRef().getName() != null ? optimalPlan.getHopRef().getName() : "null",
            optimalPlan.getHopRef().getOpString(),
            optimalPlan.getFedOutType(),
            childFedPlanPair.getRight());
        
        // Alternative parent info (if child has other parents)
        String alternativeParentInfo = "NONE";
        if (childHop != null) {
            List<Hop> parents = childHop.getParent();
            for (Hop parent : parents) {
                if (parent.getHopID() != optimalPlan.getHopID()) {
                    // Try to find alt parent's fed plan info
                    String altParentFedType = "UNKNOWN";
                    String altParentRequiredChild = "UNKNOWN";
                    
                    // Check both LOUT and FOUT plans for alt parent
                    FedPlan altParentPlanLOUT = memoTable.getFedPlanAfterPrune(parent.getHopID(), FederatedOutput.LOUT);
                    FedPlan altParentPlanFOUT = memoTable.getFedPlanAfterPrune(parent.getHopID(), FederatedOutput.FOUT);
                    
                    if (altParentPlanLOUT != null) {
                        altParentFedType = "LOUT";
                        // Find what this alt parent expects from child
                        for (Pair<Long, FederatedOutput> altChildPair : altParentPlanLOUT.getChildFedPlans()) {
                            if (altChildPair.getLeft() == childHop.getHopID()) {
                                altParentRequiredChild = altChildPair.getRight().toString();
                                break;
                            }
                        }
                    } else if (altParentPlanFOUT != null) {
                        altParentFedType = "FOUT";
                        // Find what this alt parent expects from child
                        for (Pair<Long, FederatedOutput> altChildPair : altParentPlanFOUT.getChildFedPlans()) {
                            if (altChildPair.getLeft() == childHop.getHopID()) {
                                altParentRequiredChild = altChildPair.getRight().toString();
                                break;
                            }
                        }
                    }
                    
                    alternativeParentInfo = String.format("ID:%d|Name:%s|Op:%s|FedType:%s|RequiredChild:%s", 
                        parent.getHopID(),
                        parent.getName() != null ? parent.getName() : "null",
                        parent.getOpString(),
                        altParentFedType,
                        altParentRequiredChild);
                    break;
                }
            }
        }
        
        System.err.println("[DEBUG] NULL CHILD PLAN DETECTED:");
        System.err.println("  Child:           " + childInfo);
        System.err.println("  Current Parent:  " + currentParentInfo);
        System.err.println("  Alt Parent:      " + alternativeParentInfo);
        System.err.println("  Alt Plan Exists: " + (alternativeChildPlan != null));
    }
    
    /**
     * Logs debugging information for TransRead hop rewiring process
     * @param hopName The name of the TransRead hop
     * @param hopID The ID of the TransRead hop  
     * @param childHops List of child hops found during rewiring
     * @param isEmptyChildHops Whether the child hops list is empty
     * @param logPrefix Prefix string to identify the log source
     */
    public static void logTransReadRewireDebug(String hopName, long hopID, List<Hop> childHops, 
                                             boolean isEmptyChildHops, String logPrefix) {
        if (isEmptyChildHops) {
            System.err.println("[" + logPrefix + "] (hopName: " + hopName + ", hopID: " + hopID + ") child hops is empty");
        }
    }
    
    /**
     * Logs debugging information for filtered child hops during TransRead rewiring
     * @param hopName The name of the TransRead hop
     * @param hopID The ID of the TransRead hop
     * @param filteredChildHops List of filtered child hops
     * @param isEmptyFilteredChildHops Whether the filtered child hops list is empty
     * @param logPrefix Prefix string to identify the log source
     */
    public static void logFilteredChildHopsDebug(String hopName, long hopID, List<Hop> filteredChildHops, 
                                               boolean isEmptyFilteredChildHops, String logPrefix) {
        if (isEmptyFilteredChildHops) {
            System.err.println("[" + logPrefix + "] (hopName: " + hopName + ", hopID: " + hopID + ") filtered child hops is empty");
        }
    }
    
    /**
     * Logs detailed FType mismatch error information for TransRead hop
     * @param hop The TransRead hop with FType mismatch
     * @param filteredChildHops List of filtered child hops
     * @param fTypeMap Map containing FType information for hops
     * @param expectedFType The expected FType
     * @param mismatchedFType The mismatched FType
     * @param mismatchIndex The index where mismatch occurred
     */
    public static void logFTypeMismatchError(Hop hop, List<Hop> filteredChildHops, Map<Long, FType> fTypeMap,
                                           FType expectedFType, FType mismatchedFType, int mismatchIndex) {
        String hopName = hop.getName();
        long hopID = hop.getHopID();
        
        System.err.println("[Error] FType MISMATCH DETECTED for TransRead (hopName: " + hopName + ", hopID: " + hopID + ")");
        System.err.println("[Error] TRANSREAD HOP DETAILS - Type: " + hop.getClass().getSimpleName() + 
            ", OpType: " + (hop instanceof org.apache.sysds.hops.DataOp ? 
                ((org.apache.sysds.hops.DataOp)hop).getOp() : "N/A") + 
            ", DataType: " + hop.getDataType() + 
            ", Dims: [" + hop.getDim1() + "x" + hop.getDim2() + "]");
        System.err.println("[Error] FILTERED CHILD HOPS FTYPE ANALYSIS:");
        
        for (int j = 0; j < filteredChildHops.size(); j++) {
            Hop childHop = filteredChildHops.get(j);
            FType childFType = fTypeMap.get(childHop.getHopID());
            System.err.println("[Error]   FilteredChild[" + j + "] - Name: " + childHop.getName() + 
                ", ID: " + childHop.getHopID() + 
                ", FType: " + childFType + 
                ", Type: " + childHop.getClass().getSimpleName() + 
                ", OpType: " + (childHop instanceof org.apache.sysds.hops.DataOp ? 
                    ((org.apache.sysds.hops.DataOp)childHop).getOp().toString() : "N/A") +
                ", Dims: [" + childHop.getDim1() + "x" + childHop.getDim2() + "]");
        }
        
        System.err.println("[Error] Expected FType: " + expectedFType + 
                          ", Mismatched FType: " + mismatchedFType + 
                          " at child index: " + mismatchIndex);
    }
    
    // ========== FederatedMemoTable Printing Methods ==========
    
    /**
     * Recursively prints a tree representation of the DAG starting from the given root FedPlan.
     * Includes information about hopID, fedOutType, TotalCost, SelfCost, and NetCost for each node.
     * Additionally, prints the additional total cost once at the beginning.
     *
     * @param rootFedPlan The starting point FedPlan to print
     * @param rootHopStatSet Set of root hop statistics
     * @param memoTable The memoization table containing FedPlan variants
     * @param additionalTotalCost The additional cost to be printed once
     */
    public static void printFedPlanTree(FederatedMemoTable.FedPlan rootFedPlan, Set<Long> rootHopStatSet,
                                        FederatedMemoTable memoTable, double additionalTotalCost) {
        System.out.println("Additional Cost: " + additionalTotalCost);
        Set<Long> visited = new HashSet<>();
        printFedPlanTreeRecursive(rootFedPlan, memoTable, visited, 0);

        for (Long hopID : rootHopStatSet) {
            FedPlan plan = memoTable.getFedPlanAfterPrune(hopID, FederatedOutput.LOUT);
            if (plan == null){
                plan = memoTable.getFedPlanAfterPrune(hopID, FederatedOutput.FOUT);
            }
            printNotReferencedFedPlanRecursive(plan, memoTable, visited, 1);
        }
    }

    /**
     * Helper method to recursively print the FedPlan tree for not referenced plans.
     *
     * @param plan  The current FedPlan to print
     * @param memoTable The memoization table containing FedPlan variants
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
        printFedPlan(plan, memoTable, depth, true);

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
     * @param memoTable The memoization table containing FedPlan variants
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
        printFedPlan(plan, memoTable, depth, false);
        
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

    /**
     * Prints detailed information about a FedPlan including costs, dimensions, and memory estimates.
     *
     * @param plan The FedPlan to print
     * @param memoTable The memoization table containing FedPlan variants
     * @param depth The current depth level for indentation
     * @param isNotReferenced Whether this plan is not referenced
     */
    private static void printFedPlan(FederatedMemoTable.FedPlan plan, FederatedMemoTable memoTable, int depth, boolean isNotReferenced) {
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
                if (depth == 1) {
                    sb.append("NRef(TOP)");
                } else {
                    sb.append("NRef");
                }
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

        if (childAdded)
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
                plan.getComputeWeight()));

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
        
        if (childAdded){
            sb.append(" [Edges]{");
            for (Pair<Long, FederatedOutput> childPair : plan.getChildFedPlans()){
                // Add forwarding weight for each edge
                FedPlan childPlan = memoTable.getFedPlanAfterPrune(childPair.getLeft(), childPair.getRight());
                
                if (childPlan == null) {
                    sb.append(String.format("(ID:%d, NULL)", childPair.getLeft()));
                } else {
                    String isForwardingCostOccured = "";
                    if (childPair.getRight() == plan.getFedOutType()){
                        isForwardingCostOccured = "X";
                    } else {
                        isForwardingCostOccured = "O";
                    }
                    sb.append(String.format("(ID:%d, %s, C:%.1f, F:%.1f, FW:%.1f)", childPair.getLeft(), isForwardingCostOccured, 
                                childPlan.getCumulativeCostPerParents(), 
                                plan.getChildForwardingWeight(childPlan.getLoopContext()) * childPlan.getForwardingCostPerParents(), 
                                plan.getChildForwardingWeight(childPlan.getLoopContext())));
                }
                sb.append(childAdded?",":"");
            }
            sb.append("}");
        }

        System.out.println(sb);
    }
}