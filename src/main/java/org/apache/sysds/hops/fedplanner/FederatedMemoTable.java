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
    /**
     * Represents a collection of federated execution plan variants for a specific Hop.
     * Contains cost information and references to the associated plans.
     */
    public static class FedPlanVariants {
        protected final Hop hopRef;         // Reference to the associated Hop
        protected double currentCost;       // Current execution cost (compute + memory access)
        protected double netTransferCost;   // Network transfer cost
        protected List<FedPlan> _fedPlanVariants;    // List of plan variants

        public FedPlanVariants(Hop hopRef) {
            this.hopRef = hopRef;
            this.currentCost = 0;
            this.netTransferCost = 0;
            this._fedPlanVariants = new ArrayList<>();
        }

        public void add(FedPlan fedPlan) {
            _fedPlanVariants.add(fedPlan);
        }

        public int size() {return _fedPlanVariants.size();}

        public FedPlan get(int index) {return _fedPlanVariants.get(index);}

        public List<FedPlan> getFedPlanVariants() {return _fedPlanVariants;}
    }

    /**
     * Represents a single federated execution plan with its associated costs and dependencies.
     * Contains:
     * 1. currentCost: Cost of current hop (compute + input/output memory access)
     * 2. cumulativeCost: Total cost including this plan and all child plans
     * 3. netTransferCost: Network transfer cost for this plan
     */
    public static class FedPlan {
        private double cumulativeCost;                  // Total cost including child plans
        private final FederatedOutput fedOutType;       // Output type (FOUT/LOUT)
        private final FedPlanVariants fedPlanVariantList;  // Reference to variant list
        private List<Pair<Long, FederatedOutput>> metaChildFedPlans;  // Child plan references
        private List<FedPlan> selectedFedPlans;           // Selected child plans

        public FedPlan(FederatedOutput fedOutType, List<Pair<Long, FederatedOutput>> planChilds, FedPlanVariants fedPlanVariants) {
            this.fedOutType = fedOutType;
            this.cumulativeCost = 0;
            this.metaChildFedPlans = planChilds;
            this.selectedFedPlans = new ArrayList<>();
            this.fedPlanVariantList = fedPlanVariants;
        }

        public Hop getHopRef() {return fedPlanVariantList.hopRef;}

        public FederatedOutput getFedOutType() {return fedOutType;}

        public double getCurrentCost() {return fedPlanVariantList.currentCost;}

        public double getNetTransferCost() {return fedPlanVariantList.netTransferCost;}

        public double getCumulativeCost() {return cumulativeCost;}

        /**
         * Calculates the cost from parent's perspective based on output type compatibility.
         * Returns cumulative cost if output types match, otherwise adds network transfer cost.
         */
        public double getParentViewCost(FederatedOutput parentFedOutType) {
            if (parentFedOutType == fedOutType){
                return cumulativeCost;
            }
            return cumulativeCost + fedPlanVariantList.netTransferCost;
        }
        
        public List<Pair<Long, FederatedOutput>> getMetaChildFedPlans() {return metaChildFedPlans;}

        public void setCurrentCost(double currentCost) {fedPlanVariantList.currentCost = currentCost;}

        public void setNetTransferCost(double netTransferCost) {fedPlanVariantList.netTransferCost = netTransferCost;}

        public void setCumulativeCost(double cumulativeCost) {this.cumulativeCost = cumulativeCost;}

        public void putChildFedPlan(FedPlan childFedPlan) {selectedFedPlans.add(childFedPlan);}
    }

    // Maps Hop ID and fedOutType pairs to their plan variants
    private final Map<Pair<Long, FederatedOutput>, FedPlanVariants> hopMemoTable = new HashMap<>();

    /**
     * Adds a new federated plan to the memo table.
     * Creates a new variant list if none exists for the given Hop and fedOutType.
     *
     * @param hop         The Hop node
     * @param fedOutType  The federated output type
     * @param planChilds  List of child plan references
     * @return           The newly created FedPlan
     */
    public FedPlan addFedPlan(Hop hop, FederatedOutput fedOutType, List<Pair<Long, FederatedOutput>> planChilds) {
        long hopID = hop.getHopID();
        FedPlanVariants fedPlanVariantList;

        if (contains(hopID, fedOutType)) {
            fedPlanVariantList = hopMemoTable.get(new ImmutablePair<>(hopID, fedOutType));
        } else {
            fedPlanVariantList = new FedPlanVariants(hop);
            hopMemoTable.put(new ImmutablePair<>(hopID, fedOutType), fedPlanVariantList);
        }

        FedPlan newPlan = new FedPlan(fedOutType, planChilds, fedPlanVariantList);
        fedPlanVariantList.add(newPlan);

        return newPlan;
    }

    /**
     * Retrieves the minimum cost child plan considering the parent's output type.
     * The cost is calculated using getParentViewCost to account for potential type mismatches.
     */
    public FedPlan getMinCostChildFedPlan(long childHopID, FederatedOutput childFedOutType, FederatedOutput currentFedOutType) {
        FedPlanVariants fedPlanVariantList = hopMemoTable.get(new ImmutablePair<>(childHopID, childFedOutType));
        return fedPlanVariantList._fedPlanVariants.stream()
                .min(Comparator.comparingDouble(plan -> plan.getParentViewCost(currentFedOutType)))
                .orElse(null);
    }

    public FedPlanVariants getFedPlanVariantList(long hopID, FederatedOutput fedOutType) {
        return hopMemoTable.get(new ImmutablePair<>(hopID, fedOutType));
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
}
