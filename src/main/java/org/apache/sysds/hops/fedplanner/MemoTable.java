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

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;

/**
 * A Memoization Table for managing federated plans (`FedPlan`) based on
 * combinations of Hops and FTypes. Each combination is mapped to a list
 * of possible execution plans, allowing for pruning and optimization.
 */
public class MemoTable {
    /**
     * Represents a federated execution plan with its cost and associated references.
     */
    public static class FedPlan {
        private final Hop hopRef;                                 // The associated Hop object
        private final double cost;                                // Cost of this federated plan
        private final List<Pair<Long, FTypes.FType>> planRefs;    // References to dependent plans

        public FedPlan(Hop hopRef, double cost, List<Pair<Long, FTypes.FType>> planRefs) {
            this.hopRef = hopRef;
            this.cost = cost;
            this.planRefs = planRefs;
        }

        public double getCost() {return cost;}
    }

    // Maps combinations of Hop ID and FType to lists of FedPlans
    private final Map<Pair<Long, FTypes.FType>, List<FedPlan>> hopMemoTable = new HashMap<>();

    /**
     * Adds a single FedPlan to the memo table for a given Hop and FType.
     * If the entry already exists, the new FedPlan is appended to the list.
     *
     * @param hop     The Hop object.
     * @param fType   The associated FType.
     * @param fedPlan The FedPlan to add.
     */
    public void addFedPlan(Hop hop, FTypes.FType fType, FedPlan fedPlan) {
        if (contains(hop, fType)) {
            List<FedPlan> fedPlanList = get(hop, fType);
            fedPlanList.add(fedPlan);
        } else {
            List<FedPlan> fedPlanList = new ArrayList<>();
            fedPlanList.add(fedPlan);
            hopMemoTable.put(new ImmutablePair<>(hop.getHopID(), fType), fedPlanList);
        }
    }

    /**
     * Adds multiple FedPlans to the memo table for a given Hop and FType.
     * If the entry already exists, the new FedPlans are appended to the list.
     *
     * @param hop            The Hop object.
     * @param fType          The associated FType.
     * @param newFedPlanList The list of FedPlans to add.
     */
    public void addFedPlanList(Hop hop, FTypes.FType fType, List<FedPlan> fedPlanList) {
        if (contains(hop, fType)) {
            List<FedPlan> prevFedPlanList = get(hop, fType);
            prevFedPlanList.addAll(fedPlanList);
        } else {
            assert !fedPlanList.isEmpty() : "FedPlan list should not be empty";
            hopMemoTable.put(new ImmutablePair<>(hop.getHopID(), fType), fedPlanList);
        }
    }

    /**
     * Retrieves the list of FedPlans associated with a given Hop and FType.
     *
     * @param hop   The Hop object.
     * @param fType The associated FType.
     * @return The list of FedPlans, or null if no entry exists.
     */
    public List<FedPlan> get(Hop hop, FTypes.FType fType) {
        return hopMemoTable.get(new ImmutablePair<>(hop.getHopID(), fType));
    }

    /**
     * Checks if the memo table contains an entry for a given Hop and FType.
     *
     * @param hop   The Hop object.
     * @param fType The associated FType.
     * @return True if the entry exists, false otherwise.
     */
    public boolean contains(Hop hop, FTypes.FType fType) {
        return hopMemoTable.containsKey(new ImmutablePair<>(hop.getHopID(), fType));
    }

    /**
     * Prunes the FedPlans associated with a specific Hop and FType,
     * keeping only the plan with the minimum cost.
     *
     * @param hop   The Hop object.
     * @param fType The associated FType.
     */
    public void prunePlan(Hop hop, FTypes.FType fType) {
        prunePlan(hopMemoTable.get(new ImmutablePair<>(hop.getHopID(), fType)));
    }

    /**
     * Prunes all entries in the memo table, retaining only the minimum-cost
     * FedPlan for each entry.
     */
    public void pruneAll() {
        for (Map.Entry<Pair<Long, FTypes.FType>, List<FedPlan>> entry : hopMemoTable.entrySet()) {
            prunePlan(entry.getValue());
        }
    }

    /**
     * Prunes the given list of FedPlans to retain only the plan with the minimum cost.
     *
     * @param fedPlanList The list of FedPlans to prune.
     */
    private void prunePlan(List<FedPlan> fedPlanList) {
        if (fedPlanList.size() > 1) {
            // Find the FedPlan with the minimum cost
            FedPlan minCostPlan = fedPlanList.stream()
                    .min(Comparator.comparingDouble(plan -> plan.cost))
                    .orElse(null);

            // Retain only the minimum cost plan
            fedPlanList.clear();
            fedPlanList.add(minCostPlan);
        }
    }
}
