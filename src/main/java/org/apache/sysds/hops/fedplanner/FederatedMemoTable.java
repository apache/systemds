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

 import java.util.Comparator;
 import java.util.HashMap;
 import java.util.List;
 import java.util.ArrayList;
 import java.util.Map;
 import org.apache.sysds.hops.Hop;
 import org.apache.commons.lang3.tuple.Pair;
 import org.apache.commons.lang3.tuple.ImmutablePair;
 import org.apache.sysds.runtime.instructions.fed.FEDInstruction.FederatedOutput;
 
 /**
  * A Memoization Table for managing federated plans (FedPlan) based on combinations of Hops and fedOutTypes.
  * This table stores and manages different execution plan variants for each Hop and fedOutType combination,
  * facilitating the optimization of federated execution plans.
  */
 public class FederatedMemoTable {
	 // Maps Hop ID and fedOutType pairs to their plan variants
	 private final Map<Pair<Long, FederatedOutput>, FedPlanVariants> hopMemoTable = new HashMap<>();
 
	 public void addFedPlanVariants(long hopID, FederatedOutput fedOutType, FedPlanVariants fedPlanVariants) {
		 hopMemoTable.put(new ImmutablePair<>(hopID, fedOutType), fedPlanVariants);
	 }
 
	 public FedPlanVariants getFedPlanVariants(Pair<Long, FederatedOutput> fedPlanPair) {
		 return hopMemoTable.get(fedPlanPair);
	 }
 
	 public FedPlan getFedPlanAfterPrune(long hopID, FederatedOutput fedOutType) {
		 FedPlanVariants fedPlanVariantList = hopMemoTable.get(new ImmutablePair<>(hopID, fedOutType));
		 return fedPlanVariantList._fedPlanVariants.get(0);
	 }
 
	 public FedPlan getFedPlanAfterPrune(Pair<Long, FederatedOutput> fedPlanPair) {
		 FedPlanVariants fedPlanVariantList = hopMemoTable.get(fedPlanPair);
		 return fedPlanVariantList._fedPlanVariants.get(0);
	 }
 
	 public boolean contains(long hopID, FederatedOutput fedOutType) {
		 return hopMemoTable.containsKey(new ImmutablePair<>(hopID, fedOutType));
	 }
 
	 /**
	  * Represents a single federated execution plan with its associated costs and dependencies.
	  * This class contains:
	  * 1. selfCost: Cost of the current hop (computation + input/output memory access).
	  * 2. cumulativeCost: Total cost including this plan's selfCost and all child plans' cumulativeCost.
	  * 3. forwardingCost: Network transfer cost for this plan to the parent plan.
	  * 
	  * FedPlan is linked to FedPlanVariants, which in turn uses HopCommon to manage common properties and costs.
	  */
	 public static class FedPlan {
		 private double cumulativeCost;                  // Total cost = sum of selfCost + cumulativeCost of child plans
		 private final FedPlanVariants fedPlanVariants;  // Reference to variant list
		 private final List<Pair<Long, FederatedOutput>> childFedPlans;  // Child plan references
 
		 public FedPlan(double cumulativeCost, FedPlanVariants fedPlanVariants, List<Pair<Long, FederatedOutput>> childFedPlans) {
			 this.cumulativeCost = cumulativeCost;
			 this.fedPlanVariants = fedPlanVariants;
			 this.childFedPlans = childFedPlans;			
		 }
 
		 public Hop getHopRef() {return fedPlanVariants.hopCommon.getHopRef();}
		 public long getHopID() {return fedPlanVariants.hopCommon.getHopRef().getHopID();}
		 public FederatedOutput getFedOutType() {return fedPlanVariants.getFedOutType();}
		 public double getCumulativeCost() {return cumulativeCost;}
		 public double getSelfCost() {return fedPlanVariants.hopCommon.getSelfCost();}
		 public double getForwardingCost() {return fedPlanVariants.hopCommon.getForwardingCost();}
		 public double getWeight() {return fedPlanVariants.hopCommon.getWeight();}
		 public List<Pair<Long, FederatedOutput>> getChildFedPlans() {return childFedPlans;}
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
 
		 public FedPlanVariants(HopCommon hopCommon, FederatedOutput fedOutType) {
			 this.hopCommon = hopCommon;
			 this.fedOutType = fedOutType;
			 this._fedPlanVariants = new ArrayList<>();
		 }
 
		 public boolean isEmpty() {return _fedPlanVariants.isEmpty();}
		 public void addFedPlan(FedPlan fedPlan) {_fedPlanVariants.add(fedPlan);}
		 public List<FedPlan> getFedPlanVariants() {return _fedPlanVariants;}
		 public FederatedOutput getFedOutType() {return fedOutType;}
 
		 public void pruneFedPlans() {
			 if (_fedPlanVariants.size() > 1) {
				 // Find the FedPlan with the minimum cumulative cost
				 FedPlan minCostPlan = _fedPlanVariants.stream()
						 .min(Comparator.comparingDouble(FedPlan::getCumulativeCost))
						 .orElse(null);
 
				 // Retain only the minimum cost plan
				 _fedPlanVariants.clear();
				 _fedPlanVariants.add(minCostPlan);
			 }
		 }
	 }
 
	 /**
	  * Represents common properties and costs associated with a Hop.
	  * This class holds a reference to the Hop and tracks its execution and network forwarding (transfer) costs.
	  */
	 public static class HopCommon {
		 protected final Hop hopRef; // Reference to the associated Hop
		 protected double selfCost; // Cost of the hop's computation and memory access
		 protected double forwardingCost; // Cost of forwarding the hop's output to its parent
		 protected double weight; // Weight used to calculate cost based on hop execution frequency
 
		 public HopCommon(Hop hopRef, double weight) {
			 this.hopRef = hopRef;
			 this.selfCost = 0;
			 this.forwardingCost = 0;
			 this.weight = weight;
		 }
 
		 public Hop getHopRef() {return hopRef;}
		 public double getSelfCost() {return selfCost;}
		 public double getForwardingCost() {return forwardingCost;}
		 public double getWeight() {return weight;}
 
		 protected void setSelfCost(double selfCost) {this.selfCost = selfCost;}
		 protected void setForwardingCost(double forwardingCost) {this.forwardingCost = forwardingCost;}
	 }
 }
 