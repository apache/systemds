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
import java.util.Map;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Optional;
import java.util.Set;
import java.util.HashSet;

import org.apache.commons.lang3.tuple.Pair;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.hops.fedplanner.FederatedMemoTable.HopCommon;
import org.apache.sysds.hops.fedplanner.FederatedMemoTable.FedPlan;
import org.apache.sysds.hops.fedplanner.FederatedMemoTable.FedPlanVariants;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.FunctionStatement;
import org.apache.sysds.parser.FunctionStatementBlock;
import org.apache.sysds.parser.IfStatement;
import org.apache.sysds.parser.IfStatementBlock;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.WhileStatement;
import org.apache.sysds.parser.WhileStatementBlock;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction.FederatedOutput;
import org.apache.sysds.runtime.util.UtilFunctions;

public class FederatedPlanCostEnumerator {
	private static final double DEFAULT_LOOP_WEIGHT = 10.0;
	private static final double DEFAULT_IF_ELSE_WEIGHT = 0.5;

	/**
	 * Enumerates the entire DML program to generate federated execution plans.
	 * It processes each statement block, computes the optimal federated plan,
	 * detects and resolves conflicts, and optionally prints the plan tree.
	 *
	 * @param prog The DML program to enumerate.
	 * @param isPrint A boolean indicating whether to print the federated plan tree.
	 */
	public static void enumerateProgram(DMLProgram prog, boolean isPrint) {
		FederatedMemoTable memoTable = new FederatedMemoTable();

		Map<String, List<Hop>> outerTransTable = new HashMap<>();
		Map<String, List<Hop>> formerInnerTransTable = new HashMap<>();
		Set<Hop> progRootHopSet = new HashSet<>(); // Set of hops for the root dummy node
		// TODO: Just for debug, remove later
		Set<Hop> statRootHopSet = new HashSet<>(); // Set of hops that have no parent but are not referenced
		
		for (StatementBlock sb : prog.getStatementBlocks()) {
			Optional.ofNullable(enumerateStatementBlock(sb, memoTable, outerTransTable, formerInnerTransTable, progRootHopSet, statRootHopSet, 1, false))
				.ifPresent(outerTransTable::putAll);
		}

		FedPlan optimalPlan = getMinCostRootFedPlan(progRootHopSet, memoTable);

		// Detect conflicts in the federated plans where different FedPlans have different FederatedOutput types
		double additionalTotalCost = detectAndResolveConflictFedPlan(optimalPlan, memoTable);

		// Print the federated plan tree if requested
		if (isPrint) {
			FederatedMemoTablePrinter.printFedPlanTree(optimalPlan, statRootHopSet, memoTable, additionalTotalCost);
		}
	}


	/**
	 * Enumerates the statement block and updates the transient and memoization tables.
	 * This method processes different types of statement blocks such as If, For, While, and Function blocks.
	 * It recursively enumerates the Hop DAGs within these blocks and updates the corresponding tables.
	 * The method also calculates weights recursively for if-else/loops and handles inner and outer block distinctions.
	 *
	 * @param sb The statement block to enumerate.
	 * @param memoTable The memoization table to store plan variants.
	 * @param outerTransTable The table to track immutable outer transient writes.
	 * @param formerInnerTransTable The table to track immutable former inner transient writes.
	 * @param progRootHopSet The set of hops to connect to the root dummy node.
	 * @param statRootHopSet The set of statement root hops for debugging purposes (check if not referenced).
	 * @param weight The weight associated with the current Hop.
	 * @param isInnerBlock A boolean indicating if the current block is an inner block.
	 * @return A map of inner transient writes.
	 */
	public static Map<String, List<Hop>> enumerateStatementBlock(StatementBlock sb, FederatedMemoTable memoTable, Map<String, List<Hop>> outerTransTable,
																Map<String, List<Hop>> formerInnerTransTable, Set<Hop> progRootHopSet, Set<Hop> statRootHopSet, double weight, boolean isInnerBlock) {
		Map<String, List<Hop>> innerTransTable = new HashMap<>();

		if (sb instanceof IfStatementBlock) {
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement istmt = (IfStatement)isb.getStatement(0);

			enumerateHopDAG(isb.getPredicateHops(), memoTable, outerTransTable, formerInnerTransTable, innerTransTable, progRootHopSet, statRootHopSet, weight, isInnerBlock);

			// Treat outerTransTable as immutable in inner blocks
			// Write TWrite of sb sequentially in innerTransTable, and update formerInnerTransTable after the sb ends
			// In case of if-else, create separate formerInnerTransTables for if and else, merge them after completion, and update formerInnerTransTable
			Map<String, List<Hop>> ifFormerInnerTransTable = new HashMap<>(formerInnerTransTable);
			Map<String, List<Hop>> elseFormerInnerTransTable = new HashMap<>(formerInnerTransTable);

			for (StatementBlock csb : istmt.getIfBody()){
				ifFormerInnerTransTable.putAll(enumerateStatementBlock(csb, memoTable, outerTransTable, ifFormerInnerTransTable, progRootHopSet, statRootHopSet, DEFAULT_IF_ELSE_WEIGHT * weight, true));
			}

			for (StatementBlock csb : istmt.getElseBody()){
				elseFormerInnerTransTable.putAll(enumerateStatementBlock(csb, memoTable, outerTransTable, elseFormerInnerTransTable, progRootHopSet, statRootHopSet, DEFAULT_IF_ELSE_WEIGHT * weight, true));
			}

			// If there are common keys: merge elseValue list into ifValue list
			elseFormerInnerTransTable.forEach((key, elseValue) -> {
				ifFormerInnerTransTable.merge(key, elseValue, (ifValue, newValue) -> {
					ifValue.addAll(newValue);
					return ifValue;
				});
			});
			// Update innerTransTable
			innerTransTable.putAll(ifFormerInnerTransTable);
		} else if (sb instanceof ForStatementBlock) { //incl parfor
			ForStatementBlock fsb = (ForStatementBlock) sb;
			ForStatement fstmt = (ForStatement)fsb.getStatement(0);

			// Calculate for-loop iteration count if possible
			double loopWeight = DEFAULT_LOOP_WEIGHT;
			Hop from = fsb.getFromHops().getInput().get(0);
			Hop to = fsb.getToHops().getInput().get(0);
			Hop incr = (fsb.getIncrementHops() != null) ?
					fsb.getIncrementHops().getInput().get(0) : new LiteralOp(1);

			// Calculate for-loop iteration count (weight) if from, to, and incr are literal ops (constant values)
			if( from instanceof LiteralOp && to instanceof LiteralOp && incr instanceof LiteralOp ) {
				double dfrom = HopRewriteUtils.getDoubleValue((LiteralOp) from);
				double dto = HopRewriteUtils.getDoubleValue((LiteralOp) to);
				double dincr = HopRewriteUtils.getDoubleValue((LiteralOp) incr);
				if( dfrom > dto && dincr == 1 )
					dincr = -1;
				loopWeight = UtilFunctions.getSeqLength(dfrom, dto, dincr, false);
			}
			weight *= loopWeight;

			enumerateHopDAG(fsb.getFromHops(), memoTable, outerTransTable, formerInnerTransTable, innerTransTable, progRootHopSet, statRootHopSet, weight, isInnerBlock);
			enumerateHopDAG(fsb.getToHops(), memoTable, outerTransTable, formerInnerTransTable, innerTransTable, progRootHopSet, statRootHopSet, weight, isInnerBlock);
			enumerateHopDAG(fsb.getIncrementHops(), memoTable, outerTransTable, formerInnerTransTable, innerTransTable, progRootHopSet, statRootHopSet, weight, isInnerBlock);

			enumerateStatementBlockBody(fstmt.getBody(), memoTable, outerTransTable, formerInnerTransTable, innerTransTable, progRootHopSet, statRootHopSet, weight);
		} else if (sb instanceof WhileStatementBlock) {
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement wstmt = (WhileStatement)wsb.getStatement(0);
			weight *= DEFAULT_LOOP_WEIGHT;

			enumerateHopDAG(wsb.getPredicateHops(), memoTable, outerTransTable, formerInnerTransTable, innerTransTable, progRootHopSet, statRootHopSet, weight, isInnerBlock);
			enumerateStatementBlockBody(wstmt.getBody(), memoTable, outerTransTable, formerInnerTransTable, innerTransTable, progRootHopSet, statRootHopSet, weight);
		} else if (sb instanceof FunctionStatementBlock) {
			FunctionStatementBlock fsb = (FunctionStatementBlock)sb;
			FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);

			// TODO: NOT descent multiple types (use hash set for functions using function name)
			enumerateStatementBlockBody(fstmt.getBody(), memoTable, outerTransTable, formerInnerTransTable, innerTransTable, progRootHopSet, statRootHopSet, weight);
		} else { //generic (last-level)
			if( sb.getHops() != null ){
				for(Hop c : sb.getHops())
					// In the statement block, if isInner, write hopDAG in innerTransTable, if not, write directly in outerTransTable
					enumerateHopDAG(c, memoTable, outerTransTable, formerInnerTransTable, innerTransTable, progRootHopSet, statRootHopSet, weight, isInnerBlock);
			}
		}
		return innerTransTable;
	}
		
	/**
	 * Enumerates the statement blocks within a body and updates the transient and memoization tables.
	 *
	 * @param sbList The list of statement blocks to enumerate.
	 * @param memoTable The memoization table to store plan variants.
	 * @param outerTransTable The table to track immutable outer transient writes.
	 * @param formerInnerTransTable The table to track immutable former inner transient writes.
	 * @param innerTransTable The table to track inner transient writes.
	 * @param progRootHopSet The set of hops to connect to the root dummy node.
	 * @param statRootHopSet The set of statement root hops for debugging purposes (check if not referenced).
	 * @param weight The weight associated with the current Hop.
	 */
	public static void enumerateStatementBlockBody(List<StatementBlock> sbList, FederatedMemoTable memoTable, Map<String, List<Hop>> outerTransTable,
									Map<String, List<Hop>> formerInnerTransTable, Map<String, List<Hop>> innerTransTable, Set<Hop> progRootHopSet, Set<Hop> statRootHopSet, double weight) {
		// The statement blocks within the body reference outerTransTable and formerInnerTransTable as immutable read-only,
		// and record TWrite in the innerTransTable of the statement block within the body.
		// Update the formerInnerTransTable with the contents of the returned innerTransTable.
		for (StatementBlock sb : sbList)
			formerInnerTransTable.putAll(enumerateStatementBlock(sb, memoTable, outerTransTable, formerInnerTransTable, progRootHopSet, statRootHopSet, weight, true));

		// Then update and return the innerTransTable of the statement block containing the body.
		innerTransTable.putAll(formerInnerTransTable);
	}

	/**
	 * Enumerates the statement hop DAG within a statement block.
	 * This method recursively enumerates all possible federated execution plans
	 * and identifies hops to connect to the root dummy node.
	 *
	 * @param rootHop The root Hop of the DAG to enumerate.
	 * @param memoTable The memoization table to store plan variants.
	 * @param outerTransTable The table to track transient writes.
	 * @param formerInnerTransTable The table to track immutable inner transient writes.
	 * @param innerTransTable The table to track inner transient writes.
	 * @param progRootHopSet The set of hops to connect to the root dummy node.
	 * @param statRootHopSet The set of root hops for debugging purposes.
	 * @param weight The weight associated with the current Hop.
	 * @param isInnerBlock A boolean indicating if the current block is an inner block.
	 */
	public static void enumerateHopDAG(Hop rootHop, FederatedMemoTable memoTable, Map<String, List<Hop>> outerTransTable,
										Map<String, List<Hop>> formerInnerTransTable, Map<String,List<Hop>> innerTransTable, Set<Hop> progRootHopSet, Set<Hop> statRootHopSet, double weight, boolean isInnerBlock) {
		// Recursively enumerate all possible plans
		rewireAndEnumerateFedPlan(rootHop, memoTable, outerTransTable, formerInnerTransTable, innerTransTable, weight, isInnerBlock);
	    
		// Identify hops to connect to the root dummy node
		
		if ((rootHop instanceof DataOp && (rootHop.getName().equals("__pred"))) // TWrite "__pred"
			|| (rootHop instanceof UnaryOp && ((UnaryOp)rootHop).getOp() == Types.OpOp1.PRINT)){ // u(print)
			// Connect TWrite pred and u(print) to the root dummy node
			// TODO: Should the last unreferenced TWrite be connected?
			progRootHopSet.add(rootHop);
		} else {
			// TODO: Just for debug, remove later
			// For identifying TWrites that are not referenced later
			statRootHopSet.add(rootHop);
		}
	}

	/**
	 * Rewires and enumerates federated execution plans for a given Hop.
	 * This method processes all input nodes, rewires TWrite and TRead operations,
	 * and generates federated plan variants for both inner and outer code blocks.
	 *
	 * @param hop The Hop for which to rewire and enumerate federated plans.
	 * @param memoTable The memoization table to store plan variants.
	 * @param outerTransTable The table to track transient writes.
	 * @param formerInnerTransTable The table to track immutable inner transient writes.
	 * @param innerTransTable The table to track inner transient writes.
	 * @param weight The weight associated with the current Hop.
	 * @param isInner A boolean indicating if the current block is an inner block.
	 */
	private static void rewireAndEnumerateFedPlan(Hop hop, FederatedMemoTable memoTable, Map<String,List<Hop>> outerTransTable,
										Map<String, List<Hop>> formerInnerTransTable, Map<String,List<Hop>> innerTransTable, double weight, boolean isInner) {
		// Process all input nodes first if not already in memo table
		for (Hop inputHop : hop.getInput()) {
			long inputHopID = inputHop.getHopID();
			if (!memoTable.contains(inputHopID, FederatedOutput.FOUT)
				&& !memoTable.contains(inputHopID, FederatedOutput.LOUT)) {
					rewireAndEnumerateFedPlan(inputHop, memoTable, outerTransTable, formerInnerTransTable, innerTransTable, weight, isInner);
			}
		}

		// Detect and Rewire TWrite and TRead operations
		List<Hop> childHops = hop.getInput();
		if (hop instanceof DataOp && !(hop.getName().equals("__pred"))){
			String hopName = hop.getName();

			if (isInner){ // If it's an inner code block
				if (((DataOp)hop).getOp() == Types.OpOpData.TRANSIENTWRITE){
					innerTransTable.computeIfAbsent(hopName, k -> new ArrayList<>()).add(hop);
				} else if (((DataOp)hop).getOp() == Types.OpOpData.TRANSIENTREAD){
					// Copy existing and add TWrite
					childHops = new ArrayList<>(childHops);
					List<Hop> additionalChildHops = null;
					
					// Read according to priority
					if (innerTransTable.containsKey(hopName)){
						additionalChildHops = innerTransTable.get(hopName);
					} else if (formerInnerTransTable.containsKey(hopName)){
						additionalChildHops = formerInnerTransTable.get(hopName);
					} else if (outerTransTable.containsKey(hopName)){
						additionalChildHops = outerTransTable.get(hopName);
					}

					if (additionalChildHops != null) {
						childHops.addAll(additionalChildHops);
					}
				}
			} else { // If it's an outer code block
				if (((DataOp)hop).getOp() == Types.OpOpData.TRANSIENTWRITE){
					// Add directly to outerTransTable
					outerTransTable.computeIfAbsent(hopName, k -> new ArrayList<>()).add(hop);
				} else if (((DataOp)hop).getOp() == Types.OpOpData.TRANSIENTREAD){
					childHops = new ArrayList<>(childHops);
					
					// TODO: In the case of for (i in 1:10), there is no hop that writes TWrite for i.
					// Read directly from outerTransTable and add
					List<Hop> additionalChildHops = outerTransTable.get(hopName);
					if (additionalChildHops != null) {
						childHops.addAll(additionalChildHops);
					}
				}
			}
		}

		// Enumerate the federated plan for the current Hop
		enumerateFedPlan(hop, memoTable, childHops, weight);
	}

	/**
	 * Enumerates federated execution plans for a given Hop.
	 * This method calculates the self cost and child costs for the Hop,
	 * generates federated plan variants for both LOUT and FOUT output types,
	 * and prunes redundant plans before adding them to the memo table.
	 *
	 * @param hop The Hop for which to enumerate federated plans.
	 * @param memoTable The memoization table to store plan variants.
	 * @param childHops The list of child hops.
	 * @param weight The weight associated with the current Hop.
	 */
	private static void enumerateFedPlan(Hop hop, FederatedMemoTable memoTable, List<Hop> childHops, double weight){
		long hopID = hop.getHopID();
		HopCommon hopCommon = new HopCommon(hop, weight);
		double selfCost = FederatedPlanCostEstimator.computeHopCost(hopCommon);

		FedPlanVariants lOutFedPlanVariants = new FedPlanVariants(hopCommon, FederatedOutput.LOUT);
		FedPlanVariants fOutFedPlanVariants = new FedPlanVariants(hopCommon, FederatedOutput.FOUT);

		int numInputs = childHops.size();
		int numInitInputs = hop.getInput().size();

		double[][] childCumulativeCost = new double[numInputs][2]; // # of child, LOUT/FOUT of child
		double[] childForwardingCost = new double[numInputs]; // # of child

		// The self cost follows its own weight, while the forwarding cost follows the parent's weight.
		FederatedPlanCostEstimator.getChildCosts(hopCommon, memoTable, childHops, childCumulativeCost, childForwardingCost);

		if (numInitInputs == numInputs){
			enumerateOnlyInitChildFedPlan(lOutFedPlanVariants, fOutFedPlanVariants, numInitInputs, childHops, childCumulativeCost, childForwardingCost, selfCost);
		} else {
			enumerateTReadInitChildFedPlan(lOutFedPlanVariants, fOutFedPlanVariants, numInitInputs, numInputs, childHops, childCumulativeCost, childForwardingCost, selfCost);
		}

		// Prune the FedPlans to remove redundant plans
		lOutFedPlanVariants.pruneFedPlans();
		fOutFedPlanVariants.pruneFedPlans();

		// Add the FedPlanVariants to the memo table
		memoTable.addFedPlanVariants(hopID, FederatedOutput.LOUT, lOutFedPlanVariants);
		memoTable.addFedPlanVariants(hopID, FederatedOutput.FOUT, fOutFedPlanVariants);
	}

	/**
	 * Enumerates federated execution plans for initial child hops only.
	 * This method generates all possible combinations of federated output types (LOUT and FOUT)
	 * for the initial child hops and calculates their cumulative costs.
	 *
	 * @param lOutFedPlanVariants The FedPlanVariants object for LOUT output type.
	 * @param fOutFedPlanVariants The FedPlanVariants object for FOUT output type.
	 * @param numInitInputs The number of initial input hops.
	 * @param childHops The list of child hops.
	 * @param childCumulativeCost The cumulative costs for each child hop.
	 * @param childForwardingCost The forwarding costs for each child hop.
	 * @param selfCost The self cost of the current hop.
	 */
	private static void enumerateOnlyInitChildFedPlan(FedPlanVariants lOutFedPlanVariants, FedPlanVariants fOutFedPlanVariants, int numInitInputs, List<Hop> childHops, 
				double[][] childCumulativeCost, double[] childForwardingCost, double selfCost){
		// Iterate 2^n times, generating two FedPlans (LOUT, FOUT) each time.
		for (int i = 0; i < (1 << numInitInputs); i++) {
			double[] cumulativeCost = new double[]{selfCost, selfCost};
			List<Pair<Long, FederatedOutput>> planChilds = new ArrayList<>();
			// LOUT and FOUT share the same planChilds in each iteration (only forwarding cost differs).
			enumerateInitChildFedPlan(numInitInputs, childHops, planChilds, childCumulativeCost, childForwardingCost, cumulativeCost, i);

			lOutFedPlanVariants.addFedPlan(new FedPlan(cumulativeCost[0], lOutFedPlanVariants, planChilds));
			fOutFedPlanVariants.addFedPlan(new FedPlan(cumulativeCost[1], fOutFedPlanVariants, planChilds));
		}
	}

	/**
	 * Enumerates federated execution plans for a TRead hop.
	 * This method calculates the cumulative costs for both LOUT and FOUT federated output types
	 * by considering the additional child hops, which are TWrite hops.
	 * It generates all possible combinations of federated output types for the initial child hops
	 * and adds the pre-calculated costs of the TWrite child hops to these combinations.
	 *
	 * @param lOutFedPlanVariants The FedPlanVariants object for LOUT output type.
	 * @param fOutFedPlanVariants The FedPlanVariants object for FOUT output type.
	 * @param numInitInputs The number of initial input hops.
	 * @param numInputs The total number of input hops, including additional TWrite hops.
	 * @param childHops The list of child hops.
	 * @param childCumulativeCost The cumulative costs for each child hop.
	 * @param childForwardingCost The forwarding costs for each child hop.
	 * @param selfCost The self cost of the current hop.
	 */
	private static void enumerateTReadInitChildFedPlan(FedPlanVariants lOutFedPlanVariants, FedPlanVariants fOutFedPlanVariants,
					int numInitInputs, int numInputs, List<Hop> childHops, 
					double[][] childCumulativeCost, double[] childForwardingCost, double selfCost){
		double lOutTReadCumulativeCost = selfCost;
		double fOutTReadCumulativeCost = selfCost;
		
		List<Pair<Long, FederatedOutput>> lOutTReadPlanChilds = new ArrayList<>();
		List<Pair<Long, FederatedOutput>> fOutTReadPlanChilds = new ArrayList<>();
		
		// Pre-calculate the cost for the additional child hop, which is a TWrite hop, of the TRead hop.
		// Constraint: TWrite must have the same FedOutType as TRead.
		for (int j = numInitInputs; j < numInputs; j++) {
			Hop inputHop = childHops.get(j);
			lOutTReadPlanChilds.add(Pair.of(inputHop.getHopID(), FederatedOutput.LOUT));
			fOutTReadPlanChilds.add(Pair.of(inputHop.getHopID(), FederatedOutput.FOUT));

			lOutTReadCumulativeCost += childCumulativeCost[j][0];
			fOutTReadCumulativeCost += childCumulativeCost[j][1];
			// Skip TWrite -> TRead as they have the same FedOutType.
		}

		for (int i = 0; i < (1 << numInitInputs); i++) {
			double[] cumulativeCost = new double[]{selfCost, selfCost};
			List<Pair<Long, FederatedOutput>> lOutPlanChilds = new ArrayList<>();
			enumerateInitChildFedPlan(numInitInputs, childHops, lOutPlanChilds, childCumulativeCost, childForwardingCost, cumulativeCost, i);

			// Copy lOutPlanChilds to create fOutPlanChilds and add the pre-calculated cost of the TWrite child hop.
			List<Pair<Long, FederatedOutput>> fOutPlanChilds = new ArrayList<>(lOutPlanChilds);
			
			lOutPlanChilds.addAll(lOutTReadPlanChilds);
			fOutPlanChilds.addAll(fOutTReadPlanChilds);

			cumulativeCost[0] += lOutTReadCumulativeCost;
			cumulativeCost[1] += fOutTReadCumulativeCost;

			lOutFedPlanVariants.addFedPlan(new FedPlan(cumulativeCost[0], lOutFedPlanVariants, lOutPlanChilds));
			fOutFedPlanVariants.addFedPlan(new FedPlan(cumulativeCost[1], fOutFedPlanVariants, fOutPlanChilds));
		}
	}

	// Calculates costs for initial child hops, determining FOUT or LOUT based on `i`.
	private static void enumerateInitChildFedPlan(int numInitInputs, List<Hop> childHops, List<Pair<Long, FederatedOutput>> planChilds,
				double[][] childCumulativeCost, double[] childForwardingCost, double[] cumulativeCost, int i){
		// For each input, determine if it should be FOUT or LOUT based on bit j in i
		for (int j = 0; j < numInitInputs; j++) {
			Hop inputHop = childHops.get(j);
			// Calculate the bit value to decide between FOUT and LOUT for the current input
			final int bit = (i & (1 << j)) != 0 ? 1 : 0; // Determine the bit value (decides FOUT/LOUT)
			final FederatedOutput childType = (bit == 1) ? FederatedOutput.FOUT : FederatedOutput.LOUT;
			planChilds.add(Pair.of(inputHop.getHopID(), childType));

			// Update the cumulative cost for LOUT, FOUT
			cumulativeCost[0] += childCumulativeCost[j][bit] + childForwardingCost[j] * bit;
			cumulativeCost[1] += childCumulativeCost[j][bit] + childForwardingCost[j] * (1 - bit);
		}
	}	

	// Creates a dummy root node (fedplan) and selects the FedPlan with the minimum cost to return.
	// The dummy root node does not have LOUT or FOUT.
	private static FedPlan getMinCostRootFedPlan(Set<Hop> progRootHopSet, FederatedMemoTable memoTable) {
		double cumulativeCost = 0;
		List<Pair<Long, FederatedOutput>> rootFedPlanChilds = new ArrayList<>();

		// Iterate over each Hop in the progRootHopSet
		for (Hop endHop : progRootHopSet){
			// Retrieve the pruned FedPlan for LOUT and FOUT from the memo table
			FedPlan lOutFedPlan = memoTable.getFedPlanAfterPrune(endHop.getHopID(), FederatedOutput.LOUT);
			FedPlan fOutFedPlan = memoTable.getFedPlanAfterPrune(endHop.getHopID(), FederatedOutput.FOUT);

			// Compare the cumulative costs of LOUT and FOUT FedPlans
			if (lOutFedPlan.getCumulativeCost() <= fOutFedPlan.getCumulativeCost()){
				cumulativeCost += lOutFedPlan.getCumulativeCost();
				rootFedPlanChilds.add(Pair.of(endHop.getHopID(), FederatedOutput.LOUT));
			} else{
				cumulativeCost += fOutFedPlan.getCumulativeCost();
				rootFedPlanChilds.add(Pair.of(endHop.getHopID(), FederatedOutput.FOUT));
			}
		}

		return new FedPlan(cumulativeCost, null, rootFedPlanChilds);
	}

	/**
	 * Detects and resolves conflicts in federated plans starting from the root plan.
	 * This function performs a breadth-first search (BFS) to traverse the federated plan tree.
	 * It identifies conflicts where the same plan ID has different federated output types.
	 * For each conflict, it records the plan ID and its conflicting parent plans.
	 * The function ensures that each plan ID is associated with a consistent federated output type
	 * by resolving these conflicts iteratively.
	 *
	 * The process involves:
	 * - Using a map to track conflicts, associating each plan ID with its federated output type
	 *   and a list of parent plans.
	 * - Storing detected conflicts in a linked map, each entry containing a plan ID and its
	 *   conflicting parent plans.
	 * - Performing BFS traversal starting from the root plan, checking each child plan for conflicts.
	 * - If a conflict is detected (i.e., a plan ID has different output types), the conflicting plan
	 *   is removed from the BFS queue and added to the conflict map to prevent duplicate calculations.
	 * - Resolving conflicts by ensuring a consistent federated output type across the plan.
	 * - Re-running BFS with resolved conflicts to ensure all inconsistencies are addressed.
	 *
	 * @param rootPlan The root federated plan from which to start the conflict detection.
	 * @param memoTable The memoization table used to retrieve pruned federated plans.
	 * @return The cumulative additional cost for resolving conflicts.
	 */
	private static double detectAndResolveConflictFedPlan(FedPlan rootPlan, FederatedMemoTable memoTable) {
		// Map to track conflicts: maps a plan ID to its federated output type and list of parent plans
		Map<Long, Pair<FederatedOutput, List<FedPlan>>> conflictCheckMap = new HashMap<>();

		// LinkedMap to store detected conflicts, each with a plan ID and its conflicting parent plans
		LinkedHashMap<Long, List<FedPlan>> conflictLinkedMap = new LinkedHashMap<>();

		// LinkedMap for BFS traversal starting from the root plan (Do not use value (boolean))
		LinkedHashMap<FedPlan, Boolean> bfsLinkedMap = new LinkedHashMap<>();
		bfsLinkedMap.put(rootPlan, true);

		// Array to store cumulative additional cost for resolving conflicts
		double[] cumulativeAdditionalCost = new double[]{0.0};

		while (!bfsLinkedMap.isEmpty()) {
			// Perform BFS to detect conflicts in federated plans
			while (!bfsLinkedMap.isEmpty()) {
				FedPlan currentPlan = bfsLinkedMap.keySet().iterator().next();
				bfsLinkedMap.remove(currentPlan);

				// Iterate over each child plan of the current plan
				for (Pair<Long, FederatedOutput> childPlanPair : currentPlan.getChildFedPlans()) {
					FedPlan childFedPlan = memoTable.getFedPlanAfterPrune(childPlanPair);

					// Check if the child plan ID is already visited
					if (conflictCheckMap.containsKey(childPlanPair.getLeft())) {
						// Retrieve the existing conflict pair for the child plan
						Pair<FederatedOutput, List<FedPlan>> conflictChildPlanPair = conflictCheckMap.get(childPlanPair.getLeft());
						// Add the current plan to the list of parent plans
						conflictChildPlanPair.getRight().add(currentPlan);

						// If the federated output type differs, a conflict is detected
						if (conflictChildPlanPair.getLeft() != childPlanPair.getRight()) {
							// If this is the first detection, remove conflictChildFedPlan from the BFS queue and add it to the conflict linked map (queue)
							// If the existing FedPlan is not removed from the bfsqueue or both actions are performed, duplicate calculations for the same FedPlan and its children occur
							if (!conflictLinkedMap.containsKey(childPlanPair.getLeft())) {
								conflictLinkedMap.put(childPlanPair.getLeft(), conflictChildPlanPair.getRight());
								bfsLinkedMap.remove(childFedPlan);
							}
						}
					} else {
						// If no conflict exists, create a new entry in the conflict check map
						List<FedPlan> parentFedPlanList = new ArrayList<>();
						parentFedPlanList.add(currentPlan);

						// Map the child plan ID to its output type and list of parent plans
						conflictCheckMap.put(childPlanPair.getLeft(), new ImmutablePair<>(childPlanPair.getRight(), parentFedPlanList));
						// Add the child plan to the BFS queue
						bfsLinkedMap.put(childFedPlan, true);
					}
				}
			}
			// Resolve these conflicts to ensure a consistent federated output type across the plan
			// Re-run BFS with resolved conflicts
			bfsLinkedMap = FederatedPlanCostEstimator.resolveConflictFedPlan(memoTable, conflictLinkedMap, cumulativeAdditionalCost);
			conflictLinkedMap.clear();
		}

		// Return the cumulative additional cost for resolving conflicts
		return cumulativeAdditionalCost[0];
	}
}
