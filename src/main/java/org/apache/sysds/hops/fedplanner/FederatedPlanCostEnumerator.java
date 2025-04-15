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
 import java.util.Set;
 import java.util.HashSet;

 import org.apache.commons.lang3.tuple.Pair;
 
 import org.apache.commons.lang3.tuple.ImmutablePair;
 import org.apache.sysds.common.Types;
 import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.FunctionOp.FunctionType;
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

		 List<Map<String, List<Hop>>> outerTransTableList = new ArrayList<>();
		 Map<String, List<Hop>> outerTransTable = new HashMap<>();
		 outerTransTableList.add(outerTransTable);

		 Set<Hop> progRootHopSet = new HashSet<>(); // Set of hops for the root dummy node
		 // TODO: Just for debug, remove later
		 Set<Hop> statRootHopSet = new HashSet<>(); // Set of hops that have no parent but are not referenced

		 List<Pair<Long, Double>> loopStack = new ArrayList<>();
		 Set<String> fnStack = new HashSet<>();

		 Map<Long, List<Hop>> rewireTable = FederatedPlanRewireTransTable.rewireProgram(prog);
		 
		 // Debug: Print rewireTable contents
		 System.out.println("=== RewireTable Contents ===");
		 rewireTable.forEach((hopId, hopList) -> {
			 System.out.println("HopID: " + hopId);
			 System.out.println("Connected Hops:");
			 hopList.forEach(h -> System.out.println("  - " + h.getHopID() + " (" + h.getClass().getSimpleName() + "): " + h.getName()));
			 System.out.println();
		 });
		 System.out.println("=== End RewireTable Contents ===");

		 for (StatementBlock sb : prog.getStatementBlocks()) {
			Map<String, List<Hop>> innerTransTable = enumerateStatementBlock(sb, prog, memoTable, outerTransTableList, null, fnStack, progRootHopSet, statRootHopSet, 1, loopStack);
			outerTransTableList.get(0).putAll(innerTransTable);
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
	  * @param formerTransTable The table to track immutable former inner transient writes.
	  * @param progRootHopSet The set of hops to connect to the root dummy node.
	  * @param statRootHopSet The set of statement root hops for debugging purposes (check if not referenced).
	  * @param weight The weight associated with the current Hop.
	  * @param parentLoopStack The context of parent loops for loop-level context tracking.
	  * @return A map of inner transient writes.
	  */
	 public static Map<String, List<Hop>> enumerateStatementBlock(StatementBlock sb, DMLProgram prog, FederatedMemoTable memoTable, List<Map<String, List<Hop>>> outerTransTableList,
																 Map<String, List<Hop>> formerTransTable, Set<String> fnStack, Set<Hop> progRootHopSet, Set<Hop> statRootHopSet, 
																 double weight, List<Pair<Long, Double>> parentLoopStack) {
		 List<Map<String, List<Hop>>> newOuterTransTableList = new ArrayList<>(outerTransTableList);

		 if (formerTransTable != null){
			 newOuterTransTableList.add(formerTransTable);
		 }

		 Map<String, List<Hop>> newFormerTransTable = new HashMap<>();
		 Map<String, List<Hop>> innerTransTable = new HashMap<>();

		 if (sb instanceof IfStatementBlock) {
			 IfStatementBlock isb = (IfStatementBlock) sb;
			 IfStatement istmt = (IfStatement)isb.getStatement(0);

			 Map<String, List<Hop>> elseFormerTransTable = new HashMap<>();
			 weight *= DEFAULT_IF_ELSE_WEIGHT;

			 enumerateHopDAG(isb.getPredicateHops(), prog, memoTable, newOuterTransTableList, null, innerTransTable, fnStack, progRootHopSet, statRootHopSet, weight, parentLoopStack);
			 
			 newFormerTransTable.putAll(innerTransTable);
			 elseFormerTransTable.putAll(innerTransTable);
			 
			 for (StatementBlock innerIsb : istmt.getIfBody())
				 newFormerTransTable.putAll(enumerateStatementBlock(innerIsb, prog, memoTable, newOuterTransTableList, newFormerTransTable, fnStack, progRootHopSet, statRootHopSet, weight, parentLoopStack));

			 for (StatementBlock innerIsb : istmt.getElseBody())
				 elseFormerTransTable.putAll(enumerateStatementBlock(innerIsb, prog, memoTable, newOuterTransTableList, elseFormerTransTable, fnStack, progRootHopSet, statRootHopSet, weight, parentLoopStack));

			 // If there are common keys: merge elseValue list into ifValue list
			 elseFormerTransTable.forEach((key, elseValue) -> {
				newFormerTransTable.merge(key, elseValue, (ifValue, newValue) -> {
					 ifValue.addAll(newValue);
					 return ifValue;
				 });
			 });
		 }
		 else if (sb instanceof ForStatementBlock) { //incl parfor
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

			// 현재 루프 컨텍스트 생성 (부모 컨텍스트 복사)
			List<Pair<Long, Double>> currentLoopStack = new ArrayList<>(parentLoopStack);
			currentLoopStack.add(Pair.of(sb.getSBID(), loopWeight));
			 
			 enumerateHopDAG(fsb.getFromHops(), prog, memoTable, newOuterTransTableList, null, innerTransTable, fnStack, progRootHopSet, statRootHopSet, weight, currentLoopStack);
			 enumerateHopDAG(fsb.getToHops(), prog, memoTable, newOuterTransTableList, null, innerTransTable, fnStack, progRootHopSet, statRootHopSet, weight, currentLoopStack);
			 enumerateHopDAG(fsb.getIncrementHops(), prog, memoTable, newOuterTransTableList, null, innerTransTable, fnStack, progRootHopSet, statRootHopSet, weight, currentLoopStack);
			 newFormerTransTable.putAll(innerTransTable);

			 for (StatementBlock innerFsb : fstmt.getBody())
				newFormerTransTable.putAll(enumerateStatementBlock(innerFsb, prog, memoTable, newOuterTransTableList, newFormerTransTable, fnStack, progRootHopSet, statRootHopSet, weight, currentLoopStack));
		 }
		 else if (sb instanceof WhileStatementBlock) {
			// TODO:  Loop 안의 TRead의 Parent가 Loop안에서 발생한 TWrite를 읽는 다면 동일한 fedoutputType을 가짐.
			// Question: 만약 Loop안의 Twrite을 Loop 밖에서 읽는다면?
			// 중첩 While문 일때는? 모름 자고 일어나서 하자

			 WhileStatementBlock wsb = (WhileStatementBlock) sb;
			 WhileStatement wstmt = (WhileStatement)wsb.getStatement(0);
			 weight *= DEFAULT_LOOP_WEIGHT;
			 
			// 현재 루프 컨텍스트 생성 (부모 컨텍스트 복사)
			List<Pair<Long, Double>> currentLoopStack = new ArrayList<>(parentLoopStack);
			currentLoopStack.add(Pair.of(sb.getSBID(), DEFAULT_LOOP_WEIGHT));
 
			 enumerateHopDAG(wsb.getPredicateHops(), prog, memoTable, newOuterTransTableList, null, innerTransTable, fnStack, progRootHopSet, statRootHopSet, weight, currentLoopStack);
			 newFormerTransTable.putAll(innerTransTable);

			 for (StatementBlock innerWsb : wstmt.getBody())
				newFormerTransTable.putAll(enumerateStatementBlock(innerWsb, prog, memoTable, newOuterTransTableList, newFormerTransTable, fnStack, progRootHopSet, statRootHopSet, weight, currentLoopStack));
		 }
		 else if (sb instanceof FunctionStatementBlock) {
			 FunctionStatementBlock fsb = (FunctionStatementBlock)sb;
			 FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
			 
			 for (StatementBlock innerFsb : fstmt.getBody())
				newFormerTransTable.putAll(enumerateStatementBlock(innerFsb, prog, memoTable, newOuterTransTableList, newFormerTransTable, fnStack, progRootHopSet, statRootHopSet, weight, parentLoopStack));
		 }
		 else { //generic (last-level)
			 if( sb.getHops() != null ){
				 for(Hop c : sb.getHops())
					 enumerateHopDAG(c, prog, memoTable, newOuterTransTableList, null, innerTransTable, fnStack, progRootHopSet, statRootHopSet, weight, parentLoopStack);
			 }

			 return innerTransTable;
		 }
		 return newFormerTransTable;
	 }
		 
	 /**
	  * Enumerates the statement hop DAG within a statement block.
	  * This method recursively enumerates all possible federated execution plans
	  * and identifies hops to connect to the root dummy node.
	  *
	  * @param rootHop The root Hop of the DAG to enumerate.
	  * @param memoTable The memoization table to store plan variants.
	  * @param outerTransTable The table to track transient writes.
	  * @param formerTransTable The table to track immutable inner transient writes.
	  * @param innerTransTable The table to track inner transient writes.
	  * @param progRootHopSet The set of hops to connect to the root dummy node.
	  * @param statRootHopSet The set of root hops for debugging purposes.
	  * @param weight The weight associated with the current Hop.
	  * @param loopStack The context of parent loops for loop-level context tracking.
	  */
	 public static void enumerateHopDAG(Hop rootHop, DMLProgram prog, FederatedMemoTable memoTable, List<Map<String, List<Hop>>> outerTransTableList,
										 Map<String, List<Hop>> formerTransTable, Map<String,List<Hop>> innerTransTable, Set<String> fnStack,
										 Set<Hop> progRootHopSet, Set<Hop> statRootHopSet, double weight, List<Pair<Long, Double>> loopStack) {
		 // Recursively enumerate all possible plans
		 rewireAndEnumerateFedPlan(rootHop, prog, memoTable, outerTransTableList, formerTransTable, innerTransTable, fnStack, progRootHopSet, statRootHopSet, weight, loopStack);
		 
		 // Identify hops to connect to the root dummy node
		 if ((rootHop instanceof DataOp && (rootHop.getName().equals("__pred"))) // TWrite "__pred"
			 || (rootHop instanceof UnaryOp && ((UnaryOp)rootHop).getOp() == Types.OpOp1.PRINT) // u(print)
			 || (rootHop instanceof DataOp && ((DataOp)rootHop).getOp() == Types.OpOpData.PERSISTENTWRITE)){ // PWrite
			 // Connect TWrite pred and u(print) to the root dummy node
			 // TODO: Should we check all statement-level root hops to see if they are not referenced?
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
	  * @param formerTransTable The table to track immutable inner transient writes.
	  * @param innerTransTable The table to track inner transient writes.
	  * @param weight The weight associated with the current Hop.
	  * @param loopStack The context of parent loops for loop-level context tracking.
	  */
	 private static void rewireAndEnumerateFedPlan(Hop hop, DMLProgram prog, FederatedMemoTable memoTable, List<Map<String, List<Hop>>> outerTransTableList,
												Map<String, List<Hop>> formerTransTable, Map<String, List<Hop>> innerTransTable, Set<String> fnStack, 
												Set<Hop> progRootHopSet, Set<Hop> statRootHopSet, double weight, List<Pair<Long, Double>> loopStack) {
		// Process all input nodes first if not already in memo table
		for (Hop inputHop : hop.getInput()) {
			long inputHopID = inputHop.getHopID();
			if (!memoTable.contains(inputHopID, FederatedOutput.FOUT)
					&& !memoTable.contains(inputHopID, FederatedOutput.LOUT)) {
				rewireAndEnumerateFedPlan(inputHop, prog, memoTable, outerTransTableList, formerTransTable, innerTransTable, fnStack, progRootHopSet, statRootHopSet, weight, loopStack);
			}
		}

		if( hop instanceof FunctionOp )
		{
			//maintain counters and investigate functions if not seen so far
			FunctionOp fop = (FunctionOp) hop;
			String fkey = fop.getFunctionKey();
			
			if( fop.getFunctionType() == FunctionType.DML )
			{
				FunctionStatementBlock fsb = prog.getFunctionStatementBlock(fop.getFunctionNamespace(), fop.getFunctionName());
				// Todo: progRootHopSet, statRootHopSet을 이렇게 넘겨줘야하나?
				// Todo: 재귀랑 여러번 호출되는거랑 다른 것 아닌가?
				// Todo: Input/Output이 제대로 넘겨지는 것이 맞나?
				 if(!fnStack.contains(fkey)) {
					 fnStack.add(fkey);
					 enumerateStatementBlock(fsb, prog, memoTable, outerTransTableList, null, fnStack, progRootHopSet, statRootHopSet, 1, loopStack);
				 }
			}
		}

		// Determine modified child hops based on DataOp type and transient operations
		Pair<List<Hop>, Boolean> result = rewireTransReadWrite(hop, outerTransTableList, formerTransTable, innerTransTable);
		List<Hop> childHops = result.getLeft();
		boolean isTrans = result.getRight();

		// Enumerate the federated plan for the current Hop
		enumerateFedPlan(hop, memoTable, childHops, weight, isTrans, loopStack);
	}

	private static Pair<List<Hop>, Boolean> rewireTransReadWrite(Hop hop, List<Map<String, List<Hop>>> outerTransTableList,
													Map<String, List<Hop>> formerTransTable,
													Map<String, List<Hop>> innerTransTable) {
		List<Hop> childHops = hop.getInput();
		boolean isTrans = false;
		
		// TODO: How about PWrite?
		if (!(hop instanceof DataOp) || hop.getName().equals("__pred")) {
			return Pair.of(childHops, isTrans); // Early exit for non-DataOp or __pred
		}

		DataOp dataOp = (DataOp) hop;
		Types.OpOpData opType = dataOp.getOp();
		String hopName = dataOp.getName();

		if (opType == Types.OpOpData.TRANSIENTWRITE) {
			innerTransTable.computeIfAbsent(hopName, k -> new ArrayList<>()).add(hop);
			isTrans = true;
		}
		else if (opType == Types.OpOpData.TRANSIENTREAD) {
			childHops = rewireTransRead(childHops, hopName, 
				innerTransTable, formerTransTable, outerTransTableList);
			isTrans = true;
		}

		return Pair.of(childHops, isTrans);
	}

	private static List<Hop> rewireTransRead(List<Hop> childHops, String hopName, Map<String, List<Hop>> innerTransTable,
													Map<String, List<Hop>> formerTransTable, List<Map<String, List<Hop>>> outerTransTableList) {
		List<Hop> newChildHops = new ArrayList<>(childHops);
		List<Hop> additionalChildHops = new ArrayList<>();

		// Read according to priority: inner -> former -> outer
		if (!innerTransTable.isEmpty()){
			additionalChildHops = innerTransTable.get(hopName);
		}

		if ((additionalChildHops == null || additionalChildHops.isEmpty()) && formerTransTable != null) {
			additionalChildHops = formerTransTable.get(hopName);
		}

		if (additionalChildHops == null || additionalChildHops.isEmpty()) {
			// 마지막으로 삽입된 outerTransTable부터 역순으로 순회
			for (int i = outerTransTableList.size() - 1; i >= 0; i--) {
				Map<String, List<Hop>> outerTransTable = outerTransTableList.get(i);
				additionalChildHops = outerTransTable.get(hopName);
				if (additionalChildHops != null && !additionalChildHops.isEmpty()) break;
			}
		}

		if (additionalChildHops != null && !additionalChildHops.isEmpty()) {
			newChildHops.addAll(additionalChildHops);
		}
		return newChildHops;
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
	  * @param loopStack The context of parent loops for loop-level context tracking.
	  */
	 private static void enumerateFedPlan(Hop hop, FederatedMemoTable memoTable, List<Hop> childHops, double weight, boolean isTrans, List<Pair<Long, Double>> loopStack) {
		 long hopID = hop.getHopID();
		 HopCommon hopCommon = new HopCommon(hop, weight, loopStack);
		 double selfCost = FederatedPlanCostEstimator.computeHopCost(hopCommon);
 
		 FedPlanVariants lOutFedPlanVariants = new FedPlanVariants(hopCommon, FederatedOutput.LOUT);
		 FedPlanVariants fOutFedPlanVariants = new FedPlanVariants(hopCommon, FederatedOutput.FOUT);
 
		 int numInputs = childHops.size();
		 int numInitInputs = hop.getInput().size();
 
		 double[][] childCumulativeCost = new double[numInputs][2]; // # of child, LOUT/FOUT of child
		 double[] childForwardingCost = new double[numInputs]; // # of child
 
		 // The self cost follows its own weight, while the forwarding cost follows the parent's weight.
		 FederatedPlanCostEstimator.getChildCosts(hopCommon, memoTable, childHops, childCumulativeCost, childForwardingCost);
		
		 if (isTrans){
			 enumerateTransChildFedPlan(lOutFedPlanVariants, fOutFedPlanVariants, numInitInputs, numInputs, childHops, childCumulativeCost, selfCost);
		 } else {
			 enumerateChildFedPlan(lOutFedPlanVariants, fOutFedPlanVariants, numInitInputs, childHops, childCumulativeCost, childForwardingCost, selfCost);
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
	 private static void enumerateChildFedPlan(FedPlanVariants lOutFedPlanVariants, FedPlanVariants fOutFedPlanVariants, int numInitInputs, List<Hop> childHops, 
				 double[][] childCumulativeCost, double[] childForwardingCost, double selfCost){
		 // Iterate 2^n times, generating two FedPlans (LOUT, FOUT) each time.
		 for (int i = 0; i < (1 << numInitInputs); i++) {
			 double[] cumulativeCost = new double[]{selfCost, selfCost};
			 List<Pair<Long, FederatedOutput>> planChilds = new ArrayList<>();

			 // LOUT and FOUT share the same planChilds in each iteration (only forwarding cost differs).
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
 
			 lOutFedPlanVariants.addFedPlan(new FedPlan(cumulativeCost[0], lOutFedPlanVariants, planChilds));
			 fOutFedPlanVariants.addFedPlan(new FedPlan(cumulativeCost[1], fOutFedPlanVariants, planChilds));
		 }
	 }
 
	 /**
	  * Enumerates federated execution plans for a TRead/TWrite hop.
	  * This method calculates the cumulative costs for both LOUT and FOUT federated output types
	  * considering that TRead/TWrite hops have only one child (TWrite/Child of TWrite).
	  * Since TRead, TWrite and Child of TWrite have the same federated output type, it generates only
	  * a single plan for each output type.
	  *
	  * @param lOutFedPlanVariants The FedPlanVariants object for LOUT output type.
	  * @param fOutFedPlanVariants The FedPlanVariants object for FOUT output type.
	  * @param numInitInputs The number of initial input hops.
	  * @param numInputs The total number of input hops, including additional TWrite hops.
	  * @param childHops The list of child hops.
	  * @param childCumulativeCost The cumulative costs for each child hop.
	  * @param selfCost The self cost of the current hop.
	  */
	 private static void enumerateTransChildFedPlan(FedPlanVariants lOutFedPlanVariants, FedPlanVariants fOutFedPlanVariants,
					 int numInitInputs, int numInputs, List<Hop> childHops, 
					 double[][] childCumulativeCost, double selfCost){
	 
		 double[] cumulativeCost = new double[]{selfCost, selfCost};
		 List<Pair<Long, FederatedOutput>> lOutTransPlanChilds = new ArrayList<>();
		 List<Pair<Long, FederatedOutput>> fOutTransPlanChilds = new ArrayList<>();
					
		 for (int i =0; i < numInputs; i++){
			 Hop inputHop = childHops.get(i);
			 
			 lOutTransPlanChilds.add(Pair.of(inputHop.getHopID(), FederatedOutput.LOUT));
			 fOutTransPlanChilds.add(Pair.of(inputHop.getHopID(), FederatedOutput.FOUT));

			 cumulativeCost[0] = selfCost + childCumulativeCost[0][0];
			 cumulativeCost[1] = selfCost + childCumulativeCost[0][1];
		 }
		 
		 // Generate only a single plan for each output type as "TRead, TWrite and Child of TWrite" have the same FedOutType
		 lOutFedPlanVariants.addFedPlan(new FedPlan(cumulativeCost[0], lOutFedPlanVariants, lOutTransPlanChilds));
		 fOutFedPlanVariants.addFedPlan(new FedPlan(cumulativeCost[1], fOutFedPlanVariants, fOutTransPlanChilds));
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
 