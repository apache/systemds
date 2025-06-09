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
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOpDG;
import org.apache.sysds.common.Types.OpOpN;
import org.apache.sysds.common.Types.ParamBuiltinOp;
import org.apache.sysds.hops.DataGenOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.DnnOp;
import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.ParameterizedBuiltinOp;
import org.apache.sysds.hops.FunctionOp.FunctionType;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.NaryOp;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.hops.TernaryOp;
import org.apache.sysds.common.Types.OpOp3;
import org.apache.sysds.hops.fedplanner.FederatedMemoTable.HopCommon;
import org.apache.sysds.hops.fedplanner.FederatedMemoTable.FedPlan;
import org.apache.sysds.hops.fedplanner.FederatedMemoTable.FedPlanVariants;
import org.apache.sysds.common.Opcodes;
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
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.hops.fedplanner.FTypes.Privacy;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.fedplanner.FTypes.FType;

public class FederatedPlanCostEnumerator {
	/**
	 * Enumerates the entire DML program to generate federated execution plans.
	 * It processes each statement block, computes the optimal federated plan,
	 * detects and resolves conflicts, and optionally prints the plan tree.
	 *
	 * @param prog    The DML program to enumerate.
	 * @param isPrint A boolean indicating whether to print the federated plan tree.
	 */
	public static FedPlan enumerateProgram(DMLProgram prog, FederatedMemoTable memoTable, boolean isPrint) {
		Map<Long, List<Hop>> rewireTable = new HashMap<>();
		Set<Hop> progRootHopSet = new HashSet<>();
		Set<Long> unRefTwriteSet = new HashSet<>();
		Set<Long> unRefSet = new HashSet<>();
		Map<Long, HopCommon> hopCommonTable = new HashMap<>();

		Map<Long, Privacy> privacyConstraintMap = new HashMap<>();
		List<Pair<FederatedRange, FederatedData>> fedMap = new ArrayList<>();

		FederatedPlanRewireTransTable.rewireProgram(prog, rewireTable, hopCommonTable, privacyConstraintMap, fedMap,
				unRefTwriteSet, unRefSet, progRootHopSet);

		for (long hopID : unRefTwriteSet) {
			// Todo: 나중에 더 확인해야함.
			progRootHopSet.add(hopCommonTable.get(hopID).getHopRef());
		}
		Set<String> fnStack = new HashSet<>();

		for (StatementBlock sb : prog.getStatementBlocks()) {
			enumerateStatementBlock(sb, prog, memoTable, hopCommonTable, rewireTable, privacyConstraintMap,
					unRefTwriteSet,
					fnStack, fedMap.size());
		}

		FedPlan optimalPlan = getMinCostRootFedPlan(progRootHopSet, memoTable);

		// Detect conflicts in the federated plans where different FedPlans have
		// different FederatedOutput types
		double additionalTotalCost = detectAndResolveConflictFedPlan(optimalPlan, memoTable);

		unRefSet.addAll(unRefTwriteSet);
		// Print the federated plan tree if requested
		if (isPrint) {
			FederatedMemoTablePrinter.printFedPlanTree(optimalPlan, unRefSet, memoTable, additionalTotalCost);
		}

		return optimalPlan;
	}

	public static FedPlan enumerateFunctionDynamic(FunctionStatementBlock function, FederatedMemoTable memoTable,
			boolean isPrint) {
		Map<Long, List<Hop>> rewireTable = new HashMap<>();
		Set<Hop> progRootHopSet = new HashSet<>();
		Set<Long> unRefTwriteSet = new HashSet<>();
		Set<Long> unRefSet = new HashSet<>();
		Map<Long, HopCommon> hopCommonTable = new HashMap<>();

		Map<Long, Privacy> privacyConstraintMap = new HashMap<>();
		List<Pair<FederatedRange, FederatedData>> fedMap = new ArrayList<>();

		FederatedPlanRewireTransTable.rewireFunctionDynamic(function, rewireTable, hopCommonTable, privacyConstraintMap,
				fedMap, unRefTwriteSet, unRefSet, progRootHopSet);

		Set<String> fnStack = new HashSet<>();
		enumerateStatementBlock(function, null, memoTable, hopCommonTable, rewireTable, privacyConstraintMap,
				unRefTwriteSet, fnStack, fedMap.size());

		FedPlan optimalPlan = getMinCostRootFedPlan(progRootHopSet, memoTable);

		// Detect conflicts in the federated plans where different FedPlans have
		// different FederatedOutput types
		double additionalTotalCost = detectAndResolveConflictFedPlan(optimalPlan, memoTable);

		// Print the federated plan tree if requested
		if (isPrint) {
			FederatedMemoTablePrinter.printFedPlanTree(optimalPlan, unRefTwriteSet, memoTable, additionalTotalCost);
		}

		return optimalPlan;
	}

	/**
	 * Enumerates the statement block and updates the transient and memoization
	 * tables.
	 * This method processes different types of statement blocks such as If, For,
	 * While, and Function blocks.
	 * It recursively enumerates the Hop DAGs within these blocks and updates the
	 * corresponding tables.
	 * The method also calculates weights recursively for if-else/loops and handles
	 * inner and outer block distinctions.
	 *
	 * @param sb              The statement block to enumerate.
	 * @param memoTable       The memoization table to store plan variants.
	 * @param parentLoopStack The context of parent loops for loop-level context
	 *                        tracking.
	 * @return A map of inner transient writes.
	 */
	public static void enumerateStatementBlock(StatementBlock sb, DMLProgram prog, FederatedMemoTable memoTable,
			Map<Long, HopCommon> hopCommonTable, Map<Long, List<Hop>> rewireTable,
			Map<Long, Privacy> privacyConstraintMap,
			Set<Long> unRefTwriteSet, Set<String> fnStack, int numOfWorkers) {
		if (sb instanceof IfStatementBlock) {
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement istmt = (IfStatement) isb.getStatement(0);

			enumerateHopDAG(isb.getPredicateHops(), prog, memoTable, hopCommonTable, rewireTable, privacyConstraintMap,
					unRefTwriteSet, fnStack, numOfWorkers);

			for (StatementBlock innerIsb : istmt.getIfBody())
				enumerateStatementBlock(innerIsb, prog, memoTable, hopCommonTable, rewireTable, privacyConstraintMap,
						unRefTwriteSet, fnStack, numOfWorkers);

			for (StatementBlock innerIsb : istmt.getElseBody())
				enumerateStatementBlock(innerIsb, prog, memoTable, hopCommonTable, rewireTable, privacyConstraintMap,
						unRefTwriteSet, fnStack, numOfWorkers);
		} else if (sb instanceof ForStatementBlock) { // incl parfor
			ForStatementBlock fsb = (ForStatementBlock) sb;
			ForStatement fstmt = (ForStatement) fsb.getStatement(0);

			enumerateHopDAG(fsb.getFromHops(), prog, memoTable, hopCommonTable, rewireTable, privacyConstraintMap,
					unRefTwriteSet, fnStack, numOfWorkers);
			enumerateHopDAG(fsb.getToHops(), prog, memoTable, hopCommonTable, rewireTable, privacyConstraintMap,
					unRefTwriteSet, fnStack, numOfWorkers);
			if (fsb.getIncrementHops() != null) {
				enumerateHopDAG(fsb.getIncrementHops(), prog, memoTable, hopCommonTable, rewireTable,
						privacyConstraintMap,
						unRefTwriteSet, fnStack, numOfWorkers);
			}

			for (StatementBlock innerFsb : fstmt.getBody())
				enumerateStatementBlock(innerFsb, prog, memoTable, hopCommonTable, rewireTable, privacyConstraintMap,
						unRefTwriteSet, fnStack, numOfWorkers);
		} else if (sb instanceof WhileStatementBlock) {
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement wstmt = (WhileStatement) wsb.getStatement(0);

			enumerateHopDAG(wsb.getPredicateHops(), prog, memoTable, hopCommonTable, rewireTable, privacyConstraintMap,
					unRefTwriteSet, fnStack, numOfWorkers);

			for (StatementBlock innerWsb : wstmt.getBody())
				enumerateStatementBlock(innerWsb, prog, memoTable, hopCommonTable, rewireTable, privacyConstraintMap,
						unRefTwriteSet, fnStack, numOfWorkers);
		} else if (sb instanceof FunctionStatementBlock) {
			FunctionStatementBlock fsb = (FunctionStatementBlock) sb;
			FunctionStatement fstmt = (FunctionStatement) fsb.getStatement(0);

			for (StatementBlock innerFsb : fstmt.getBody())
				enumerateStatementBlock(innerFsb, prog, memoTable, hopCommonTable, rewireTable, privacyConstraintMap,
						unRefTwriteSet, fnStack, numOfWorkers);
		} else { // generic (last-level)
			if (sb.getHops() != null) {
				for (Hop c : sb.getHops())
					enumerateHopDAG(c, prog, memoTable, hopCommonTable, rewireTable, privacyConstraintMap,
							unRefTwriteSet,
							fnStack, numOfWorkers);
			}
		}
	}

	/**
	 * Rewires and enumerates federated execution plans for a given Hop.
	 * This method processes all input nodes, rewires TWrite and TRead operations,
	 * and generates federated plan variants for both inner and outer code blocks.
	 *
	 * @param hop       The Hop for which to rewire and enumerate federated plans.
	 * @param memoTable The memoization table to store plan variants.
	 * @param loopStack The context of parent loops for loop-level context tracking.
	 */
	private static void enumerateHopDAG(Hop hop, DMLProgram prog, FederatedMemoTable memoTable,
			Map<Long, HopCommon> hopCommonTable, Map<Long, List<Hop>> rewireTable,
			Map<Long, Privacy> privacyConstraintMap, Set<Long> unRefTwriteSet, Set<String> fnStack, int numOfWorkers) {
		// Process all input nodes first if not already in memo table
		for (Hop inputHop : hop.getInput()) {
			long inputHopID = inputHop.getHopID();
			if (!memoTable.contains(inputHopID, FederatedOutput.FOUT)
					&& !memoTable.contains(inputHopID, FederatedOutput.LOUT)) {
				enumerateHopDAG(inputHop, prog, memoTable, hopCommonTable, rewireTable, privacyConstraintMap,
						unRefTwriteSet, fnStack, numOfWorkers);
			}
		}

		if (hop instanceof FunctionOp) {
			// maintain counters and investigate functions if not seen so far
			FunctionOp fop = (FunctionOp) hop;
			if (fop.getFunctionType() == FunctionType.DML) {
				String fkey = fop.getFunctionKey();
				for (Hop inputHop : fop.getInput()) {
					fkey += "," + inputHop.getName();
				}

				if (!fnStack.contains(fkey)) {
					fnStack.add(fkey);
					FunctionStatementBlock fsb = prog.getFunctionStatementBlock(fop.getFunctionNamespace(),
							fop.getFunctionName());
					// Todo (Future): hop reconstruction을 안하면 memoTable 따로 써야함.
					enumerateStatementBlock(fsb, prog, memoTable, hopCommonTable, rewireTable, privacyConstraintMap,
							unRefTwriteSet, fnStack, numOfWorkers);
				}
			}
		}

		// Enumerate the federated plan for the current Hop
		enumerateHop(hop, memoTable, hopCommonTable, rewireTable, privacyConstraintMap, unRefTwriteSet, fnStack,
				numOfWorkers);
	}

	/**
	 * Enumerates federated execution plans for a given Hop.
	 * This method calculates the self cost and child costs for the Hop,
	 * generates federated plan variants for both LOUT and FOUT output types,
	 * and prunes redundant plans before adding them to the memo table.
	 *
	 * @param hop       The Hop for which to enumerate federated plans.
	 * @param memoTable The memoization table to store plan variants.
	 * @param loopStack The context of parent loops for loop-level context tracking.
	 */
	private static void enumerateHop(Hop hop, FederatedMemoTable memoTable, Map<Long, HopCommon> hopCommonTable,
			Map<Long, List<Hop>> rewireTable, Map<Long, Privacy> privacyConstraintMap,
			Set<Long> unRefTwriteSet, Set<String> fnStack, int numOfWorkers) {
		long hopID = hop.getHopID();
		List<Hop> childHops = new ArrayList<>(hop.getInput());
		int numParentHops = hop.getParent().size();
		boolean isTrans = false;

		if ((hop instanceof DataOp) && !hop.getName().equals("__pred")
				&& !(((DataOp) hop).getOp() == Types.OpOpData.PERSISTENTWRITE)) {
			Types.OpOpData opType = ((DataOp) hop).getOp();
			if (opType == Types.OpOpData.TRANSIENTWRITE) {
				List<Hop> transParentHops = rewireTable.get(hop.getHopID());
				if (transParentHops != null) {
					numParentHops += transParentHops.size();
					isTrans = true;
				}
			} else if (opType == Types.OpOpData.TRANSIENTREAD) {
				List<Hop> transChildHops = rewireTable.get(hop.getHopID());
				if (transChildHops != null) {
					childHops.addAll(transChildHops);
				}
				isTrans = true;
			}
		} else {
			for (Hop parentHop : hop.getParent()) {
				if (parentHop instanceof DataOp
						&& ((DataOp) parentHop).getOp() == Types.OpOpData.TRANSIENTWRITE
						&& !parentHop.getName().equals("__pred")
						&& unRefTwriteSet.contains(parentHop.getHopID())) {
					numParentHops--;
				}
			}
		}
		HopCommon hopCommon = hopCommonTable.get(hopID);
		hopCommon.setNumOfParentHops(numParentHops);
		double selfCost = FederatedPlanCostEstimator.computeHopCost(hopCommon);

		FedPlanVariants lOutFedPlanVariants = new FedPlanVariants(hopCommon, FederatedOutput.LOUT);
		FedPlanVariants fOutFedPlanVariants = new FedPlanVariants(hopCommon, FederatedOutput.FOUT);

		int numInputs = childHops.size();

		double[][] childCumulativeCost = new double[numInputs][2]; // # of child, LOUT/FOUT of child
		double[] childForwardingCost = new double[numInputs]; // # of child

		List<Hop> lOUTOnlyinputHops = new ArrayList<>();
		List<Double> lOUTOnlychildCumulativeCost = new ArrayList<>();
		List<Double> lOUTOnlychildForwardingCost = new ArrayList<>();

		List<Hop> fOUTOnlyinputHops = new ArrayList<>();
		List<Double> fOUTOnlychildCumulativeCost = new ArrayList<>();
		List<Double> fOUTOnlychildForwardingCost = new ArrayList<>();

		// The self cost follows its own weight, while the forwarding cost follows the
		// parent's weight.
		FederatedPlanCostEstimator.getChildCosts(hopCommon, memoTable, childHops, childCumulativeCost,
				childForwardingCost, lOUTOnlyinputHops, lOUTOnlychildCumulativeCost, lOUTOnlychildForwardingCost,
				fOUTOnlyinputHops, fOUTOnlychildCumulativeCost, fOUTOnlychildForwardingCost);

		Privacy privacyConstraint = privacyConstraintMap.get(hopID);

		if (isTrans) {
			// Todo 뭔가 Trans에서 안 꼬이나 확인해야함.
			enumerateTransChildFedPlan(lOutFedPlanVariants, fOutFedPlanVariants, childHops, childCumulativeCost,
					lOUTOnlyinputHops, lOUTOnlychildCumulativeCost, fOUTOnlyinputHops, fOUTOnlychildCumulativeCost,
					selfCost, numOfWorkers);

			lOutFedPlanVariants.pruneFedPlans();
			fOutFedPlanVariants.pruneFedPlans();

			memoTable.addFedPlanVariants(hopID, FederatedOutput.FOUT, fOutFedPlanVariants);
			memoTable.addFedPlanVariants(hopID, FederatedOutput.LOUT, lOutFedPlanVariants);
		} else {
			if (isLocalForcedHop(hop)) {
				singleTypeEnumerateChildFedPlan(lOutFedPlanVariants, FederatedOutput.LOUT,
						childHops, childCumulativeCost, childForwardingCost, lOUTOnlyinputHops,
						lOUTOnlychildCumulativeCost, lOUTOnlychildForwardingCost, fOUTOnlyinputHops,
						fOUTOnlychildCumulativeCost, fOUTOnlychildForwardingCost, selfCost, numOfWorkers);

				lOutFedPlanVariants.pruneFedPlans();
				memoTable.addFedPlanVariants(hopID, FederatedOutput.LOUT, lOutFedPlanVariants);
			} else if (privacyConstraint == Privacy.PRIVATE || privacyConstraint == Privacy.PRIVATE_AGGREGATE) {
				singleTypeEnumerateChildFedPlan(fOutFedPlanVariants, FederatedOutput.FOUT, childHops,
						childCumulativeCost, childForwardingCost, lOUTOnlyinputHops, lOUTOnlychildCumulativeCost,
						lOUTOnlychildForwardingCost, fOUTOnlyinputHops, fOUTOnlychildCumulativeCost,
						fOUTOnlychildForwardingCost, selfCost, numOfWorkers);

				fOutFedPlanVariants.pruneFedPlans();
				memoTable.addFedPlanVariants(hopID, FederatedOutput.FOUT, fOutFedPlanVariants);
			} else {
				enumerateChildFedPlan(lOutFedPlanVariants, fOutFedPlanVariants, childHops, childCumulativeCost,
						childForwardingCost, lOUTOnlyinputHops, lOUTOnlychildCumulativeCost,
						lOUTOnlychildForwardingCost,
						fOUTOnlyinputHops, fOUTOnlychildCumulativeCost, fOUTOnlychildForwardingCost, selfCost,
						numOfWorkers);

				lOutFedPlanVariants.pruneFedPlans();
				fOutFedPlanVariants.pruneFedPlans();

				memoTable.addFedPlanVariants(hopID, FederatedOutput.LOUT, lOutFedPlanVariants);
				memoTable.addFedPlanVariants(hopID, FederatedOutput.FOUT, fOutFedPlanVariants);
			}
		}
	}

	/**
	 * Enumerates federated execution plans for initial child hops only.
	 * This method generates all possible combinations of federated output types
	 * (LOUT and FOUT)
	 * for the initial child hops and calculates their cumulative costs.
	 *
	 * @param lOutFedPlanVariants The FedPlanVariants object for LOUT output type.
	 * @param fOutFedPlanVariants The FedPlanVariants object for FOUT output type.
	 * @param childHops           The list of child hops.
	 * @param childCumulativeCost The cumulative costs for each child hop.
	 * @param childForwardingCost The forwarding costs for each child hop.
	 * @param selfCost            The self cost of the current hop.
	 */
	private static void enumerateChildFedPlan(FedPlanVariants lOutFedPlanVariants, FedPlanVariants fOutFedPlanVariants,
			List<Hop> childHops, double[][] childCumulativeCost, double[] childForwardingCost,
			List<Hop> lOUTOnlyinputHops, List<Double> lOUTOnlychildCumulativeCost,
			List<Double> lOUTOnlychildForwardingCost,
			List<Hop> fOUTOnlyinputHops, List<Double> fOUTOnlychildCumulativeCost,
			List<Double> fOUTOnlychildForwardingCost,
			double selfCost, int numOfWorkers) {
		// Iterate 2^n times, generating two FedPlans (LOUT, FOUT) each time.
		int numInputs = childHops.size();
		int numLoutOnlyInputs = lOUTOnlyinputHops.size();
		int numFoutOnlyInputs = fOUTOnlyinputHops.size();

		for (int i = 0; i < (1 << numInputs); i++) {
			double[] cumulativeCost = new double[] { selfCost, selfCost / numOfWorkers };
			List<Pair<Long, FederatedOutput>> planChilds = new ArrayList<>();

			// LOUT and FOUT share the same planChilds in each iteration (only forwarding
			// cost differs).
			for (int j = 0; j < numInputs; j++) {
				Hop inputHop = childHops.get(j);
				// Calculate the bit value to decide between FOUT and LOUT for the current input
				final int bit = (i & (1 << j)) != 0 ? 1 : 0; // Determine the bit value (decides FOUT/LOUT)
				final FederatedOutput childType = (bit == 1) ? FederatedOutput.FOUT : FederatedOutput.LOUT;
				planChilds.add(Pair.of(inputHop.getHopID(), childType));

				// Update the cumulative cost for LOUT, FOUT
				cumulativeCost[0] += childCumulativeCost[j][bit] + childForwardingCost[j] * bit;
				cumulativeCost[1] += childCumulativeCost[j][bit] + childForwardingCost[j] * (1 - bit);
			}

			for (int j = 0; j < numLoutOnlyInputs; j++) {
				Hop inputHop = lOUTOnlyinputHops.get(j);
				planChilds.add(Pair.of(inputHop.getHopID(), FederatedOutput.LOUT));
				// Update the cumulative cost for LOUT, FOUT
				cumulativeCost[0] += lOUTOnlychildCumulativeCost.get(j);
				cumulativeCost[1] += lOUTOnlychildCumulativeCost.get(j) + lOUTOnlychildForwardingCost.get(j);
			}

			for (int j = 0; j < numFoutOnlyInputs; j++) {
				Hop inputHop = fOUTOnlyinputHops.get(j);
				planChilds.add(Pair.of(inputHop.getHopID(), FederatedOutput.FOUT));
				// Update the cumulative cost for LOUT, FOUT
				cumulativeCost[0] += fOUTOnlychildCumulativeCost.get(j) + fOUTOnlychildForwardingCost.get(j);
				cumulativeCost[1] += fOUTOnlychildCumulativeCost.get(j);
			}

			lOutFedPlanVariants.addFedPlan(new FedPlan(cumulativeCost[0], lOutFedPlanVariants, planChilds));
			fOutFedPlanVariants.addFedPlan(new FedPlan(cumulativeCost[1], fOutFedPlanVariants, planChilds));
		}
	}

	private static void singleTypeEnumerateChildFedPlan(FedPlanVariants fedPlanVariants, FederatedOutput fedOutType,
			List<Hop> childHops, double[][] childCumulativeCost, double[] childForwardingCost,
			List<Hop> lOUTOnlyinputHops, List<Double> lOUTOnlychildCumulativeCost,
			List<Double> lOUTOnlychildForwardingCost,
			List<Hop> fOUTOnlyinputHops, List<Double> fOUTOnlychildCumulativeCost,
			List<Double> fOUTOnlychildForwardingCost, double selfCost, int numOfWorkers) {
		// Iterate 2^n times, generating two FedPlans (LOUT, FOUT) each time.
		int numInputs = childHops.size();
		int numLoutOnlyInputs = lOUTOnlyinputHops.size();
		int numFoutOnlyInputs = fOUTOnlyinputHops.size();

		for (int i = 0; i < (1 << numInputs); i++) {
			double cumulativeCost = fedOutType == FederatedOutput.LOUT ? selfCost : selfCost / numOfWorkers;
			List<Pair<Long, FederatedOutput>> planChilds = new ArrayList<>();

			// LOUT and FOUT share the same planChilds in each iteration (only forwarding
			// cost differs).
			for (int j = 0; j < numInputs; j++) {
				Hop inputHop = childHops.get(j);
				// Calculate the bit value to decide between FOUT and LOUT for the current input
				final int bit = (i & (1 << j)) != 0 ? 1 : 0; // Determine the bit value (decides FOUT/LOUT)
				final FederatedOutput childType = (bit == 1) ? FederatedOutput.FOUT : FederatedOutput.LOUT;
				planChilds.add(Pair.of(inputHop.getHopID(), childType));

				// Update the cumulative cost for LOUT, FOUT
				// LOUT
				cumulativeCost += childCumulativeCost[j][bit];
				cumulativeCost += fedOutType == FederatedOutput.LOUT ? childForwardingCost[j] * (bit)
						: childForwardingCost[j] * (1 - bit);
			}

			for (int j = 0; j < numLoutOnlyInputs; j++) {
				Hop inputHop = lOUTOnlyinputHops.get(j);
				planChilds.add(Pair.of(inputHop.getHopID(), FederatedOutput.LOUT));
				// Update the cumulative cost for LOUT, FOUT
				cumulativeCost += lOUTOnlychildCumulativeCost.get(j);
				cumulativeCost += fedOutType == FederatedOutput.LOUT ? 0 : lOUTOnlychildForwardingCost.get(j);
			}

			for (int j = 0; j < numFoutOnlyInputs; j++) {
				Hop inputHop = fOUTOnlyinputHops.get(j);
				planChilds.add(Pair.of(inputHop.getHopID(), FederatedOutput.FOUT));
				// Update the cumulative cost for LOUT, FOUT
				cumulativeCost += fOUTOnlychildCumulativeCost.get(j);
				cumulativeCost += fedOutType == FederatedOutput.LOUT ? fOUTOnlychildForwardingCost.get(j) : 0;
			}

			fedPlanVariants.addFedPlan(new FedPlan(cumulativeCost, fedPlanVariants, planChilds));
		}
	}

	/**
	 * Enumerates federated execution plans for a TRead/TWrite hop.
	 * This method calculates the cumulative costs for both LOUT and FOUT federated
	 * output types
	 * considering that TRead/TWrite hops have only one child (TWrite/Child of
	 * TWrite).
	 * Since TRead, TWrite and Child of TWrite have the same federated output type,
	 * it generates only
	 * a single plan for each output type
	 * 
	 * @param lOutFedPlanVariants The FedPlanVariants object for LOUT output type.
	 * @param fOutFedPlanVariants The FedPlanVariants object for FOUT output type.
	 * @param numInputs           The total number of input hops, including
	 *                            additional TWrite hops.
	 * @param childHops           The list of child hops.
	 * @param childCumulativeCost The cumulative costs for each child hop.
	 * @param selfCost            The self cost of the current hop.
	 */
	private static void enumerateTransChildFedPlan(FedPlanVariants lOutFedPlanVariants,
			FedPlanVariants fOutFedPlanVariants,
			List<Hop> childHops, double[][] childCumulativeCost,
			List<Hop> lOUTOnlyinputHops, List<Double> lOUTOnlychildCumulativeCost,
			List<Hop> fOUTOnlyinputHops, List<Double> fOUTOnlychildCumulativeCost,
			double selfCost, int numOfWorkers) {

		int numInputs = childHops.size();
		int numLoutOnlyInputs = lOUTOnlyinputHops.size();
		int numFoutOnlyInputs = fOUTOnlyinputHops.size();

		if (numLoutOnlyInputs > 0 && numFoutOnlyInputs > 0) {
			// Todo: 왜 이렇게 했는지 확인하고 해결해야 함.
			System.out.println("=== LOUT Only Input Hops ===");
			for (Hop hop : lOUTOnlyinputHops) {
				System.out.println("Name: " + hop.getName() + ", ID: " + hop.getHopID());
			}
			System.out.println("=== FOUT Only Input Hops ===");
			for (Hop hop : fOUTOnlyinputHops) {
				System.out.println("Name: " + hop.getName() + ", ID: " + hop.getHopID());
			}
		}

		if (numLoutOnlyInputs > 0) {
			double lOUTcumulativeCost = selfCost;
			List<Pair<Long, FederatedOutput>> lOutTransPlanChilds = new ArrayList<>();

			for (int i = 0; i < numInputs; i++) {
				Hop inputHop = childHops.get(i);
				lOutTransPlanChilds.add(Pair.of(inputHop.getHopID(), FederatedOutput.LOUT));
				lOUTcumulativeCost += childCumulativeCost[i][0];
			}

			for (int j = 0; j < numLoutOnlyInputs; j++) {
				Hop inputHop = lOUTOnlyinputHops.get(j);
				lOutTransPlanChilds.add(Pair.of(inputHop.getHopID(), FederatedOutput.LOUT));
				lOUTcumulativeCost += lOUTOnlychildCumulativeCost.get(j);
			}
			// Generate only a single plan for each output type as "TRead, TWrite and Child
			// of TWrite" have the same FedOutType
			lOutFedPlanVariants.addFedPlan(new FedPlan(lOUTcumulativeCost, lOutFedPlanVariants, lOutTransPlanChilds));
			return;
		}

		if (numFoutOnlyInputs > 0) {
			double fOUTcumulativeCost = selfCost / numOfWorkers;
			List<Pair<Long, FederatedOutput>> fOutTransPlanChilds = new ArrayList<>();

			for (int i = 0; i < numInputs; i++) {
				Hop inputHop = childHops.get(i);
				fOutTransPlanChilds.add(Pair.of(inputHop.getHopID(), FederatedOutput.FOUT));
				fOUTcumulativeCost += childCumulativeCost[i][1];
			}

			for (int j = 0; j < numFoutOnlyInputs; j++) {
				Hop inputHop = fOUTOnlyinputHops.get(j);
				fOutTransPlanChilds.add(Pair.of(inputHop.getHopID(), FederatedOutput.FOUT));
				fOUTcumulativeCost += fOUTOnlychildCumulativeCost.get(j);
			}
			// Generate only a single plan for each output type as "TRead, TWrite and Child
			// of TWrite" have the same FedOutType
			fOutFedPlanVariants.addFedPlan(new FedPlan(fOUTcumulativeCost, fOutFedPlanVariants, fOutTransPlanChilds));
			return;
		}

		double[] cumulativeCost = new double[] { selfCost, selfCost / numOfWorkers };
		List<Pair<Long, FederatedOutput>> lOutTransPlanChilds = new ArrayList<>();
		List<Pair<Long, FederatedOutput>> fOutTransPlanChilds = new ArrayList<>();

		for (int i = 0; i < numInputs; i++) {
			Hop inputHop = childHops.get(i);

			lOutTransPlanChilds.add(Pair.of(inputHop.getHopID(), FederatedOutput.LOUT));
			fOutTransPlanChilds.add(Pair.of(inputHop.getHopID(), FederatedOutput.FOUT));

			cumulativeCost[0] += childCumulativeCost[i][0];
			cumulativeCost[1] += childCumulativeCost[i][1];
		}

		// Generate only a single plan for each output type as "TRead, TWrite and Child
		// of TWrite" have the same FedOutType
		lOutFedPlanVariants.addFedPlan(new FedPlan(cumulativeCost[0], lOutFedPlanVariants, lOutTransPlanChilds));
		fOutFedPlanVariants.addFedPlan(new FedPlan(cumulativeCost[1], fOutFedPlanVariants, fOutTransPlanChilds));
	}

	// Creates a dummy root node (fedplan) and selects the FedPlan with the minimum
	// cost to return.
	// The dummy root node does not have LOUT or FOUT.
	private static FedPlan getMinCostRootFedPlan(Set<Hop> progRootHopSet, FederatedMemoTable memoTable) {
		double cumulativeCost = 0;
		List<Pair<Long, FederatedOutput>> rootFedPlanChilds = new ArrayList<>();

		// Iterate over each Hop in the progRootHopSet
		for (Hop endHop : progRootHopSet) {
			// Retrieve the pruned FedPlan for LOUT and FOUT from the memo table
			FedPlan lOutFedPlan = memoTable.getFedPlanAfterPrune(endHop.getHopID(), FederatedOutput.LOUT);
			FedPlan fOutFedPlan = memoTable.getFedPlanAfterPrune(endHop.getHopID(), FederatedOutput.FOUT);

			if (fOutFedPlan == null) {
				cumulativeCost += lOutFedPlan.getCumulativeCost();
				rootFedPlanChilds.add(Pair.of(endHop.getHopID(), FederatedOutput.LOUT));
			} else if (lOutFedPlan == null) {
				cumulativeCost += fOutFedPlan.getCumulativeCost();
				rootFedPlanChilds.add(Pair.of(endHop.getHopID(), FederatedOutput.FOUT));
			} else {
				// Compare the cumulative costs of LOUT and FOUT FedPlans
				if (lOutFedPlan.getCumulativeCost() <= fOutFedPlan.getCumulativeCost()) {
					cumulativeCost += lOutFedPlan.getCumulativeCost();
					rootFedPlanChilds.add(Pair.of(endHop.getHopID(), FederatedOutput.LOUT));
				} else {
					cumulativeCost += fOutFedPlan.getCumulativeCost();
					rootFedPlanChilds.add(Pair.of(endHop.getHopID(), FederatedOutput.FOUT));
				}
			}
		}

		return new FedPlan(cumulativeCost, null, rootFedPlanChilds);
	}

	private static boolean isLocalForcedHop(Hop hop) {
//		if (hop instanceof AggUnaryOp) {
//			AggUnaryOp aggUnaryOp = (AggUnaryOp) hop;
//
//			// 1) 입력이 Federated 인지 확인 (빠른 체크)
//			Hop in = hop.getInput().get(0);
//			if (!in.isFederated())          // 시스템 DS 1.3+  : Hop 에 내장
//				return false;
//
//			// 2) 집계 방향이 분할 축과 충돌하는지 확인
//			boolean isColAgg = aggUnaryOp.getDirection().isCol();
//			if ( (in.getFederatedOutputType() == FType.ROW && isColAgg) ||
//				 (in.getFederatedOutputType() == FType.COL && !isColAgg) )
//				return true;                // 여기서 CP 로 강제
//		}
		
		// DnnOp -> all local
		if (hop instanceof DnnOp) {
			return true;
		}
		// FunctionOp: all local, except for transformencode
		else if (hop instanceof FunctionOp) {
			FunctionOp fop = (FunctionOp) hop;
			return fop.getFunctionName().equalsIgnoreCase(Opcodes.TRANSFORMENCODE.toString());
		}
		// NaryOp operations
		else if (hop instanceof NaryOp) {
			OpOpN op = ((NaryOp) hop).getOp();
			return op == OpOpN.PRINTF || op == OpOpN.EVAL || op == OpOpN.LIST
			// cbind/rbind of lists only support in CP right now
					|| (op == OpOpN.CBIND && hop.getInput().get(0).getDataType().isList())
					|| (op == OpOpN.RBIND && hop.getInput().get(0).getDataType().isList());
		}
		// ParameterizedBuiltin operations
		else if (hop instanceof ParameterizedBuiltinOp) {
			ParamBuiltinOp op = ((ParameterizedBuiltinOp) hop).getOp();
			return op == ParamBuiltinOp.TOSTRING || op == ParamBuiltinOp.LIST
					|| op == ParamBuiltinOp.CDF || op == ParamBuiltinOp.INVCDF
					|| op == ParamBuiltinOp.PARAMSERV || op == ParamBuiltinOp.REXPAND
					|| op == ParamBuiltinOp.REPLACE;
		}
		// UnaryOp operations
		else if (hop instanceof UnaryOp) {
			UnaryOp uop = (UnaryOp) hop;
			OpOp1 op = uop.getOp();
			return op == OpOp1.PRINT || op == OpOp1.ASSERT || op == OpOp1.STOP
					|| op == OpOp1.TYPEOF || op == OpOp1.INVERSE || op == OpOp1.EIGEN
					|| op == OpOp1.CHOLESKY || op == OpOp1.DET || op == OpOp1.SVD
					|| op == OpOp1.SQRT_MATRIX_JAVA || op == OpOp1.LOG || op == OpOp1.ROUND
					|| hop.getInput().get(0).getDataType() == DataType.LIST
					|| uop.isMetadataOperation();
		}
		// DataGenOp operations
		else if (hop instanceof DataGenOp) {
			OpOpDG op = ((DataGenOp) hop).getOp();
			return op == OpOpDG.TIME || op == OpOpDG.SINIT || op == OpOpDG.RAND || op == OpOpDG.SEQ;
		} else if (hop instanceof TernaryOp) {
			OpOp3 op = ((TernaryOp) hop).getOp();
			return op == OpOp3.CTABLE || op == OpOp3.IFELSE;
		} else if (hop instanceof BinaryOp) {
			OpOp2 op = ((BinaryOp) hop).getOp();
			return op == OpOp2.MIN;
		}
		return false;
	}

	/**
	 * Detects and resolves conflicts in federated plans starting from the root
	 * plan.
	 * This function performs a breadth-first search (BFS) to traverse the federated
	 * plan tree.
	 * It identifies conflicts where the same plan ID has different federated output
	 * types.
	 * For each conflict, it records the plan ID and its conflicting parent plans.
	 * The function ensures that each plan ID is associated with a consistent
	 * federated output type
	 * by resolving these conflicts iteratively.
	 *
	 * The process involves:
	 * - Using a map to track conflicts, associating each plan ID with its federated
	 * output type
	 * and a list of parent plans.
	 * - Storing detected conflicts in a linked map, each entry containing a plan ID
	 * and its
	 * conflicting parent plans.
	 * - Performing BFS traversal starting from the root plan, checking each child
	 * plan for conflicts.
	 * - If a conflict is detected (i.e., a plan ID has different output types), the
	 * conflicting plan
	 * is removed from the BFS queue and added to the conflict map to prevent
	 * duplicate calculations.
	 * - Resolving conflicts by ensuring a consistent federated output type across
	 * the plan.
	 * - Re-running BFS with resolved conflicts to ensure all inconsistencies are
	 * addressed.
	 *
	 * @param rootPlan  The root federated plan from which to start the conflict
	 *                  detection.
	 * @param memoTable The memoization table used to retrieve pruned federated
	 *                  plans.
	 * @return The cumulative additional cost for resolving conflicts.
	 */
	private static double detectAndResolveConflictFedPlan(FedPlan rootPlan, FederatedMemoTable memoTable) {
		// Map to track conflicts: maps a plan ID to its federated output type and list
		// of parent plans
		Map<Long, Pair<FederatedOutput, List<FedPlan>>> conflictCheckMap = new HashMap<>();

		// LinkedMap to store detected conflicts, each with a plan ID and its
		// conflicting parent plans
		LinkedHashMap<Long, List<FedPlan>> conflictLinkedMap = new LinkedHashMap<>();

		// LinkedMap for BFS traversal starting from the root plan (Do not use value
		// (boolean))
		LinkedHashMap<FedPlan, Boolean> bfsLinkedMap = new LinkedHashMap<>();
		bfsLinkedMap.put(rootPlan, true);

		// Array to store cumulative additional cost for resolving conflicts
		double[] cumulativeAdditionalCost = new double[] { 0.0 };

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
						Pair<FederatedOutput, List<FedPlan>> conflictChildPlanPair = conflictCheckMap
								.get(childPlanPair.getLeft());
						// Add the current plan to the list of parent plans
						conflictChildPlanPair.getRight().add(currentPlan);

						// If the federated output type differs, a conflict is detected
						if (conflictChildPlanPair.getLeft() != childPlanPair.getRight()) {
							// If this is the first detection, remove conflictChildFedPlan from the BFS
							// queue and add it to the conflict linked map (queue)
							// If the existing FedPlan is not removed from the bfsqueue or both actions are
							// performed, duplicate calculations for the same FedPlan and its children occur
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
						conflictCheckMap.put(childPlanPair.getLeft(),
								new ImmutablePair<>(childPlanPair.getRight(), parentFedPlanList));
						// Add the child plan to the BFS queue
						bfsLinkedMap.put(childFedPlan, true);
					}
				}
			}
			// Resolve these conflicts to ensure a consistent federated output type across
			// the plan
			// Re-run BFS with resolved conflicts
			bfsLinkedMap = FederatedPlanCostEstimator.resolveConflictFedPlan(memoTable, conflictLinkedMap,
					cumulativeAdditionalCost);
			conflictLinkedMap.clear();
		}

		// Return the cumulative additional cost for resolving conflicts
		return cumulativeAdditionalCost[0];
	}
}
