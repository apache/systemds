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
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.fedplanner.FTypes.FType;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.ReorgOp;
import org.apache.sysds.hops.TernaryOp;
import org.apache.sysds.hops.cost.HopRel;
import org.apache.sysds.hops.ipa.FunctionCallGraph;
import org.apache.sysds.hops.ipa.FunctionCallSizeInfo;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.FunctionStatement;
import org.apache.sysds.parser.FunctionStatementBlock;
import org.apache.sysds.parser.IfStatement;
import org.apache.sysds.parser.IfStatementBlock;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.WhileStatement;
import org.apache.sysds.parser.WhileStatementBlock;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction;

public class FederatedPlannerCostbased extends AFederatedPlanner {
	private static final Log LOG = LogFactory.getLog(FederatedPlannerCostbased.class.getName());

	private final static MemoTable hopRelMemo = new MemoTable();
	/**
	 * IDs of hops for which the final fedout value has been set.
	 */
	private final static Set<Long> hopRelUpdatedFinal = new HashSet<>();
	/**
	 * Terminal hops in DML program given to this rewriter.
	 */
	private final static List<Hop> terminalHops = new ArrayList<>();
	private final static Map<String, Hop> transientWrites = new HashMap<>();
	Map<Long, FTypes.FType> memo = new HashMap<>();
	Map<String, FTypes.FType> fedVars = new HashMap<>();

	public List<Hop> getTerminalHops(){
		return terminalHops;
	}

	@Override
	public void rewriteProgram( DMLProgram prog, FunctionCallGraph fgraph, FunctionCallSizeInfo fcallSizes ) {
		prog.updateRepetitionEstimates();
		rewriteStatementBlocks(prog, prog.getStatementBlocks());
		setFinalFedouts();
	}
	
	/**
	 * Estimates cost and enumerates federated execution plans in hopRelMemo.
	 * The method calls the contained statement blocks recursively.
	 *
	 * @param prog dml program
	 * @param sbs  list of statement blocks
	 * @return list of statement blocks with the federated output value updated for each hop
	 */
	private ArrayList<StatementBlock> rewriteStatementBlocks(DMLProgram prog, List<StatementBlock> sbs) {
		ArrayList<StatementBlock> rewrittenStmBlocks = new ArrayList<>();
		for(StatementBlock stmBlock : sbs)
			rewrittenStmBlocks.addAll(rewriteStatementBlock(prog, stmBlock));
		return rewrittenStmBlocks;
	}

	/**
	 * Estimates cost and enumerates federated execution plans in hopRelMemo.
	 * The method calls the contained statement blocks recursively.
	 *
	 * @param prog dml program
	 * @param sb   statement block
	 * @return list of statement blocks with the federated output value updated for each hop
	 */
	public ArrayList<StatementBlock> rewriteStatementBlock(DMLProgram prog, StatementBlock sb) {
		if(sb instanceof WhileStatementBlock)
			return rewriteWhileStatementBlock(prog, (WhileStatementBlock) sb);
		else if(sb instanceof IfStatementBlock)
			return rewriteIfStatementBlock(prog, (IfStatementBlock) sb);
		else if(sb instanceof ForStatementBlock) {
			// This also includes ParForStatementBlocks
			return rewriteForStatementBlock(prog, (ForStatementBlock) sb);
		}
		else if(sb instanceof FunctionStatementBlock)
			return rewriteFunctionStatementBlock(prog, (FunctionStatementBlock) sb);
		else {
			// StatementBlock type (no subclass)
			return rewriteDefaultStatementBlock(prog, sb);
		}
	}

	private ArrayList<StatementBlock> rewriteWhileStatementBlock(DMLProgram prog, WhileStatementBlock whileSB) {
		Hop whilePredicateHop = whileSB.getPredicateHops();
		selectFederatedExecutionPlan(whilePredicateHop);
		for(Statement stm : whileSB.getStatements()) {
			WhileStatement whileStm = (WhileStatement) stm;
			whileStm.setBody(rewriteStatementBlocks(prog, whileStm.getBody()));
		}
		return new ArrayList<>(Collections.singletonList(whileSB));
	}

	private ArrayList<StatementBlock> rewriteIfStatementBlock(DMLProgram prog, IfStatementBlock ifSB) {
		selectFederatedExecutionPlan(ifSB.getPredicateHops());
		for(Statement statement : ifSB.getStatements()) {
			IfStatement ifStatement = (IfStatement) statement;
			ifStatement.setIfBody(rewriteStatementBlocks(prog, ifStatement.getIfBody()));
			ifStatement.setElseBody(rewriteStatementBlocks(prog, ifStatement.getElseBody()));
		}
		return new ArrayList<>(Collections.singletonList(ifSB));
	}

	private ArrayList<StatementBlock> rewriteForStatementBlock(DMLProgram prog, ForStatementBlock forSB) {
		selectFederatedExecutionPlan(forSB.getFromHops());
		selectFederatedExecutionPlan(forSB.getToHops());
		selectFederatedExecutionPlan(forSB.getIncrementHops());
		for(Statement statement : forSB.getStatements()) {
			ForStatement forStatement = ((ForStatement) statement);
			forStatement.setBody(rewriteStatementBlocks(prog, forStatement.getBody()));
		}
		return new ArrayList<>(Collections.singletonList(forSB));
	}

	private ArrayList<StatementBlock> rewriteFunctionStatementBlock(DMLProgram prog, FunctionStatementBlock funcSB) {
		for(Statement statement : funcSB.getStatements()) {
			FunctionStatement funcStm = (FunctionStatement) statement;
			funcStm.setBody(rewriteStatementBlocks(prog, funcStm.getBody()));
		}
		return new ArrayList<>(Collections.singletonList(funcSB));
	}

	private ArrayList<StatementBlock> rewriteDefaultStatementBlock(DMLProgram prog, StatementBlock sb) {
		if(sb.hasHops()) {
			for(Hop sbHop : sb.getHops()) {
				if(sbHop instanceof FunctionOp) {
					String funcName = ((FunctionOp) sbHop).getFunctionName();
					FunctionStatementBlock sbFuncBlock = prog.getBuiltinFunctionDictionary().getFunction(funcName);
					rewriteStatementBlock(prog, sbFuncBlock);
				}
				else
					selectFederatedExecutionPlan(sbHop);
			}
		}
		return new ArrayList<>(Collections.singletonList(sb));
	}

	/**
	 * Set final fedouts of all hops starting from terminal hops.
	 */
	private void setFinalFedouts(){
		for ( Hop root : terminalHops)
			setFinalFedout(root);
	}

	/**
	 * Sets FederatedOutput field of all hops in DAG starting from given root.
	 * The FederatedOutput chosen for root is the minimum cost HopRel found in memo table for the given root.
	 * The FederatedOutput values chosen for the inputs to the root are chosen based on the input dependencies.
	 *
	 * @param root hop for which FederatedOutput needs to be set
	 */
	private void setFinalFedout(Hop root) {
		HopRel optimalRootHopRel = hopRelMemo.getMinCostAlternative(root);
		setFinalFedout(root, optimalRootHopRel);
	}

	/**
	 * Update the FederatedOutput value and cost based on information stored in given rootHopRel.
	 *
	 * @param root       hop for which FederatedOutput is set
	 * @param rootHopRel from which FederatedOutput value and cost is retrieved
	 */
	private void setFinalFedout(Hop root, HopRel rootHopRel) {
		if ( hopRelUpdatedFinal.contains(root.getHopID()) ){
			if((rootHopRel.hasLocalOutput() ^ root.hasLocalOutput()) && hopRelMemo.hasFederatedOutputAlternative(root)){
				// Update with FOUT alternative without visiting inputs
				updateFederatedOutput(root, hopRelMemo.getFederatedOutputAlternative(root));
				root.activatePrefetch();
			}
			else {
				// Update without visiting inputs
				updateFederatedOutput(root, rootHopRel);
			}
		}
		else {
			updateFederatedOutput(root, rootHopRel);
			visitInputDependency(rootHopRel);
		}
	}

	/**
	 * Sets FederatedOutput value for each of the inputs of rootHopRel
	 *
	 * @param rootHopRel which has its input values updated
	 */
	private void visitInputDependency(HopRel rootHopRel) {
		List<HopRel> hopRelInputs = rootHopRel.getInputDependency();
		for(HopRel input : hopRelInputs)
			setFinalFedout(input.getHopRef(), input);
	}

	/**
	 * Updates FederatedOutput value and cost estimate based on updateHopRel values.
	 *
	 * @param root         which has its values updated
	 * @param updateHopRel from which the values are retrieved
	 */
	private void updateFederatedOutput(Hop root, HopRel updateHopRel) {
		root.setFederatedOutput(updateHopRel.getFederatedOutput());
		root.setFederatedCost(updateHopRel.getCostObject());
		forceFixedFedOut(root);
		LOG.trace("Updated fedOut to " + updateHopRel.getFederatedOutput() + " for hop "
			+ root.getHopID() + " opcode: " + root.getOpString());
		hopRelUpdatedFinal.add(root.getHopID());
	}

	/**
	 * Set federated output to fixed value if FEDERATED_SPECS is activated for root hop.
	 * @param root hop set to fixed fedout value as loaded from FEDERATED_SPECS
	 */
	private void forceFixedFedOut(Hop root){
		if ( OptimizerUtils.FEDERATED_SPECS.containsKey(root.getBeginLine()) ){
			FEDInstruction.FederatedOutput fedOutSpec = OptimizerUtils.FEDERATED_SPECS.get(root.getBeginLine());
			root.setFederatedOutput(fedOutSpec);
			if ( fedOutSpec.isForcedFederated() )
				root.deactivatePrefetch();
		}
	}

	/**
	 * Select federated execution plan for every Hop in the DAG starting from given roots.
	 * The cost estimates of the hops are also updated when FederatedOutput is updated in the hops.
	 *
	 * @param roots starting point for going through the Hop DAG to update the FederatedOutput fields.
	 */
	@SuppressWarnings("unused")
	private void selectFederatedExecutionPlan(ArrayList<Hop> roots){
		for ( Hop root : roots )
			selectFederatedExecutionPlan(root);
	}

	/**
	 * Select federated execution plan for every Hop in the DAG starting from given root.
	 *
	 * @param root starting point for going through the Hop DAG to update the federatedOutput fields
	 */
	private void selectFederatedExecutionPlan(Hop root) {
		if ( root != null ){
			visitFedPlanHop(root);
			if ( HopRewriteUtils.isTerminalHop(root) )
				terminalHops.add(root);
			else if ( HopRewriteUtils.isData(root, Types.OpOpData.TRANSIENTWRITE) )
				transientWrites.put(root.getName(), root);
		}
	}

	/**
	 * Go through the Hop DAG and set the FederatedOutput field and cost estimate for each Hop from leaf to given currentHop.
	 *
	 * @param currentHop the Hop from which the DAG is visited
	 */
	private void visitFedPlanHop(Hop currentHop) {
		// If the currentHop is in the hopRelMemo table, it means that it has been visited
		if(hopRelMemo.containsHop(currentHop))
			return;
		// If the currentHop has input, then the input should be visited depth-first
		debugLog(currentHop);
		for(Hop input : currentHop.getInput())
			visitFedPlanHop(input);
		updateMemo(currentHop);
		// Put FOUT, LOUT, and None HopRels into the memo table
		ArrayList<HopRel> hopRels = getFedPlans(currentHop);
		if(hopRels.isEmpty())
			hopRels.add(new HopRel(currentHop, FEDInstruction.FederatedOutput.NONE, hopRelMemo));
		addTrace(hopRels);
		hopRelMemo.put(currentHop, hopRels);
	}

	private void updateMemo(Hop currentHop){
		if( HopRewriteUtils.isData(currentHop, Types.OpOpData.FEDERATED) )
			memo.put(currentHop.getHopID(), deriveFType((DataOp)currentHop));
		if( HopRewriteUtils.isData(currentHop, Types.OpOpData.TRANSIENTWRITE) )
			fedVars.put(currentHop.getName(), memo.get(currentHop.getInput(0).getHopID()));
		else if( HopRewriteUtils.isData(currentHop, Types.OpOpData.TRANSIENTREAD) ){
			//Hop tWrite = fedVars.get(currentHop.getName());
			//FTypes.FType tWriteFtype = memo.get(tWrite.getHopID());
			memo.put(currentHop.getHopID(), fedVars.get(currentHop.getName()));
		}
		else if ( allowsFederated(currentHop, memo) )
			memo.put(currentHop.getHopID(), getFederatedOut(currentHop, memo));
		else {
			memo.put(currentHop.getHopID(), null);
		}
	}

	/**
	 * Get the alternative plans regarding the federated output for given currentHop.
	 * @param currentHop for which alternative federated plans are generated
	 * @return list of alternative plans
	 */
	private ArrayList<HopRel> getFedPlans(Hop currentHop){
		ArrayList<HopRel> hopRels = new ArrayList<>();
		if ( isFedInstSupportedHop(currentHop) ){
			ArrayList<Hop> inputHops = currentHop.getInput();
			if ( HopRewriteUtils.isData(currentHop, Types.OpOpData.TRANSIENTREAD) ){
				Hop tWriteHop = transientWrites.get(currentHop.getName());
				if ( tWriteHop == null )
					throw new DMLRuntimeException("Transient write not found for " + currentHop);
				inputHops = new ArrayList<>(Collections.singletonList(tWriteHop));
			}
			if ( isFOUTSupported(currentHop, inputHops) )
				hopRels.addAll(generateHopRels(currentHop, inputHops));
			if ( isLOUTSupported(currentHop) )
				hopRels.add(new HopRel(currentHop, FEDInstruction.FederatedOutput.LOUT, hopRelMemo, inputHops));
		}
		return hopRels;
	}

	/**
	 * Generate a collection of FOUT HopRels representing the different possible FType outputs.
	 * For each FType output, only the minimum cost input combination is chosen.
	 * @param currentHop for which HopRels are generated
	 * @param inputHops to currentHop
	 * @return collection of FOUT HopRels with different FType outputs
	 */
	private Collection<HopRel> generateHopRels(Hop currentHop, List<Hop> inputHops){
		List<List<FType>> validFTypes = getValidFTypes(inputHops);
		List<List<FType>> inputFTypeCombinations = getAllCombinations(validFTypes);
		Map<FType,HopRel> foutHopRelMap = new HashMap<>();
		for ( List<FType> inputCombination : inputFTypeCombinations){
			FType outputFType = getFederatedOut(currentHop, inputCombination.toArray(new FType[0]));
			HopRel alt = new HopRel(currentHop, FEDInstruction.FederatedOutput.FOUT, outputFType, hopRelMemo, inputHops, inputCombination);
			if ( foutHopRelMap.containsKey(alt.getFType()) ){
				foutHopRelMap.computeIfPresent(alt.getFType(),
					(key,currentVal) -> (currentVal.getCost() < alt.getCost()) ? currentVal : alt);
			} else {
				foutHopRelMap.put(outputFType, alt);
			}
		}
		return foutHopRelMap.values();
	}

	private List<List<FType>> getValidFTypes(List<Hop> inputHops){
		List<List<FType>> validFTypes = new ArrayList<>();
		for ( Hop inputHop : inputHops )
			validFTypes.add(hopRelMemo.getFTypes(inputHop));
		return validFTypes;
	}

	public List<List<FType>> getAllCombinations(List<List<FType>> validFTypes){
		List<List<FType>> resultList = new ArrayList<>();
		buildCombinations(validFTypes, resultList, 0, new ArrayList<>());
		return resultList;
	}

	public void buildCombinations(List<List<FType>> validFTypes, List<List<FType>> result, int currentIndex, List<FType> currentResult){
		if ( currentIndex == validFTypes.size() ){
			result.add(currentResult);
		} else {
			for (FType currentType : validFTypes.get(currentIndex)){
				List<FType> currentPass = new ArrayList<>(currentResult);
				currentPass.add(currentType);
				buildCombinations(validFTypes, result, currentIndex+1, currentPass);
			}
		}
	}

	/**
	 * Checks to see if the currentHop supports FOUT.
	 * @param currentHop for which FOUT support is checked
	 * @param inputHops of currentHop
	 * @return true if FOUT is supported by the currentHop
	 */
	private boolean isFOUTSupported(Hop currentHop, ArrayList<Hop> inputHops){
		//if ( !currentHop.isFederatedDataOp() && !allowsFederated(currentHop, memo) )
		//	return false;
		if (currentHop.isScalar())
			return false;
		// If the output of AggUnaryOp is a scalar, the operation cannot be FOUT
		if(currentHop instanceof AggUnaryOp && currentHop.isScalar())
			return false;
		// It can only be FOUT if at least one of the inputs are FOUT, except if it is a federated DataOp
		if(inputHops.stream().noneMatch(hopRelMemo::hasFederatedOutputAlternative)
			&& !currentHop.isFederatedDataOp())
			return false;
		return true;
	}

	/**
	 * Write HOP visit to debug log if debug is activated.
	 * @param currentHop hop written to log
	 */
	private void debugLog(Hop currentHop){
		if ( LOG.isDebugEnabled() ){
			LOG.debug("Visiting HOP: " + currentHop + " Input size: " + currentHop.getInput().size());
			int index = 0;
			for ( Hop hop : currentHop.getInput()){
				if ( hop == null )
					LOG.debug("Input at index is null: " + index);
				else
					LOG.debug("HOP input: " + hop + " at index " + index + " of " + currentHop);
				index++;
			}
		}
	}

	private void addTrace(ArrayList<HopRel> hopRels){
		if (LOG.isTraceEnabled()){
			for(HopRel hr : hopRels){
				LOG.trace(hr);
			}
		}
	}

	/**
	 * Checks if the instructions related to the given hop supports FOUT/LOUT processing.
	 *
	 * @param hop to check for federated support
	 * @return true if federated instructions related to hop supports FOUT/LOUT processing
	 */
	private boolean isFedInstSupportedHop(Hop hop) {
		// The following operations are supported given that the above conditions have not returned already
		return (hop instanceof AggBinaryOp || hop instanceof BinaryOp || hop instanceof ReorgOp
			|| hop instanceof AggUnaryOp || hop instanceof TernaryOp || hop instanceof DataOp);
	}

	/**
	 * Checks to see if the associatedHop supports LOUT.
	 * It supports LOUT if the output has no privacy constraints.
	 *
	 * @param associatedHop for which LOUT support is checked.
	 * @return true if LOUT is supported by the associatedHop
	 */
	private boolean isLOUTSupported(Hop associatedHop) {
		return associatedHop.getPrivacy() == null || !associatedHop.getPrivacy().hasConstraints();
	}
}
