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
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.fedplanner.FTypes.FType;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.cost.HopRel;
import org.apache.sysds.hops.ipa.FunctionCallGraph;
import org.apache.sysds.hops.ipa.FunctionCallSizeInfo;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DataIdentifier;
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
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.IntObject;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction.FederatedOutput;
import org.apache.sysds.utils.Explain;
import org.apache.sysds.utils.Explain.ExplainType;

public class FederatedPlannerCostbased extends AFederatedPlanner {
	private static final Log LOG = LogFactory.getLog(FederatedPlannerCostbased.class.getName());

	private final MemoTable hopRelMemo = new MemoTable();
	/**
	 * IDs of hops for which the final fedout value has been set.
	 */
	private final Set<Long> hopRelUpdatedFinal = new HashSet<>();
	/**
	 * Terminal hops in DML program given to this rewriter.
	 */
	private final List<Hop> terminalHops = new ArrayList<>();
	private final Map<String, Hop> transientWrites = new HashMap<>();
	private LocalVariableMap localVariableMap = new LocalVariableMap();

	public List<Hop> getTerminalHops(){
		return terminalHops;
	}

	@Override
	public void rewriteProgram( DMLProgram prog, FunctionCallGraph fgraph, FunctionCallSizeInfo fcallSizes ) {
		prog.updateRepetitionEstimates();
		rewriteStatementBlocks(prog, prog.getStatementBlocks(), null);
		setFinalFedouts();
		updateExplain();
	}

	@Override
	public void rewriteFunctionDynamic(FunctionStatementBlock function, LocalVariableMap funcArgs) {
		localVariableMap = funcArgs;
		rewriteStatementBlock(function.getDMLProg(), function, null);
		setFinalFedouts();
		updateExplain();
	}

	/**
	 * Estimates cost and enumerates federated execution plans in hopRelMemo.
	 * The method calls the contained statement blocks recursively.
	 *
	 * @param prog dml program
	 * @param sbs  list of statement blocks
	 * @param paramMap map of parameters in function call
	 * @return list of statement blocks with the federated output value updated for each hop
	 */
	private ArrayList<StatementBlock> rewriteStatementBlocks(DMLProgram prog, List<StatementBlock> sbs, Map<String, Hop> paramMap) {
		ArrayList<StatementBlock> rewrittenStmBlocks = new ArrayList<>();
		for(StatementBlock stmBlock : sbs)
			rewrittenStmBlocks.addAll(rewriteStatementBlock(prog, stmBlock, paramMap));
		return rewrittenStmBlocks;
	}

	/**
	 * Estimates cost and enumerates federated execution plans in hopRelMemo.
	 * The method calls the contained statement blocks recursively.
	 *
	 * @param prog dml program
	 * @param sb   statement block
	 * @param paramMap map of parameters in function call
	 * @return list of statement blocks with the federated output value updated for each hop
	 */
	public ArrayList<StatementBlock> rewriteStatementBlock(DMLProgram prog, StatementBlock sb, Map<String, Hop> paramMap) {
		if(sb instanceof WhileStatementBlock)
			return rewriteWhileStatementBlock(prog, (WhileStatementBlock) sb, paramMap);
		else if(sb instanceof IfStatementBlock)
			return rewriteIfStatementBlock(prog, (IfStatementBlock) sb, paramMap);
		else if(sb instanceof ForStatementBlock) {
			// This also includes ParForStatementBlocks
			return rewriteForStatementBlock(prog, (ForStatementBlock) sb, paramMap);
		}
		else if(sb instanceof FunctionStatementBlock)
			return rewriteFunctionStatementBlock(prog, (FunctionStatementBlock) sb, paramMap);
		else {
			// StatementBlock type (no subclass)
			return rewriteDefaultStatementBlock(prog, sb, paramMap);
		}
	}

	private ArrayList<StatementBlock> rewriteWhileStatementBlock(DMLProgram prog, WhileStatementBlock whileSB, Map<String, Hop> paramMap) {
		Hop whilePredicateHop = whileSB.getPredicateHops();
		selectFederatedExecutionPlan(whilePredicateHop, paramMap);
		for(Statement stm : whileSB.getStatements()) {
			WhileStatement whileStm = (WhileStatement) stm;
			whileStm.setBody(rewriteStatementBlocks(prog, whileStm.getBody(), paramMap));
		}
		return new ArrayList<>(Collections.singletonList(whileSB));
	}

	private ArrayList<StatementBlock> rewriteIfStatementBlock(DMLProgram prog, IfStatementBlock ifSB, Map<String, Hop> paramMap) {
		selectFederatedExecutionPlan(ifSB.getPredicateHops(), paramMap);
		for(Statement statement : ifSB.getStatements()) {
			IfStatement ifStatement = (IfStatement) statement;
			ifStatement.setIfBody(rewriteStatementBlocks(prog, ifStatement.getIfBody(), paramMap));
			ifStatement.setElseBody(rewriteStatementBlocks(prog, ifStatement.getElseBody(), paramMap));
		}
		return new ArrayList<>(Collections.singletonList(ifSB));
	}

	private ArrayList<StatementBlock> rewriteForStatementBlock(DMLProgram prog, ForStatementBlock forSB, Map<String, Hop> paramMap) {
		selectFederatedExecutionPlan(forSB.getFromHops(), paramMap);
		selectFederatedExecutionPlan(forSB.getToHops(), paramMap);
		selectFederatedExecutionPlan(forSB.getIncrementHops(), paramMap);

		// add iter variable to local variable map allowing us to reason over transient reads in the HOP DAG
		DataIdentifier iterVar = ((ForStatement) forSB.getStatement(0)).getIterablePredicate().getIterVar();
		LocalVariableMap tmpLocalVariableMap = localVariableMap;
		localVariableMap = (LocalVariableMap) localVariableMap.clone();
		// value doesn't matter, localVariableMap is just used to check if the variable is federated
		localVariableMap.put(iterVar.getName(), new IntObject(-1));
		for(Statement statement : forSB.getStatements()) {
			ForStatement forStatement = ((ForStatement) statement);
			forStatement.setBody(rewriteStatementBlocks(prog, forStatement.getBody(), paramMap));
		}
		localVariableMap = tmpLocalVariableMap;
		return new ArrayList<>(Collections.singletonList(forSB));
	}

	private ArrayList<StatementBlock> rewriteFunctionStatementBlock(DMLProgram prog, FunctionStatementBlock funcSB, Map<String, Hop> paramMap) {
		for(Statement statement : funcSB.getStatements()) {
			FunctionStatement funcStm = (FunctionStatement) statement;
			funcStm.setBody(rewriteStatementBlocks(prog, funcStm.getBody(), paramMap));
		}
		return new ArrayList<>(Collections.singletonList(funcSB));
	}

	private ArrayList<StatementBlock> rewriteDefaultStatementBlock(DMLProgram prog, StatementBlock sb, Map<String, Hop> paramMap) {
		if(sb.hasHops()) {
			for(Hop sbHop : sb.getHops()) {
				selectFederatedExecutionPlan(sbHop, paramMap);
				if(sbHop instanceof FunctionOp) {
					String funcName = ((FunctionOp) sbHop).getFunctionName();
					String funcNamespace = ((FunctionOp) sbHop).getFunctionNamespace();
					Map<String, Hop> funcParamMap = FederatedPlannerUtils.getParamMap((FunctionOp) sbHop);
					if ( paramMap != null && funcParamMap != null)
						funcParamMap.putAll(paramMap);
					paramMap = funcParamMap;
					FunctionStatementBlock sbFuncBlock = prog.getFunctionDictionary(funcNamespace)
						.getFunction(funcName);
					rewriteStatementBlock(prog, sbFuncBlock, paramMap);

					FunctionStatement funcStatement = (FunctionStatement) sbFuncBlock.getStatement(0);
					mapFunctionOutputs((FunctionOp) sbHop, funcStatement);
				}
			}
		}
		return new ArrayList<>(Collections.singletonList(sb));
	}

	/**
	 * Saves the HOPs (TWrite) of the function return values for
	 * the variable name used when calling the function.
	 *
	 * Example:
	 * <code>
	 *     f = function() return (matrix[double] model) {a = rand(1, 1);}
	 *     b = f();
	 * </code>
	 * This function saves the HOP writing to <code>a</code> for identifier <code>b</code>.
	 *
	 * @param sbHop The <code>FunctionOp</code> for the call
	 * @param funcStatement The <code>FunctionStatement</code> of the called function
	 */
	private void mapFunctionOutputs(FunctionOp sbHop, FunctionStatement funcStatement) {
		for (int i = 0; i < sbHop.getOutputVariableNames().length; ++i) {
			Hop outputWrite = transientWrites.get(funcStatement.getOutputParams().get(i).getName());
			transientWrites.put(sbHop.getOutputVariableNames()[i], outputWrite);
		}
	}

	/**
	 * Return parameter map containing the mapping from parameter name to input hop
	 * for all parameters of the function hop.
	 * @param funcOp hop for which the mapping of parameter names to input hops are made
	 * @return parameter map or empty map if function has no parameters
	 */
	private Map<String,Hop> getParamMap(FunctionOp funcOp){
		String[] inputNames = funcOp.getInputVariableNames();
		Map<String,Hop> paramMap = new HashMap<>();
		if ( inputNames != null ){
			for ( int i = 0; i < funcOp.getInput().size(); i++ )
				paramMap.put(inputNames[i],funcOp.getInput(i));
		}
		return paramMap;
	}

	/**
	 * Set final fedouts of all hops starting from terminal hops.
	 */
	public void setFinalFedouts(){
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
		root.setForcedExecType(updateHopRel.getExecType());
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
			FederatedOutput fedOutSpec = OptimizerUtils.FEDERATED_SPECS.get(root.getBeginLine());
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
	 * @param paramMap map of parameters in function call
	 */
	@SuppressWarnings("unused")
	private void selectFederatedExecutionPlan(ArrayList<Hop> roots, Map<String, Hop> paramMap){
		for ( Hop root : roots )
			selectFederatedExecutionPlan(root, paramMap);
	}

	/**
	 * Select federated execution plan for every Hop in the DAG starting from given root.
	 *
	 * @param root starting point for going through the Hop DAG to update the federatedOutput fields
	 * @param paramMap map of parameters in function call
	 */
	private void selectFederatedExecutionPlan(Hop root, Map<String, Hop> paramMap) {
		if ( root != null ){
			visitFedPlanHop(root, paramMap);
			if ( HopRewriteUtils.isTerminalHop(root) )
				terminalHops.add(root);
		}
	}

	/**
	 * Go through the Hop DAG and set the FederatedOutput field and cost estimate for each Hop from leaf to given currentHop.
	 *
	 * @param currentHop the Hop from which the DAG is visited
	 * @param paramMap map of parameters in function call
	 */
	private void visitFedPlanHop(Hop currentHop, Map<String, Hop> paramMap) {
		// If the currentHop is in the hopRelMemo table, it means that it has been visited
		if(hopRelMemo.containsHop(currentHop))
			return;
		debugLog(currentHop);
		// If the currentHop has input, then the input should be visited depth-first
		for(Hop input : currentHop.getInput())
			visitFedPlanHop(input, paramMap);
		// Put FOUT and LOUT HopRels into the memo table
		ArrayList<HopRel> hopRels = getFedPlans(currentHop, paramMap);
		// Put NONE HopRel into memo table if no FOUT or LOUT HopRels were added
		if(hopRels.isEmpty())
			hopRels.add(getNONEHopRel(currentHop, paramMap));
		addTrace(hopRels);
		hopRelMemo.put(currentHop, hopRels);
	}

	private ArrayList<Hop> getHopInputs(Hop currentHop, Map<String, Hop> paramMap){
		if ( HopRewriteUtils.isData(currentHop, Types.OpOpData.TRANSIENTREAD) )
			return FederatedPlannerUtils.getTransientInputs(currentHop, paramMap, transientWrites);
		else
			return currentHop.getInput();
	}

	private HopRel getNONEHopRel(Hop currentHop, Map<String, Hop> paramMap){
		ArrayList<Hop> inputs = getHopInputs(currentHop, paramMap);
		HopRel noneHopRel = new HopRel(currentHop, FederatedOutput.NONE, hopRelMemo, inputs);
		FType[] inputFType = noneHopRel.getInputDependency().stream().map(HopRel::getFType).toArray(FType[]::new);
		FType outputFType = getFederatedOut(currentHop, inputFType);
		noneHopRel.setFType(outputFType);
		return noneHopRel;
	}

	/**
	 * Get the alternative plans regarding the federated output for given currentHop.
	 * @param currentHop for which alternative federated plans are generated
	 * @param paramMap map of parameters in function call
	 * @return list of alternative plans
	 */
	private ArrayList<HopRel> getFedPlans(Hop currentHop, Map<String, Hop> paramMap){
		ArrayList<HopRel> hopRels = new ArrayList<>();
		ArrayList<Hop> inputHops = currentHop.getInput();
		if ( HopRewriteUtils.isData(currentHop, Types.OpOpData.TRANSIENTREAD) ) {
			inputHops = getTransientInputs(currentHop, paramMap);
			if (inputHops == null) {
				// check if transient read on a runtime variable (only when planning during dynamic recompilation)
				return createHopRelsFromRuntimeVars(currentHop, hopRels);
			}
		}
		if ( HopRewriteUtils.isData(currentHop, Types.OpOpData.TRANSIENTWRITE) )
			transientWrites.put(currentHop.getName(), currentHop);
		if ( HopRewriteUtils.isData(currentHop, Types.OpOpData.FEDERATED) )
			hopRels.add(new HopRel(currentHop, FederatedOutput.FOUT, deriveFType((DataOp)currentHop), hopRelMemo, inputHops));
		else
			hopRels.addAll(generateHopRels(currentHop, inputHops));
		if ( isLOUTSupported(currentHop) )
			hopRels.add(new HopRel(currentHop, FederatedOutput.LOUT, hopRelMemo, inputHops));
		return hopRels;
	}

	private ArrayList<HopRel> createHopRelsFromRuntimeVars(Hop currentHop, ArrayList<HopRel> hopRels) {
		Data variable = localVariableMap.get(currentHop.getName());
		if (variable == null) {
			throw new DMLRuntimeException("Transient write not found for " + currentHop);
		}
		FederationMap fedMapping = null;
		if (variable instanceof CacheableData<?>) {
			CacheableData<?> cacheable = (CacheableData<?>) variable;
			fedMapping = cacheable.getFedMapping();
		}
		if(fedMapping != null)
			hopRels.add(new HopRel(currentHop, FederatedOutput.FOUT, fedMapping.getType(), hopRelMemo,
				new ArrayList<>()));
		else
			hopRels.add(new HopRel(currentHop, FederatedOutput.LOUT, hopRelMemo, new ArrayList<>()));
		return hopRels;
	}

	/**
	 * Get transient inputs from either paramMap or transientWrites.
	 * Inputs from paramMap has higher priority than inputs from transientWrites.
	 * @param currentHop hop for which inputs are read from maps
	 * @param paramMap of local parameters
	 * @return inputs of currentHop
	 */
	private ArrayList<Hop> getTransientInputs(Hop currentHop, Map<String, Hop> paramMap){
		// FIXME: does not work for function calls (except when the return names match the variables their results are assigned to)
		//  `model = l2svm(...)` works (because `m_l2svm = function(...) return(Matrix[Double] model)`),
		//  `m = l2svm(...)` does not
		Hop tWriteHop = null;
		if ( paramMap != null)
			tWriteHop = paramMap.get(currentHop.getName());
		if ( tWriteHop == null )
			tWriteHop = transientWrites.get(currentHop.getName());
		if ( tWriteHop == null ) {
			if(localVariableMap.get(currentHop.getName()) != null)
				return null;
			else
				throw new DMLRuntimeException("Transient write not found for " + currentHop);
		}
		else
			return new ArrayList<>(Collections.singletonList(tWriteHop));
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
		for ( List<FType> inputCombination : inputFTypeCombinations ){
			if ( allowsFederated(currentHop, inputCombination.toArray(FType[]::new)) ){
				FType outputFType = getFederatedOut(currentHop, inputCombination.toArray(new FType[0]));
				if ( outputFType != null ){
					HopRel alt = new HopRel(currentHop, FederatedOutput.FOUT, outputFType, hopRelMemo, inputHops, inputCombination);
					if ( foutHopRelMap.containsKey(alt.getFType()) ){
						foutHopRelMap.computeIfPresent(alt.getFType(),
							(key,currentVal) -> (currentVal.getCost() < alt.getCost()) ? currentVal : alt);
					} else {
						foutHopRelMap.put(outputFType, alt);
					}
				} else {
					LOG.trace("Allows federated, but FOUT is not allowed: " + currentHop + " input FTypes: " + inputCombination);
				}
			} else {
				LOG.trace("Does not allow federated: " + currentHop + " input FTypes: " + inputCombination);
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
	 * Add hopRelMemo to Explain class to get explain info related to federated enumeration.
	 */
	public void updateExplain(){
		if (DMLScript.EXPLAIN == ExplainType.HOPS)
			Explain.setMemo(hopRelMemo);
	}

	/**
	 * Write HOP visit to debug log if debug is activated.
	 * @param currentHop hop written to log
	 */
	private void debugLog(Hop currentHop){
		if ( LOG.isDebugEnabled() ){
			LOG.debug("Visiting HOP: " + currentHop + " Input size: " + currentHop.getInput().size());
			if (currentHop.getPrivacy() != null)
				LOG.debug(currentHop.getPrivacy());
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
				LOG.trace("Adding to memo: " + hr);
			}
		}
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
