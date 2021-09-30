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

package org.apache.sysds.hops.rewrite;

import org.apache.sysds.api.DMLException;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.ReorgOp;
import org.apache.sysds.hops.TernaryOp;
import org.apache.sysds.hops.cost.HopRel;
import org.apache.sysds.hops.ipa.FunctionCallGraph;
import org.apache.sysds.hops.ipa.FunctionCallSizeInfo;
import org.apache.sysds.hops.ipa.IPAPass;
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
import org.apache.sysds.runtime.instructions.fed.FEDInstruction;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class IPAPassRewriteFederatedPlan extends IPAPass {

	private final static Map<Long, List<HopRel>> hopRelMemo = new HashMap<>();

	/**
	 * Indicates if an IPA pass is applicable for the current
	 * configuration such as global flags or the chosen execution
	 * mode (e.g., HYBRID).
	 *
	 * @param fgraph function call graph
	 * @return true if applicable.
	 */
	@Override public boolean isApplicable(FunctionCallGraph fgraph) {
		return OptimizerUtils.FEDERATED_COMPILATION;
	}

	/**
	 * Rewrites the given program or its functions in place,
	 * with access to the read-only function call graph.
	 *
	 * @param prog       dml program
	 * @param fgraph     function call graph
	 * @param fcallSizes function call size infos
	 * @return true if function call graph should be rebuild
	 */
	@Override public boolean rewriteProgram(DMLProgram prog, FunctionCallGraph fgraph,
		FunctionCallSizeInfo fcallSizes) {
		rewriteStatementBlocks(prog.getStatementBlocks());
		//TODO: Set final fedout of Hops
		return false;
	}

	/**
	 * TODO: Change this documentation
	 * Handle an arbitrary statement block. Specific type constraints have to be ensured
	 * within the individual rewrites. If a rewrite does not apply to individual blocks, it
	 * should simply return the input block.
	 *
	 * @param sb    statement block
	 * @return list of statement blocks
	 */
	public ArrayList<StatementBlock> rewriteStatementBlock(StatementBlock sb) {
		if ( sb instanceof WhileStatementBlock)
			return rewriteWhileStatementBlock((WhileStatementBlock) sb);
		else if ( sb instanceof IfStatementBlock)
			return rewriteIfStatementBlock((IfStatementBlock) sb);
		else if ( sb instanceof ForStatementBlock){
			// This also includes ParForStatementBlocks
			return rewriteForStatementBlock((ForStatementBlock) sb);
		}
		else if ( sb instanceof FunctionStatementBlock)
			return rewriteFunctionStatementBlock((FunctionStatementBlock) sb);
		else {
			// StatementBlock type (no subclass)
			sb.setHops(selectFederatedExecutionPlan(sb.getHops()));
		}

		return new ArrayList<>(Arrays.asList(sb));
	}

	private ArrayList<StatementBlock> rewriteWhileStatementBlock(WhileStatementBlock whileSB){
		Hop whilePredicateHop = whileSB.getPredicateHops();
		selectFederatedExecutionPlan(whilePredicateHop);
		for ( Statement stm : whileSB.getStatements() ){
			WhileStatement whileStm = (WhileStatement) stm;
			whileStm.setBody(rewriteStatementBlocks(whileStm.getBody()));
		}
		return new ArrayList<>(Arrays.asList(whileSB));
	}

	private ArrayList<StatementBlock> rewriteIfStatementBlock(IfStatementBlock ifSB){
		selectFederatedExecutionPlan(ifSB.getPredicateHops());
		for ( Statement statement : ifSB.getStatements() ){
			IfStatement ifStatement = (IfStatement) statement;
			ifStatement.setIfBody(rewriteStatementBlocks(ifStatement.getIfBody()));
			ifStatement.setElseBody(rewriteStatementBlocks(ifStatement.getElseBody()));
		}
		return new ArrayList<>(Arrays.asList(ifSB));
	}

	private ArrayList<StatementBlock> rewriteForStatementBlock(ForStatementBlock forSB){
		selectFederatedExecutionPlan(forSB.getFromHops());
		selectFederatedExecutionPlan(forSB.getToHops());
		selectFederatedExecutionPlan(forSB.getIncrementHops());
		for ( Statement statement : forSB.getStatements() ){
			ForStatement forStatement = ((ForStatement)statement);
			forStatement.setBody(rewriteStatementBlocks(forStatement.getBody()));
		}
		return new ArrayList<>(Arrays.asList(forSB));
	}

	private ArrayList<StatementBlock> rewriteFunctionStatementBlock(FunctionStatementBlock funcSB){
		for ( Statement statement : funcSB.getStatements() ){
			FunctionStatement funcStm = (FunctionStatement) statement;
			funcStm.setBody(rewriteStatementBlocks(funcStm.getBody()));
		}
		return new ArrayList<>(Arrays.asList(funcSB));
	}

	/**
	 * TODO: Rewrite this documentation
	 * Handle a list of statement blocks. Specific type constraints have to be ensured
	 * within the individual rewrites. If a rewrite does not require sequence access, it
	 * should simply return the input list of statement blocks.
	 *
	 * @param sbs   list of statement blocks
	 * @return list of statement blocks
	 */
	public ArrayList<StatementBlock> rewriteStatementBlocks(List<StatementBlock> sbs) {
		ArrayList<StatementBlock> rewrittenStmBlocks = new ArrayList<>();
		for ( StatementBlock stmBlock : sbs )
			rewrittenStmBlocks.addAll(rewriteStatementBlock(stmBlock));
		return rewrittenStmBlocks;
	}


	private void setFinalFedout(Hop root){
		HopRel optimalRootHopRel = hopRelMemo.get(root.getHopID()).stream().min(Comparator.comparingDouble(HopRel::getCost))
			.orElseThrow(() -> new DMLException("Hop root " + root + " has no feasible federated output alternatives"));
		setFinalFedout(root, optimalRootHopRel);
	}

	private void setFinalFedout(Hop root, HopRel rootHopRel){
		updateFederatedOutput(root, rootHopRel);
		visitInputDependency(rootHopRel);
	}

	private void visitInputDependency(HopRel rootHopRel){
		List<HopRel> hopRelInputs = rootHopRel.getInputDependency();
		for ( HopRel input : hopRelInputs )
			setFinalFedout(input.getHopRef(), input);
	}

	private void updateFederatedOutput(Hop root, HopRel updateHopRel){
		root.setFederatedOutput(updateHopRel.getFederatedOutput());
		root.setFederatedCost(updateHopRel.getCostObject());
	}

	/**
	 * Select federated execution plan for every Hop in the DAG starting from given roots.
	 * @param roots starting point for going through the Hop DAG to update the FederatedOutput fields.
	 * @return the list of roots with updated FederatedOutput fields.
	 */
	private ArrayList<Hop> selectFederatedExecutionPlan(ArrayList<Hop> roots){
		for ( Hop root : roots ){
			selectFederatedExecutionPlan(root);
		}
		return roots;
	}

	private void selectFederatedExecutionPlan(Hop root){
		visitFedPlanHop(root);
		setFinalFedout(root);
	}

	/**
	 * Go through the Hop DAG and set the FederatedOutput field for each Hop from leaf to given currentHop.
	 * @param currentHop the Hop from which the DAG is visited
	 */
	private void visitFedPlanHop(Hop currentHop){
		// If the currentHop is in the hopRelMemo table, it means that it has been visited
		if ( hopRelMemo.containsKey(currentHop.getHopID()) )
			return; //TODO: The memo table could contain the ID of the hop, but not have all possible fedouts. This should also be checked and then the missing fedouts should be added without overwriting the existing values.
		// If the currentHop has input, then the input should be visited depth-first
		if ( currentHop.getInput() != null && currentHop.getInput().size() > 0 ){
			for ( Hop input : currentHop.getInput() )
				visitFedPlanHop(input);
		}
		// Put FOUT, LOUT, and None HopRels into the memo table
		ArrayList<HopRel> hopRels = new ArrayList<>();
		if ( isFedInstSupportedHop(currentHop) ){
			for ( FEDInstruction.FederatedOutput fedoutValue : FEDInstruction.FederatedOutput.values() )
				if ( isFedOutSupported(currentHop, fedoutValue) )
					hopRels.add(new HopRel(currentHop,fedoutValue, hopRelMemo));
		}
		if ( hopRels.isEmpty() )
			hopRels.add(new HopRel(currentHop, FEDInstruction.FederatedOutput.NONE, hopRelMemo));
		hopRelMemo.put(currentHop.getHopID(), hopRels);

		/*
		if ( ( isFedInstSupportedHop(currentHop) ) ){
			// The Hop can be FOUT or LOUT or None. Check utility of FOUT vs LOUT vs None.
			currentHop.setFederatedOutput(getHighestUtilFedOut(currentHop));
		}
		else
			currentHop.setFederatedOutput(FEDInstruction.FederatedOutput.NONE);*/
		currentHop.setVisited();
	}

	/**
	 * Returns the FederatedOutput with the highest utility out of the valid FederatedOutput values.
	 * @param hop for which the utility is found
	 * @return the FederatedOutput value with highest utility for the given Hop
	 */
	private FEDInstruction.FederatedOutput getHighestUtilFedOut(Hop hop){
		Map<FEDInstruction.FederatedOutput,Long> fedOutUtilMap = new EnumMap<>(FEDInstruction.FederatedOutput.class);
		if ( isFOUTSupported(hop) )
			fedOutUtilMap.put(FEDInstruction.FederatedOutput.FOUT, getUtilFout());
		if ( isLOUTSupported(hop) )
			fedOutUtilMap.put(FEDInstruction.FederatedOutput.LOUT, getUtilLout(hop));
		fedOutUtilMap.put(FEDInstruction.FederatedOutput.NONE, 0L);

		Map.Entry<FEDInstruction.FederatedOutput, Long> fedOutMax = Collections.max(fedOutUtilMap.entrySet(), Map.Entry.comparingByValue());
		return fedOutMax.getKey();
	}

	/**
	 * Utility if hop is FOUT. This is a simple version where it always returns 1.
	 * @return utility if hop is FOUT
	 */
	private long getUtilFout(){
		//TODO: Make better utility estimation
		return 1;
	}

	/**
	 * Utility if hop is LOUT. This is a simple version only based on dimensions.
	 * @param hop for which utility is calculated
	 * @return utility if hop is LOUT
	 */
	private long getUtilLout(Hop hop){
		//TODO: Make better utility estimation
		return -(long)hop.getMemEstimate();
	}

	private boolean isFedInstSupportedHop(Hop hop){

		// Check that some input is FOUT, otherwise none of the fed instructions will run unless it is fedinit
		//if ( (!hop.isFederatedDataOp()) && hop.getInput().stream().noneMatch(Hop::hasFederatedOutput) )
			//return false;

		// The following operations are supported given that the above conditions have not returned already
		return ( hop instanceof AggBinaryOp || hop instanceof BinaryOp || hop instanceof ReorgOp
			|| hop instanceof AggUnaryOp || hop instanceof TernaryOp || hop instanceof DataOp);
	}

	private boolean isFedOutSupported(Hop associatedHop, FEDInstruction.FederatedOutput fedOut){
		switch(fedOut){
			case FOUT:
				return isFOUTSupported(associatedHop);
			case LOUT:
				return isLOUTSupported(associatedHop);
			case NONE:
				return false;
			default:
				return true;
		}
	}

	/**
	 * Checks to see if the associatedHop supports FOUT.
	 * @param associatedHop for which FOUT support is checked
	 * @return true if FOUT is supported by the associatedHop
	 */
	private boolean isFOUTSupported(Hop associatedHop){
		// If the output of AggUnaryOp is a scalar, the operation cannot be FOUT
		if ( associatedHop instanceof AggUnaryOp && associatedHop.isScalar() )
			return false;
		// If one of the parents is a federated DataOp, all the inputs have to be LOUT.
		if (associatedHop.getParent().stream().anyMatch(Hop::isFederatedDataOp))
			return false;
		// It can only be FOUT if at least one of the inputs are FOUT
		if ( !(associatedHop.getInput().stream().anyMatch(
			input -> hopRelMemo.get(input.getHopID()).stream().anyMatch(HopRel::hasFederatedOutput) ))
			&& !associatedHop.isFederatedDataOp() )
			return false;
		return true;
	}

	/**
	 * Checks to see if the associatedHop supports LOUT.
	 * It supports LOUT if the output has no privacy constraints.
	 * @param associatedHop for which LOUT support is checked.
	 * @return true if LOUT is supported by the associatedHop
	 */
	private boolean isLOUTSupported(Hop associatedHop){
		return associatedHop.getPrivacy() == null
			|| (associatedHop.getPrivacy() != null && !associatedHop.getPrivacy().hasConstraints());
	}
}
