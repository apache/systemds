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

package org.apache.sysds.hops.ipa;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.Hop;
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

/**
 * This rewrite eliminates unnecessary sub-DAGs that produce
 * transient write outputs which are never consumed subsequently.
 * 
 */
public class IPAPassEliminateDeadCode extends IPAPass
{
	@Override
	public boolean isApplicable(FunctionCallGraph fgraph) {
		return InterProceduralAnalysis.ELIMINATE_DEAD_CODE;
	}
	
	@Override
	public boolean rewriteProgram( DMLProgram prog, FunctionCallGraph fgraph, FunctionCallSizeInfo fcallSizes ) {
		// step 1: backwards pass over main program to track used and remove unused vars
		findAndRemoveDeadCode(prog.getStatementBlocks(), new HashSet<>(), fgraph);
		
		// step 2: backwards pass over functions to track used and remove unused vars
		for( FunctionStatementBlock fsb : prog.getFunctionStatementBlocks() ) {
			// mark function outputs as used variables
			Set<String> usedVars = new HashSet<>();
			FunctionStatement fstmt = (FunctionStatement) fsb.getStatement(0);
			fstmt.getOutputParams().stream().forEach(d -> usedVars.add(d.getName()));
			
			// backward pass over function to track used and remove unused vars
			findAndRemoveDeadCode(fstmt.getBody(), usedVars, fgraph);
		}
		return false;
	}
	
	private static void findAndRemoveDeadCode(List<StatementBlock> sbs, Set<String> usedVars, FunctionCallGraph fgraph) {
		for( int i=sbs.size()-1; i >= 0; i-- ) {
			// remove unused assignments
			if( HopRewriteUtils.isLastLevelStatementBlock(sbs.get(i)) ) {
				List<Hop> roots = sbs.get(i).getHops();
				for( int j=0; j<roots.size(); j++ ) {
					Hop root = roots.get(j);
					boolean isTWrite = HopRewriteUtils.isData(root, OpOpData.TRANSIENTWRITE);
					boolean isFCall = isFunctionCallWithUnusedOutputs(root, usedVars, fgraph);
					if( (isTWrite && !usedVars.contains(root.getName())) || isFCall ) {
						if( isFCall ) {
							String fkey = ((FunctionOp) root).getFunctionKey();
							fgraph.removeFunctionCall(fkey, (FunctionOp) root, sbs.get(i));
						}
						roots.remove(j); j--;
						rRemoveOpFromDAG(root);
					}
				}
			}
			
			// maintain used variables (in terms of existing transient reads because 
			// other rewrites such as simplification, DAG splits, and code motion might
			// have removed/added reads and not consistently updated variablesRead()
			usedVars.addAll(rCollectReadVariableNames(sbs.get(i), new HashSet<>()));
		}
	}
	
	private static boolean isFunctionCallWithUnusedOutputs(Hop hop, Set<String> varNames, FunctionCallGraph fgraph) {
		return hop instanceof FunctionOp
			&& fgraph.isSideEffectFreeFunction(((FunctionOp)hop).getFunctionKey())
			&& Arrays.stream(((FunctionOp) hop).getOutputVariableNames())
				.allMatch(var -> !varNames.contains(var));
	}
	
	private static void rRemoveOpFromDAG(Hop current) {
		// cleanup child to parent links and
		// recurse on operators ready for cleanup
		for( Hop input : current.getInput() ) {
			input.getParent().remove(current);
			if( input.getParent().isEmpty() )
				rRemoveOpFromDAG(input);
		}
		//cleanup parent to child links
		current.getInput().clear();
	}
	
	private static Set<String> rCollectReadVariableNames(StatementBlock sb, Set<String> varNames) {
		if( sb instanceof WhileStatementBlock ) {
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement wstmt = (WhileStatement) sb.getStatement(0);
			collectReadVariableNames(wsb.getPredicateHops(), varNames);
			for( StatementBlock csb : wstmt.getBody() )
				rCollectReadVariableNames(csb, varNames);
		}
		else if( sb instanceof ForStatementBlock ) {
			ForStatementBlock fsb = (ForStatementBlock) sb;
			ForStatement fstmt = (ForStatement) sb.getStatement(0);
			collectReadVariableNames(fsb.getFromHops(), varNames);
			collectReadVariableNames(fsb.getToHops(), varNames);
			collectReadVariableNames(fsb.getIncrementHops(), varNames);
			for( StatementBlock csb : fstmt.getBody() )
				rCollectReadVariableNames(csb, varNames);
		}
		else if( sb instanceof IfStatementBlock ) {
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement istmt = (IfStatement) sb.getStatement(0);
			collectReadVariableNames(isb.getPredicateHops(), varNames);
			for( StatementBlock csb : istmt.getIfBody() )
				rCollectReadVariableNames(csb, varNames);
			if( istmt.getElseBody() != null )
				for( StatementBlock csb : istmt.getElseBody() )
					rCollectReadVariableNames(csb, varNames);
		}
		else if( sb.getHops() != null ) {
			Hop.resetVisitStatus(sb.getHops());
			for( Hop hop : sb.getHops() )
				rCollectReadVariableNames(hop, varNames);
		}
		return varNames;
	}
	
	private static Set<String> collectReadVariableNames(Hop hop, Set<String> varNames) {
		if( hop == null )
			return varNames;
		hop.resetVisitStatus();
		return rCollectReadVariableNames(hop, varNames);
	}
	
	private static Set<String> rCollectReadVariableNames(Hop hop, Set<String> varNames) {
		if( hop.isVisited() )
			return varNames;
		for( Hop c : hop.getInput() )
			rCollectReadVariableNames(c, varNames);
		if( HopRewriteUtils.isData(hop, OpOpData.TRANSIENTREAD) )
			varNames.add(hop.getName());
		hop.setVisited();
		return varNames;
	}
}
