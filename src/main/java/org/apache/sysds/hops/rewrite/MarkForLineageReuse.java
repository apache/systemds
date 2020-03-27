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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.FunctionStatement;
import org.apache.sysds.parser.FunctionStatementBlock;
import org.apache.sysds.parser.IfStatement;
import org.apache.sysds.parser.IfStatementBlock;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.WhileStatement;
import org.apache.sysds.parser.WhileStatementBlock;
import org.apache.sysds.runtime.lineage.LineageCacheConfig;

public class MarkForLineageReuse extends StatementBlockRewriteRule 
{
	@Override
	public boolean createsSplitDag() {
		return false;
	}

	@Override
	public List<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus status) 
	{
		if (!HopRewriteUtils.isLoopStatementBlock(sb) || LineageCacheConfig.ReuseCacheType.isNone())
			return Arrays.asList(sb);    //early abort

		if (sb instanceof ForStatementBlock) {
			ForStatement fstmt = (ForStatement)sb.getStatement(0);
			Set<String> loopVar = new HashSet<>(Arrays.asList(fstmt.getIterablePredicate().getIterVar().getName()));
			HashSet<String> deproots = new HashSet<>();
			rUnmarkLoopDepVarsSB(fstmt.getBody(), deproots, loopVar);
		}
		if (sb instanceof WhileStatementBlock) {
			WhileStatement wstmt = (WhileStatement)sb.getStatement(0);
			// intersection of updated and conditional variables are the loop variables
			Set<String> loopVar = sb.variablesUpdated().getVariableNames().stream()
					.filter(v -> wstmt.getConditionalPredicate().variablesRead().containsVariable(v))
					.collect(Collectors.toSet());
			HashSet<String> deproots = new HashSet<>();
			rUnmarkLoopDepVarsSB(wstmt.getBody(), deproots, loopVar);
		}
		return Arrays.asList(sb);
	}
	
	private void rUnmarkLoopDepVarsSB(ArrayList<StatementBlock> sbs, HashSet<String> deproots, Set<String> loopVar)
	{
		HashSet<String> newdepsbs = new HashSet<>();
		int lim = 0;
		do {
			newdepsbs.clear();
			newdepsbs.addAll(deproots);
			for (StatementBlock sb : sbs) {
				if (sb instanceof ForStatementBlock) {
					ForStatement fstmt = (ForStatement)sb.getStatement(0);
					rUnmarkLoopDepVarsSB(fstmt.getBody(), newdepsbs, loopVar);
					//TODO: nested loops.
				}
				else if (sb instanceof WhileStatementBlock) {
					WhileStatement wstmt = (WhileStatement)sb.getStatement(0);
					rUnmarkLoopDepVarsSB(wstmt.getBody(), newdepsbs, loopVar); 
				}
				else if (sb instanceof IfStatementBlock) {
					IfStatement ifstmt = (IfStatement)sb.getStatement(0);
					rUnmarkLoopDepVarsSB(ifstmt.getIfBody(), newdepsbs, loopVar); 
					if (ifstmt.getElseBody() != null)
						rUnmarkLoopDepVarsSB(ifstmt.getElseBody(), newdepsbs, loopVar); 
				}
				else if (sb instanceof FunctionStatementBlock) {
					FunctionStatement fnstmt = (FunctionStatement)sb.getStatement(0);
					rUnmarkLoopDepVarsSB(fnstmt.getBody(), newdepsbs, loopVar);
				}
				else {
					if (sb.getHops() != null)
						for (int j=0; j<sb.variablesUpdated().getSize(); j++) {
							HashSet<String> newdeproots = new HashSet<>(deproots);
							for (Hop hop : sb.getHops()) {
								// find the loop dependent DAG roots
								Hop.resetVisitStatus(sb.getHops());
								HashSet<Long> dephops = new HashSet<>();
								rUnmarkLoopDepVars(hop, loopVar, newdeproots, dephops);
							}
							if (!deproots.isEmpty() && deproots.equals(newdeproots))
								// break if loop dependent DAGs are converged to a unvarying set
								break;
							else
								// iterate to propagate the loop dependents across all the DAGs in this SB
								deproots.addAll(newdeproots);
						}
				}
			}
			deproots.addAll(newdepsbs);
			lim++;
		}
		// iterate to propagate the loop dependents across all the SBs
		while (lim < sbs.size() && (deproots.isEmpty() || !deproots.equals(newdepsbs)));
	}
	
	private void rUnmarkLoopDepVars(Hop hop, Set<String> loopVar, HashSet<String> deproots, HashSet<Long> dephops)
	{
		if (hop.isVisited())
			return;

		for (Hop hi : hop.getInput()) 
			rUnmarkLoopDepVars(hi, loopVar, deproots, dephops);

		// unmark operation if this itself or any of its inputs are loop dependent
		boolean loopdephop = loopVar.contains(hop.getName())
				|| (HopRewriteUtils.isData(hop, OpOpData.TRANSIENTREAD)
				&& deproots.contains(hop.getName()));
		for (Hop hi : hop.getInput()) 
			loopdephop |= dephops.contains(hi.getHopID());
		
		if (loopdephop) {
			dephops.add(hop.getHopID());
			hop.setRequiresLineageCaching(false); 
			//TODO: extend all the hops to propagate till variablecp output
		}
		// TODO: logic to separate out partially reusable cases (e.g cbind-tsmm)
		
		if (HopRewriteUtils.isData(hop, OpOpData.TRANSIENTWRITE) 
			&& !dephops.isEmpty())
			// copy to propagate across
			deproots.add(hop.getName());

		hop.setVisited();
	}
	
	@Override
	public List<StatementBlock> rewriteStatementBlocks(List<StatementBlock> sbs, ProgramRewriteStatus status) {
		return sbs;
	}
}
