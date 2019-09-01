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

package org.tugraz.sysds.hops.rewrite;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.stream.Collectors;

import org.tugraz.sysds.hops.DataOp;
import org.tugraz.sysds.hops.FunctionOp;
import org.tugraz.sysds.hops.Hop;
import org.tugraz.sysds.hops.LiteralOp;
import org.tugraz.sysds.hops.Hop.DataGenMethod;
import org.tugraz.sysds.hops.Hop.DataOpTypes;
import org.tugraz.sysds.hops.recompile.Recompiler;
import org.tugraz.sysds.parser.DataIdentifier;
import org.tugraz.sysds.parser.ForStatement;
import org.tugraz.sysds.parser.ForStatementBlock;
import org.tugraz.sysds.parser.IfStatementBlock;
import org.tugraz.sysds.parser.StatementBlock;
import org.tugraz.sysds.parser.VariableSet;
import org.tugraz.sysds.parser.WhileStatement;
import org.tugraz.sysds.parser.WhileStatementBlock;

/**
 * Rule: Simplify program structure by hoisting loop-invariant operations
 * out of while, for, or parfor loops.
 */
public class RewriteHoistLoopInvariantOperations extends StatementBlockRewriteRule
{
	private final boolean _sideEffectFreeFuns;
	
	public RewriteHoistLoopInvariantOperations() {
		this(false);
	}
	
	public RewriteHoistLoopInvariantOperations(boolean noSideEffects) {
		_sideEffectFreeFuns = noSideEffects;
	}
	
	@Override
	public boolean createsSplitDag() {
		return true;
	}
	
	@Override
	public List<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus state) {
		//early abort if possible
		if( sb == null || !HopRewriteUtils.isLoopStatementBlock(sb) )
			return Arrays.asList(sb); //rewrite only applies to loops
		
		//step 1: determine read-only variables
		Set<String> candInputs = sb.variablesRead().getVariableNames().stream()
			.filter(v -> !sb.variablesUpdated().containsVariable(v))
			.collect(Collectors.toSet());
		
		//step 2: collect loop-invariant operations along with their tmp names
		Map<String, Hop> invariantOps = new HashMap<>();
		collectOperations(sb, candInputs, invariantOps);
		
		//step 3: create new statement block for all temporary intermediates
		return invariantOps.isEmpty() ? Arrays.asList(sb) :
			Arrays.asList(createStatementBlock(sb, invariantOps), sb);
	}
	
	@Override
	public List<StatementBlock> rewriteStatementBlocks(List<StatementBlock> sbs, ProgramRewriteStatus state) {
		return sbs;
	}
	
	private void collectOperations(StatementBlock sb, Set<String> candInputs, Map<String, Hop> invariantOps) {
		
		if( sb instanceof WhileStatementBlock ) {
			WhileStatement wstmt = (WhileStatement) sb.getStatement(0);
			for( StatementBlock csb : wstmt.getBody() )
				collectOperations(csb, candInputs, invariantOps);
		}
		else if( sb instanceof ForStatementBlock ) {
			ForStatement fstmt = (ForStatement) sb.getStatement(0);
			for( StatementBlock csb : fstmt.getBody() )
				collectOperations(csb, candInputs, invariantOps);
		}
		else if( sb instanceof IfStatementBlock ) {
			//note: for now we do not pull loop-invariant code out of
			//if statement blocks because these operations are conditionally
			//executed, so unconditional execution might be counter productive
		}
		else if( sb.getHops() != null ) {
			//step a: bottom-up flagging of loop-invariant operations
			//(these are defined operations whose inputs are read only
			//variables or other loop-invariant operations)
			Hop.resetVisitStatus(sb.getHops());
			HashSet<Long> memo = new HashSet<>();
			for( Hop hop : sb.getHops() )
				rTagLoopInvariantOperations(hop, candInputs, memo);
			
			//step b: copy hop sub dag and replace it via tread
			Hop.resetVisitStatus(sb.getHops());
			for( Hop hop : sb.getHops() )
				rCollectAndReplaceOperations(hop, candInputs, memo, invariantOps);
			
			if( !memo.isEmpty() ) {
				LOG.debug("Applied hoistLoopInvariantOperations (lines "
					+sb.getBeginLine()+"-"+sb.getEndLine()+"): "+memo.size()+".");
			}
		}
	}
	
	private void rTagLoopInvariantOperations(Hop hop, Set<String> candInputs, Set<Long> memo) {
		if( hop.isVisited() )
			return;
		
		//process inputs first (depth first)
		for( Hop c : hop.getInput() )
			rTagLoopInvariantOperations(c, candInputs, memo);
		
		//flag operation if all inputs are loop invariant
		boolean invariant = !HopRewriteUtils.isDataGenOp(hop, DataGenMethod.RAND)
			&& (!(hop instanceof FunctionOp) || _sideEffectFreeFuns)
			&& !HopRewriteUtils.isData(hop, DataOpTypes.TRANSIENTREAD)
			&& !HopRewriteUtils.isData(hop, DataOpTypes.TRANSIENTWRITE);
		for( Hop c : hop.getInput() ) {
			invariant &= (candInputs.contains(c.getName()) 
				|| memo.contains(c.getHopID()) || c instanceof LiteralOp);
		}
		if( invariant )
			memo.add(hop.getHopID());
		
		hop.setVisited();
	}
	
	private void rCollectAndReplaceOperations(Hop hop, Set<String> candInputs, Set<Long> memo, Map<String, Hop> invariantOps) {
		if( hop.isVisited() )
			return;
		
		//replace amenable inputs or process recursively
		//(without iterators due to parent-child modifications)
		for( int i=0; i<hop.getInput().size(); i++ ) {
			Hop c = hop.getInput().get(i);
			if( memo.contains(c.getHopID()) ) {
				String tmpName = createCutVarName(false);
				Hop tmp = Recompiler.deepCopyHopsDag(c);
				tmp.getParent().clear();
				invariantOps.put(tmpName, tmp);
				
				//create read and replace all parent references
				DataOp tread = HopRewriteUtils.createTransientRead(tmpName, c);
				List<Hop> parents = new ArrayList<>(c.getParent());
				for( Hop p : parents )
					HopRewriteUtils.replaceChildReference(p, c, tread);
			}
			else {
				rCollectAndReplaceOperations(c, candInputs, memo, invariantOps);
			}
		}
		
		hop.setVisited();
	}
	
	private StatementBlock createStatementBlock(StatementBlock sb, Map<String, Hop> invariantOps) {
		//create empty last-level statement block
		StatementBlock ret = new StatementBlock();
		ret.setDMLProg(sb.getDMLProg());
		ret.setParseInfo(sb);
		ret.setLiveIn(new VariableSet(sb.liveIn()));
		ret.setLiveOut(new VariableSet(sb.liveIn()));
		
		//append hops with custom
		ArrayList<Hop> hops = new ArrayList<>();
		for( Entry<String, Hop> e : invariantOps.entrySet() ) {
			Hop h = e.getValue();
			DataOp twrite = HopRewriteUtils.createTransientWrite(e.getKey(), h);
			hops.add(twrite);
			//update live variable analysis
			DataIdentifier diVar = new DataIdentifier(e.getKey());
			diVar.setDimensions(h.getDim1(), h.getDim2());
			diVar.setBlocksize(h.getBlocksize());
			diVar.setDataType(h.getDataType());
			diVar.setValueType(h.getValueType());
			ret.liveOut().addVariable(e.getKey(), diVar);
			sb.liveIn().addVariable(e.getKey(), diVar);
		}
		ret.setHops(hops);
		return ret;
	}
}
