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

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;

import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.HopsException;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.FunctionStatement;
import org.apache.sysds.parser.FunctionStatementBlock;
import org.apache.sysds.parser.IfStatement;
import org.apache.sysds.parser.IfStatementBlock;
import org.apache.sysds.parser.LanguageException;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.WhileStatement;
import org.apache.sysds.parser.WhileStatementBlock;
import org.apache.sysds.runtime.lineage.LineageCacheConfig;

public class IPAPassFlagNonDeterminism extends IPAPass {
	@Override
	public boolean isApplicable(FunctionCallGraph fgraph) {
		return InterProceduralAnalysis.REMOVE_UNUSED_FUNCTIONS
			&& !fgraph.containsSecondOrderCall();
	}

	@Override
	public boolean rewriteProgram (DMLProgram prog, FunctionCallGraph fgraph, FunctionCallSizeInfo fcallSizes) 
	{
		if (!LineageCacheConfig.isMultiLevelReuse())
			return false;
		
		try {
			// Find the individual functions and statementblocks with non-determinism.
			HashSet<String> ndfncs = new HashSet<>();
			for (String fkey : fgraph.getReachableFunctions()) {
				FunctionStatementBlock fsblock = prog.getFunctionStatementBlock(fkey);
				FunctionStatement fnstmt = (FunctionStatement)fsblock.getStatement(0);
				String fname = DMLProgram.splitFunctionKey(fkey)[1];
				if (rIsNonDeterministicFnc(fname, fnstmt.getBody()))
					ndfncs.add(fkey);
			}

			// Find the callers of the nondeterministic functions.
			propagate2Callers(fgraph, ndfncs, new HashSet<String>(), null);
			
			// Mark the corresponding FunctionStatementBlocks
			ndfncs.forEach(fkey -> {
				FunctionStatementBlock fsblock = prog.getFunctionStatementBlock(fkey); 
				fsblock.setNondeterministic(true);
			});
			
			// Find and mark the StatementBlocks having calls to nondeterministic functions.
			rMarkNondeterministicSBs(prog.getStatementBlocks(), ndfncs);
			for (String fkey : fgraph.getReachableFunctions()) {
				FunctionStatementBlock fsblock = prog.getFunctionStatementBlock(fkey);
				FunctionStatement fnstmt = (FunctionStatement)fsblock.getStatement(0);
				rMarkNondeterministicSBs(fnstmt.getBody(), ndfncs);
			}
		}
		catch( LanguageException ex ) {
			throw new HopsException(ex);
		}
		return false;
	}

	private boolean rIsNonDeterministicFnc (String fname, ArrayList<StatementBlock> sbs) 
	{
		boolean isND = false;
		for (StatementBlock sb : sbs)
		{
			if (isND)
				break;

			if (sb instanceof ForStatementBlock) {
				ForStatement fstmt = (ForStatement)sb.getStatement(0);
				isND = rIsNonDeterministicFnc(fname, fstmt.getBody());
			}
			else if (sb instanceof WhileStatementBlock) {
				WhileStatement wstmt = (WhileStatement)sb.getStatement(0);
				isND = rIsNonDeterministicFnc(fname, wstmt.getBody());
			}
			else if (sb instanceof IfStatementBlock) {
				IfStatement ifstmt = (IfStatement)sb.getStatement(0);
				isND = rIsNonDeterministicFnc(fname, ifstmt.getIfBody());
				if (ifstmt.getElseBody() != null)
					isND = rIsNonDeterministicFnc(fname, ifstmt.getElseBody());
			}
			else {
				if (sb.getHops() != null) {
					Hop.resetVisitStatus(sb.getHops());
					for (Hop hop : sb.getHops()) 
						isND |= rIsNonDeterministicHop(hop);
					Hop.resetVisitStatus(sb.getHops());
					// Mark the statementblock
					sb.setNondeterministic(isND);
				}
			}
		}
		return isND;
	}
	
	private void rMarkNondeterministicSBs (ArrayList<StatementBlock> sbs, HashSet<String> ndfncs)
	{
		for (StatementBlock sb : sbs)
		{
			if (sb instanceof ForStatementBlock) {
				ForStatement fstmt = (ForStatement)sb.getStatement(0);
				rMarkNondeterministicSBs(fstmt.getBody(), ndfncs);
			}
			else if (sb instanceof WhileStatementBlock) {
				WhileStatement wstmt = (WhileStatement)sb.getStatement(0);
				rMarkNondeterministicSBs(wstmt.getBody(), ndfncs);
			}
			else if (sb instanceof IfStatementBlock) {
				IfStatement ifstmt = (IfStatement)sb.getStatement(0);
				rMarkNondeterministicSBs(ifstmt.getIfBody(), ndfncs);
				if (ifstmt.getElseBody() != null)
					rMarkNondeterministicSBs(ifstmt.getElseBody(), ndfncs);
			}
			else {
				if (sb.getHops() != null) {
					boolean callsND = false;
					Hop.resetVisitStatus(sb.getHops());
					for (Hop hop : sb.getHops())
						callsND |= rMarkNondeterministicHop(hop, ndfncs);
					Hop.resetVisitStatus(sb.getHops());
					if (callsND)
						sb.setNondeterministic(callsND);
				}
			}
		}
	}
	
	private boolean rMarkNondeterministicHop(Hop hop, HashSet<String> ndfncs) {
		if (hop.isVisited())
			return false;

		boolean callsND = hop instanceof FunctionOp && ndfncs.contains(hop.getName());
			
		if (!callsND)
			for (Hop hi : hop.getInput())
				callsND |= rMarkNondeterministicHop(hi, ndfncs);
		hop.setVisited();
		return callsND;
	}
	
	private boolean rIsNonDeterministicHop(Hop hop) {
		if (hop.isVisited())
			return false;

		boolean isND = HopRewriteUtils.isDataGenOpWithNonDeterminism(hop);
		
		if (!isND)
			for (Hop hi : hop.getInput())
				isND |= rIsNonDeterministicHop(hi);
		hop.setVisited();
		return isND;
	}
	
	private void propagate2Callers (FunctionCallGraph fgraph, HashSet<String> ndfncs, HashSet<String> fstack, String fkey) {
		Collection<String> cfkeys = fgraph.getCalledFunctions(fkey);
		if (cfkeys != null) {
			for (String cfkey : cfkeys) {
				if (fstack.contains(cfkey) && fgraph.isRecursiveFunction(cfkey)) {
					if (ndfncs.contains(cfkey) && fkey !=null)
						ndfncs.add(fkey);
				}
				else {
					fstack.add(cfkey);
					propagate2Callers(fgraph, ndfncs, fstack, cfkey);
					fstack.remove(cfkey);
					if (ndfncs.contains(cfkey) && fkey !=null)
						ndfncs.add(fkey);
				}
			}
		}
	}
}
