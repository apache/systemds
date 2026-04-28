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

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.hops.Hop;
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
 * Prune stale parent links by keeping only parent references reachable from the statement block roots/predicates.
 */
public class IPAPassPruneUnreachableHops extends IPAPass {
	@Override
	public boolean isApplicable(FunctionCallGraph fgraph) {
		return DMLScript.USE_OOC;
	}

	@Override
	public boolean rewriteProgram(DMLProgram prog, FunctionCallGraph fgraph, FunctionCallSizeInfo fcallSizes) {
		pruneStatementBlocks(prog.getStatementBlocks());
		for(FunctionStatementBlock fsb : prog.getFunctionStatementBlocks())
			pruneStatementBlocks(((FunctionStatement) fsb.getStatement(0)).getBody());
		return false;
	}

	private static void pruneStatementBlocks(List<StatementBlock> sbs) {
		for(StatementBlock sb : sbs) {
			if(sb instanceof WhileStatementBlock) {
				WhileStatementBlock wsb = (WhileStatementBlock) sb;
				WhileStatement wstmt = (WhileStatement) sb.getStatement(0);
				pruneHops(wsb.getPredicateHops());
				pruneStatementBlocks(wstmt.getBody());
			}
			else if(sb instanceof IfStatementBlock) {
				IfStatementBlock isb = (IfStatementBlock) sb;
				IfStatement istmt = (IfStatement) sb.getStatement(0);
				pruneHops(isb.getPredicateHops());
				pruneStatementBlocks(istmt.getIfBody());
				if(istmt.getElseBody() != null)
					pruneStatementBlocks(istmt.getElseBody());
			}
			else if(sb instanceof ForStatementBlock) {
				ForStatementBlock fsb = (ForStatementBlock) sb;
				ForStatement fstmt = (ForStatement) sb.getStatement(0);
				pruneHops(fsb.getFromHops());
				pruneHops(fsb.getToHops());
				pruneHops(fsb.getIncrementHops());
				pruneStatementBlocks(fstmt.getBody());
			}
			else if(sb instanceof FunctionStatementBlock) {
				FunctionStatement fstmt = (FunctionStatement) sb.getStatement(0);
				pruneStatementBlocks(fstmt.getBody());
			}
			else {
				pruneHops(sb.getHops());
			}
		}
	}

	private static void pruneHops(Hop root) {
		if(root == null)
			return;
		Set<Long> reachable = new HashSet<>();
		collectReachable(root, reachable);
		pruneParents(root, reachable, new HashSet<Long>());
	}

	private static void pruneHops(List<Hop> roots) {
		if(roots == null || roots.isEmpty())
			return;

		Set<Long> reachable = new HashSet<>();
		for(Hop root : roots)
			collectReachable(root, reachable);

		for(Hop root : roots)
			pruneParents(root, reachable, new HashSet<Long>());
	}

	private static void collectReachable(Hop hop, Set<Long> reachable) {
		if(hop == null || !reachable.add(hop.getHopID()))
			return;
		for(Hop in : hop.getInput())
			collectReachable(in, reachable);
	}

	private static void pruneParents(Hop hop, Set<Long> reachable, Set<Long> visited) {
		if(hop == null || !visited.add(hop.getHopID()))
			return;
		hop.getParent().removeIf(p -> !reachable.contains(p.getHopID()));
		for(Hop in : hop.getInput())
			pruneParents(in, reachable, visited);
	}
}
