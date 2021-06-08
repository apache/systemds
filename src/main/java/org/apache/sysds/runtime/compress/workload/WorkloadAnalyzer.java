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

package org.apache.sysds.runtime.compress.workload;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.hops.rewrite.RewriteCompressedReblock;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.FunctionStatement;
import org.apache.sysds.parser.FunctionStatementBlock;
import org.apache.sysds.parser.IfStatement;
import org.apache.sysds.parser.IfStatementBlock;
import org.apache.sysds.parser.ParForStatementBlock;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.WhileStatement;
import org.apache.sysds.parser.WhileStatementBlock;
import org.apache.sysds.runtime.compress.workload.AWTreeNode.WTNodeType;

public class WorkloadAnalyzer {
	protected static final Log LOG = LogFactory.getLog(WorkloadAnalyzer.class.getName());

	public static Map<Long, WTreeRoot> getAllCandidateWorkloads(DMLProgram prog) {
		// extract all compression candidates from program
		List<Hop> candidates = getCandidates(prog);

		// for each candidate, create pruned workload tree
		// TODO memoization of processed subtree if overlap
		Map<Long, WTreeRoot> map = new HashMap<>();
		for(Hop cand : candidates) {
			WTreeRoot tree = createWorkloadTree(prog, cand);
			pruneWorkloadTree(tree);
			map.put(cand.getHopID(), tree);
		}

		return map;
	}

	private static List<Hop> getCandidates(DMLProgram prog) {
		List<Hop> candidates = new ArrayList<>();
		for(StatementBlock sb : prog.getStatementBlocks())
			getCandidates(sb, prog, candidates, new HashSet<>());
		return candidates;
	}

	private static WTreeRoot createWorkloadTree(DMLProgram prog, Hop candidate) {
		WTreeRoot main = new WTreeRoot(candidate);
		// TODO generalize, below line assumes only pread candidates (at bottom on DAGs)
		Set<String> compressed = new HashSet<>();
		compressed.add(candidate.getName());
		for(StatementBlock sb : prog.getStatementBlocks())
			main.addChild(createWorkloadTree(sb, prog, compressed, new HashSet<>()));
		return main;
	}

	private static boolean pruneWorkloadTree(AWTreeNode node) {
		// recursively process sub trees
		Iterator<WTreeNode> iter = node.getChildNodes().iterator();
		while(iter.hasNext()) {
			if(pruneWorkloadTree(iter.next()))
				iter.remove();
		}

		// indicate that node can be removed
		return node.isEmpty();
	}

	private static void getCandidates(StatementBlock sb, DMLProgram prog, List<Hop> cands, Set<String> fStack) {
		if(sb instanceof FunctionStatementBlock) {
			FunctionStatementBlock fsb = (FunctionStatementBlock) sb;
			FunctionStatement fstmt = (FunctionStatement) fsb.getStatement(0);
			for(StatementBlock csb : fstmt.getBody())
				getCandidates(csb, prog, cands, fStack);
		}
		else if(sb instanceof WhileStatementBlock) {
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement wstmt = (WhileStatement) wsb.getStatement(0);
			for(StatementBlock csb : wstmt.getBody())
				getCandidates(csb, prog, cands, fStack);
		}
		else if(sb instanceof IfStatementBlock) {
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement istmt = (IfStatement) isb.getStatement(0);
			for(StatementBlock csb : istmt.getIfBody())
				getCandidates(csb, prog, cands, fStack);
			for(StatementBlock csb : istmt.getElseBody())
				getCandidates(csb, prog, cands, fStack);
		}
		else if(sb instanceof ForStatementBlock) { // incl parfor
			ForStatementBlock fsb = (ForStatementBlock) sb;
			ForStatement fstmt = (ForStatement) fsb.getStatement(0);
			for(StatementBlock csb : fstmt.getBody())
				getCandidates(csb, prog, cands, fStack);
		}
		else { // generic (last-level)
			if(sb.getHops() == null)
				return;
			Hop.resetVisitStatus(sb.getHops());
			for(Hop hop : sb.getHops())
				getCandidates(hop, prog, cands, fStack);
			Hop.resetVisitStatus(sb.getHops());
		}
	}

	private static void getCandidates(Hop hop, DMLProgram prog, List<Hop> cands, Set<String> fStack) {
		if(hop.isVisited())
			return;

		// evaluate and add candidates (type and size)
		if(RewriteCompressedReblock.satisfiesCompressionCondition(hop))
			cands.add(hop);

		// recursively process children (inputs)
		for(Hop c : hop.getInput())
			getCandidates(c, prog, cands, fStack);

		// process function calls with awareness of the current
		// call stack to avoid endless loops in recursive functions
		if(hop instanceof FunctionOp) {
			FunctionOp fop = (FunctionOp) hop;
			if(!fStack.contains(fop.getFunctionKey())) {
				fStack.add(fop.getFunctionKey());
				getCandidates(prog.getFunctionStatementBlock(fop.getFunctionKey()), prog, cands, fStack);
				fStack.remove(fop.getFunctionKey());
			}
		}

		hop.setVisited();
	}

	private static WTreeNode createWorkloadTree(StatementBlock sb, DMLProgram prog, Set<String> compressed,
		Set<String> fStack) {
		WTreeNode node = null;
		if(sb instanceof FunctionStatementBlock) {
			FunctionStatementBlock fsb = (FunctionStatementBlock) sb;
			FunctionStatement fstmt = (FunctionStatement) fsb.getStatement(0);
			node = new WTreeNode(WTNodeType.FCALL);
			for(StatementBlock csb : fstmt.getBody())
				node.addChild(createWorkloadTree(csb, prog, compressed, fStack));
		}
		else if(sb instanceof WhileStatementBlock) {
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement wstmt = (WhileStatement) wsb.getStatement(0);
			node = new WTreeNode(WTNodeType.WHILE);
			createWorkloadTree(wsb.getPredicateHops(), prog, node, compressed, fStack);
			for(StatementBlock csb : wstmt.getBody())
				node.addChild(createWorkloadTree(csb, prog, compressed, fStack));
		}
		else if(sb instanceof IfStatementBlock) {
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement istmt = (IfStatement) isb.getStatement(0);
			node = new WTreeNode(WTNodeType.IF);
			createWorkloadTree(isb.getPredicateHops(), prog, node, compressed, fStack);
			for(StatementBlock csb : istmt.getIfBody())
				node.addChild(createWorkloadTree(csb, prog, compressed, fStack));
			for(StatementBlock csb : istmt.getElseBody())
				node.addChild(createWorkloadTree(csb, prog, compressed, fStack));
		}
		else if(sb instanceof ForStatementBlock) { // incl parfor
			ForStatementBlock fsb = (ForStatementBlock) sb;
			ForStatement fstmt = (ForStatement) fsb.getStatement(0);
			node = new WTreeNode(sb instanceof ParForStatementBlock ? WTNodeType.PARFOR : WTNodeType.FOR);
			createWorkloadTree(fsb.getFromHops(), prog, node, compressed, fStack);
			createWorkloadTree(fsb.getToHops(), prog, node, compressed, fStack);
			createWorkloadTree(fsb.getIncrementHops(), prog, node, compressed, fStack);
			for(StatementBlock csb : fstmt.getBody())
				node.addChild(createWorkloadTree(csb, prog, compressed, fStack));
		}
		else { // generic (last-level)
			node = new WTreeNode(WTNodeType.BASIC_BLOCK);
			if(sb.getHops() != null) {
				Hop.resetVisitStatus(sb.getHops());
				// process hop DAG to collect operations
				Set<Long> compressed2 = new HashSet<>();
				for(Hop hop : sb.getHops())
					createWorkloadTree(hop, prog, node, compressed, compressed2, fStack);
				// maintain hop DAG outputs (compressed or not compressed)
				for(Hop hop : sb.getHops()) {
					if(hop instanceof FunctionOp) {
						FunctionOp fop = (FunctionOp) hop;
						if(!fStack.contains(fop.getFunctionKey())) {
							fStack.add(fop.getFunctionKey());
							FunctionStatementBlock fsb = prog.getFunctionStatementBlock(fop.getFunctionKey());
							FunctionStatement fstmt = (FunctionStatement) fsb.getStatement(0);
							Set<String> fCompressed = new HashSet<>();
							// handle propagation of compressed intermediates into functions
							List<DataIdentifier> fArgs = fstmt.getInputParams();
							for(int i = 0; i < fArgs.size(); i++)
								if(compressed2.contains(fop.getInput(i).getHopID()))
									fCompressed.add(fArgs.get(i).getName());
							node.addChild(createWorkloadTree(fsb, prog, fCompressed, fStack));
							fStack.remove(fop.getFunctionKey());
						}
					}
					else if(HopRewriteUtils.isData(hop, OpOpData.TRANSIENTWRITE)) {
						// handle propagation of compressed intermediates across blocks
						if(compressed.contains(hop.getName()) && !compressed2.contains(hop.getHopID()))
							compressed.remove(hop.getName());
						if(!compressed.contains(hop.getName()) && compressed2.contains(hop.getHopID()))
							compressed.add(hop.getName());
					}
				}
				Hop.resetVisitStatus(sb.getHops());
			}
		}

		return node;
	}

	private static void createWorkloadTree(Hop hop, DMLProgram prog, WTreeNode parent, Set<String> compressed,
		Set<String> fStack) {
		if(hop == null)
			return;
		hop.resetVisitStatus();
		createWorkloadTree(hop, prog, parent, compressed, new HashSet<>(), fStack); // see below
		hop.resetVisitStatus();
	}

	private static void createWorkloadTree(Hop hop, DMLProgram prog, WTreeNode parent, Set<String> compressed,
		Set<Long> compressed2, Set<String> fStack) {
		if(hop == null || hop.isVisited())
			return;

		// recursively process children (inputs first for propagation of compression status)
		for(Hop c : hop.getInput())
			createWorkloadTree(c, prog, parent, compressed, compressed2, fStack);

		// map statement block propagation to hop propagation
		if(HopRewriteUtils.isData(hop, OpOpData.PERSISTENTREAD, OpOpData.TRANSIENTREAD) &&
			compressed.contains(hop.getName())) {
			compressed2.add(hop.getHopID());
		}

		// collect operations on compressed intermediates or inputs
		// if any input is compressed we collect this hop as a compressed operation
		if(hop.getInput().stream().anyMatch(h -> compressed2.contains(h.getHopID()))) {
			if(!HopRewriteUtils.isData(hop, OpOpData.PERSISTENTREAD, // all, but data ops
				OpOpData.TRANSIENTREAD, OpOpData.TRANSIENTWRITE)) {
				LOG.error(hop.getClass().getSimpleName());
				if(hop instanceof AggBinaryOp) {
					AggBinaryOp agbhop = (AggBinaryOp) hop;
					List<Hop> in = agbhop.getInput();
					boolean left = compressed2.contains(in.get(0).getHopID());
					boolean right = compressed2.contains(in.get(1).getHopID());
					parent.addOp(new OpSided(hop, left, right));
				}
				else {
					parent.addOp(new Op(hop));
				}
			}

			// if the output size also qualifies for compression, we propagate this status
			if(RewriteCompressedReblock.satisfiesSizeConstraintsForCompression(hop) && hop.getDataType().isMatrix()) {
				compressed2.add(hop.getHopID());
			}
		}

		hop.setVisited();
	}
}
