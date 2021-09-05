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
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.common.Types.ReOrgOp;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.IndexingOp;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.ReorgOp;
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
import org.apache.sysds.runtime.compress.cost.CostEstimatorBuilder;
import org.apache.sysds.runtime.compress.workload.AWTreeNode.WTNodeType;

public class WorkloadAnalyzer {
	private static final Log LOG = LogFactory.getLog(WorkloadAnalyzer.class.getName());
	// indicator for more aggressive compression of intermediates
	public static boolean ALLOW_INTERMEDIATE_CANDIDATES = false;
	// avoid wtree construction for assumptionly already compressed intermediates
	// (due to conditional control flow this might miss compression opportunities)
	public static boolean PRUNE_COMPRESSED_INTERMEDIATES = true;
	
	private final Set<Hop> visited;
	private final Set<Long> compressed;
	private final Set<Long> transposed;
	private final Set<String> transientCompressed;
	private final Set<Long> overlapping;
	private final Set<String> transientOverlapping;
	private final DMLProgram prog;
	private final List<Hop> decompressHops;

	public static Map<Long, WTreeRoot> getAllCandidateWorkloads(DMLProgram prog) {
		// extract all compression candidates from program (in program order)
		List<Hop> candidates = getCandidates(prog);
		
		// for each candidate, create pruned workload tree
		List<WorkloadAnalyzer> allWAs = new LinkedList<>();
		Map<Long, WTreeRoot> map = new HashMap<>();
		for(Hop cand : candidates) {
			//prune already covered candidate (intermediate already compressed)
			if( PRUNE_COMPRESSED_INTERMEDIATES )
				if( allWAs.stream().anyMatch(w -> w.containsCompressed(cand)) )
					continue; //intermediate already compressed
			
			//construct workload tree for candidate
			WorkloadAnalyzer wa = new WorkloadAnalyzer(prog);
			WTreeRoot tree = wa.createWorkloadTree(cand);
			map.put(cand.getHopID(), tree);
			allWAs.add(wa);
		}

		return map;
	}

	protected WorkloadAnalyzer(DMLProgram prog) {
		this.prog = prog;
		this.visited = new HashSet<>();
		this.compressed = new HashSet<>();
		this.transposed = new HashSet<>();
		this.transientCompressed = new HashSet<>();
		this.overlapping = new HashSet<>();
		this.transientOverlapping = new HashSet<>();
		this.decompressHops = new ArrayList<>();
	}

	protected WorkloadAnalyzer(DMLProgram prog, Set<Long> overlapping) {
		this.prog = prog;
		this.visited = new HashSet<>();
		this.compressed = new HashSet<>();
		this.transposed = new HashSet<>();
		this.transientCompressed = new HashSet<>();
		this.overlapping = overlapping;
		this.transientOverlapping = new HashSet<>();
		this.decompressHops = new ArrayList<>();
	}

	protected WorkloadAnalyzer(DMLProgram prog, Set<Long> compressed, Set<String> transientCompressed,
		Set<Long> transposed, Set<Long> overlapping, Set<String> transientOverlapping) {
		this.prog = prog;
		this.visited = new HashSet<>();
		this.compressed = compressed;
		this.transposed = transposed;
		this.transientCompressed = transientCompressed;
		this.overlapping = overlapping;
		this.transientOverlapping = transientOverlapping;
		this.decompressHops = new ArrayList<>();
	}

	protected WTreeRoot createWorkloadTree(Hop candidate) {
		WTreeRoot main = new WTreeRoot(candidate, decompressHops);
		compressed.add(candidate.getHopID());
		transientCompressed.add(candidate.getName());
		for(StatementBlock sb : prog.getStatementBlocks())
			createWorkloadTree(main, sb, prog, new HashSet<>());
		pruneWorkloadTree(main);
		return main;
	}

	protected boolean containsCompressed(Hop hop) {
		return compressed.contains(hop.getHopID());
	}
	
	private static List<Hop> getCandidates(DMLProgram prog) {
		List<Hop> candidates = new ArrayList<>();
		for(StatementBlock sb : prog.getStatementBlocks()) {
			getCandidates(sb, prog, candidates, new HashSet<>());
		}
		return candidates;
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
		if(sb == null)
			return;
		else if(sb instanceof FunctionStatementBlock) {
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
		if( (  RewriteCompressedReblock.satisfiesAggressiveCompressionCondition(hop)
				& ALLOW_INTERMEDIATE_CANDIDATES)
			|| RewriteCompressedReblock.satisfiesCompressionCondition(hop))
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

	private void createWorkloadTree(AWTreeNode n, StatementBlock sb, DMLProgram prog, Set<String> fStack) {
		WTreeNode node;
		if(sb instanceof FunctionStatementBlock) {
			FunctionStatementBlock fsb = (FunctionStatementBlock) sb;
			FunctionStatement fstmt = (FunctionStatement) fsb.getStatement(0);
			node = new WTreeNode(WTNodeType.FCALL, 1);
			for(StatementBlock csb : fstmt.getBody())
				createWorkloadTree(node, csb, prog, fStack);
		}
		else if(sb instanceof WhileStatementBlock) {
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement wstmt = (WhileStatement) wsb.getStatement(0);
			node = new WTreeNode(WTNodeType.WHILE, 10);
			createWorkloadTree(wsb.getPredicateHops(), prog, node, fStack);

			for(StatementBlock csb : wstmt.getBody())
				createWorkloadTree(node, csb, prog, fStack);
		}
		else if(sb instanceof IfStatementBlock) {
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement istmt = (IfStatement) isb.getStatement(0);
			node = new WTreeNode(WTNodeType.IF, 1);
			createWorkloadTree(isb.getPredicateHops(), prog, node, fStack);

			for(StatementBlock csb : istmt.getIfBody())
				createWorkloadTree(node, csb, prog, fStack);
			for(StatementBlock csb : istmt.getElseBody())
				createWorkloadTree(node, csb, prog, fStack);
		}
		else if(sb instanceof ForStatementBlock) { // incl parfor
			ForStatementBlock fsb = (ForStatementBlock) sb;
			ForStatement fstmt = (ForStatement) fsb.getStatement(0);
			node = new WTreeNode(sb instanceof ParForStatementBlock ? WTNodeType.PARFOR : WTNodeType.FOR,
				fsb.getEstimateReps());
			createWorkloadTree(fsb.getFromHops(), prog, node, fStack);
			createWorkloadTree(fsb.getToHops(), prog, node, fStack);
			createWorkloadTree(fsb.getIncrementHops(), prog, node, fStack);
			for(StatementBlock csb : fstmt.getBody())
				createWorkloadTree(node, csb, prog, fStack);

		}
		else { // generic (last-level)
			if(sb.getHops() != null) {

				// process hop DAG to collect operations that are compressed.
				for(Hop hop : sb.getHops())
					createWorkloadTree(hop, prog, n, fStack);

				// maintain hop DAG outputs (compressed or not compressed)
				for(Hop hop : sb.getHops()) {
					if(hop instanceof FunctionOp) {
						FunctionOp fop = (FunctionOp) hop;
						if(!fStack.contains(fop.getFunctionKey())) {
							fStack.add(fop.getFunctionKey());
							FunctionStatementBlock fsb = prog.getFunctionStatementBlock(fop.getFunctionKey());
							if(fsb == null)
								continue;
							FunctionStatement fstmt = (FunctionStatement) fsb.getStatement(0);
							Set<String> fCompressed = new HashSet<>();
							// handle propagation of compressed intermediates into functions
							List<DataIdentifier> fArgs = fstmt.getInputParams();
							for(int i = 0; i < fArgs.size(); i++)
								if(compressed.contains(fop.getInput(i).getHopID()) ||
									transientCompressed.contains(fop.getInput(i).getName()))
									fCompressed.add(fArgs.get(i).getName());
							WorkloadAnalyzer fa = new WorkloadAnalyzer(prog, compressed, fCompressed, transposed,
								overlapping, transientOverlapping);
							fa.createWorkloadTree(n, fsb, prog, fStack);
							List<DataIdentifier> fOut = fstmt.getOutputParams();
							String[] outs = fop.getOutputVariableNames();
							for(int i = 0; i < outs.length; i++)
								if(fCompressed.contains(fOut.get(i).getName())) {
									transientCompressed.add(outs[i]);
								}
							fStack.remove(fop.getFunctionKey());
						}
					}

				}
			}
			return;
		}
		n.addChild(node);
	}

	private void createWorkloadTree(Hop hop, DMLProgram prog, AWTreeNode parent, Set<String> fStack) {
		if(hop == null || visited.contains(hop) || isNoOp(hop))
			return;

		// DFS: recursively process children (inputs first for propagation of compression status)
		for(Hop c : hop.getInput())
			createWorkloadTree(c, prog, parent, fStack);

		// map statement block propagation to hop propagation
		if(HopRewriteUtils.isData(hop, OpOpData.PERSISTENTREAD, OpOpData.TRANSIENTREAD) &&
			transientCompressed.contains(hop.getName())) {
			compressed.add(hop.getHopID());
			if(transientOverlapping.contains(hop.getName()))
				overlapping.add(hop.getHopID());
		}

		if(LOG.isTraceEnabled()) {
			LOG.trace("\n" + compressed + "\n" + transientCompressed + "\n" + getHopIds(hop.getInput()) + "\n"
				+ hop.getInput() + "\n\n");
		}

		// collect operations on compressed intermediates or inputs
		// if any input is compressed we collect this hop as a compressed operation
		if(hop.getInput().stream().anyMatch(h -> compressed.contains(h.getHopID()))) {

			if(isCompressedOp(hop)) {
				Op o = createOp(hop);
				parent.addOp(o);
				if(o.isCompressedOutput())
					compressed.add(hop.getHopID());
			}
			else if(HopRewriteUtils.isData(hop, OpOpData.TRANSIENTWRITE)) {
				Hop in = hop.getInput().get(0);
				if(compressed.contains(hop.getHopID()) || compressed.contains(in.getHopID()) ||
					transientCompressed.contains(in.getName())) {
					transientCompressed.add(hop.getName());
				}
				if(overlapping.contains(hop.getHopID()) || overlapping.contains(in.getHopID()) ||
					transientOverlapping.contains(in.getName())) {
					transientOverlapping.add(hop.getName());
				}
			}
		}

		visited.add(hop);
	}

	private Op createOp(Hop hop) {
		if(hop.getDataType().isMatrix()) {
			if(hop instanceof ReorgOp && ((ReorgOp) hop).getOp() == ReOrgOp.TRANS) {
				transposed.add(hop.getHopID());
				compressed.add(hop.getHopID());
				transientCompressed.add(hop.getName());
				return new OpMetadata(hop);
			}
			else if(hop instanceof AggBinaryOp) {
				AggBinaryOp agbhop = (AggBinaryOp) hop;
				List<Hop> in = agbhop.getInput();
				boolean transposedLeft = transposed.contains(in.get(0).getHopID());
				boolean transposedRight = transposed.contains(in.get(1).getHopID());
				boolean left = compressed.contains(in.get(0).getHopID()) ||
					transientCompressed.contains(in.get(0).getName());
				boolean right = compressed.contains(in.get(1).getHopID()) ||
					transientCompressed.contains(in.get(1).getName());
				OpSided ret = new OpSided(hop, left, right, transposedLeft, transposedRight);
				if(ret.isRightMM()) {
					HashSet<Long> overlapping2 = new HashSet<>();
					overlapping2.add(hop.getHopID());
					WorkloadAnalyzer overlappingAnalysis = new WorkloadAnalyzer(prog, overlapping2);
					WTreeRoot r = overlappingAnalysis.createWorkloadTree(hop);

					CostEstimatorBuilder b = new CostEstimatorBuilder(r);
					if(LOG.isTraceEnabled())
						LOG.trace("Workload for overlapping: " + r + "\n" + b);

					if(b.shouldUseOverlap())
						overlapping.add(hop.getHopID());
					else {
						decompressHops.add(hop);
						ret.setOverlappingDecompression(true);
					}
				}

				return ret;
			}
			else if(HopRewriteUtils.isBinary(hop, OpOp2.CBIND)) {
				ArrayList<Hop> in = hop.getInput();
				if(isOverlapping(in.get(0)) || isOverlapping(in.get(1)))
					overlapping.add(hop.getHopID());
				return new OpNormal(hop, true);
			}
			else if(HopRewriteUtils.isBinary(hop, OpOp2.RBIND)) {
				ArrayList<Hop> in = hop.getInput();
				if(isOverlapping(in.get(0)) || isOverlapping(in.get(1)))
					return new OpOverlappingDecompress(hop);
				else
					return new OpDecompressing(hop);
			}
			else if(HopRewriteUtils.isBinaryMatrixScalarOperation(hop) ||
				HopRewriteUtils.isBinaryMatrixRowVectorOperation(hop)) {
				ArrayList<Hop> in = hop.getInput();
				if(isOverlapping(in.get(0)) || isOverlapping(in.get(1)))
					overlapping.add(hop.getHopID());

				return new OpNormal(hop, true);
			}
			else if(hop instanceof IndexingOp) {
				IndexingOp idx = (IndexingOp) hop;
				final boolean isOverlapping = isOverlapping(hop.getInput(0));
				final boolean fullColumn = HopRewriteUtils.isFullColumnIndexing(idx);
				if(fullColumn && isOverlapping)
					overlapping.add(hop.getHopID());

				if(fullColumn)
					return new OpNormal(hop, RewriteCompressedReblock.satisfiesSizeConstraintsForCompression(hop));
				else
					return new OpDecompressing(hop);
			}
			else if(HopRewriteUtils.isBinaryMatrixMatrixOperation(hop) ||
				HopRewriteUtils.isBinaryMatrixColVectorOperation(hop)) {
				ArrayList<Hop> in = hop.getInput();
				if(isOverlapping(in.get(0)) || isOverlapping(in.get(1)))
					return new OpOverlappingDecompress(hop);

				return new OpDecompressing(hop);
			}

			// if the output size also qualifies for compression, we propagate this status
			// return new OpNormal(hop, RewriteCompressedReblock.satisfiesSizeConstraintsForCompression(hop));
			return new OpNormal(hop, RewriteCompressedReblock.satisfiesSizeConstraintsForCompression(hop));
		}
		else
			return new OpNormal(hop, false);
	}

	private boolean isOverlapping(Hop hop) {
		return overlapping.contains(hop.getHopID()) || transientOverlapping.contains(hop.getName());
	}

	private static boolean isCompressedOp(Hop hop) {
		return !(HopRewriteUtils.isData(hop, OpOpData.PERSISTENTREAD, // all, but data ops
			OpOpData.TRANSIENTREAD, OpOpData.TRANSIENTWRITE));
	}

	private static boolean isNoOp(Hop hop) {
		return hop instanceof LiteralOp || HopRewriteUtils.isUnary(hop, OpOp1.NROW, OpOp1.NCOL);
	}

	private static String getHopIds(List<Hop> hops) {
		StringBuilder sb = new StringBuilder();
		for(Hop h : hops) {
			sb.append(h.getHopID() + " , ");
		}
		return sb.toString();
	}
}
