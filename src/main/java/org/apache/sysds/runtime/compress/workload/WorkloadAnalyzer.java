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
import java.util.Objects;
import java.util.Set;
import java.util.Stack;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.OpOp3;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.common.Types.ParamBuiltinOp;
import org.apache.sysds.common.Types.ReOrgOp;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.IndexingOp;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.NaryOp;
import org.apache.sysds.hops.ParameterizedBuiltinOp;
import org.apache.sysds.hops.ReorgOp;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.hops.rewrite.RewriteCompressedReblock;
import org.apache.sysds.parser.DMLProgram;
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
import org.apache.sysds.utils.Explain;

public class WorkloadAnalyzer {
	private static final Log LOG = LogFactory.getLog(WorkloadAnalyzer.class.getName());
	// indicator for more aggressive compression of intermediates
	public static boolean ALLOW_INTERMEDIATE_CANDIDATES = false;
	// avoid w-tree construction for already compressed intermediates
	// (due to conditional control flow this might miss compression opportunities)
	public static boolean PRUNE_COMPRESSED_INTERMEDIATES = true;

	private final Set<Hop> visited;
	private final Set<Long> compressed;
	private final Set<Long> transposed;
	private final Map<String, Long> transientCompressed;
	private final Set<Long> overlapping;
	private final DMLProgram prog;
	private final Map<Long, Op> treeLookup;
	private final Stack<Hop> stack;
	private Stack<StatementBlock> lineage = new Stack<>();

	public static Map<Long, WTreeRoot> getAllCandidateWorkloads(DMLProgram prog) {
		// extract all compression candidates from program (in program order)
		String configValue = ConfigurationManager.getDMLConfig()
				.getTextValue(DMLConfig.COMPRESSED_LINALG_INTERMEDIATE);
		// if set update it, otherwise keep it as set before
		ALLOW_INTERMEDIATE_CANDIDATES = configValue != null && Objects.equals(configValue.toUpperCase(), "TRUE") ||
										configValue == null && ALLOW_INTERMEDIATE_CANDIDATES;
		List<Hop> candidates = getCandidates(prog);

		// for each candidate, create pruned workload tree
		List<WorkloadAnalyzer> allWAs = new LinkedList<>();
		Map<Long, WTreeRoot> map = new HashMap<>();
		for(Hop cand : candidates) {
			// prune already covered candidate (intermediate already compressed)
			if(PRUNE_COMPRESSED_INTERMEDIATES)
				if(allWAs.stream().anyMatch(w -> w.containsCompressed(cand)))
					continue; // intermediate already compressed

			// construct workload tree for candidate
			WorkloadAnalyzer wa = new WorkloadAnalyzer(prog);
			WTreeRoot tree = wa.createWorkloadTreeRoot(cand);

			map.put(cand.getHopID(), tree);
			allWAs.add(wa);
		}

		return map;
	}

	private WorkloadAnalyzer(DMLProgram prog) {
		this.prog = prog;
		this.visited = new HashSet<>();
		this.compressed = new HashSet<>();
		this.transposed = new HashSet<>();
		this.transientCompressed = new HashMap<>();
		this.overlapping = new HashSet<>();
		this.treeLookup = new HashMap<>();
		this.stack = new Stack<>();
		this.lineage = new Stack<>();
	}

	private WorkloadAnalyzer(DMLProgram prog, Set<Long> compressed, HashMap<String, Long> transientCompressed,
		Set<Long> transposed, Set<Long> overlapping, Map<Long, Op> treeLookup) {
		this.prog = prog;
		this.visited = new HashSet<>();
		this.compressed = compressed;
		this.transposed = transposed;
		this.transientCompressed = transientCompressed;
		this.overlapping = overlapping;
		this.treeLookup = treeLookup;
		this.stack = new Stack<>();
	}

	private WTreeRoot createWorkloadTreeRoot(Hop candidate) {
		WTreeRoot main = new WTreeRoot(candidate);
		compressed.add(candidate.getHopID());
		if(HopRewriteUtils.isTransformEncode(candidate)) {
			Hop matrix = ((FunctionOp) candidate).getOutputs().get(0);
			compressed.add(matrix.getHopID());
			transientCompressed.put(matrix.getName(), matrix.getHopID());
		}
		for(StatementBlock sb : prog.getStatementBlocks())
			createWorkloadTreeNodes(main, sb, prog, new HashSet<>());

		pruneWorkloadTree(main);
		return main;
	}

	private boolean containsCompressed(Hop hop) {
		return compressed.contains(hop.getHopID());
	}

	private static List<Hop> getCandidates(DMLProgram prog) {
		List<Hop> candidates = new ArrayList<>();
		for(StatementBlock sb : prog.getStatementBlocks())
			getCandidates(sb, prog, candidates, new HashSet<>());

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
			ArrayList<Hop> hops = sb.getHops();
			if(hops != null) {
				Hop.resetVisitStatus(hops);
				for(Hop hop : hops)
					getCandidates(hop, prog, cands, fStack);
				Hop.resetVisitStatus(hops);
			}
		}
	}

	private static void getCandidates(Hop hop, DMLProgram prog, List<Hop> cands, Set<String> fStack) {
		if(hop.isVisited())
			return;
		// evaluate and add candidates (type and size)
		if((ALLOW_INTERMEDIATE_CANDIDATES && RewriteCompressedReblock.satisfiesAggressiveCompressionCondition(hop)) ||
			RewriteCompressedReblock.satisfiesCompressionCondition(hop))
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

	private void createWorkloadTreeNodes(AWTreeNode n, StatementBlock sb, DMLProgram prog, Set<String> fStack) {
		WTreeNode node;
		lineage.add(sb);
		if(sb instanceof FunctionStatementBlock) {
			FunctionStatementBlock fsb = (FunctionStatementBlock) sb;
			FunctionStatement fstmt = (FunctionStatement) fsb.getStatement(0);
			node = new WTreeNode(WTNodeType.FCALL, 1);
			for(StatementBlock csb : fstmt.getBody())
				createWorkloadTreeNodes(node, csb, prog, fStack);
		}
		else if(sb instanceof WhileStatementBlock) {
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement wstmt = (WhileStatement) wsb.getStatement(0);
			node = new WTreeNode(WTNodeType.WHILE, 100);
			createWorkloadTree(wsb.getPredicateHops(), prog, node, fStack);

			for(StatementBlock csb : wstmt.getBody())
				createWorkloadTreeNodes(node, csb, prog, fStack);
		}
		else if(sb instanceof IfStatementBlock) {
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement istmt = (IfStatement) isb.getStatement(0);
			node = new WTreeNode(WTNodeType.IF, 1);
			createWorkloadTree(isb.getPredicateHops(), prog, node, fStack);

			for(StatementBlock csb : istmt.getIfBody())
				createWorkloadTreeNodes(node, csb, prog, fStack);
			for(StatementBlock csb : istmt.getElseBody())
				createWorkloadTreeNodes(node, csb, prog, fStack);
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
				createWorkloadTreeNodes(node, csb, prog, fStack);

		}
		else { // generic (last-level)
			ArrayList<Hop> hops = sb.getHops();

			if(hops != null) {

				// process hop DAG to collect operations that are compressed.
				for(Hop hop : hops) {
					createWorkloadTree(hop, prog, n, fStack);
					// createStack(hop);
					// processStack(prog, n, fStack);
				}

				// maintain hop DAG outputs (compressed or not compressed)
				for(Hop hop : hops) {
					if(hop instanceof FunctionOp) {
						FunctionOp fop = (FunctionOp) hop;
						if(HopRewriteUtils.isTransformEncode(fop))
							break;
						else if(!fStack.contains(fop.getFunctionKey())) {
							fStack.add(fop.getFunctionKey());
							FunctionStatementBlock fsb = prog.getFunctionStatementBlock(fop.getFunctionKey());
							if(fsb == null)
								continue;

							HashMap<String, Long> fCompressed = new HashMap<>();
							// handle propagation of compressed intermediates into functions

							String[] ins = fop.getInputVariableNames();
							for(int i = 0; i < ins.length; i++) {
								final String name = ins[i];
								final Long outsideID = fop.getInput(i).getHopID();
								if(compressed.contains(outsideID))
									fCompressed.put(name, outsideID);
							}

							WorkloadAnalyzer fa = new WorkloadAnalyzer(prog, compressed, fCompressed, transposed, overlapping,
								treeLookup);
							fa.createWorkloadTreeNodes(n, fsb, prog, fStack);
							String[] outs = fop.getOutputVariableNames();
							for(int i = 0; i < outs.length; i++) {
								Long id = fCompressed.get(outs[i]);
								if(id != null)
									transientCompressed.put(outs[i], id);
							}
							fStack.remove(fop.getFunctionKey());
						}
					}
				}
			}
			lineage.pop();
			return;
		}
		n.addChild(node);
		lineage.pop();
	}

	private void createStack(Hop hop) {
		if(hop == null || visited.contains(hop) || isNoOp(hop))
			return;
		stack.add(hop);
		for(Hop c : hop.getInput())
			createStack(c);

		visited.add(hop);
	}

	private void createWorkloadTree(Hop hop, DMLProgram prog, AWTreeNode parent, Set<String> fStack) {
		createStack(hop);
		processStack(prog, parent, fStack);
	}

	private void processStack(DMLProgram prog, AWTreeNode parent, Set<String> fStack) {

		while(!stack.isEmpty()) {
			Hop hop = stack.pop();

			// map statement block propagation to hop propagation
			if(HopRewriteUtils.isData(hop, OpOpData.PERSISTENTREAD, OpOpData.TRANSIENTREAD) &&
				transientCompressed.containsKey(hop.getName())) {
				compressed.add(hop.getHopID());
				treeLookup.put(hop.getHopID(), treeLookup.get(transientCompressed.get(hop.getName())));
			}
			else {

				// collect operations on compressed intermediates or inputs
				// if any input is compressed we collect this hop as a compressed operation
				if(hop.getInput().stream().anyMatch(h -> compressed.contains(h.getHopID())))
					createOp(hop, parent);

			}
		}

	}

	private void createOp(Hop hop, AWTreeNode parent) {

		if(hop.getDataType().isMatrix()) {
			Op o = null;
			if(HopRewriteUtils.isData(hop, OpOpData.PERSISTENTREAD, OpOpData.TRANSIENTREAD))
				return;
			else if(HopRewriteUtils.isData(hop, OpOpData.TRANSIENTWRITE, OpOpData.PERSISTENTWRITE)) {
				transientCompressed.put(hop.getName(), hop.getInput(0).getHopID());
				compressed.add(hop.getHopID());
				o = new OpMetadata(hop, hop.getInput(0));
				if(isOverlapping(hop.getInput(0)))
					o.setOverlapping();
			}
			else if(hop instanceof ReorgOp && ((ReorgOp) hop).getOp() == ReOrgOp.TRANS) {
				transposed.add(hop.getHopID());
				compressed.add(hop.getHopID());
				// hack add to transient compressed since the decompression is marking the parents.
				transientCompressed.put(hop.getName(), hop.getHopID());
				// transientCompressed.add(hop.getName());
				o = new OpMetadata(hop, hop.getInput(0));
				if(isOverlapping(hop.getInput(0)))
					o.setOverlapping();
			}
			else if(hop instanceof AggUnaryOp) {
				if((isOverlapping(hop.getInput().get(0)) && !HopRewriteUtils.isAggUnaryOp(hop, AggOp.SUM, AggOp.MEAN)) ||
					HopRewriteUtils.isAggUnaryOp(hop, AggOp.TRACE)) {
					setDecompressionOnAllInputs(hop, parent);
					return;
				}
				else {
					boolean compressedOut = false;
					Hop parentHop = hop.getInput(0);
					if(HopRewriteUtils.isBinary(parentHop, OpOp2.EQUAL, OpOp2.NOTEQUAL, OpOp2.LESS,
							OpOp2.LESSEQUAL, OpOp2.GREATER, OpOp2.GREATEREQUAL)){
						Hop leftIn = parentHop.getInput(0);
						Hop rightIn = parentHop.getInput(1);
						// input ops might be not in the current statement block -> check for transient reads
						if(HopRewriteUtils.isAggUnaryOp(leftIn, AggOp.MIN, AggOp.MAX) ||
								HopRewriteUtils.isAggUnaryOp(rightIn, AggOp.MIN, AggOp.MAX) ||
								checkTransientRead(hop, leftIn) ||
								checkTransientRead(hop, rightIn)
						)
							compressedOut = true;

					}
					o = new OpNormal(hop, compressedOut);
				}
			}
			else if(hop instanceof UnaryOp) {
				if(!HopRewriteUtils.isUnary(hop, OpOp1.MULT2, OpOp1.MINUS1_MULT, OpOp1.MINUS_RIGHT, OpOp1.CAST_AS_MATRIX)) {
					if(isOverlapping(hop.getInput(0))) {
						treeLookup.get(hop.getInput(0).getHopID()).setDecompressing();
						return;
					}

				}
				else if(HopRewriteUtils.isUnary(hop, OpOp1.DETECTSCHEMA)) {
					o = new OpNormal(hop, false);
				}
			}
			else if(hop instanceof AggBinaryOp) {
				AggBinaryOp agbhop = (AggBinaryOp) hop;
				List<Hop> in = agbhop.getInput();
				boolean transposedLeft = transposed.contains(in.get(0).getHopID());
				boolean transposedRight = transposed.contains(in.get(1).getHopID());
				boolean left = compressed.contains(in.get(0).getHopID()) ||
					transientCompressed.containsKey(in.get(0).getName());
				boolean right = compressed.contains(in.get(1).getHopID()) ||
					transientCompressed.containsKey(in.get(1).getName());
				OpSided ret = new OpSided(hop, left, right, transposedLeft, transposedRight);

				if(ret.isRightMM()) {
					overlapping.add(hop.getHopID());
					ret.setOverlapping();
					if(!ret.isCompressedOutput())
						ret.setDecompressing();
				}
				o = ret;
			}
			else if(hop instanceof BinaryOp) {
				if(HopRewriteUtils.isBinary(hop, OpOp2.CBIND)) {
					List<Hop> in = hop.getInput();
					o = new OpNormal(hop, true);
					if(isOverlapping(in.get(0)) || isOverlapping(in.get(1))) {
						overlapping.add(hop.getHopID());
						o.setOverlapping();
					}
					// assume that CBind have to decompress, but only such that it also have the compressed version
					// available. Therefore add a new OpNormal, set to decompressing.
					o.setDecompressing();
				}
				else if(HopRewriteUtils.isBinary(hop, OpOp2.RBIND)) {
					setDecompressionOnAllInputs(hop, parent);
					return;
				}
				else if(HopRewriteUtils.isBinary(hop, OpOp2.APPLY_SCHEMA)) {
					o = new OpNormal(hop, true);
				}
				else {
					List<Hop> in = hop.getInput();
					final boolean ol0 = isOverlapping(in.get(0));
					final boolean ol1 = isOverlapping(in.get(1));
					final boolean ol = ol0 || ol1;

					// shortcut instead of comparing to MatrixScalar or RowVector.
					if(in.get(1).getDim1() == 1 || in.get(1).isScalar() || in.get(0).isScalar()) {

						if(ol && HopRewriteUtils.isBinary(hop, OpOp2.PLUS, OpOp2.MULT, OpOp2.DIV, OpOp2.MINUS)) {
							overlapping.add(hop.getHopID());
							o = new OpNormal(hop, true);
							o.setOverlapping();
						}
						else if(ol) {
							if(in.get(0) != null) {
								Op oo = treeLookup.get(in.get(0).getHopID());
								if(oo != null)
									oo.setDecompressing();
							}
							return;
						}
						else {
							o = new OpNormal(hop, true);
						}
						if(!HopRewriteUtils.isBinarySparseSafe(hop))
							o.setDensifying();

					} else if(HopRewriteUtils.isBinaryMatrixColVectorOperation(hop) ) {
						Hop leftIn = hop.getInput(0);
						Hop rightIn = hop.getInput(1);
						if(HopRewriteUtils.isBinary(hop, OpOp2.DIV) && rightIn instanceof AggUnaryOp && leftIn == rightIn.getInput(0)){
							o = new OpNormal(hop, true);
						} else {
							setDecompressionOnAllInputs(hop, parent);
							return;
						}
					}
					else if(HopRewriteUtils.isBinaryMatrixMatrixOperation(hop) ||
						HopRewriteUtils.isBinaryMatrixMatrixOperationWithSharedInput(hop)) {
						setDecompressionOnAllInputs(hop, parent);
						return;
					}
					else if(ol0 || ol1) {
						setDecompressionOnAllInputs(hop, parent);
						return;
					}
					else {
						String ex = "Setting decompressed because input Binary Op dimensions is unknown:\n"
							+ Explain.explain(hop);
						LOG.warn(ex);
						setDecompressionOnAllInputs(hop, parent);
						return;
					}
				}

			}
			else if(hop instanceof IndexingOp) {
				final boolean isOverlapping = isOverlapping(hop.getInput(0));
				o = new OpNormal(hop, true);
				if(isOverlapping) {
					overlapping.add(hop.getHopID());
					o.setOverlapping();
				}
			}
			else if(HopRewriteUtils.isTernary(hop, OpOp3.MINUS_MULT, OpOp3.PLUS_MULT, OpOp3.QUANTILE, OpOp3.CTABLE)) {
				setDecompressionOnAllInputs(hop, parent);
				return;
			}
			else if(HopRewriteUtils.isTernary(hop, OpOp3.IFELSE)) {
				final Hop o1 = hop.getInput(1);
				final Hop o2 = hop.getInput(2);
				if(isCompressed(o1) && isCompressed(o2)) {
					o = new OpMetadata(hop, o1);
					if(isOverlapping(o1) || isOverlapping(o2))
						o.setOverlapping();
				}
				else if(isCompressed(o1)) {
					o = new OpMetadata(hop, o1);
					if(isOverlapping(o1))
						o.setOverlapping();
				}
				else if(isCompressed(o2)) {
					o = new OpMetadata(hop, o2);
					if(isOverlapping(o2))
						o.setOverlapping();
				}
				else {
					setDecompressionOnAllInputs(hop, parent);
				}
			}
			else if(hop instanceof ParameterizedBuiltinOp) {
				if(HopRewriteUtils.isParameterizedBuiltinOp(hop, ParamBuiltinOp.REPLACE, ParamBuiltinOp.TRANSFORMAPPLY)) {
					o = new OpNormal(hop, true);
				}
				else {
					LOG.warn("Unknown ParameterizedBuiltinOp Hop:" + hop.getClass().getSimpleName() + "\n" + Explain.explain(hop));
					setDecompressionOnAllInputs(hop, parent);
					return;
				}
			}
			else if(hop instanceof NaryOp) {
				setDecompressionOnAllInputs(hop, parent);
				return;
			}
			else if(hop instanceof ReorgOp){
				setDecompressionOnAllInputs(hop, parent);
				return;
			}
			else {
				LOG.warn("Unknown Matrix Hop:" + hop.getClass().getSimpleName() + "\n" + Explain.explain(hop));
				setDecompressionOnAllInputs(hop, parent);
				return;
			}

			o = o != null ? o : new OpNormal(hop, RewriteCompressedReblock.satisfiesSizeConstraintsForCompression(hop));
			treeLookup.put(hop.getHopID(), o);
			parent.addOp(o);

			if(o.isCompressedOutput())
				compressed.add(hop.getHopID());
		}
		else if(hop.getDataType().isFrame()) {
			Op o = null;
			if(HopRewriteUtils.isData(hop, OpOpData.PERSISTENTREAD, OpOpData.TRANSIENTREAD))
				return;
			else if(HopRewriteUtils.isData(hop, OpOpData.TRANSIENTWRITE, OpOpData.PERSISTENTWRITE)) {
				transientCompressed.put(hop.getName(), hop.getInput(0).getHopID());
				compressed.add(hop.getHopID());
				o = new OpMetadata(hop, hop.getInput(0));
				if(isOverlapping(hop.getInput(0)))
					o.setOverlapping();
			}
			else if(HopRewriteUtils.isUnary(hop, OpOp1.DETECTSCHEMA)) {
				o = new OpNormal(hop, false);
			}
			else if(HopRewriteUtils.isBinary(hop, OpOp2.APPLY_SCHEMA)) {
				o = new OpNormal(hop, true);
			}
			else if(hop instanceof AggUnaryOp) {
				o = new OpNormal(hop, false);
			}
			else {
				LOG.warn("Unknown Frame Hop:" + hop.getClass().getSimpleName() + "\n" + Explain.explain(hop));
				setDecompressionOnAllInputs(hop, parent);
				return;
			}

			o = o != null ? o : new OpNormal(hop, RewriteCompressedReblock.satisfiesSizeConstraintsForCompression(hop));
			treeLookup.put(hop.getHopID(), o);
			parent.addOp(o);
			if(o.isCompressedOutput())
				compressed.add(hop.getHopID());
		}
		else if(HopRewriteUtils.isTransformEncode(hop)) {
			Hop matrix = ((FunctionOp) hop).getOutputs().get(0);
			compressed.add(matrix.getHopID());
			transientCompressed.put(matrix.getName(), matrix.getHopID());
			parent.addOp(new OpNormal(hop, true));
		}
		else if(hop instanceof FunctionOp && ((FunctionOp) hop).getFunctionNamespace().equals(".builtinNS")) {
			parent.addOp(new OpNormal(hop, false));
		}
		else if(hop instanceof AggUnaryOp) {
			if((isOverlapping(hop.getInput().get(0)) && !HopRewriteUtils.isAggUnaryOp(hop, AggOp.SUM, AggOp.MEAN)) ||
					HopRewriteUtils.isAggUnaryOp(hop, AggOp.TRACE)) {
					setDecompressionOnAllInputs(hop, parent);
					return;
				}
				else {
					Op o  = new OpNormal(hop, false);
					treeLookup.put(hop.getHopID(), o);
					parent.addOp(o);
				}
		}
		else {
			LOG.warn(
				"Unknown Matrix or Frame Hop:" + hop.getClass().getSimpleName() + "\n" + hop.getDataType() + "\n" + Explain.explain(hop));
			parent.addOp(new OpNormal(hop, false));
		}
	}

	private boolean checkTransientRead(Hop hop, Hop input) {
		// op is not in current statement block
		if(HopRewriteUtils.isData(input, OpOpData.TRANSIENTREAD)){
			String varName = input.getName();
			StatementBlock csb = lineage.peek();
			StatementBlock parentStatement = lineage.get(lineage.size() -2);

			if(parentStatement instanceof WhileStatementBlock) {
				WhileStatementBlock wsb = (WhileStatementBlock) parentStatement;
				WhileStatement wstmt = (WhileStatement) wsb.getStatement(0);
				ArrayList<StatementBlock> stmts = wstmt.getBody();
				boolean foundCurrent = false;
				StatementBlock sb;

				// traverse statement blocks in reverse to find the statement block, which came before the current
				// if we iterate in default order, we might find an earlier updated version of the current variable
				for (int i = stmts.size()-1; i >= 0; i--) {
					sb = stmts.get(i);
					if(foundCurrent && sb.variablesUpdated().containsVariable(varName)) {
						for(Hop cand : sb.getHops()){
							if(HopRewriteUtils.isData(cand, OpOpData.TRANSIENTWRITE) && cand.getName().equals(varName)
									&& HopRewriteUtils.isAggUnaryOp( cand.getInput(0), AggOp.MIN, AggOp.MAX)){
								return true;
							}
						}
					} else if(sb == csb){
						foundCurrent = true;
					}
				}
			}
		}
		return false;
	}

	private boolean isCompressed(Hop hop) {
		return compressed.contains(hop.getHopID());
	}

	private void setDecompressionOnAllInputs(Hop hop, AWTreeNode parent) {
		if(parent instanceof WTreeRoot)
			((WTreeRoot) parent).setDecompressing();
		for(Hop h : hop.getInput()) {
			Op ol = treeLookup.get(h.getHopID());
			if(ol != null) {
				while(ol instanceof OpMetadata) {
					// go up through operations and mark the first known as decompressing.
					// The first known usually is the root of the work tree.
					Op oln = treeLookup.get(((OpMetadata) ol).getParent().getHopID());
					if(oln == null)
						break;
					else
						ol = oln;
				}
				ol.setDecompressing();
			}
		}
	}

	private boolean isOverlapping(Hop hop) {
		Op o = treeLookup.get(hop.getHopID());
		if(o != null)
			return o.isOverlapping();
		else
			return false;
	}

	private static boolean isNoOp(Hop hop) {
		return hop instanceof LiteralOp || HopRewriteUtils.isUnary(hop, OpOp1.NROW, OpOp1.NCOL);
	}
}
