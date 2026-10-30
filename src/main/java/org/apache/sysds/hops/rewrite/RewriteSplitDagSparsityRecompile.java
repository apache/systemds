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

import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.VariableSet;

/**
 * Rule: Split Hop DAG after reads with unknown nnz. This is important to create recompile
 * hooks if mtd has unspecified nnz.
 */
public class RewriteSplitDagSparsityRecompile extends SplitStatementBlockRewriteRule {
	@Override
	public List<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus state) {
		return rRewriteStatementBlock(sb, state);
	}

	private static Hop getFirstTransientWriteParent(Hop hop) {
		for(Hop p : hop.getParent()) {
			if(p instanceof DataOp && ((DataOp)p).getOp() == OpOpData.TRANSIENTWRITE)
				return p;
		}
		return null;
	}

	public List<StatementBlock> rRewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus state) {
		// split only if sparsity rewrites are enabled (splitting is pointless w/o rewrite)
		if(!ConfigurationManager.getDMLConfig().getBooleanValue(DMLConfig.SPARSITY_REWRITES))
			return Arrays.asList(sb);

		ArrayList<StatementBlock> ret = new ArrayList<>();

		// collect all candidate hops in levels based on their data dependencies
		ArrayList<Hop> cand = new ArrayList<Hop>();
		collectNnzComputationCandidates(sb.getHops(), cand);

		// split hop dag on demand
		if(!cand.isEmpty()) {
			//collect child operators of candidates (to prevent rewrite anomalies)
			//For example, B = rmEmpty(A), C = rmEmpty(B), op(C, nrow(B)) needs to
			//preserve the link rmEmpty(B) in the split-off DAG. Therefore, the
			//candidates themselves need to be part of this anomaly filter as well.
			HashSet<Hop> candChilds = new HashSet<>();
			collectCandidateChildOperators(cand, candChilds);
			candChilds.addAll(cand);

			// duplicate sb incl live variable sets
			StatementBlock sbSplit = new StatementBlock();
			sbSplit.setDMLProg(sb.getDMLProg());
			sbSplit.setParseInfo(sb);
			sbSplit.setLiveIn(new VariableSet());
			sbSplit.setLiveOut(new VariableSet());

			// move candidate hops to new statement block
			// (and replace original persistent read with transient read)
			ArrayList<Hop> sbSplitHops = new ArrayList<>();
			for(Hop cHop : cand) {
				// create new transient read
				String varname;
				DataOp tWrite = (DataOp)getFirstTransientWriteParent(cHop);
				if(cHop instanceof DataOp && ((DataOp)cHop).isRead() && tWrite != null) {
					varname = tWrite.getName();
				}
				else {
					varname = createCutVarName(false);
				}
				DataOp tRead = HopRewriteUtils.createTransientRead(varname, cHop);
				tRead.setRequiresRecompile();

				// replace hop with transient read operation
				ArrayList<Hop> parents = new ArrayList<>(cHop.getParent());
				for(int i = 0; i < parents.size(); i++) {
					// prevent concurrent modification by index access
					Hop parent = parents.get(i);
					if(!candChilds.contains(parent)) { // anomaly filter
						if(parent != tWrite)
							HopRewriteUtils.replaceChildReference(parent, cHop, tRead);
						else
							sb.getHops().remove(parent);
					}
				}

				if(tWrite == null)
					tWrite = HopRewriteUtils.createTransientWrite(varname, cHop);
				sbSplitHops.add(tWrite);

				// update live in and out of new statement block (for piggybacking)
				DataIdentifier diVar = new DataIdentifier(varname);
				diVar.setDimensions(cHop.getDim1(), cHop.getDim2());
				diVar.setBlocksize(cHop.getBlocksize());
				diVar.setDataType(cHop.getDataType());
				diVar.setValueType(cHop.getValueType());
				sbSplit.liveOut().addVariable(varname, new DataIdentifier(diVar));
				sb.liveIn().addVariable(varname, new DataIdentifier(diVar));
				sb.variablesRead().addVariable(varname, new DataIdentifier(diVar));
			}

			// ensure disjoint operators across DAGs (prevent replicated operations)
			handleReplicatedOperators(sbSplitHops, sb.getHops(), sbSplit.liveOut(), sb.liveIn());

			sbSplit.setHops(Recompiler.deepCopyHopsDag(sbSplitHops));
			sbSplit.updateRecompilationFlag();
			sbSplit.setSplitDag(true); // avoid later merge by other rewrites

			//recursive application of rewrite rule
			List<StatementBlock> tmp = rRewriteStatementBlock(sbSplit, state);

			ret.addAll(tmp); // statement block with permanent read and transient write
			sb.setSplitDag(true); // avoid later merge by other rewrites
			LOG.debug("RewriteSplitDagSparsityRecompile: Split DAG into " + (ret.size() + 1) + " statement blocks.");
		}

		ret.add(sb);
		return ret;
	}

	@Override
	public List<StatementBlock> rewriteStatementBlocks(List<StatementBlock> sbs, ProgramRewriteStatus state) {
		return sbs;
	}

	private void collectNnzComputationCandidates(ArrayList<Hop> roots, ArrayList<Hop> cand) {
		if(roots == null)
			return;
		Hop.resetVisitStatus(roots);
		for(Hop root : roots) {
			rCollectNnzComputationCandidates(root, cand, null);
		}
	}

	public static boolean anyParentIsMatrixMultiply(Hop hop) {
		return hop.getParent().stream().anyMatch(parentHop -> HopRewriteUtils.isMatrixMultiply(parentHop));
	}

	public static boolean allParentsAreMatrixMultiply(Hop hop) {
		return hop.getParent().stream().allMatch(parentHop -> HopRewriteUtils.isMatrixMultiply(parentHop));
	}

	public static boolean anyChildIsMatrixMultiply(Hop hop) {
		return hop.getInput().stream().anyMatch(childHop -> HopRewriteUtils.isMatrixMultiply(childHop));
	}

	public static boolean allChildsAreMatrixMultiply(Hop hop) {
		return hop.getInput().stream().allMatch(childHop -> HopRewriteUtils.isMatrixMultiply(childHop));
	}

	public static boolean isMMChain(Hop hop) {
		// hop is part of mm chain with at least two matrix multplications
		return HopRewriteUtils.isMatrixMultiply(hop)
			&& (anyParentIsMatrixMultiply(hop) || anyChildIsMatrixMultiply(hop));
	}

	public static boolean anyParentIsMMChain(Hop hop) {
		List<Hop> parentHops = hop.getParent();
		if(parentHops == null)
			return false;
		return parentHops.stream().anyMatch(pHop -> isMMChain(pHop));
	}

	private void rCollectNnzComputationCandidates(Hop hop, ArrayList<Hop> cand, Hop parentHop) {
		if(hop.isVisited())
			return;
		// has already been split off
		boolean noSplitRequired = HopRewriteUtils.hasOnlyWriteParents(hop, true, false);

		if(!noSplitRequired) {
			if(hop instanceof DataOp && ((DataOp) hop).getOp() == OpOpData.PERSISTENTREAD && !hop.nnzKnown()) {
				// split persistent reads with unknown nnz to obtain their nnz before continuing rewrites
				cand.add(hop);
			}
			else if(parentHop != null && !HopRewriteUtils.isMatrixMultiply(hop)
				&& !(hop instanceof DataOp && ((DataOp) hop).getOp() == OpOpData.TRANSIENTREAD)
				&& anyParentIsMMChain(hop)) { // split only mm chains (i.e., at least two matrix multiplications)
				// split before mm chain to ensure correct sparsity information for mm chain sparsity estimation
				cand.add(hop);
			}
			else if((!HopRewriteUtils.isMatrixMultiply(parentHop) || !allParentsAreMatrixMultiply(hop))
				&& isMMChain(hop) // split only mm chains (i.e., at least two matrix multiplications)
				&& !((AggBinaryOp) hop).hasLeftPMInput()) {
				// split after mm chain to force the mm chain into a separate statment block for isolated optimization
				cand.add(hop);
			}
			else {
				rCollectNnzComputationCandidatesChildren(hop, cand);
			}
		}
		else {
			rCollectNnzComputationCandidatesChildren(hop, cand);
		}

		hop.setVisited();
	}

	private void rCollectNnzComputationCandidatesChildren(Hop parentHop, ArrayList<Hop> cand) {
		// process children
		if(parentHop.getInput() != null) {
			for(Hop c : parentHop.getInput())
				rCollectNnzComputationCandidates(c, cand, parentHop);
		}
	}
}
