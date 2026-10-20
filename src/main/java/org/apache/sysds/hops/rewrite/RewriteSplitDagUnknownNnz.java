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
import java.util.List;

import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.VariableSet;

/**
 * Rule: Split Hop DAG after reads with unknown nnz. This is important to create recompile
 * hooks if mtd has unspecified nnz.
 */
public class RewriteSplitDagUnknownNnz extends StatementBlockRewriteRule {
	@Override
	public boolean createsSplitDag() {
		return true;
	}

	@Override
	public List<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus state) {
		return rRewriteStatementBlock(sb, state);
	}

	public List<StatementBlock> rRewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus state) {
		ArrayList<StatementBlock> ret = new ArrayList<>();

		// collect all candidate hops in levels based on their data dependencies
		ArrayList<Hop> cand = new ArrayList<Hop>();
		collectNnzComputationCandidates(sb.getHops(), cand);

		// split hop dag on demand and if sparsity rewrites are enabled (splitting is pointless w/o rewrite)
		if(!cand.isEmpty() && ConfigurationManager.getDMLConfig().getBooleanValue(DMLConfig.SPARSITY_REWRITES)) {
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
				DataOp tRead = HopRewriteUtils.createTransientRead(cHop.getName(), cHop);
				tRead.setRequiresRecompile();

				// replace persistent read with transient read
				List<Hop> parents = cHop.getParent();
				for(int i = 0; i < parents.size(); i++) {
					Hop parent = parents.get(i);
					HopRewriteUtils.replaceChildReference(parent, cHop, tRead);
				}

				// add reblock sub dag to first statement block
				DataOp tWrite = HopRewriteUtils.createTransientWrite(cHop.getName(), cHop);
				sbSplitHops.add(tWrite);

				// update live in and out of new statement block (for piggybacking)
				DataIdentifier diVar = sb.variablesRead().getVariable(cHop.getName());
				if(diVar != null) { // var read should always exist because persistent read
					sbSplit.liveOut().addVariable(cHop.getName(), new DataIdentifier(diVar));
					sb.liveIn().addVariable(cHop.getName(), new DataIdentifier(diVar));
				}
			}

			sbSplit.setHops(sbSplitHops);
			sbSplit.updateRecompilationFlag();
			sbSplit.setSplitDag(true); // avoid later merge by other rewrites

			//recursive application of rewrite rule
			List<StatementBlock> tmp = rRewriteStatementBlock(sbSplit, state);

			ret.addAll(tmp); // statement block with permanent read and transient write
			sb.setSplitDag(true); // avoid later merge by other rewrites
			LOG.debug("RewriteSplitDagUnknownNnz: Split DAG into " + (ret.size() + 1) + " statement blocks.");
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

	private void rCollectNnzComputationCandidates(Hop hop, ArrayList<Hop> cand, Hop parentHop) {
		if(hop.isVisited())
			return;
		// has already been split off
		boolean noSplitRequired = HopRewriteUtils.hasOnlyWriteParents(hop, true, false);

		if(!noSplitRequired) {
			if(hop instanceof DataOp && ((DataOp) hop).getOp() == OpOpData.PERSISTENTREAD && !hop.nnzKnown()) {
				// split persistent reads with unknown nnz to obtain their nnz before continuing rewrites
				// splitAndProcessChildren(hop, candListIter);
				cand.add(hop);
			}
			else if(parentHop != null && anyParentIsMatrixMultiply(hop) && !HopRewriteUtils.isMatrixMultiply(hop)) {
				// split before mm chain to ensure correct sparsity information for mm chain sparsity estimation
				// splitAndProcessChildren(hop, candListIter);
				cand.add(hop);
			}
			else if((!HopRewriteUtils.isMatrixMultiply(parentHop) || !allParentsAreMatrixMultiply(hop))
				&& HopRewriteUtils.isMatrixMultiply(hop) && !((AggBinaryOp) hop).hasLeftPMInput()) {
				// split after mm chain to force the mm chain into a separate statment block for isolated optimization
				// splitAndProcessChildren(hop, candListIter);
				cand.add(hop);
			}
			else {
				rCollectCandidatesChildren(hop, cand);
			}
		}
		else {
			rCollectCandidatesChildren(hop, cand);
		}

		hop.setVisited();
	}

	private void rCollectCandidatesChildren(Hop parentHop, ArrayList<Hop> cand) {
		// process children
		if(parentHop.getInput() != null) {
			for(Hop c : parentHop.getInput())
				rCollectNnzComputationCandidates(c, cand, parentHop);
		}
	}
}
