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
import java.util.Collections;
import java.util.ListIterator;
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
import org.apache.sysds.runtime.DMLRuntimeException;

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
		ArrayList<StatementBlock> ret = new ArrayList<>();

		// collect all candidate hops in levels based on their data dependencies
		ArrayList<ArrayList<Hop>> candidateLists = new ArrayList<ArrayList<Hop>>();
		collectNnzComputationCandidates(sb.getHops(), candidateLists);
		Collections.reverse(candidateLists); // reverse order of dependency levels

		// split hop dag on demand and if sparsity rewrites are enabled (splitting is pointless w/o rewrite)
		if(!candidateLists.isEmpty() && ConfigurationManager.getDMLConfig().getBooleanValue(DMLConfig.SPARSITY_REWRITES)) {
			for(ArrayList<Hop> cand : candidateLists) {
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
					ArrayList<Hop> parents = new ArrayList<>(cHop.getParent());
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
				ret.add(sbSplit); // statement block with permanent read and transient write
				sbSplit.setSplitDag(true); // avoid later merge by other rewrites
			}
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

	private void rSplitMMChain(Hop hop, ListIterator<ArrayList<Hop>> candListIter) {
		if(hop.isVisited())
			return;
		if(HopRewriteUtils.isMatrixMultiply(hop)) {
			rSplitMMChainChildren(hop, candListIter);
		}
		else {
			// split again when operation is different from matrix multiply (i.e., before mm chain)
			getNextLevel(candListIter).add(hop);
			rCollectNnzComputationCandidatesChildren(hop, candListIter);
			candListIter.previous();
		}
		hop.setVisited();
	}

	private void rSplitMMChainChildren(Hop parentHop, ListIterator<ArrayList<Hop>> candListIter) {
		if(parentHop.getInput() == null)
			throw new DMLRuntimeException("rSplitMMChain should be called for MM chain hops only. " +
				"An MM hop must have inputs.");
		// process children
		for(Hop childHop : parentHop.getInput()) {
			rSplitMMChain(childHop, candListIter);
		}
	}

	private void collectNnzComputationCandidates(ArrayList<Hop> roots, ArrayList<ArrayList<Hop>> candidateLists) {
		if(roots == null)
			return;
		Hop.resetVisitStatus(roots);
		ListIterator<ArrayList<Hop>> candListIter = candidateLists.listIterator();
		for(Hop root : roots) {
			rCollectNnzComputationCandidates(root, candListIter);
		}
	}
	
	private ArrayList<Hop> getNextLevel(ListIterator<ArrayList<Hop>> liter) {
		if(liter.hasNext()) {
			return liter.next();
		}
		else {
			ArrayList<Hop> newLevel = new ArrayList<Hop>();
			liter.add(newLevel);
			return newLevel;
		}
	}

	private void rCollectNnzComputationCandidates(Hop hop, ListIterator<ArrayList<Hop>> candListIter) {
		if(hop.isVisited())
			return;

		// collect persistent reads with unknown nnz
		if(hop instanceof DataOp && ((DataOp) hop).getOp() == OpOpData.PERSISTENTREAD && !((DataOp) hop).nnzKnown()) {
				getNextLevel(candListIter).add(hop);
				candListIter.previous(); // reset level increment (no children to process)
		}
		else if(HopRewriteUtils.isMatrixMultiply(hop) && !((AggBinaryOp) hop).hasLeftPMInput()) {
			getNextLevel(candListIter).add(hop);
			rSplitMMChainChildren(hop, candListIter);
			candListIter.previous(); // reset level increment after processing children
		}
		else{
			rCollectNnzComputationCandidatesChildren(hop, candListIter);
		}

		hop.setVisited();
	}

	private void rCollectNnzComputationCandidatesChildren(Hop parentHop, ListIterator<ArrayList<Hop>> candListIter) {
		// process children
		if(parentHop.getInput() != null) {
			for(Hop c : parentHop.getInput())
				rCollectNnzComputationCandidates(c, candListIter);
		}
	}
}
