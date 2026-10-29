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
import java.util.HashMap;
import java.util.HashSet;

import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.parser.VariableSet;
import org.apache.sysds.runtime.matrix.data.Pair;

public abstract class SplitStatementBlockRewriteRule extends StatementBlockRewriteRule {
	@Override
	public boolean createsSplitDag() {
		return true;
	}

	protected void handleReplicatedOperators(ArrayList<Hop> rootsSbSplit, ArrayList<Hop> rootsSbOrigin,
		VariableSet sbSplitOut, VariableSet sbOriginIn) {
		// step 1: create probe set SB1
		HashSet<Hop> probeSet = new HashSet<>();
		Hop.resetVisitStatus(rootsSbSplit);
		for(Hop h : rootsSbSplit) {
			rAddHopsToProbeSet(h, probeSet);
		}

		// step 2: probe SB2 operators top-down (collect cut candidates)
		HashSet<Pair<Hop,Hop>> candSet = new HashSet<>();
		Hop.resetVisitStatus(rootsSbOrigin);
		for(Hop h : rootsSbOrigin) {
			rProbeAndAddHopsToCandidateSet(h, probeSet, candSet);
		}

		// step 3: create additional cuts with reuse for common references
		HashMap<Long, DataOp> reuseTRead = new HashMap<>();
		for(Pair<Hop,Hop> p : candSet) {
			Hop hop = p.getKey();
			Hop c = p.getValue();

			DataOp tRead = reuseTRead.get(c.getHopID());
			if(tRead == null) {
				String varname = createCutVarName(false);

				tRead = HopRewriteUtils.createTransientRead(varname, c);
				reuseTRead.put(c.getHopID(), tRead);

				DataOp twrite = HopRewriteUtils.createTransientWrite(varname, c);

				// update live in and out of new statement block (for piggybacking)
				DataIdentifier diVar = new DataIdentifier(varname);
				diVar.setDimensions(c.getDim1(), c.getDim2());
				diVar.setBlocksize(c.getBlocksize());
				diVar.setDataType(c.getDataType());
				diVar.setValueType(c.getValueType());
				sbSplitOut.addVariable(varname, new DataIdentifier(diVar));
				sbOriginIn.addVariable(varname, new DataIdentifier(diVar));

				rootsSbSplit.add(twrite);
			}

			// create additional cut by rewriting both hop dags
			int pos = HopRewriteUtils.getChildReferencePos(hop, c);
			HopRewriteUtils.removeChildReferenceByPos(hop, c, pos);
			HopRewriteUtils.addChildReference(hop, tRead, pos);
		}
	}

	private void rAddHopsToProbeSet(Hop hop, HashSet<Hop> probeSet) {
		if(hop.isVisited())
			return;

		// prevent cuts for no-ops
		if(!(hop instanceof DataOp && !((DataOp)hop).isPersistentReadWrite()) && !(hop instanceof LiteralOp))
			probeSet.add(hop);

		if(hop.getInput() != null) {
			for(Hop c : hop.getInput()) {
				rAddHopsToProbeSet(c, probeSet);
			}
		}

		hop.setVisited();
	}

	/**
	 * NOTE: candset is a set of parent-child pairs because a parent might have
	 * multiple references to replicated hops.
	 *
	 * @param hop high-level operator
	 * @param probeSet probe set?
	 * @param candSet candidate set?
	 */
	private void rProbeAndAddHopsToCandidateSet(Hop hop, HashSet<Hop> probeSet, HashSet<Pair<Hop,Hop>> candSet) {
		if(hop.isVisited())
			return;

		if(hop.getInput() != null)
			for(Hop c : hop.getInput()) {
				// probe for replicated operator, if any child is replicated, keep parent
				// for cut between parent-child; otherwise recursively descend.
				if(!probeSet.contains(c))
					rProbeAndAddHopsToCandidateSet(c, probeSet, candSet);
				else
					candSet.add(new Pair<>(hop,c));
			}

		hop.setVisited();
	}

	protected void collectCandidateChildOperators(ArrayList<Hop> cand, HashSet<Hop> candChilds) {
		Hop.resetVisitStatus(cand);
		if(cand != null) {
			for(Hop root : cand) {
				rCollectCandidateChildOperators(root, cand, candChilds, false);
			}
		}

		// Immediately reset the visit status because candidates might be inner nodes in the DAG.
		// Subsequent resets on the root nodes of the DAG would otherwise not necessarily reach
		// these nodes which could lead to missing checks on subsequent passes (e.g., when checking
		// for replicated operators).
		Hop.resetVisitStatus(cand);
	}

	private void rCollectCandidateChildOperators(Hop hop, ArrayList<Hop> cand, HashSet<Hop> candChilds, boolean collect) {
		if(hop.isVisited())
			return;

		// collect operator if necessary
		if(collect)
			candChilds.add(hop);

		// activate collection if we passed a hop from the candidates
		boolean passedFlag = collect;
		if(cand.contains(hop))
			passedFlag = true;

		// process childs recursively
		if(hop.getInput() != null) {
			for(Hop childHop : hop.getInput()) {
				rCollectCandidateChildOperators(childHop, cand, candChilds, passedFlag);
			}
		}

		hop.setVisited();
	}
}
