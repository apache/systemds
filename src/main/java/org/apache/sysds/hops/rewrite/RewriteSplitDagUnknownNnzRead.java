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
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.VariableSet;

/**
 * Rule: Split Hop DAG after reads with unknown nnz. This is important to create recompile
 * hooks if mtd has unspecified nnz.
 */
public class RewriteSplitDagUnknownNnzRead extends StatementBlockRewriteRule {
	@Override
	public boolean createsSplitDag() {
		return true;
	}

	@Override
	public List<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus state) {
		ArrayList<StatementBlock> ret = new ArrayList<>();

		// collect all read hops w/ unknown nnz
		ArrayList<Hop> cand = new ArrayList<>();
		collectReadHopsUnknownNnz(sb.getHops(), cand);

		// split hop dag on demand
		if(!cand.isEmpty()) {
			// duplicate sb incl live variable sets
			StatementBlock sb1 = new StatementBlock();
			sb1.setDMLProg(sb.getDMLProg());
			sb1.setParseInfo(sb);
			sb1.setLiveIn(new VariableSet());
			sb1.setLiveOut(new VariableSet());

			// move reads incl reblock to new statement block
			// (and replace original persistent read with transient read)
			ArrayList<Hop> sb1hops = new ArrayList<>();
			for(Hop reblock : cand) {
				// replace reblock inputs to avoid dangling references across dags
				// (otherwise, for instance, literal ops are shared across dags)
				for(int i = 0; i < reblock.getInput().size(); i++) {
					if(reblock.getInput().get(i) instanceof LiteralOp) {
						HopRewriteUtils.replaceChildReference(reblock, reblock.getInput().get(i),
							new LiteralOp((LiteralOp) reblock.getInput().get(i)));
					}
				}

				// create new transient read
				DataOp tRead = HopRewriteUtils.createTransientRead(reblock.getName(), reblock);
				tRead.setRequiresRecompile();

				// replace reblock with transient read
				ArrayList<Hop> parents = new ArrayList<>(reblock.getParent());
				for(int i = 0; i < parents.size(); i++) {
					Hop parent = parents.get(i);
					HopRewriteUtils.replaceChildReference(parent, reblock, tRead);
				}

				// add reblock sub dag to first statement block
				DataOp tWrite = HopRewriteUtils.createTransientWrite(reblock.getName(), reblock);
				sb1hops.add(tWrite);

				// update live in and out of new statement block (for piggybacking)
				DataIdentifier diVar = sb.variablesRead().getVariable(reblock.getName());
				if(diVar != null) { // var read should always exist because persistent read
					sb1.liveOut().addVariable(reblock.getName(), new DataIdentifier(diVar));
					sb.liveIn().addVariable(reblock.getName(), new DataIdentifier(diVar));
				}
			}

			sb1.setHops(sb1hops);
			sb1.updateRecompilationFlag();
			ret.add(sb1); // statement block with permanent read and transient write
			ret.add(sb); // statement block with transient read
			sb.setSplitDag(true); // avoid later merge by other rewrites
			LOG.debug("Applied splitDagUnknownNnzRead.");
		}
		// keep original hop dag
		else {
			ret.add(sb);
		}

		return ret;
	}

	@Override
	public List<StatementBlock> rewriteStatementBlocks(List<StatementBlock> sbs, ProgramRewriteStatus state) {
		return sbs;
	}

	private void collectReadHopsUnknownNnz(ArrayList<Hop> roots, ArrayList<Hop> cand) {
		if(roots == null)
			return;
		Hop.resetVisitStatus(roots);
		for(Hop root : roots)
			collectReadHopsUnknownNnz(root, cand);
	}

	private void collectReadHopsUnknownNnz(Hop hop, ArrayList<Hop> cand) {
		if(hop.isVisited())
			return;

		// collect persistent reads with unknown nnz
		if(hop instanceof DataOp) {
			DataOp dop = (DataOp) hop;
			if(dop.getOp() == OpOpData.PERSISTENTREAD && !dop.nnzKnown()) {
				cand.add(dop);
			}
		}

		// process children
		if(hop.getInput() != null) {
			for(Hop c : hop.getInput()) {
				collectReadHopsUnknownNnz(c, cand);
			}
		}

		hop.setVisited();
	}
}
