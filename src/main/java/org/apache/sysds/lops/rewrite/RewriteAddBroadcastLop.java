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

package org.apache.sysds.lops.rewrite;

import org.apache.sysds.common.Types;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.lops.Data;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.OperatorOrderingUtils;
import org.apache.sysds.lops.UnaryCP;
import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.FunctionStatement;
import org.apache.sysds.parser.FunctionStatementBlock;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.WhileStatement;
import org.apache.sysds.parser.WhileStatementBlock;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class RewriteAddBroadcastLop extends LopRewriteRule
{
	boolean MULTI_BLOCK_REWRITE = false;

	@Override
	public List<StatementBlock> rewriteLOPinStatementBlock(StatementBlock sb)
	{
		if (!ConfigurationManager.isBroadcastEnabled())
			return List.of(sb);

		ArrayList<Lop> lops = OperatorOrderingUtils.getLopList(sb);
		if (lops == null)
			return List.of(sb);

		ArrayList<Lop> nodesWithBroadcast = new ArrayList<>();
		for (Lop l : lops) {
			nodesWithBroadcast.add(l);
			if (isBroadcastNeeded(l)) {
				List<Lop> oldOuts = new ArrayList<>(l.getOutputs());
				// Construct a Broadcast lop that takes this CP node as an input
				UnaryCP bc = new UnaryCP(l, Types.OpOp1.BROADCAST, l.getDataType(), l.getValueType(), Types.ExecType.CP);
				bc.setAsynchronous(true);
				// FIXME: Wire Broadcast only with the necessary outputs
				for (Lop outSP : oldOuts) {
					// Rewire l -> outSP to l -> Broadcast -> outSP
					bc.addOutput(outSP);
					outSP.replaceInput(l, bc);
					l.removeOutput(outSP);
					// FIXME: Rewire _inputParams when needed (e.g. GroupedAggregate)
				}
				//Place it immediately after the Spark lop in the node list
				nodesWithBroadcast.add(bc);
			}
		}
		// New node is added inplace in the Lop DAG
		return Arrays.asList(sb);
	}

	@Override
	public List<StatementBlock> rewriteLOPinStatementBlocks(List<StatementBlock> sbs)
	{
		if (!MULTI_BLOCK_REWRITE)
			return sbs;
		// FIXME: Enable after handling of rmvar of asynchronous inputs

		if (!ConfigurationManager.isBroadcastEnabled())
			return sbs;
		if (sbs == null || sbs.isEmpty())
			return sbs;
		// The first statement block has to be a basic block
		// TODO: Remove this constraints
		StatementBlock sb1 = sbs.get(0);
		if (!HopRewriteUtils.isLastLevelStatementBlock(sb1))
			return sbs;
		if (sb1.getLops() == null || sb1.getLops().isEmpty())
			return sbs;

		// Gather the twrite names of the potential broadcast candidates from the first block
		// TODO: Replace repetitive rewrite calls with a single one to place all prefetches
		HashMap<String, List<Boolean>> twrites = new HashMap<>();
		HashMap<String, Lop> broadcastCandidates = new HashMap<>();
		for (Lop root : sb1.getLops()) {
			if (root instanceof Data && ((Data)root).getOperationType().isTransientWrite()) {
				Lop written = root.getInputs().get(0);
				if (written.getExecType() == Types.ExecType.CP && written.getDataType().isMatrix()) {
					// Potential broadcast candidate. Save in the twrite map
					twrites.put(root.getOutputParameters().getLabel(), new ArrayList<>());
					broadcastCandidates.put(root.getOutputParameters().getLabel(), written);
				}
			}
		}
		if (broadcastCandidates.isEmpty())
			return sbs;

		// Recursively check the consumers in the bellow blocks to find if broadcast is required
		for (int i=1; i< sbs.size(); i++)
			findConsumers(sbs.get(i), twrites);

		// Place a broadcast if any of the consumers are Spark
		for (Map.Entry<String, Lop> entry : broadcastCandidates.entrySet()) {
			if (twrites.get(entry.getKey()).stream().anyMatch(outBC -> (outBC == true))) {
				Lop candidate = entry.getValue();
				List<Lop> oldOuts = new ArrayList<>(candidate.getOutputs());
				// Construct a broadcast lop that takes this CP node as an input
				UnaryCP bc = new UnaryCP(candidate, Types.OpOp1.BROADCAST, candidate.getDataType(),
					candidate.getValueType(), Types.ExecType.CP);
				bc.setAsynchronous(true);
				// FIXME: Wire Broadcast only with the necessary outputs
				for (Lop outSP : oldOuts) {
					// Rewire l -> outSP to l -> Broadcast -> outSP
					bc.addOutput(outSP);
					outSP.replaceInput(candidate, bc);
					candidate.removeOutput(outSP);
					// FIXME: Rewire _inputParams when needed (e.g. GroupedAggregate)
				}
			}
		}
		return sbs;
	}

	private static boolean isBroadcastNeeded(Lop lop) {
		// Asynchronously broadcast a matrix if that is produced by a CP instruction,
		// and at least one Spark parent needs to broadcast this intermediate (eg. mapmm)
		boolean isBcOutput = lop.getOutputs().stream()
			.anyMatch(out -> (out.getBroadcastInput() == lop));
		// TODO: Early broadcast objects that are bigger than a single block
		boolean isCPInput = lop.getExecType() == Types.ExecType.CP;
		return isCPInput && isBcOutput && lop.getDataType() == Types.DataType.MATRIX;
	}

	private void findConsumers(StatementBlock sb, HashMap<String, List<Boolean>> twrites) {
		if (sb instanceof FunctionStatementBlock) {
			FunctionStatementBlock fsb = (FunctionStatementBlock)sb;
			FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
			for (StatementBlock input : fstmt.getBody())
				findConsumers(input, twrites);
		}
		else if (sb instanceof WhileStatementBlock) {
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement wstmt = (WhileStatement) wsb.getStatement(0);
			for (StatementBlock input : wstmt.getBody())
				findConsumers(input, twrites);
		}
		else if (sb instanceof ForStatementBlock) { //incl parfor
			ForStatementBlock fsb = (ForStatementBlock) sb;
			ForStatement fstmt = (ForStatement) fsb.getStatement(0);
			for (StatementBlock input : fstmt.getBody())
				findConsumers(input, twrites);
		}

		// Find the execution types of the consumers
		ArrayList<Lop> lops = OperatorOrderingUtils.getLopList(sb);
		if (lops == null)
			return;
		for (Lop l : lops) {
			// Find consumers in this basic block
			if (l instanceof Data && ((Data) l).getOperationType().isTransientRead()
				&& twrites.containsKey(l.getOutputParameters().getLabel())) {
				// Check if the consumers satisfy broadcast conditions
				for (Lop consumer : l.getOutputs())
					if (consumer.getBroadcastInput() == l)
						twrites.get(l.getOutputParameters().getLabel()).add(true);
			}
		}

	}
}
