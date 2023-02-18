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
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.lops.CSVReBlock;
import org.apache.sysds.lops.CentralMoment;
import org.apache.sysds.lops.Checkpoint;
import org.apache.sysds.lops.CoVariance;
import org.apache.sysds.lops.Data;
import org.apache.sysds.lops.DataGen;
import org.apache.sysds.lops.GroupedAggregate;
import org.apache.sysds.lops.GroupedAggregateM;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.MMTSJ;
import org.apache.sysds.lops.MMZip;
import org.apache.sysds.lops.MapMultChain;
import org.apache.sysds.lops.OperatorOrderingUtils;
import org.apache.sysds.lops.ParameterizedBuiltin;
import org.apache.sysds.lops.PickByCount;
import org.apache.sysds.lops.ReBlock;
import org.apache.sysds.lops.SpoofFused;
import org.apache.sysds.lops.UAggOuterChain;
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

public class RewriteAddPrefetchLop extends LopRewriteRule
{
	@Override
	public List<StatementBlock> rewriteLOPinStatementBlock(StatementBlock sb)
	{
		if (!ConfigurationManager.isPrefetchEnabled())
			return List.of(sb);

		ArrayList<Lop> lops = OperatorOrderingUtils.getLopList(sb);
		if (lops == null)
			return List.of(sb);

		ArrayList<Lop> nodesWithPrefetch = new ArrayList<>();
		//Find the Spark nodes with all CP outputs
		for (Lop l : lops) {
			nodesWithPrefetch.add(l);
			if (isPrefetchNeeded(l) && !l.prefetchActivated()) {
				List<Lop> oldOuts = new ArrayList<>(l.getOutputs());
				//Construct a Prefetch lop that takes this Spark node as an input
				UnaryCP prefetch = new UnaryCP(l, Types.OpOp1.PREFETCH, l.getDataType(), l.getValueType(), Types.ExecType.CP);
				prefetch.setAsynchronous(true);
				//Reset asynchronous flag for the input if already set (e.g. mapmm -> prefetch)
				l.setAsynchronous(false);
				l.activatePrefetch();
				for (Lop outCP : oldOuts) {
					//Rewire l -> outCP to l -> Prefetch -> outCP
					prefetch.addOutput(outCP);
					outCP.replaceInput(l, prefetch);
					l.removeOutput(outCP);
					//FIXME: Rewire _inputParams when needed (e.g. GroupedAggregate)
				}
				//Place it immediately after the Spark lop in the node list
				nodesWithPrefetch.add(prefetch);
			}
		}
		//New node is added inplace in the Lop DAG
		return Arrays.asList(sb);
	}

	@Override
	public List<StatementBlock> rewriteLOPinStatementBlocks(List<StatementBlock> sbs)
	{
		if (!ConfigurationManager.isPrefetchEnabled())
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

		// Gather the twrite names of the potential prefetch candidates from the first block
		// TODO: Replace repetitive rewrite calls with a single one to place all prefetches
		HashMap<String, List<Boolean>> twrites = new HashMap<>();
		HashMap<String, Lop> prefetchCandidates = new HashMap<>();
		for (Lop root : sb1.getLops()) {
			if (root instanceof Data && ((Data)root).getOperationType().isTransientWrite()) {
				Lop written = root.getInputs().get(0);
				if (isTransformOP(written) && !hasParameterizedOut(written) && written.getDataType().isMatrix()) {
					// Potential prefetch candidate. Save in the twrite map
					twrites.put(root.getOutputParameters().getLabel(), new ArrayList<>());
					prefetchCandidates.put(root.getOutputParameters().getLabel(), written);
				}
			}
		}
		if (prefetchCandidates.isEmpty())
			return sbs;

		// Recursively check the consumers in the bellow blocks to find if prefetch is required
		for (int i=1; i< sbs.size(); i++)
			findConsumers(sbs.get(i), twrites);

		// Place a prefetch if all the consumers are CP
		for (Map.Entry<String, Lop> entry : prefetchCandidates.entrySet()) {
			if (twrites.get(entry.getKey()).stream().allMatch(outCP -> (outCP == true))) {
				Lop candidate = entry.getValue();
				// Add prefetch after prefetch candidate
				List<Lop> oldOuts = new ArrayList<>(candidate.getOutputs());
				// Construct a Prefetch lop that takes this Spark node as an input
				UnaryCP prefetch = new UnaryCP(candidate, Types.OpOp1.PREFETCH, candidate.getDataType(),
					candidate.getValueType(), Types.ExecType.CP);
				prefetch.setAsynchronous(true);
				// Reset asynchronous flag for the input if already set (e.g. mapmm -> prefetch)
				candidate.setAsynchronous(false);
				candidate.activatePrefetch();
				for (Lop outCP : oldOuts) {
					// Rewire l -> outCP to l -> Prefetch -> outCP
					prefetch.addOutput(outCP);
					outCP.replaceInput(candidate, prefetch);
					candidate.removeOutput(outCP);
				}
			}
		}
		return sbs;
	}

	private boolean isPrefetchNeeded(Lop lop) {
		// Run Prefetch for a Spark instruction if the instruction is a Transformation
		// and the output is consumed by only CP instructions.
		boolean transformOP = isTransformOP(lop);
		//FIXME: Rewire _inputParams when needed (e.g. GroupedAggregate)
		boolean hasParameterizedOut = hasParameterizedOut(lop);
		//TODO: support non-matrix outputs
		return transformOP && !hasParameterizedOut
			&& (lop.isAllOutputsCP() || OperatorOrderingUtils.isCollectForBroadcast(lop))
			&& lop.getDataType().isMatrix();
	}

	private boolean isTransformOP(Lop lop) {
		boolean transformOP = lop.getExecType() == Types.ExecType.SPARK && lop.getAggType() != AggBinaryOp.SparkAggType.SINGLE_BLOCK
			// Always Action operations
			&& !(lop.getDataType() == Types.DataType.SCALAR)
			&& !(lop instanceof MapMultChain) && !(lop instanceof PickByCount)
			&& !(lop instanceof MMZip) && !(lop instanceof CentralMoment)
			&& !(lop instanceof CoVariance)
			// Not qualified for prefetching
			&& !(lop instanceof Checkpoint) && !(lop instanceof ReBlock)
			&& !(lop instanceof CSVReBlock) && !(lop instanceof DataGen)
			// Cannot filter Transformation cases from Actions (FIXME)
			&& !(lop instanceof MMTSJ) && !(lop instanceof UAggOuterChain)
			&& !(lop instanceof ParameterizedBuiltin) && !(lop instanceof SpoofFused);
		return transformOP;
	}

	private boolean hasParameterizedOut(Lop lop) {
		return lop.getOutputs().stream()
			.anyMatch(out -> ((out instanceof ParameterizedBuiltin)
				|| (out instanceof GroupedAggregate)
				|| (out instanceof GroupedAggregateM)));
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
				// Check if the consumers satisfy prefetch conditions
				for (Lop consumer : l.getOutputs())
					if (consumer.getExecType() == Types.ExecType.CP
						|| consumer.getBroadcastInput()==l)
						twrites.get(l.getOutputParameters().getLabel()).add(true);
			}
		}

	}
}
