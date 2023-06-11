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
import org.apache.sysds.lops.CSVReBlock;
import org.apache.sysds.lops.CentralMoment;
import org.apache.sysds.lops.Checkpoint;
import org.apache.sysds.lops.CoVariance;
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
import org.apache.sysds.parser.StatementBlock;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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
			if (isPrefetchNeeded(l)) {
				List<Lop> oldOuts = new ArrayList<>(l.getOutputs());
				//Construct a Prefetch lop that takes this Spark node as a input
				UnaryCP prefetch = new UnaryCP(l, Types.OpOp1.PREFETCH, l.getDataType(), l.getValueType(), Types.ExecType.CP);
				prefetch.setAsynchronous(true);
				//Reset asynchronous flag for the input if already set (e.g. mapmm -> prefetch)
				l.setAsynchronous(false);
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
	public List<StatementBlock> rewriteLOPinStatementBlocks(List<StatementBlock> sbs) {
		return sbs;
	}

	private boolean isPrefetchNeeded(Lop lop) {
		// Run Prefetch for a Spark instruction if the instruction is a Transformation
		// and the output is consumed by only CP instructions.
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

		// Exclude List consumers. List is just a metadata handle.
		boolean anyOutputList = lop.getOutputs().stream()
			.anyMatch(out -> out.getDataType() == Types.DataType.LIST);

		//FIXME: Rewire _inputParams when needed (e.g. GroupedAggregate)
		boolean hasParameterizedOut = lop.getOutputs().stream()
			.anyMatch(out -> ((out instanceof ParameterizedBuiltin)
				|| (out instanceof GroupedAggregate)
				|| (out instanceof GroupedAggregateM)));
		//TODO: support non-matrix outputs
		return transformOP && !hasParameterizedOut && !anyOutputList
			&& (lop.isAllOutputsCP() || OperatorOrderingUtils.isCollectForBroadcast(lop))
			&& lop.getDataType() == Types.DataType.MATRIX;
	}
}
