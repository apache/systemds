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
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.OperatorOrderingUtils;
import org.apache.sysds.lops.UnaryCP;
import org.apache.sysds.parser.StatementBlock;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class RewriteAddBroadcastLop extends LopRewriteRule
{
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
				// Construct a Broadcast lop that takes this Spark node as an input
				UnaryCP bc = new UnaryCP(l, Types.OpOp1.BROADCAST, l.getDataType(), l.getValueType(), Types.ExecType.CP);
				bc.setAsynchronous(true);
				//FIXME: Wire Broadcast only with the necessary outputs
				for (Lop outCP : oldOuts) {
					// Rewire l -> outCP to l -> Broadcast -> outCP
					bc.addOutput(outCP);
					outCP.replaceInput(l, bc);
					l.removeOutput(outCP);
					//FIXME: Rewire _inputParams when needed (e.g. GroupedAggregate)
				}
				//Place it immediately after the Spark lop in the node list
				nodesWithBroadcast.add(bc);
			}
		}
		// New node is added inplace in the Lop DAG
		return Arrays.asList(sb);
	}

	@Override
	public List<StatementBlock> rewriteLOPinStatementBlocks(List<StatementBlock> sbs) {
		return sbs;
	}

	private static boolean isBroadcastNeeded(Lop lop) {
		// Asynchronously broadcast a matrix if that is produced by a CP instruction,
		// and at least one Spark parent needs to broadcast this intermediate (eg. mapmm)
		boolean isBc = lop.getOutputs().stream()
			.anyMatch(out -> (out.getBroadcastInput() == lop));
		//TODO: Early broadcast objects that are bigger than a single block
		boolean isCP = lop.getExecType() == Types.ExecType.CP;
		return isCP && isBc && lop.getDataType() == Types.DataType.MATRIX;
	}
}
