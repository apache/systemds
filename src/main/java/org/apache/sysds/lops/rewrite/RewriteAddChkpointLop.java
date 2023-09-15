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

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.lops.Checkpoint;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.OperatorOrderingUtils;
import org.apache.sysds.parser.StatementBlock;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class RewriteAddChkpointLop extends LopRewriteRule
{
	@Override
	public List<StatementBlock> rewriteLOPinStatementBlock(StatementBlock sb)
	{
		if (!ConfigurationManager.isCheckpointEnabled())
			return List.of(sb);

		ArrayList<Lop> lops = OperatorOrderingUtils.getLopList(sb);
		if (lops == null)
			return List.of(sb);

		// Collect the Spark roots and #Spark instructions in each subDAG
		HashSet<Lop> sparkRoots = new HashSet<>();
		Map<Long, Integer> sparkOpCount = new HashMap<>();
		List<Lop> roots = lops.stream().filter(OperatorOrderingUtils::isLopRoot).collect(Collectors.toList());
		roots.forEach(r -> OperatorOrderingUtils.collectSparkRoots(r, sparkOpCount, sparkRoots));
		if (sparkRoots.isEmpty())
			return List.of(sb);

		// Add Chkpoint lops after the expensive Spark operators, which are
		// shared among multiple Spark jobs. Only consider operators with
		// Spark consumers for now.
		Map<Long, Integer> operatorJobCount = new HashMap<>();
		//markPersistableSparkOps(sparkRoots, operatorJobCount);
		OperatorOrderingUtils.markSharedSparkOps(sparkRoots, operatorJobCount);
		// TODO: A rewrite pass to remove less effective checkpoints
		addChkpointLop(lops, operatorJobCount);
		placeCompiledCheckpoints(lops, sb);
		//New node is added inplace in the Lop DAG
		return List.of(sb);
	}

	@Override
	public List<StatementBlock> rewriteLOPinStatementBlocks(List<StatementBlock> sbs) {
		return sbs;
	}

	private void addChkpointLop(List<Lop> nodes, Map<Long, Integer> operatorJobCount) {
		for (Lop l : nodes) {
			// Increment the job counter if this node benefits from persisting
			// and reachable from multiple job roots
			if(operatorJobCount.containsKey(l.getID())
				&& operatorJobCount.get(l.getID()) > 1
				&& OperatorOrderingUtils.isPersistableSparkOp(l)) {
				// This operation is expensive and shared between Spark jobs
				List<Lop> oldOuts = new ArrayList<>(l.getOutputs());
				// Construct a chkpoint lop that takes this Spark node as an input
				Lop chkpoint = new Checkpoint(l, l.getDataType(), l.getValueType(),
					Checkpoint.getDefaultStorageLevelString(), false);
				for (Lop out : oldOuts) {
					//Rewire l -> out to l -> chkpoint -> out
					chkpoint.addOutput(out);
					out.replaceInput(l, chkpoint);
					l.removeOutput(out);
				}
			}
		}
	}

	private void placeCompiledCheckpoints(List<Lop> nodes, StatementBlock sb) {
		if (sb.getCheckpointPositions() == null)
			return;

		for (Lop l : nodes) {
			// Check if the compiler placed and saved a checkpoint
			// TODO: Call recompiler on the loops
			if (isCheckpointed(l, sb)) {
				List<Lop> oldOuts = new ArrayList<>(l.getOutputs());
				// Construct a chkpoint lop that takes this Spark node as an input
				Lop chkpoint = new Checkpoint(l, l.getDataType(), l.getValueType(),
					Checkpoint.getDefaultStorageLevelString(), false);
				for (Lop out : oldOuts) {
					//Rewire l -> out to l -> chkpoint -> out
					chkpoint.addOutput(out);
					out.replaceInput(l, chkpoint);
					l.removeOutput(out);
				}
			}
		}
	}

	private boolean isCheckpointed(Lop lop, StatementBlock sb) {
		var cpPositions = sb.getCheckpointPositions();
		if (cpPositions == null)
			return false;

		if (cpPositions.containsKey(lop.getType())) {
			List<Lop.Type> outputsT = cpPositions.get(lop.getType());
			List<Lop> outputs = new ArrayList<>(lop.getOutputs());
			if (outputs.size() != outputsT.size())
				return false;
			for (int i=0; i< outputs.size(); i++) {
				if (outputs.get(i).getType() != outputsT.get(i)
					|| !outputs.get(i).isExecSpark())
					return false;
			}
		}
		else
			return false;

		return true;
	}
}
