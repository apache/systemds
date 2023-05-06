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
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.lops.Checkpoint;
import org.apache.sysds.lops.Data;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.OperatorOrderingUtils;
import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.WhileStatement;
import org.apache.sysds.parser.WhileStatementBlock;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class RewriteAddChkpointInLoop extends LopRewriteRule
{
	@Override
	public List<StatementBlock> rewriteLOPinStatementBlock(StatementBlock sb) {
		if (!ConfigurationManager.isCheckpointEnabled())
			return List.of(sb);

		if (sb == null || !HopRewriteUtils.isLastLevelLoopStatementBlock(sb))
			return List.of(sb);
		// TODO: support If-Else block inside loop. Consumers inside branches.

		// This rewrite adds checkpoints for the Spark intermediates, which
		// are updated in each iteration of a loop. Without the checkpoints,
		// CP consumers in the loop body will trigger long Spark jobs containing
		// all previous iterations. Note, a checkpoint is counterproductive if
		// there is no consumer in the loop body, i.e. all iterations combine
		// to form a single Spark job triggered from outside the loop.

		// Find the variables which are read and updated in each iteration
		Set<String> readUpdatedVars = sb.variablesRead().getVariableNames().stream()
			.filter(v -> sb.variablesUpdated().containsVariable(v))
			.collect(Collectors.toSet());
		if (readUpdatedVars.isEmpty())
			return List.of(sb);

		// Collect the Spark roots in the loop body (assuming single block)
		StatementBlock csb = sb instanceof WhileStatementBlock
			? ((WhileStatement) sb.getStatement(0)).getBody().get(0)
			: ((ForStatement) sb.getStatement(0)).getBody().get(0);
		ArrayList<Lop> lops = OperatorOrderingUtils.getLopList(csb);
		List<Lop> roots = lops.stream().filter(OperatorOrderingUtils::isLopRoot).collect(Collectors.toList());
		HashSet<Lop> sparkRoots = new HashSet<>();
		roots.forEach(r -> OperatorOrderingUtils.collectSparkRoots(r, new HashMap<>(), sparkRoots));
		if (sparkRoots.isEmpty())
			return List.of(sb);

		// Mark the Spark intermediates which are read and updated in each iteration
		Map<Long, Integer> operatorJobCount = new HashMap<>();
		findOverlappingJobs(sparkRoots, readUpdatedVars, operatorJobCount);
		if (operatorJobCount.isEmpty())
			return List.of(sb);

		// Add checkpoint Lops after the shared operators
		addChkpointLop(lops, operatorJobCount);
		// TODO: A rewrite pass to remove less effective checkpoints
		return List.of(sb);
	}

	@Override
	public List<StatementBlock> rewriteLOPinStatementBlocks(List<StatementBlock> sbs) {
		return sbs;
	}

	private void addChkpointLop(List<Lop> nodes, Map<Long, Integer> operatorJobCount) {
		for (Lop l : nodes) {
			if(operatorJobCount.containsKey(l.getID()) && operatorJobCount.get(l.getID()) > 1) {
				// TODO: Check if this lop leads to one of those variables
				// This operation is shared between Spark jobs
				List<Lop> oldOuts = new ArrayList<>(l.getOutputs());
				// Construct a chkpoint lop that takes this Spark node as an input
				Lop checkpoint = new Checkpoint(l, l.getDataType(), l.getValueType(),
					Checkpoint.getDefaultStorageLevelString(), false);
				for (Lop out : oldOuts) {
					//Rewire l -> out to l -> checkpoint -> out
					checkpoint.addOutput(out);
					out.replaceInput(l, checkpoint);
					l.removeOutput(out);
				}
			}
		}
	}

	private void findOverlappingJobs(HashSet<Lop> sparkRoots, Set<String> ruVars, Map<Long, Integer> operatorJobCount) {
		HashSet<Lop> sharedRoots = new HashSet<>();
		// Find the Spark jobs which are sharing these variables
		for (String var : ruVars) {
			for (Lop root : sparkRoots) {
				if(ifJobContains(root, var))
					sharedRoots.add(root);
				root.resetVisitStatus();
			}
			// Mark the operators shared by these Spark jobs
			if (!sharedRoots.isEmpty())
				OperatorOrderingUtils.markSharedSparkOps(sharedRoots, operatorJobCount);
			sharedRoots.clear();
		}
	}

	// Check if this Spark job has the passed variable as a leaf node
	private boolean ifJobContains(Lop root, String var) {
		if (root.isVisited())
			return false;

		for (Lop input : root.getInputs()) {
			if (!(input instanceof Data) && (!input.isExecSpark() || root.getBroadcastInput() == input))
				continue; //consider only Spark operator chains
			if (ifJobContains(input, var)) {
				root.setVisited();
				return true;
			}
		}

		if (root instanceof Data && ((Data) root).isTransientRead())
			if (root.getOutputParameters().getLabel().equalsIgnoreCase(var)) {
				root.setVisited();
				return true;
			}

		root.setVisited();
		return false;
	}

}
