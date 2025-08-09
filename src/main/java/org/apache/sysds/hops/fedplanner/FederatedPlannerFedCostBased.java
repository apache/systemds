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

package org.apache.sysds.hops.fedplanner;

import java.util.Set;

import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.hops.ipa.FunctionCallGraph;
import org.apache.sysds.hops.ipa.FunctionCallSizeInfo;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.FunctionStatementBlock;
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction;
import org.apache.sysds.hops.fedplanner.FederatedMemoTable.FedPlan;
import org.apache.commons.lang3.tuple.Pair;

import java.util.HashSet;
import java.util.List;
/**
 * Baseline federated planner that compiles all hops
 * that support federated execution on federated inputs to
 * forced federated operations.
 */
public class FederatedPlannerFedCostBased extends AFederatedPlanner {
	@Override
	public void rewriteProgram( DMLProgram prog, FunctionCallGraph fgraph, FunctionCallSizeInfo fcallSizes )
	{
		FederatedMemoTable memoTable = new FederatedMemoTable();
		FedPlan optimalPlan = FederatedPlanCostEnumerator.enumerateProgram(prog, memoTable, true);
		Set<Long> visited = new HashSet<>();

		List<Pair<Long, FEDInstruction.FederatedOutput>> childFedPlanPairs = optimalPlan.getChildFedPlans();
		 for (Pair<Long, FEDInstruction.FederatedOutput> childFedPlanPair : childFedPlanPairs) {
			FedPlan childPlan = memoTable.getFedPlanAfterPrune(childFedPlanPair);
			rewriteHop(childPlan, memoTable, visited);
		 }
	}

	@Override
	public void rewriteFunctionDynamic(FunctionStatementBlock function, LocalVariableMap funcArgs) {
		FederatedMemoTable memoTable = new FederatedMemoTable();
		FedPlan optimalPlan = FederatedPlanCostEnumerator.enumerateFunctionDynamic(function, memoTable, true);
		Set<Long> visited = new HashSet<>();
		rewriteHop(optimalPlan, memoTable, visited);
	}

	private void rewriteHop(FedPlan optimalPlan, FederatedMemoTable memoTable, Set<Long> visited) {
		long hopID = optimalPlan.getHopRef().getHopID();

        if (visited.contains(hopID)) {
            return;
        } else {
            visited.add(hopID);
        }

        for (Pair<Long, FEDInstruction.FederatedOutput> childFedPlanPair : optimalPlan.getChildFedPlans()) {
            FedPlan childPlan = memoTable.getFedPlanAfterPrune(childFedPlanPair);
            
            // DEBUG: Check if getFedPlanAfterPrune returns null
            if (childPlan == null) {
				FederatedPlannerLogger.logNullChildPlanDebug(childFedPlanPair, optimalPlan, memoTable);
                continue;
            }
            
			rewriteHop(childPlan, memoTable, visited);
        }

		if (optimalPlan.getFedOutType() == FEDInstruction.FederatedOutput.LOUT) {
			optimalPlan.setFederatedOutput(FEDInstruction.FederatedOutput.LOUT);
			optimalPlan.setForcedExecType(ExecType.CP);
		} else {
			optimalPlan.setFederatedOutput(FEDInstruction.FederatedOutput.FOUT);
			optimalPlan.setForcedExecType(ExecType.FED);
		}
	}
}
