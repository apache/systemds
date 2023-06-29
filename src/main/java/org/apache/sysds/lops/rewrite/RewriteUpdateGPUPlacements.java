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
import org.apache.sysds.lops.Data;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.MatMultCP;
import org.apache.sysds.lops.OperatorOrderingUtils;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.runtime.matrix.data.LibMatrixNative;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.util.ArrayList;
import java.util.List;

public class RewriteUpdateGPUPlacements extends LopRewriteRule
{
	@Override
	public List<StatementBlock> rewriteLOPinStatementBlock(StatementBlock sb)
	{
		// Return if rule-based GPU placement is disabled
		if (!ConfigurationManager.isRuleBasedGPUPlacement())
			return List.of(sb);

		// Return if all operators are CP
		ArrayList<Lop> lops = OperatorOrderingUtils.getLopList(sb);
		if (lops == null || lops.stream().noneMatch(Lop::isExecGPU))
			return List.of(sb);

		// Iterate the DAGs and apply the rules on the GPU operators
		// TODO: Iterate multiple times to propagate the updates
		List<Lop> roots = sb.getLops();
		roots.forEach(this::rUpdateExecType);
		roots.forEach(Lop::resetVisitStatus);

		return List.of(sb);
	}

	@Override
	public List<StatementBlock> rewriteLOPinStatementBlocks(List<StatementBlock> sbs) {
		return sbs;
	}

	private void updateExecTypeGPU2CP(Lop lop) {
		// Return if not a GPU op
		if (!lop.isExecGPU())
			return;

		// Rule1: Place only dense operators at GPU (no sparse inputs)
		// Ignore this check if dimensions and nnz are unknown
		for (Lop in : lop.getInputs()) {
			if (in.getNnz() >= 0
				&& MatrixBlock.evalSparseFormatInMemory(in.getNumRows(), in.getNumCols(), in.getNnz())) {
				// Sparse input. Change to CP. This also avoids s2d and d2s conversions.
				lop.setExecType(Types.ExecType.CP);
				return;
			}
		}

		// Rule2: Place compute-intensive MatMults at GPU regardless inputs' locations
		if (lop instanceof MatMultCP) {
			boolean memBound = LibMatrixNative.isMatMultMemoryBound((int) lop.getInput(0).getNumRows(),
				(int) lop.getInput(0).getNumCols(), (int) lop.getInput(1).getNumCols());
			if (!memBound) // Compute bound. Stays at GPU
				return;
		}

		// Rule3: Location aware placement
		// TODO: Propagate GPU execution types to DataOps (in hop level or lop level).
		//  For now, skip this rule if the input is a DataOp.
		if (lop.getInputs().size() == 2) { //binary operator
			// Estimate sizes
			long size1 = MatrixBlock.estimateSizeInMemory(lop.getInput(0).getNumRows(),
				lop.getInput(0).getNumCols(), lop.getInput(0).getNnz());
			long size2 = MatrixBlock.estimateSizeInMemory(lop.getInput(1).getNumRows(),
				lop.getInput(1).getNumCols(), lop.getInput(1).getNnz());
			// Move to CP if the larger input and all output intermediates are at host
			if (size1 > size2 && !((lop.getInput(0)) instanceof Data)
				&& !lop.getInput(0).isExecGPU() && !lop.isAllOutputsGPU())
				lop.setExecType(Types.ExecType.CP);
			if (size2 > size1 && !((lop.getInput(1)) instanceof Data)
				&& !lop.getInput(1).isExecGPU() && !lop.isAllOutputsGPU())
				lop.setExecType(Types.ExecType.CP);
			// If same sized, move to CP if both the inputs and outputs are CP
			if (size1 == size2 &&!(lop.getInput(0) instanceof Data)
				&& !(lop.getInput(1) instanceof Data) && !lop.getInput(0).isExecGPU()
				&& !lop.getInput(1).isExecGPU() && !lop.isAllOutputsGPU())
				lop.setExecType(Types.ExecType.CP);
		}

		// For unary, move to CP if the input and the outputs are CP
		if (lop.getInputs().size() == 1)
			if (!(lop.getInput(0) instanceof Data)
				&& !lop.getInput(0).isExecGPU()
				&& !lop.isAllOutputsGPU())
				lop.setExecType(Types.ExecType.CP);

		// For ternary, move to CP if most inputs and outputs are CP
		if (lop.getInputs().size() > 2) {
			int numGPUInputs = 0;
			int numCPInputs = 0;
			for (Lop in : lop.getInputs()) {
				if (!(in instanceof Data) && in.isExecGPU())
					numGPUInputs++;
				if (!(in instanceof Data) && in.isExecCP())
					numCPInputs++;
			}
			if (numCPInputs > numGPUInputs && !lop.isAllOutputsGPU())
				lop.setExecType(Types.ExecType.CP);
		}
	}

	private void rUpdateExecType(Lop root) {
		if (root.isVisited())
			return;

		for (Lop input : root.getInputs()) {
			if (input instanceof Data)
				continue;
			rUpdateExecType(input);
		}
		updateExecTypeGPU2CP(root);
		root.setVisited();
	}
}
