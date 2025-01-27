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

package org.apache.sysds.lops.compile.linearization;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.hops.AggBinaryOp.SparkAggType;
import org.apache.sysds.lops.CSVReBlock;
import org.apache.sysds.lops.CentralMoment;
import org.apache.sysds.lops.Checkpoint;
import org.apache.sysds.lops.CoVariance;
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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class LinearizerMaxParallelism extends IDagLinearizer
{
	// Place the Spark operation chains first (more expensive to less expensive),
	// followed by asynchronously triggering operators and CP chains.
	@Override
	public List<Lop> linearize(List<Lop> v) {
		List<Lop> v2 = v;
		boolean hasSpark = v.stream().anyMatch(LinearizerMaxParallelism::isDistributedOp);
		boolean hasGPU = v.stream().anyMatch(LinearizerMaxParallelism::isGPUOp);

		// Fallback to default depth-first if all operators are CP
		if (!hasSpark && !hasGPU)
			return new LinearizerDepthFirst().linearize(v);

		if (hasSpark) {
			// Step 1: Collect the Spark roots and #Spark instructions in each subDAG
			Map<Long, Integer> sparkOpCount = new HashMap<>();
			List<Lop> roots = v.stream().filter(OperatorOrderingUtils::isLopRoot).collect(Collectors.toList());
			HashSet<Lop> sparkRoots = new HashSet<>();
			roots.forEach(r -> OperatorOrderingUtils.collectSparkRoots(r, sparkOpCount, sparkRoots));
			sparkRoots.forEach(sr -> sr.setAsynchronous(true));

			// Step 2: Depth-first linearization of Spark roots.
			// Maintain the default order (by ID) to trigger independent Spark jobs first
			// This allows parallel execution of the jobs in the cluster
			ArrayList<Lop> operatorList = new ArrayList<>();
			sparkRoots.forEach(r -> depthFirst(r, operatorList, sparkOpCount, false));

			// Step 3: Place the rest of the operators (CP). Sort the CP roots based on
			// #Spark operators in ascending order, i.e. execute the independent CP legs first
			roots.forEach(r -> depthFirst(r, operatorList, sparkOpCount, false));
			roots.forEach(Lop::resetVisitStatus);

			v2 = operatorList;
		}

		if (hasGPU) {
			// Step 1: Collect the GPU roots and #GPU instructions in each subDAG
			Map<Long, Integer> gpuOpCount = new HashMap<>();
			List<Lop> roots = v2.stream().filter(OperatorOrderingUtils::isLopRoot).collect(Collectors.toList());
			HashSet<Lop> gpuRoots = new HashSet<>();
			roots.forEach(r -> OperatorOrderingUtils.collectGPURoots(r, gpuOpCount, gpuRoots));
			gpuRoots.forEach(sr -> sr.setAsynchronous(true));

			// Step 2: Depth-first linearization of GPU roots.
			// Maintain the default order (by ID) to trigger independent GPU OP chains first
			ArrayList<Lop> operatorList = new ArrayList<>();
			gpuRoots.forEach(r -> depthFirst(r, operatorList, gpuOpCount, false));

			// Step 3: Place the rest of the operators (CP).
			roots.forEach(r -> depthFirst(r, operatorList, gpuOpCount, false));
			roots.forEach(Lop::resetVisitStatus);

			v2 = operatorList;
		}
		return v2;
	}
	

	// Place the operators in a depth-first manner, but order
	// the DAGs based on number of Spark operators
	private static void depthFirst(Lop root, ArrayList<Lop> opList, Map<Long, Integer> sparkOpCount, boolean sparkFirst) {
		if (root.isVisited())
			return;

		if (root.getInputs().isEmpty()) {  //leaf node
			opList.add(root);
			root.setVisited();
			return;
		}
		// Sort the inputs based on number of Spark operators
		Lop[] sortedInputs = root.getInputs().toArray(new Lop[0]);
		if (sparkFirst) //to place the child DAG with more Spark OPs first
			Arrays.sort(sortedInputs, (l1, l2) -> sparkOpCount.get(l2.getID()) - sparkOpCount.get(l1.getID()));
		else //to place the child DAG with more CP OPs first
			Arrays.sort(sortedInputs, Comparator.comparingInt(l -> sparkOpCount.get(l.getID())));

		for (Lop input : sortedInputs)
			depthFirst(input, opList, sparkOpCount, sparkFirst);

		opList.add(root);
		root.setVisited();
	}

	private static boolean isDistributedOp(Lop lop) {
		return lop.isExecSpark()
			|| (lop instanceof UnaryCP
			&& (((UnaryCP) lop).getOpCode().equalsIgnoreCase(Opcodes.PREFETCH.toString())
			|| ((UnaryCP) lop).getOpCode().equalsIgnoreCase(Opcodes.BROADCAST.toString())));
	}

	private static boolean isGPUOp(Lop lop) {
		return lop.isExecGPU()
			|| (lop instanceof UnaryCP
			&& (((UnaryCP) lop).getOpCode().equalsIgnoreCase(Opcodes.PREFETCH.toString())
			|| ((UnaryCP) lop).getOpCode().equalsIgnoreCase(Opcodes.BROADCAST.toString())));
	}

	@SuppressWarnings("unused")
	private static List<Lop> addAsyncEagerCheckpointLop(List<Lop> nodes) {
		List<Lop> nodesWithCheckpoint = new ArrayList<>();
		 // Find the Spark action nodes
		for (Lop l : nodes) {
			if (isCheckpointNeeded(l)) {
				List<Lop> oldInputs = new ArrayList<>(l.getInputs());
				// Place a Checkpoint node just below this node (Spark action)
				for (Lop in : oldInputs) {
					if (in.getExecType() != ExecType.SPARK)
						continue;
					// Rewire in -> l to in -> Checkpoint -> l
					//UnaryCP checkpoint = new UnaryCP(in, OpOp1.TRIGREMOTE, in.getDataType(), in.getValueType(), ExecType.CP);
					Lop checkpoint = new Checkpoint(in, in.getDataType(), in.getValueType(),
						Checkpoint.getDefaultStorageLevelString(), true);
					checkpoint.addOutput(l);
					l.replaceInput(in, checkpoint);
					in.removeOutput(l);
					nodesWithCheckpoint.add(checkpoint);
				}
			}
			nodesWithCheckpoint.add(l);
		}
		return nodesWithCheckpoint;
	}

	private static boolean isCheckpointNeeded(Lop lop) {
		// Place checkpoint_e just before a Spark action (FIXME)
		boolean actionOP = lop.getExecType() == ExecType.SPARK
				&& ((lop.getAggType() == SparkAggType.SINGLE_BLOCK)
				// Always Action operations
				|| (lop.getDataType() == DataType.SCALAR)
				|| (lop instanceof MapMultChain) || (lop instanceof PickByCount)
				|| (lop instanceof MMZip) || (lop instanceof CentralMoment)
				|| (lop instanceof CoVariance) || (lop instanceof MMTSJ))
				// Not qualified for Checkpoint
				&& !(lop instanceof Checkpoint) && !(lop instanceof ReBlock)
				&& !(lop instanceof CSVReBlock)
				// Cannot filter Transformation cases from Actions (FIXME)
				&& !(lop instanceof UAggOuterChain)
				&& !(lop instanceof ParameterizedBuiltin) && !(lop instanceof SpoofFused);

		//FIXME: Rewire _inputParams when needed (e.g. GroupedAggregate)
		boolean hasParameterizedOut = lop.getOutputs().stream()
				.anyMatch(out -> ((out instanceof ParameterizedBuiltin)
					|| (out instanceof GroupedAggregate)
					|| (out instanceof GroupedAggregateM)));
		return actionOP && !hasParameterizedOut;
	}
}
