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

import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.Data;
import org.apache.sysds.lops.Lop;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Stack;
import java.util.stream.Collectors;

public class LinearizerCostBased extends IDagLinearizer
{
	@Override
	public List<Lop> linearize(List<Lop> v) {
		// Simplify DAG by removing literals and transient inputs and outputs
		List<Lop> removedLeaves = new ArrayList<>();
		List<Lop> removedRoots = new ArrayList<>();
		HashMap<Long, ArrayList<Lop>> removedInputs = new HashMap<>();
		HashMap<Long, ArrayList<Lop>> removedOutputs = new HashMap<>();
		simplifyDag(v, removedLeaves, removedRoots, removedInputs, removedOutputs);
		// TODO: Partition the DAG if connected by a single node. Optimize separately

		// Collect the leaf nodes of the simplified DAG
		List<Lop> leafNodes = v.stream().filter(l -> l.getInputs().isEmpty()).collect(Collectors.toList());

		// For each leaf, find all possible orders starting from the given leaf
		List<Order> finalOrders = new ArrayList<>();
		for (Lop leaf : leafNodes)
			generateOrders(leaf, leafNodes, finalOrders, v.size());

		// TODO: Handle distributed and GPU operators (0 compute cost, memory overhead on collect)
		// TODO: Asynchronous operators (max of compute costs, total operation memory overhead)

		// TODO: Select the order with minimum compute cost and buffer pool evictions
		int randInd = (int) (Math.random() * finalOrders.size());
		List<Lop> best = finalOrders.get(randInd).getOrder();

		// Add the removed leaf and root nodes back to the list
		addRemovedNodes(best, removedLeaves, removedRoots, removedInputs, removedOutputs);

		return best;
	}

	private static void generateOrders(Lop leaf, List<Lop> leafNodes, List<Order> finalOrders, int count)
	{
		// Create a stack to store the partial solutions
		Stack<Order> stack = new Stack<>();
		stack.push(new Order(leaf));

		while (!stack.isEmpty())
		{
			// Pop a partial order of Lops
			Order partialOrder = stack.pop();

			// If the partial order contains all nodes, move it to the full solution list
			if (partialOrder.size() == count) {
				finalOrders.add(partialOrder);
				continue;
			}

			// Create new partial orders and push to the stack
			List<Lop> distinctOutputs = new ArrayList<>();
			// Collect the distinct set of outputs of the already listed nodes
			for (Lop lop : partialOrder.getOrder()) {
				for(Lop out : lop.getOutputs()) {
					if(!out.isVisited() && allInputsLinearized(out, partialOrder) && !partialOrder.contains(out)) {
						out.setVisited();
						distinctOutputs.add(out);
					}
				}
			}
			// Create new partial orders with the outputs of the already listed nodes
			for (Lop out : distinctOutputs) {
				out.resetVisitStatus();
				stack.push(copyAndAdd(partialOrder, out, true));
			}

			// Create new partial orders with the disconnected leaves
			for (Lop otherLeaf : leafNodes) {
				if (!partialOrder.contains(otherLeaf)) {
					stack.push(copyAndAdd(partialOrder, otherLeaf, false));
				}
			}
		}
	}

	private static boolean allInputsLinearized(Lop lop, Order partialOrder) {
		List<Lop> order = partialOrder.getOrder();
		for (Lop input : lop.getInputs()) {
			if (!order.contains(input))
				return false;
		}
		return true;
	}

	private static Order copyAndAdd(Order partialOrder, Lop node, boolean allInputsLinearized) {
		Order newEntry = new Order(partialOrder);
		// Add the new operator and maintain memory and compute estimates
		newEntry.addOperator(node, allInputsLinearized);
		return newEntry;
	}

	private static void simplifyDag(List<Lop> lops, List<Lop> removedLeaves, List<Lop> removedRoots,
		HashMap<Long, ArrayList<Lop>> removedInputs, HashMap<Long, ArrayList<Lop>> removedOutputs) {
		// Store the removed nodes and the full input/output arrays (order preserving)
		for (Lop lop : lops) {
			if (lop.getInputs().isEmpty()
				&& ((lop instanceof Data && ((Data) lop).isTransientRead())
				|| lop.getDataType() == Types.DataType.SCALAR)) {
				removedLeaves.add(lop);
				for (Lop out : lop.getOutputs()) {
					removedInputs.putIfAbsent(out.getID(), new ArrayList<>(out.getInputs()));
					out.removeInput(lop);
				}
			}
			if (lop.getOutputs().isEmpty()
				&& lop instanceof Data && ((Data) lop).isTransientWrite()) {
				removedRoots.add(lop);
				for (Lop in : lop.getInputs()) {
					removedOutputs.putIfAbsent(in.getID(), new ArrayList<>(in.getOutputs()));
					in.removeOutput(lop);
				}
			}
		}
		// Remove the insignificant nodes from the main list
		lops.removeAll(removedLeaves);
		lops.removeAll(removedRoots);
	}

	private static void addRemovedNodes(List<Lop> lops, List<Lop> removedLeaves, List<Lop> removedRoots,
		// Add the nodes, removed during simplification back
		HashMap<Long, ArrayList<Lop>> removedInputs, HashMap<Long, ArrayList<Lop>> removedOutputs) {
		for (Lop leaf : removedLeaves)
			leaf.getOutputs().forEach(out -> out.replaceAllInputs(removedInputs.get(out.getID())));
		lops.addAll(0, removedLeaves);

		for (Lop root : removedRoots)
			root.getInputs().forEach(in -> in.replaceAllOutputs(removedOutputs.get(in.getID())));
		lops.addAll(removedRoots);
	}

	private static class Order
	{
		private List<Lop> _order;
		private double _pinnedMemEstimate;
		private double _bufferpoolEstimate;
		private int _numEvictions;
		private double _computeCost;

		public Order(List<Lop> lops, double pin, double bp, double comp) {
			_order = new ArrayList<>(lops);
			_pinnedMemEstimate = pin;
			_bufferpoolEstimate = bp;
			_numEvictions = 0;
			_computeCost = comp;
		}

		public Order(Lop lop) {
			// Initiate the memory estimates for the first operator
			this(Arrays.asList(lop), lop.getOutputMemoryEstimate(), 0, lop.getComputeEstimate());
		}

		public Order(Order that) {
			_order = that.getOrder();
			_pinnedMemEstimate = that._pinnedMemEstimate;
			_bufferpoolEstimate = that._bufferpoolEstimate;
			_numEvictions = that._numEvictions;
			_computeCost = that._computeCost;
		}

		public void addOperator(Lop lop, boolean allInputsLinearized) {
			_order.add(lop);
			// Update total compute cost for this partial order
			_computeCost += lop.getComputeEstimate();
			// Estimate buffer pool state after executing this operator
			_bufferpoolEstimate += lop.getOutputMemoryEstimate();
			if (allInputsLinearized) {
				lop.getInputs().forEach(in ->_bufferpoolEstimate -= in.getOutputMemoryEstimate());
				_bufferpoolEstimate = _bufferpoolEstimate < 0 ? 0 : _bufferpoolEstimate;
			}
			// Maintain total eviction count for this order
			if (_bufferpoolEstimate > OptimizerUtils.getBufferPoolLimit())
				_numEvictions++;
			// TODO: Add IO time to compute cost for evictions
			// Estimate operational memory state during the execution of this operator
			_pinnedMemEstimate = lop.getTotalMemoryEstimate();
		}

		protected List<Lop> getOrder() {
			return _order;
		}

		@SuppressWarnings("unused")
		protected double getComputeCost() {
			return _computeCost;
		}

		protected boolean contains(Lop lop) {
			return _order.contains(lop);
		}

		protected int size() {
			return _order.size();
		}
	}
}
