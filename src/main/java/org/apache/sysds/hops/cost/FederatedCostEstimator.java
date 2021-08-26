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

package org.apache.sysds.hops.cost;

import org.apache.sysds.hops.Hop;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.FunctionStatementBlock;
import org.apache.sysds.parser.IfStatementBlock;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.WhileStatementBlock;

import java.util.ArrayList;

public class FederatedCostEstimator {
	private static final int DEFAULT_ITERATION_NUMBER = 15;
	private static final double WORKER_NETWORK_BANDWIDTH_BYTES = 1024*1024*1024; //Default network bandwidth in bytes per second
	private static final double WORKER_COMPUTE_BANDWITH_FLOPS = 2.5*1024*1024*1024; //Default compute bandwidth in FLOPS
	private static final double WORKER_DEGREE_OF_PARALLELISM = 8; //Default number of parallel processes for workers

	public FederatedCost costEstimate(DMLProgram dmlProgram){
		FederatedCost programTotalCost = new FederatedCost();
		for ( StatementBlock stmBlock : dmlProgram.getStatementBlocks() )
			programTotalCost.addInputTotalCost(costEstimate(stmBlock).getTotal());
		return programTotalCost;
	}

	private FederatedCost costEstimate(StatementBlock sb){
		if ( sb instanceof WhileStatementBlock){
			WhileStatementBlock whileSB = (WhileStatementBlock) sb;
			FederatedCost whileSBCost = addInitialInputCost(sb);
			whileSBCost.addRepititionCost(DEFAULT_ITERATION_NUMBER);
			return whileSBCost;
		}
		else if ( sb instanceof IfStatementBlock){
			IfStatementBlock ifSB = (IfStatementBlock) sb;
			//Get cost of if-block + else-block and divide by two
			// since only one of the code blocks will be executed in the end
			FederatedCost ifSBCost = addInitialInputCost(sb);
			ifSBCost.setInputTotalCost(ifSBCost.getInputTotalCost() / 2);
			return ifSBCost;
		}
		else if ( sb instanceof ForStatementBlock){
			// This also includes ParForStatementBlocks
			ForStatementBlock forSB = (ForStatementBlock) sb;
			FederatedCost forCost = addInitialInputCost(sb);
			forCost.addRepititionCost(forSB.getEstimateReps());
			return forCost;
		}
		else if ( sb instanceof FunctionStatementBlock){
			FederatedCost funcCost = addInitialInputCost(sb);
			FunctionStatementBlock funcSB = (FunctionStatementBlock) sb;
			return funcCost;
		}
		else {
			// StatementBlock type (no subclass)
			return costEstimate(sb.getHops());
		}
	}

	private FederatedCost addInitialInputCost(StatementBlock sb){
		FederatedCost basicCost = new FederatedCost();
		for ( StatementBlock childSB : sb.getDMLProg().getStatementBlocks() )
			basicCost.addInputTotalCost(costEstimate(childSB).getTotal());
		return basicCost;
	}

	private FederatedCost costEstimate(ArrayList<Hop> roots){
		FederatedCost basicCost = new FederatedCost();
		for ( Hop root : roots )
			basicCost.addInputTotalCost(costEstimate(root).getTotal());
		return basicCost;
	}

	/**
	 * Return cost estimate of Hop DAG starting from given root.
	 * @param root of Hop DAG for which cost is estimated
	 * @return cost estimation of Hop DAG starting from given root
	 */
	private FederatedCost costEstimate(Hop root){
		if ( root.federatedCostInitialized() )
			return root.getFederatedCost();
		else {
			// If no input has FOUT, the root will be processed by the coordinator
			boolean hasFederatedInput = root.someInputFederated();
			//the input cost is included the first time the input hop is used
			//for additional usage, the additional cost is zero (disregarding potential read cost)
			double inputCosts = root.getInput().stream()
				.mapToDouble( in -> in.federatedCostInitialized() ? 0 : costEstimate(in).getTotal() )
				.sum();
			double inputTransferCost = hasFederatedInput ? root.getInput().stream()
				.filter(Hop::hasLocalOutput)
				.mapToDouble(Hop::getOutputMemEstimate)
				.map(inMem -> inMem/WORKER_NETWORK_BANDWIDTH_BYTES)
				.sum() : 0;
			double computingCost = ComputeCost.getHOPComputeCost(root);
			if ( hasFederatedInput ){
				//Find the number of inputs that has FOUT set.
				int numWorkers = (int)root.getInput().stream().filter(Hop::hasFederatedOutput).count();
				//divide memory usage by the number of workers the computation would be split to multiplied by
				//the number of parallel processes at each worker multiplied by the FLOPS of each process
				//This assumes uniform workload among the workers with FOUT data involved in the operation
				//and assumes that the degree of parallelism and compute bandwidth are equal for all workers
				computingCost = computingCost / (numWorkers*WORKER_DEGREE_OF_PARALLELISM*WORKER_COMPUTE_BANDWITH_FLOPS);
			}
			//Calculate output transfer cost if the operation is computed at federated workers and the output is forced to the coordinator
			double outputTransferCost = ( root.hasLocalOutput() && hasFederatedInput ) ?
				root.getOutputMemEstimate() / WORKER_NETWORK_BANDWIDTH_BYTES : 0;
			double readCost = root.getInputMemEstimate();

			FederatedCost rootFedCost =
				new FederatedCost(readCost, inputTransferCost, outputTransferCost, computingCost, inputCosts);
			root.setFederatedCost(rootFedCost);
			return rootFedCost;
		}
	}
}
