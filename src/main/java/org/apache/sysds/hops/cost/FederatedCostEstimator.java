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
import org.apache.sysds.hops.ipa.MemoTable;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.FunctionStatement;
import org.apache.sysds.parser.FunctionStatementBlock;
import org.apache.sysds.parser.IfStatement;
import org.apache.sysds.parser.IfStatementBlock;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.WhileStatement;
import org.apache.sysds.parser.WhileStatementBlock;

import java.util.ArrayList;

/**
 * Cost estimator for federated executions with methods and constants for going through DML programs to estimate costs.
 */
public class FederatedCostEstimator {
	public int DEFAULT_MEMORY_ESTIMATE = 8;
	public int DEFAULT_ITERATION_NUMBER = 15;
	public double WORKER_NETWORK_BANDWIDTH_BYTES_PS = 1024*1024*1024; //Default network bandwidth in bytes per second
	public double WORKER_COMPUTE_BANDWIDTH_FLOPS = 2.5*1024*1024*1024; //Default compute bandwidth in FLOPS
	public double WORKER_DEGREE_OF_PARALLELISM = 8; //Default number of parallel processes for workers
	public double WORKER_READ_BANDWIDTH_BYTES_PS = 3.5*1024*1024*1024; //Default read bandwidth in bytes per second

	public boolean printCosts = false; //Temporary for debugging purposes

	/**
	 * Estimate cost of given DML program in bytes.
	 * @param dmlProgram for which the cost is estimated
	 * @return federated cost object with cost estimate in bytes
	 */
	public FederatedCost costEstimate(DMLProgram dmlProgram){
		FederatedCost programTotalCost = new FederatedCost();
		for ( StatementBlock stmBlock : dmlProgram.getStatementBlocks() )
			programTotalCost.addInputTotalCost(costEstimate(stmBlock).getTotal());
		return programTotalCost;
	}

	/**
	 * Cost estimate in bytes of given statement block.
	 * @param sb statement block
	 * @return federated cost object with cost estimate in bytes
	 */
	private FederatedCost costEstimate(StatementBlock sb){
		if ( sb instanceof WhileStatementBlock){
			WhileStatementBlock whileSB = (WhileStatementBlock) sb;
			FederatedCost whileSBCost = costEstimate(whileSB.getPredicateHops());
			for ( Statement statement : whileSB.getStatements() ){
				WhileStatement whileStatement = (WhileStatement) statement;
				for ( StatementBlock bodyBlock : whileStatement.getBody() )
					whileSBCost.addInputTotalCost(costEstimate(bodyBlock));
			}
			whileSBCost.addRepetitionCost(DEFAULT_ITERATION_NUMBER);
			return whileSBCost;
		}
		else if ( sb instanceof IfStatementBlock){
			//Get cost of if-block + else-block and divide by two
			// since only one of the code blocks will be executed in the end
			IfStatementBlock ifSB = (IfStatementBlock) sb;
			FederatedCost ifSBCost = new FederatedCost();
			for ( Statement statement : ifSB.getStatements() ){
				IfStatement ifStatement = (IfStatement) statement;
				for ( StatementBlock ifBodySB : ifStatement.getIfBody() )
					ifSBCost.addInputTotalCost(costEstimate(ifBodySB));
				for ( StatementBlock elseBodySB : ifStatement.getElseBody() )
					ifSBCost.addInputTotalCost(costEstimate(elseBodySB));
			}
			ifSBCost.setInputTotalCost(ifSBCost.getInputTotalCost()/2);
			ifSBCost.addInputTotalCost(costEstimate(ifSB.getPredicateHops()));
			return ifSBCost;
		}
		else if ( sb instanceof ForStatementBlock){
			// This also includes ParForStatementBlocks
			ForStatementBlock forSB = (ForStatementBlock) sb;
			ArrayList<Hop> predicateHops = new ArrayList<>();
			predicateHops.add(forSB.getFromHops());
			predicateHops.add(forSB.getToHops());
			predicateHops.add(forSB.getIncrementHops());
			FederatedCost forSBCost = costEstimate(predicateHops);
			for ( Statement statement : forSB.getStatements() ){
				ForStatement forStatement = (ForStatement) statement;
				for ( StatementBlock forStatementBlockBody : forStatement.getBody() )
					forSBCost.addInputTotalCost(costEstimate(forStatementBlockBody));
			}
			forSBCost.addRepetitionCost(forSB.getEstimateReps());
			return forSBCost;
		}
		else if ( sb instanceof FunctionStatementBlock){
			FederatedCost funcCost = addInitialInputCost(sb);
			FunctionStatementBlock funcSB = (FunctionStatementBlock) sb;
			for(Statement statement : funcSB.getStatements()) {
				FunctionStatement funcStatement = (FunctionStatement) statement;
				for ( StatementBlock funcStatementBody : funcStatement.getBody() )
					funcCost.addInputTotalCost(costEstimate(funcStatementBody));
			}
			return funcCost;
		}
		else {
			// StatementBlock type (no subclass)
			return costEstimate(sb.getHops());
		}
	}

	/**
	 * Creates new FederatedCost object and adds all child statement block cost estimates to the object.
	 * @param sb statement block
	 * @return new FederatedCost estimate object with all estimates of child statement blocks added
	 */
	private FederatedCost addInitialInputCost(StatementBlock sb){
		FederatedCost basicCost = new FederatedCost();
		for ( StatementBlock childSB : sb.getDMLProg().getStatementBlocks() )
			basicCost.addInputTotalCost(costEstimate(childSB).getTotal());
		return basicCost;
	}

	/**
	 * Cost estimate in bytes of given list of roots.
	 * The individual cost estimates of the hops are summed.
	 * @param roots list of hops
	 * @return new FederatedCost object with sum of cost estimates of given hops
	 */
	private FederatedCost costEstimate(ArrayList<Hop> roots){
		FederatedCost basicCost = new FederatedCost();
		for ( Hop root : roots )
			basicCost.addInputTotalCost(costEstimate(root));
		return basicCost;
	}

	/**
	 * Return cost estimate in bytes of Hop DAG starting from given root.
	 * @param root of Hop DAG for which cost is estimated
	 * @return cost estimation of Hop DAG starting from given root
	 */
	public FederatedCost costEstimate(Hop root){
		if ( root.federatedCostInitialized() )
			return root.getFederatedCost();
		else {
			// If no input has FOUT, the root will be processed by the coordinator
			boolean hasFederatedInput = root.someInputFederated();
			// The input cost is included the first time the input hop is used.
			// For additional usage, the additional cost is zero (disregarding potential read cost).
			double inputCosts = root.getInput().stream()
				.mapToDouble( in -> in.federatedCostInitialized() ? 0 : costEstimate(in).getTotal() )
				.sum();
			double inputTransferCost = inputTransferCostEstimate(hasFederatedInput, root);
			double computingCost = ComputeCost.getHOPComputeCost(root);
			if ( hasFederatedInput ){
				// Find the number of inputs that has FOUT set.
				int numWorkers = (int)root.getInput().stream().filter(Hop::hasFederatedOutput).count();
				// Divide memory usage by the number of workers the computation would be split to multiplied by
				// the number of parallel processes at each worker multiplied by the FLOPS of each process.
				// This assumes uniform workload among the workers with FOUT data involved in the operation
				// and assumes that the degree of parallelism and compute bandwidth are equal for all workers
				computingCost = computingCost / (numWorkers*WORKER_DEGREE_OF_PARALLELISM* WORKER_COMPUTE_BANDWIDTH_FLOPS);
			} else computingCost = computingCost / (WORKER_DEGREE_OF_PARALLELISM* WORKER_COMPUTE_BANDWIDTH_FLOPS);
			// Calculate output transfer cost if the operation is computed at federated workers and the output is forced to the coordinator
			double outputTransferCost = ( root.hasLocalOutput() && (hasFederatedInput || root.isFederatedDataOp()) ) ?
				root.getOutputMemEstimate(DEFAULT_MEMORY_ESTIMATE) / WORKER_NETWORK_BANDWIDTH_BYTES_PS : 0;
			double readCost = root.getInputMemEstimate(DEFAULT_MEMORY_ESTIMATE) / WORKER_READ_BANDWIDTH_BYTES_PS;

			FederatedCost rootFedCost =
				new FederatedCost(readCost, inputTransferCost, outputTransferCost, computingCost, inputCosts);
			root.setFederatedCost(rootFedCost);

			if ( printCosts )
				printCosts(root);

			return rootFedCost;
		}
	}

	/**
	 * Return cost estimate in bytes of Hop DAG starting from given root HopRel.
	 * @param root HopRel of Hop DAG for which cost is estimated
	 * @param hopRelMemo memo table of HopRels for calculating input costs
	 * @return cost estimation of Hop DAG starting from given root HopRel
	 */
	public FederatedCost costEstimate(HopRel root, MemoTable hopRelMemo){
		// Check if root is in memo table.
		if ( hopRelMemo.containsHopRel(root) ){
			return root.getCostObject();
		}
		else {
			// If no input has FOUT, the root will be processed by the coordinator
			boolean hasFederatedInput = root.inputDependency.stream().anyMatch(in -> in.hopRef.hasFederatedOutput());
			// The input cost is included the first time the input hop is used.
			// For additional usage, the additional cost is zero (disregarding potential read cost).
			double inputCosts = root.inputDependency.stream()
				.mapToDouble( in -> {
					double inCost = in.existingCostPointer(root.hopRef.getHopID()) ?
						0 : costEstimate(in, hopRelMemo).getTotal();
					in.addCostPointer(root.hopRef.getHopID());
					return inCost;
				} )
				.sum();
			double inputTransferCost = inputTransferCostEstimate(hasFederatedInput, root);
			double computingCost = ComputeCost.getHOPComputeCost(root.hopRef);
			if ( hasFederatedInput ){
				// Find the number of inputs that has FOUT set.
				int numWorkers = (int)root.inputDependency.stream().filter(HopRel::hasFederatedOutput).count();
				// Divide memory usage by the number of workers the computation would be split to multiplied by
				// the number of parallel processes at each worker multiplied by the FLOPS of each process
				// This assumes uniform workload among the workers with FOUT data involved in the operation
				// and assumes that the degree of parallelism and compute bandwidth are equal for all workers
				computingCost = computingCost / (numWorkers*WORKER_DEGREE_OF_PARALLELISM* WORKER_COMPUTE_BANDWIDTH_FLOPS);
			} else computingCost = computingCost / (WORKER_DEGREE_OF_PARALLELISM* WORKER_COMPUTE_BANDWIDTH_FLOPS);
			// Calculate output transfer cost if the operation is computed at federated workers and the output is forced to the coordinator
			// If the root is a federated DataOp, the data is forced to the coordinator even if no input is LOUT
			double outputTransferCost = ( root.hasLocalOutput() && (hasFederatedInput || root.hopRef.isFederatedDataOp()) ) ?
				root.hopRef.getOutputMemEstimate(DEFAULT_MEMORY_ESTIMATE) / WORKER_NETWORK_BANDWIDTH_BYTES_PS : 0;
			double readCost = root.hopRef.getInputMemEstimate(DEFAULT_MEMORY_ESTIMATE) / WORKER_READ_BANDWIDTH_BYTES_PS;

			return new FederatedCost(readCost, inputTransferCost, outputTransferCost, computingCost, inputCosts);
		}
	}

	/**
	 * Returns input transfer cost estimate.
	 * The input transfer cost estimate is based on the memory estimate of LOUT when some input is FOUT
	 * except if root is a federated DataOp, since all input for this has to be at the coordinator.
	 * When no input is FOUT, the input transfer cost is always 0.
	 * @param hasFederatedInput true if root has any FOUT input
	 * @param root hopRel for which cost is estimated
	 * @return input transfer cost estimate
	 */
	private double inputTransferCostEstimate(boolean hasFederatedInput, HopRel root){
		if ( hasFederatedInput )
			return root.inputDependency.stream()
				.filter(input -> (root.hopRef.isFederatedDataOp()) ? input.hasFederatedOutput() : input.hasLocalOutput() )
				.mapToDouble(in -> in.hopRef.getOutputMemEstimate(DEFAULT_MEMORY_ESTIMATE))
				.sum() / WORKER_NETWORK_BANDWIDTH_BYTES_PS;
		else return 0;
	}

	/**
	 * Returns input transfer cost estimate.
	 * The input transfer cost estimate is based on the memory estimate of LOUT when some input is FOUT
	 * except if root is a federated DataOp, since all input for this has to be at the coordinator.
	 * When no input is FOUT, the input transfer cost is always 0.
	 * @param hasFederatedInput true if root has any FOUT input
	 * @param root hop for which cost is estimated
	 * @return input transfer cost estimate
	 */
	private double inputTransferCostEstimate(boolean hasFederatedInput, Hop root){
		if ( hasFederatedInput )
			return root.getInput().stream()
				.filter(input -> (root.isFederatedDataOp()) ? input.hasFederatedOutput() : input.hasLocalOutput() )
				.mapToDouble(in -> in.getOutputMemEstimate(DEFAULT_MEMORY_ESTIMATE))
				.sum() / WORKER_NETWORK_BANDWIDTH_BYTES_PS;
		else return 0;
	}

	/**
	 * Prints costs and information about root for debugging purposes
	 * @param root hop for which information is printed
	 */
	private static void printCosts(Hop root){
		System.out.println("===============================");
		System.out.println(root);
		System.out.println("Is federated: " + root.isFederated());
		System.out.println("Has federated output: " + root.hasFederatedOutput());
		System.out.println(root.getText());
		System.out.println("Pure computeCost: " + ComputeCost.getHOPComputeCost(root));
		System.out.println("Dim1: " + root.getDim1() + " Dim2: " + root.getDim2());
		System.out.println(root.getFederatedCost().toString());
		System.out.println("===============================");
	}
}
