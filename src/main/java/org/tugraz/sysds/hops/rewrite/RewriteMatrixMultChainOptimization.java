/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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
 
package org.tugraz.sysds.hops.rewrite;

import java.util.ArrayList;
import java.util.Arrays;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import org.tugraz.sysds.hops.AggBinaryOp;
import org.tugraz.sysds.hops.Hop;
import org.tugraz.sysds.hops.HopsException;
import org.tugraz.sysds.utils.Explain;

/**
 * Rule: Determine the optimal order of execution for a chain of
 * matrix multiplications 
 * 
 * Solution: Classic Dynamic Programming
 * Approach: Currently, the approach based only on matrix dimensions
 * Goal: To reduce the number of computations in the run-time
 * (map-reduce) layer
 */
public class RewriteMatrixMultChainOptimization extends HopRewriteRule
{
	protected static final Log LOG = LogFactory.getLog(RewriteMatrixMultChainOptimization.class.getName());
	private static final boolean LDEBUG = false;
	
	static {
		// for internal debugging only
		if( LDEBUG ) {
			Logger.getLogger("org.tugraz.sysds.hops.rewrite.RewriteMatrixMultChainOptimization")
				  .setLevel((Level) Level.TRACE);
		}
	}
	
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) 
	{
		if( roots == null )
			return null;

		// Find the optimal order for the chain whose result is the current HOP
		for( Hop h : roots ) 
			rule_OptimizeMMChains(h, state);
		
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state)
	{
		if( root == null )
			return null;

		// Find the optimal order for the chain whose result is the current HOP
		rule_OptimizeMMChains(root, state);
		
		return root;
	}
	
	/**
	 * rule_OptimizeMMChains(): This method recurses through all Hops in the DAG
	 * to find chains that need to be optimized.
	 * 
	 * @param hop high-level operator
	 */
	private void rule_OptimizeMMChains(Hop hop, ProgramRewriteStatus state) 
	{
		if( hop.isVisited() )
			return;
		
		if( HopRewriteUtils.isMatrixMultiply(hop)
			&& !((AggBinaryOp)hop).hasLeftPMInput() && !hop.isVisited() ) 
		{
			// Try to find and optimize the chain in which current Hop is the
			// last operator
			prepAndOptimizeMMChain(hop, state);
		}
		
		for( Hop hi : hop.getInput() )
			rule_OptimizeMMChains(hi, state);

		hop.setVisited();
	}

	
	/**
	 * optimizeMMChain(): It optimizes the matrix multiplication chain in which
	 * the last Hop is "this". Step-1) Identify the chain (mmChain). (Step-2) clear all
	 * links among the Hops that are involved in mmChain. (Step-3) Find the
	 * optimal ordering (dynamic programming) (Step-4) Relink the hops in
	 * mmChain.
	 * 
	 * @param hop high-level operator
	 */
	private void prepAndOptimizeMMChain( Hop hop, ProgramRewriteStatus state )
	{
		if( LOG.isTraceEnabled() ) {
			LOG.trace("MM Chain Optimization for HOP: (" + hop.getClass().getSimpleName()
				+ ", " + hop.getHopID() + ", " + hop.getName() + ")");
		}
		
		ArrayList<Hop> mmChain = new ArrayList<>();
		ArrayList<Hop> mmOperators = new ArrayList<>();
		ArrayList<Hop> tempList;

		// Step 1: Identify the chain (mmChain) & clear all links among the Hops
		// that are involved in mmChain.

		// Initialize mmChain with my inputs
		mmOperators.add(hop);
		for( Hop hi : hop.getInput() )
			mmChain.add(hi);

		// expand each Hop in mmChain to find the entire matrix multiplication
		// chain
		int i = 0;
		while( i < mmChain.size() )
		{
			boolean expandable = false;

			Hop h = mmChain.get(i);
			/*
			 * Check if mmChain[i] is expandable: 
			 * 1) It must be MATMULT 
			 * 2) It must not have been visited already 
			 *    (one MATMULT should get expanded only in one chain)
			 * 3) Its output should not be used in multiple places
			 *    (either within chain or outside the chain)
			 */
			
			if ( HopRewriteUtils.isMatrixMultiply(h)
				&& !((AggBinaryOp)hop).hasLeftPMInput() && !h.isVisited() ) 
			{
				// check if the output of "h" is used at multiple places. If yes, it can
				// not be expanded.
				expandable = !(h.getParent().size() > 1 
					|| inputCount(h.getParent().get(0), h) > 1);
				if( !expandable )
					break;
			}

			h.setVisited();

			if( !expandable ) {
				i = i + 1;
			}
			else {
				tempList = mmChain.get(i).getInput();
				if( tempList.size() != 2 ) {
					throw new HopsException(hop.printErrorLocation() + "Hops::rule_OptimizeMMChain(): AggBinary must have exactly two inputs.");
				}

				// add current operator to mmOperators, and its input nodes to mmChain
				mmOperators.add(mmChain.get(i));
				mmChain.set(i, tempList.get(0));
				mmChain.add(i + 1, tempList.get(1));
			}
		}

		// print the MMChain
		if( LOG.isTraceEnabled() ) {
			LOG.trace("Identified MM Chain: ");
			for( Hop h : mmChain ) {
				logTraceHop(h, 1);
			}
		}

		//core mmchain optimization (potentially overridden)
		if( mmChain.size() == 2 ) 
			return; //nothing to optimize
		else
			optimizeMMChain(hop, mmChain, mmOperators, state);
	}
	
	protected void optimizeMMChain(Hop hop, ArrayList<Hop> mmChain, ArrayList<Hop> mmOperators, ProgramRewriteStatus state) {
		// Step 2: construct dims array
		double[] dimsArray = new double[mmChain.size() + 1];
		boolean dimsKnown = getDimsArray( hop, mmChain, dimsArray );
		
		if( dimsKnown ) {
			// Step 3: clear the links among Hops within the identified chain
			clearLinksWithinChain ( hop, mmOperators );
			
			// Step 4: Find the optimal ordering via dynamic programming.
			
			// Invoke Dynamic Programming
			int size = mmChain.size();
			int[][] split = mmChainDP(dimsArray, mmChain.size());
			
			 // Step 5: Relink the hops using the optimal ordering (split[][]) found from DP.
			LOG.trace("Optimal MM Chain: ");
			mmChainRelinkHops(mmOperators.get(0), 0, size - 1, mmChain, mmOperators, 1, split, 1);
		}
	}
	
	/**
	 * mmChainDP(): Core method to perform dynamic programming on a given array
	 * of matrix dimensions.
	 * 
	 * Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein
	 * Introduction to Algorithms, Third Edition, MIT Press, page 395.
	 */
	private static int[][] mmChainDP(double[] dimArray, int size) 
	{
		double[][] dpMatrix = new double[size][size]; //min cost table
		int[][] split = new int[size][size]; //min cost index table

		//init minimum costs for chains of length 1
		for( int i = 0; i < size; i++ ) {
			Arrays.fill(dpMatrix[i], 0);
			Arrays.fill(split[i], -1);
		}

		//compute cost-optimal chains for increasing chain sizes 
		for( int l = 2; l <= size; l++ ) { // chain length
			for( int i = 0; i < size - l + 1; i++ ) {
				int j = i + l - 1;
				// find cost of (i,j)
				dpMatrix[i][j] = Double.MAX_VALUE;
				for( int k = i; k <= j - 1; k++ ) 
				{
					//recursive cost computation
					double cost = dpMatrix[i][k] + dpMatrix[k + 1][j] 
						+ (dimArray[i] * dimArray[k + 1] * dimArray[j + 1]);
					
					//prune suboptimal
					if( cost < dpMatrix[i][j] ) {
						dpMatrix[i][j] = cost;
						split[i][j] = k;
					}
				}

				if( LOG.isTraceEnabled() ){
					LOG.trace("mmchainopt [i="+(i+1)+",j="+(j+1)+"]: costs = "+dpMatrix[i][j]+", split = "+(split[i][j]+1));
				}
			}
		}

		return split;
	}

	/**
	 * mmChainRelinkHops(): This method gets invoked after finding the optimal
	 * order (split[][]) from dynamic programming. It relinks the Hops that are
	 * part of the mmChain. mmChain : basic operands in the entire matrix
	 * multiplication chain. mmOperators : Hops that store the intermediate
	 * results in the chain. For example: A = B %*% (C %*% D) there will be
	 * three Hops in mmChain (B,C,D), and two Hops in mmOperators (one for each
	 * %*%) .
	 */
	protected final void mmChainRelinkHops(Hop h, int i, int j, ArrayList<Hop> mmChain, ArrayList<Hop> mmOperators,
			int opIndex, int[][] split, int level) 
	{
		//single matrix - end of recursion
		if( i == j ) {
			logTraceHop(h, level);
			return;
		}

		if( LOG.isTraceEnabled() ){
			String offset = Explain.getIdentation(level);
			LOG.trace(offset + "(");
		}
		
		// Set Input1 for current Hop h
		if( i == split[i][j] ) {
			h.getInput().add(mmChain.get(i));
			mmChain.get(i).getParent().add(h);
		}
		else {
			h.getInput().add(mmOperators.get(opIndex));
			mmOperators.get(opIndex).getParent().add(h);
			opIndex = opIndex + 1;
		}

		// Set Input2 for current Hop h
		if( split[i][j] + 1 == j ) {
			h.getInput().add(mmChain.get(j));
			mmChain.get(j).getParent().add(h);
		} 
		else {
			h.getInput().add(mmOperators.get(opIndex));
			mmOperators.get(opIndex).getParent().add(h);
			opIndex = opIndex + 1;
		}

		// Find children for both the inputs
		mmChainRelinkHops(h.getInput().get(0), i, split[i][j], mmChain, mmOperators, opIndex, split, level+1);
		mmChainRelinkHops(h.getInput().get(1), split[i][j] + 1, j, mmChain, mmOperators, opIndex, split, level+1);

		// Propagate properties of input hops to current hop h
		h.refreshSizeInformation();
		
		if( LOG.isTraceEnabled() ){
			String offset = Explain.getIdentation(level);
			LOG.trace(offset + ")");
		}
	}

	protected static void clearLinksWithinChain( Hop hop, ArrayList<Hop> operators ) 
	{
		for( int i=0; i < operators.size(); i++ ) {
			Hop op = operators.get(i);
			if( op.getInput().size() != 2 || (i != 0 && op.getParent().size() > 1 ) ) {
				throw new HopsException(hop.printErrorLocation() + 
					"Unexpected error while applying optimization on matrix-mult chain. \n");
			}
			Hop input1 = op.getInput().get(0);
			Hop input2 = op.getInput().get(1);
			
			op.getInput().clear();
			input1.getParent().remove(op);
			input2.getParent().remove(op);
		}
	}

	/**
	 * Obtains all dimension information of the chain and constructs the dimArray.
	 * If all dimensions are known it returns true; othrewise the mmchain rewrite
	 * should be ended without modifications.
	 * 
	 * @param hop high-level operator
	 * @param chain list of high-level operators
	 * @param dimArray dimension array
	 * @return true if all dimensions known
	 */
	protected static boolean getDimsArray( Hop hop, ArrayList<Hop> chain, double[] dimsArray )
	{
		boolean dimsKnown = true;
		
		// Build the array containing dimensions from all matrices in the chain		
		// check the dimensions in the matrix chain to insure all dimensions are known
		for( int i=0; i< chain.size(); i++ )
			if( chain.get(i).getDim1() <= 0 || chain.get(i).getDim2() <= 0 )
				dimsKnown = false;
		
		if( dimsKnown ) { //populate dims array if all dims known
			for( int i = 0; i < chain.size(); i++ ) {
				if (i == 0) {
					dimsArray[i] = chain.get(i).getDim1();
					if (dimsArray[i] <= 0) {
						throw new HopsException(hop.printErrorLocation() + 
								"Hops::optimizeMMChain() : Invalid Matrix Dimension: "+ dimsArray[i]);
					}
				}
				else if (chain.get(i - 1).getDim2() != chain.get(i).getDim1()) {
					throw new HopsException(hop.printErrorLocation() +
						"Hops::optimizeMMChain() : Matrix Dimension Mismatch: " + 
						chain.get(i - 1).getDim2()+" != "+chain.get(i).getDim1());
				}
				
				dimsArray[i + 1] = chain.get(i).getDim2();
				if( dimsArray[i + 1] <= 0 ) {
					throw new HopsException(hop.printErrorLocation() + 
							"Hops::optimizeMMChain() : Invalid Matrix Dimension: " + dimsArray[i + 1]);
				}
			}
		}
		
		return dimsKnown;
	}

	private static int inputCount( Hop p, Hop h ) {
		return CollectionUtils.cardinality(h, p.getInput());
	}
	
	private static void logTraceHop( Hop hop, int level ) {
		if( LOG.isTraceEnabled() ) {
			String offset = Explain.getIdentation(level);
			LOG.trace(offset+ "Hop " + hop.getName() + "(" + hop.getClass().getSimpleName() 
				+ ", " + hop.getHopID() + ")" + " " + hop.getDim1() + "x" + hop.getDim2());
		}
	}
}
