/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;
import java.util.Arrays;

import com.ibm.bi.dml.hops.AggBinaryOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;

/**
 * Rule: Determine the optimal order of execution for a chain of
 * matrix multiplications Solution: Classic Dynamic Programming
 * Approach Currently, the approach based only on matrix dimensions
 * Goal: To reduce the number of computations in the run-time
 * (map-reduce) layer
 */
public class RewriteMatrixMultChainOptimization extends HopRewriteRule
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots) 
		throws HopsException
	{
		if( roots == null )
			return null;

		for( Hop h : roots ) 
		{
			// Find the optimal order for the chain whose result is the current HOP
			rule_OptimizeMMChains(h);
		}		
		
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root)
		throws HopsException
	{
		if( root == null )
			return null;

		// Find the optimal order for the chain whose result is the current HOP
		rule_OptimizeMMChains(root);
		
		return root;
	}
	
	/**
	 * rule_OptimizeMMChains(): This method recurses through all Hops in the DAG
	 * to find chains that need to be optimized.
	 */
	private void rule_OptimizeMMChains(Hop hop) 
		throws HopsException 
	{
		if(hop.get_visited() == Hop.VISIT_STATUS.DONE)
				return;
		
		if (hop.getKind() == Hop.Kind.AggBinaryOp && ((AggBinaryOp) hop).isMatrixMultiply()
				&& hop.get_visited() != Hop.VISIT_STATUS.DONE) {
			// Try to find and optimize the chain in which current Hop is the
			// last operator
			optimizeMMChain(hop);
		}
		
		for (Hop hi : hop.getInput())
			rule_OptimizeMMChains(hi);

		hop.set_visited(Hop.VISIT_STATUS.DONE);
	}

	/**
	 * mmChainDP(): Core method to perform dynamic programming on a given array
	 * of matrix dimensions.
	 * 
	 * Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein
	 * Introduction to Algorithms, Third Edition, MIT Press, page 395.
	 */
	private int[][] mmChainDP(double dimArray[], int size) 
	{
		double dpMatrix[][] = new double[size][size]; //min cost table
		int split[][] = new int[size][size]; //min cost index table

		//init minimum costs for chains of length 1
		for (int i = 0; i < size; i++) {
			Arrays.fill(dpMatrix[i], 0);
			Arrays.fill(split[i], -1);
		}

		//compute cost-optimal chains for increasing chain sizes 
		for (int l = 2; l <= size; l++) { // chain length
			for (int i = 0; i < size - l + 1; i++) {
				int j = i + l - 1;
				// find cost of (i,j)
				dpMatrix[i][j] = Double.MAX_VALUE;
				for (int k = i; k <= j - 1; k++) 
				{
					//recursive cost computation
					double cost = dpMatrix[i][k] + dpMatrix[k + 1][j] 
							  + (dimArray[i] * dimArray[k + 1] * dimArray[j + 1]);
					
					//prune suboptimal
					if (cost < dpMatrix[i][j]) {
						dpMatrix[i][j] = cost;
						split[i][j] = k;
					}
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
	private void mmChainRelinkHops(Hop h, int i, int j, ArrayList<Hop> mmChain, ArrayList<Hop> mmOperators,
			int opIndex, int[][] split) 
	{
		if (i == j)
			return;

		Hop input1, input2;
		// Set Input1 for current Hop h
		if (i == split[i][j]) {
			input1 = mmChain.get(i);
			h.getInput().add(mmChain.get(i));
			mmChain.get(i).getParent().add(h);
		} else {
			input1 = mmOperators.get(opIndex);
			h.getInput().add(mmOperators.get(opIndex));
			mmOperators.get(opIndex).getParent().add(h);
			opIndex = opIndex + 1;
		}

		// Set Input2 for current Hop h
		if (split[i][j] + 1 == j) {
			input2 = mmChain.get(j);
			h.getInput().add(mmChain.get(j));
			mmChain.get(j).getParent().add(h);
		} else {
			input2 = mmOperators.get(opIndex);
			h.getInput().add(mmOperators.get(opIndex));
			mmOperators.get(opIndex).getParent().add(h);
			opIndex = opIndex + 1;
		}

		// Find children for both the inputs
		mmChainRelinkHops(h.getInput().get(0), i, split[i][j], mmChain, mmOperators, opIndex, split);
		mmChainRelinkHops(h.getInput().get(1), split[i][j] + 1, j, mmChain, mmOperators, opIndex, split);

		// Propagate properties of input hops to current hop h
		h.set_dim1(input1.get_dim1());
		h.set_rows_in_block(input1.get_rows_in_block());
		h.set_dim2(input2.get_dim2());
		h.set_cols_in_block(input2.get_cols_in_block());

	}

	/**
	 * 
	 * @param operators
	 * @throws HopsException
	 */
	private void clearLinksWithinChain ( Hop hop, ArrayList<Hop> operators ) 
		throws HopsException 
	{
		Hop op, input1, input2;
		
		for ( int i=0; i < operators.size(); i++ ) {
			op = operators.get(i);
			if ( op.getInput().size() != 2 || (i != 0 && op.getParent().size() > 1 ) ) {
				throw new HopsException(hop.printErrorLocation() + "Unexpected error while applying optimization on matrix-mult chain. \n");
			}
			input1 = op.getInput().get(0);
			input2 = op.getInput().get(1);
			
			op.getInput().clear();
			input1.getParent().remove(op);
			input2.getParent().remove(op);
		}
	}

	/**
	 * 
	 * @param chain
	 * @return
	 * @throws HopsException
	 */
	private double [] getDimArray ( Hop hop, ArrayList<Hop> chain ) 
		throws HopsException 
	{
		// Build the array containing dimensions from all matrices in the
		// chain
		
		double dimArray[] = new double[chain.size() + 1];
		
		// check the dimensions in the matrix chain to insure all dimensions are known
		boolean shortCircuit = false;
		for (int i=0; i< chain.size(); i++){
			if (chain.get(i).get_dim1() <= 0 || chain.get(i).get_dim2() <= 0)
				shortCircuit = true;
		}
		if (shortCircuit){
			for (int i=0; i< dimArray.length; i++){
				dimArray[i] = 1;
			}	
			LOG.trace("short-circuit optimizeMMChain() for matrices with unknown size");
			return dimArray;
		}
		
		
		
		for (int i = 0; i < chain.size(); i++) {
			if (i == 0) {
				dimArray[i] = chain.get(i).get_dim1();
				if (dimArray[i] <= 0) {
					throw new HopsException(hop.printErrorLocation() + 
							"Hops::optimizeMMChain() : Invalid Matrix Dimension: "
									+ dimArray[i]);
				}
			} else {
				if (chain.get(i - 1).get_dim2() != chain.get(i)
						.get_dim1()) {
					throw new HopsException(hop.printErrorLocation() +
							"Hops::optimizeMMChain() : Matrix Dimension Mismatch");
				}
			}
			dimArray[i + 1] = chain.get(i).get_dim2();
			if (dimArray[i + 1] <= 0) {
				throw new HopsException(hop.printErrorLocation() + 
						"Hops::optimizeMMChain() : Invalid Matrix Dimension: "
								+ dimArray[i + 1]);
			}
		}

		return dimArray;
	}

	
	/**
	 * 
	 * @param p
	 * @param h
	 * @return
	 */
	private int inputCount ( Hop p, Hop h ) {
		int count = 0;
		for ( int i=0; i < p.getInput().size(); i++ )
			if ( p.getInput().get(i).equals(h) )
				count++;
		return count;
	}
	
	/**
	 * optimizeMMChain(): It optimizes the matrix multiplication chain in which
	 * the last Hop is "this". Step-1) Identify the chain (mmChain). (Step-2) clear all
	 * links among the Hops that are involved in mmChain. (Step-3) Find the
	 * optimal ordering (dynamic programming) (Step-4) Relink the hops in
	 * mmChain.
	 */
	private void optimizeMMChain( Hop hop ) throws HopsException 
	{
		LOG.trace("MM Chain Optimization for HOP: (" + " " + hop.getKind() + ", " + hop.getHopID() + ", "
					+ hop.get_name() + ")");
		
		ArrayList<Hop> mmChain = new ArrayList<Hop>();
		ArrayList<Hop> mmOperators = new ArrayList<Hop>();
		ArrayList<Hop> tempList;

		/*
		 * Step-1: Identify the chain (mmChain) & clear all links among the Hops
		 * that are involved in mmChain.
		 */

		mmOperators.add(hop);
		// Initialize mmChain with my inputs
		for (Hop hi : hop.getInput()) {
			mmChain.add(hi);
		}

		// expand each Hop in mmChain to find the entire matrix multiplication
		// chain
		int i = 0;
		while (i < mmChain.size()) {

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

			if (h.getKind() == Hop.Kind.AggBinaryOp && ((AggBinaryOp) h).isMatrixMultiply()
					&& h.get_visited() != Hop.VISIT_STATUS.DONE) {
				// check if the output of "h" is used at multiple places. If yes, it can
				// not be expanded.
				if (h.getParent().size() > 1 || inputCount( (Hop) ((h.getParent().toArray())[0]), h) > 1 ) {
					expandable = false;
					break;
				}
				else 
					expandable = true;
			}

			h.set_visited(Hop.VISIT_STATUS.DONE);

			if ( !expandable ) {
				i = i + 1;
			} else {
				tempList = mmChain.get(i).getInput();
				if (tempList.size() != 2) {
					throw new HopsException(hop.printErrorLocation() + "Hops::rule_OptimizeMMChain(): AggBinary must have exactly two inputs.");
				}

				// add current operator to mmOperators, and its input nodes to mmChain
				mmOperators.add(mmChain.get(i));
				mmChain.set(i, tempList.get(0));
				mmChain.add(i + 1, tempList.get(1));
			}
		}

		// print the MMChain
		if (LOG.isTraceEnabled()) {
			LOG.trace("Identified MM Chain: ");
			//System.out.print("MMChain_" + getHopID() + " (" + mmChain.size() + "): ");
			for (Hop h : mmChain) {
				LOG.trace("Hop " + h.get_name() + "(" + h.getKind() + ", " + h.getHopID() + ")" + " "
						+ h.get_dim1() + "x" + h.get_dim2());
				//System.out.print("[" + h.get_name() + "(" + h.getKind() + ", " + h.getHopID() + ")" + " " + h.get_dim1() + "x" + h.get_dim2() + "]  ");
			}
			//System.out.println("");
			LOG.trace("--End of MM Chain--");
		}

		if (mmChain.size() == 2) {
			// If the chain size is 2, then there is nothing to optimize.
			return;
		} 
		else {
			 // Step-2: clear the links among Hops within the identified chain
			clearLinksWithinChain ( hop, mmOperators );
			
			 // Step-3: Find the optimal ordering via dynamic programming.
			
			double dimArray[] = getDimArray( hop, mmChain );
			
			// Invoke Dynamic Programming
			int size = mmChain.size();
			int[][] split = new int[size][size];
			split = mmChainDP(dimArray, mmChain.size());
			
			 // Step-4: Relink the hops using the optimal ordering (split[][]) found from DP.
			mmChainRelinkHops(mmOperators.get(0), 0, size - 1, mmChain, mmOperators, 1, split);
		}
		//System.out.println("  .");
	}
}
