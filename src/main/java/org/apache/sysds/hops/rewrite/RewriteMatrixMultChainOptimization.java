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
 
package org.apache.sysds.hops.rewrite;

import java.util.ArrayList;
import java.util.Arrays;

import org.apache.commons.lang3.mutable.MutableInt;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.HopsException;
import org.apache.sysds.runtime.util.CollectionUtils;
import org.apache.sysds.utils.Explain;

/**
 * <strong>Rule</strong>: Determine the optimal order of execution for a chain of
 * matrix multiplications <br>
 * <strong>Solution</strong>: Classic Dynamic Programming <br>
 * <strong>Approach</strong>: Currently, the approach based only on matrix dimensions <br>
 * <strong>Goal</strong>: To reduce the number of computations in the run-time
 * (map-reduce) layer
 */
public class RewriteMatrixMultChainOptimization extends HopRewriteRule
{
	private static final Boolean PUSH_DOWN_TRANSPOSE = true;

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
	 * rule_OptimizeMMChains(): This method goes through all Hops in the DAG
	 * to find chains that need to be optimized.
	 * 
	 * @param hop high-level operator
	 */
	private void rule_OptimizeMMChains(Hop hop, ProgramRewriteStatus state)
	{
		if( !hop.isVisited() ) {

			if (HopRewriteUtils.isMatrixMultiply(hop) && !((AggBinaryOp) hop).hasLeftPMInput()) {
				// Try to find and optimize the chain in which current Hop is the
				// last operator
				prepAndOptimizeMMChain(hop, state);
			}

			for (Hop hi : hop.getInput())
				rule_OptimizeMMChains(hi, state);

			hop.setVisited();
		}
	}

	/**
	 * optimizeMMChain(): It optimizes the matrix multiplication chain in which
	 * the last Hop is "this".
	 * <ul><li>Step 1: Identify the chain (mmChain).</li>
	 * <li>Step 2: Clear all links among the Hops that are involved in mmChain.</li>
	 * <li>Step 3: Find the optimal ordering via dynamic programming.</li>
	 * <li>Step 4: Relink the hops in mmChain.</li></ul>
	 * @param hop high-level operator
	 */
	private void prepAndOptimizeMMChain( Hop hop, ProgramRewriteStatus state )
	{
		if( LOG.isTraceEnabled() ) {
			LOG.trace("MM Chain Optimization for HOP: (" + hop.getClass().getSimpleName()
				+ ", " + hop.getHopID() + ", " + hop.getName() + ")");
		}

		// Step 1: Identify the chain (mmChain) & clear all links among the Hops
		// that are involved in mmChain.

		// Initialize mmChain with current hop's inputs
		ArrayList<Hop> mmOperators = new ArrayList<>();
		mmOperators.add(hop);
		ArrayList<Hop> mmChain = new ArrayList<>(hop.getInput());

		if (PUSH_DOWN_TRANSPOSE) {
			checkChainForTransposeAndRewrite(mmChain, hop);
		}

		int mmChainIndex = 0;

		// Expand each Hop in mmChain to find the entire matrix multiplication chain
		while( mmChainIndex < mmChain.size() )
		{
			boolean expandable = false;

			Hop h = mmChain.get(mmChainIndex);
			/*
			 * Check if mmChain[i] is expandable: 
			 * 1) It must be MATMULT 
			 * 2) It must not have been visited already 
			 *    (one MATMULT should get expanded only in one chain)
			 * 3) Its output should not be used in multiple places
			 *    (either within chain or outside the chain)
			 */
			
			if ( HopRewriteUtils.isMatrixMultiply(h) && !h.isVisited() )
			{
				// check if the output of "h" is used at multiple places. If yes, it can
				// not be expanded.
				expandable = !(h.getParent().size() > 1 || inputCount(h.getParent().get(0), h) > 1);
				if( !expandable )
					break;
			}

			h.setVisited();

			if( !expandable ) {
				mmChainIndex++;
			}
			else {
				ArrayList<Hop> tempList = mmChain.get(mmChainIndex).getInput();
				if( tempList.size() != 2 ) {
					throw new HopsException(hop.printErrorLocation() + "Hops::rule_OptimizeMMChain(): AggBinary must have exactly two inputs.");
				}

				// add current operator to mmOperators, and its input nodes to mmChain
				mmOperators.add(mmChain.get(mmChainIndex));
				mmChain.set(mmChainIndex, tempList.get(0));
				mmChain.add(mmChainIndex + 1, tempList.get(1));
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
		if( mmChain.size() != 2 )
			optimizeMMChain(hop, mmChain, mmOperators, state);
	}
	
	protected void optimizeMMChain(Hop hop, ArrayList<Hop> mmChain, ArrayList<Hop> mmOperators, ProgramRewriteStatus state) {
		// Step 2: construct dims array
		double[] dimsArray = new double[mmChain.size() + 1];
		boolean dimsKnown = getDimsArray( hop, mmChain, dimsArray );
		
		if( dimsKnown ) {
			// Step 3: Clear the links among Hops within the identified chain
			clearLinksWithinChain ( hop, mmOperators );
			
			// Step 4: Find the optimal ordering via dynamic programming.
			
			// Invoke Dynamic Programming
			int size = mmChain.size();
			int[][] split = mmChainDP(dimsArray, mmChain.size());
			
			 // Step 5: Relink the hops using the optimal ordering (split[][]) found from DP.
			LOG.trace("Optimal MM Chain: ");
			mmChainRelinkHops(mmOperators.get(0), 0, size - 1, mmChain, mmOperators, new MutableInt(1), split, 1);
		}
	}
	
	/**
	 * mmChainDP(): Core method to perform dynamic programming on a given array
	 * of matrix dimensions. <br>
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
	 * part of the mmChain.
	 * @param mmChain basic operands in the entire matrix multiplication chain
	 * @param mmOperators Hops that store the intermediate results in the chain.
	 *                      <strong>For example:</strong> A = B %*% (C %*% D) there will be three
	 *                      Hops in mmChain (B,C,D), and two Hops in mmOperators
	 *                     (one for each * %*%).
	 * @param h high level operator
	 * @param i array index i
	 * @param j array index j
	 * @param opIndex operator index
	 * @param split optimal order
	 * @param level log level
	 */
	protected final void mmChainRelinkHops(Hop h, int i, int j, ArrayList<Hop> mmChain,
		ArrayList<Hop> mmOperators, MutableInt opIndex, int[][] split, int level)
	{
		//NOTE: the opIndex is a MutableInt in order to get the correct positions
		//in ragged chains like ((((a, b), c), (D, E), f), e) that might be given
		//like that by the original scripts variable assignments

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
			int ix = opIndex.getValue();
			opIndex.increment();
			h.getInput().add(mmOperators.get(ix));
			mmOperators.get(ix).getParent().add(h);
		}

		// Set Input2 for current Hop h
		if( split[i][j] + 1 == j ) {
			h.getInput().add(mmChain.get(j));
			mmChain.get(j).getParent().add(h);
		} 
		else {
			int ix = opIndex.getValue();
			opIndex.increment();
			h.getInput().add(mmOperators.get(ix));
			mmOperators.get(ix).getParent().add(h);
		}

		// Find children for both the inputs
		mmChainRelinkHops(h.getInput(0), i, split[i][j], mmChain, mmOperators, opIndex, split, level+1);
		mmChainRelinkHops(h.getInput(1), split[i][j] + 1, j, mmChain, mmOperators, opIndex, split, level+1);

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
			if( op.getInput().size() != 2 || (i > 0 && op.getParent().size() > 1 ) ) {
				throw new HopsException(hop.printErrorLocation() + 
					"Unexpected error while applying optimization on matrix-mult chain. \n");
			}
			Hop input1 = op.getInput(0);
			Hop input2 = op.getInput(1);

			op.getInput().clear();
			input1.getParent().remove(op);
			input2.getParent().remove(op);
		}
	}

	/**
	 * Obtains all dimension information of the chain and constructs the dimArray.
	 * If all dimensions are known it returns true; otherwise the mmchain rewrite
	 * should be ended without modifications.
	 * 
	 * @param hop high-level operator
	 * @param chain list of high-level operators
	 * @param dimsArray dimension array
	 * @return true if all dimensions known
	 */
	protected static boolean getDimsArray( Hop hop, ArrayList<Hop> chain, double[] dimsArray )
	{
		boolean dimsKnown = true;
		
		// Build the array containing dimensions from all matrices in the chain		
		// check the dimensions in the matrix chain to insure all dimensions are known
		for (Hop value : chain)
			if (value.getDim1() <= 0 || value.getDim2() <= 0)
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

	/**
	 * Transforms a transpose operator into matrixmult and adjusts
	 * all the respective attributes of the other operators, also creates a second transpose operator.
	 * Thus, we can achieve larger optimization space for the transformed chain.<br>
	 * <strong>Idea:</strong> t(A %*% B) -> t(B) %*% t(A)
	 *
	 * @param transposeHop the transpose operator, which contains all useful data for the transformation
	 * @return the new matrixmult operator
	 */
	private Hop rewriteChainOnTransposeOperator(Hop transposeHop) {
		Hop matrixMultHop = transposeHop.getInput(0);
		Hop firstMatrix = matrixMultHop.getInput(0);
		Hop secondMatrix = matrixMultHop.getInput(1);

		// Clone transpose operator for the overwritten chain
		Hop secondTransposeHop = null;
		try {
			secondTransposeHop = (Hop) transposeHop.clone();
		} catch (CloneNotSupportedException ex) {
			System.err.println("Error on cloning transpose operator: " + ex.getMessage());
		}
		assert secondTransposeHop!= null;

		// Set parent to the other operators accordingly
		updateParentOfHop(firstMatrix, transposeHop);
		updateParentOfHop(secondMatrix, secondTransposeHop);
		updateParentOfHop(transposeHop, matrixMultHop);
		updateParentOfHop(secondTransposeHop, matrixMultHop);

		// Set input to all operators and update attributes accordingly
		ArrayList<Hop> inputList = new ArrayList<>();
		inputList.add(firstMatrix);
		updateAttributesOfHop(transposeHop, inputList, firstMatrix.getName());

		inputList.set(0, secondMatrix);
		updateAttributesOfHop(secondTransposeHop, inputList, secondMatrix.getName());

		inputList.set(0, secondTransposeHop);
		inputList.add(transposeHop);
		updateAttributesOfHop(matrixMultHop, inputList, firstMatrix.getName());

		return matrixMultHop;
	}

	private void checkChainForTransposeAndRewrite(ArrayList<Hop> mmChain, Hop parentOfChain) {
		int mmChainIndex = 0;
		while (mmChainIndex < mmChain.size())
		{
			Hop currentChainHop = mmChain.get(mmChainIndex);

			// Check if current hop is a transpose operator,
			// if it has been visited,
			// and if it has only one input, which is a matrixmult operator
			boolean isTransposeOperator = HopRewriteUtils.isReorg(currentChainHop, Types.ReOrgOp.TRANS);

			if (isTransposeOperator && !currentChainHop.isVisited() && currentChainHop.getInput().size() == 1)
			{
				Hop transposeOperatorChild = currentChainHop.getInput(0);
				if (HopRewriteUtils.isMatrixMultiply(transposeOperatorChild)
					&& hasOnlyTwoReadsAsInput(transposeOperatorChild) && transposeOperatorChild.getParent().size() == 1)
				{
					int indexInParentInput = parentOfChain.getInput().indexOf(currentChainHop);

					// Set transpose operator's parent as new one for matrix multiplication operator
					Hop matrixMultHop = rewriteChainOnTransposeOperator(currentChainHop);
					updateParentOfHop(matrixMultHop, parentOfChain);

					// Update input of transpose operator's parent
					parentOfChain.getInput().set(indexInParentInput, matrixMultHop);

					// Replace transpose operator with the matrixmult one in the mmchain
					mmChain.set(mmChainIndex, matrixMultHop);
				}
			}
			mmChainIndex++;
		}
	}

	private void updateParentOfHop(Hop hopToUpdate, Hop parentToSet) {
		hopToUpdate.getParent().clear();
		hopToUpdate.getParent().add(parentToSet);
	}

	/**
	 * Updates input list, dimensions of matrix and text of a given Hop.
	 *
	 * @param hopToUpdate the hop that will be updated
	 * @param inputList new input list that will be set
	 * @param text new text of the operator
	 */
	private void updateAttributesOfHop(Hop hopToUpdate, ArrayList<Hop> inputList, String text) {
		hopToUpdate.getInput().clear();

		for (Hop input : inputList) {
			hopToUpdate.getInput().add(input);
		}

		if (HopRewriteUtils.isMatrixMultiply(hopToUpdate)) {
			// Here we add dimensions of a matrixmult operator
			hopToUpdate.setDim1(inputList.get(0).getDim1());
			hopToUpdate.setDim2(inputList.get(1).getDim2());
		} else {
			// Here we add dimensions of a transpose operator
			hopToUpdate.setDim1(inputList.get(0).getDim2());
			hopToUpdate.setDim2(inputList.get(0).getDim1());
		}

		//hopToUpdate.setText(String.format("t(%s)", text));
	}

	private boolean hasOnlyTwoReadsAsInput(Hop transposeOperatorChild) {
		if (transposeOperatorChild.getInput().size() == 2) {
			for(Hop hop: transposeOperatorChild.getInput()) {
				if (!HopRewriteUtils.isData(hop, Types.OpOpData.TRANSIENTREAD, Types.OpOpData.PERSISTENTREAD))
					return false;
			}
			return true;
		}
		return false;
	}
}
