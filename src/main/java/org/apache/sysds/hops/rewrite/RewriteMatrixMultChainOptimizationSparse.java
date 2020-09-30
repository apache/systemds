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
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.HopsException;
import org.apache.sysds.hops.estim.MMNode;
import org.apache.sysds.hops.estim.EstimatorMatrixHistogram;
import org.apache.sysds.hops.estim.EstimatorMatrixHistogram.MatrixHistogram;
import org.apache.sysds.hops.estim.SparsityEstimator.OpCode;
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Rule: Determine the optimal order of execution for a chain of
 * matrix multiplications 
 * 
 * Solution: Classic Dynamic Programming
 * Approach: Currently, the approach based only on matrix dimensions
 * and sparsity estimates using the MNC sketch
 * Goal: To reduce the number of computations in the run-time
 * (map-reduce) layer
 */
public class RewriteMatrixMultChainOptimizationSparse extends RewriteMatrixMultChainOptimization
{
	@Override
	protected void optimizeMMChain(Hop hop, ArrayList<Hop> mmChain, ArrayList<Hop> mmOperators, ProgramRewriteStatus state) {
		// Step 2: construct dims array and input matrices
		double[] dimsArray = new double[mmChain.size() + 1];
		boolean dimsKnown = getDimsArray( hop, mmChain, dimsArray );
		MMNode[] sketchArray = new MMNode[mmChain.size() + 1];
		boolean inputsAvail = getInputMatrices(hop, mmChain, sketchArray, state);
		
		if( dimsKnown && inputsAvail ) {
			// Step 3: clear the links among Hops within the identified chain
			clearLinksWithinChain ( hop, mmOperators );
			
			// Step 4: Find the optimal ordering via dynamic programming.
			
			// Invoke Dynamic Programming
			int size = mmChain.size();
			int[][] split = mmChainDPSparse(dimsArray, sketchArray, mmChain.size());
			
			 // Step 5: Relink the hops using the optimal ordering (split[][]) found from DP.
			LOG.trace("Optimal MM Chain: ");
			mmChainRelinkHops(mmOperators.get(0), 0, size - 1, mmChain, mmOperators, new MutableInt(1), split, 1);
		}
	}
	
	/**
	 * mmChainDP(): Core method to perform dynamic programming on a given array
	 * of matrix dimensions.
	 * 
	 * Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein
	 * Introduction to Algorithms, Third Edition, MIT Press, page 395.
	 */
	private static int[][] mmChainDPSparse(double[] dimArray, MMNode[] sketchArray, int size) 
	{
		double[][] dpMatrix = new double[size][size]; //min cost table
		MMNode[][] dpMatrixS = new MMNode[size][size]; //min sketch table
		int[][] split = new int[size][size]; //min cost index table

		//init minimum costs for chains of length 1
		for( int i = 0; i < size; i++ ) {
			Arrays.fill(dpMatrix[i], 0);
			Arrays.fill(split[i], -1);
			dpMatrixS[i][i] = sketchArray[i];
		}

		//compute cost-optimal chains for increasing chain sizes 
		EstimatorMatrixHistogram estim = new EstimatorMatrixHistogram(true);
		for( int l = 2; l <= size; l++ ) { // chain length
			for( int i = 0; i < size - l + 1; i++ ) {
				int j = i + l - 1;
				// find cost of (i,j)
				dpMatrix[i][j] = Double.MAX_VALUE;
				for( int k = i; k <= j - 1; k++ ) 
				{
					//construct estimation nodes (w/ lazy propagation and memoization)
					MMNode tmp = new MMNode(dpMatrixS[i][k], dpMatrixS[k+1][j], OpCode.MM);
					estim.estim(tmp, false);
					MatrixHistogram lhs = (MatrixHistogram) dpMatrixS[i][k].getSynopsis();
					MatrixHistogram rhs = (MatrixHistogram) dpMatrixS[k+1][j].getSynopsis();
					
					//recursive cost computation
					double cost = dpMatrix[i][k] + dpMatrix[k + 1][j] 
						+ dotProduct(lhs.getColCounts(), rhs.getRowCounts());
					
					//prune suboptimal
					if( cost < dpMatrix[i][j] ) {
						dpMatrix[i][j] = cost;
						dpMatrixS[i][j] = tmp;
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
	
	private static boolean getInputMatrices(Hop hop, ArrayList<Hop> chain, MMNode[] sketchArray, ProgramRewriteStatus state) {
		boolean inputsAvail = true;
		LocalVariableMap vars = state.getVariables();
		
		for( int i=0; i<chain.size(); i++ ) {
			inputsAvail &= HopRewriteUtils.isData(chain.get(0), OpOpData.TRANSIENTREAD);
			if( inputsAvail )
				sketchArray[i] = new MMNode(getMatrix(chain.get(i).getName(), vars));
			else 
				break;
		}
		
		return inputsAvail;
	}
	
	private static MatrixBlock getMatrix(String name, LocalVariableMap vars) {
		Data dat = vars.get(name);
		if( !(dat instanceof MatrixObject) )
			throw new HopsException("Input '"+name+"' not a matrix: "+dat.getDataType());
		return ((MatrixObject)dat).acquireReadAndRelease();
	}
	
	private static double dotProduct(int[] h1cNnz, int[] h2rNnz) {
		long fp = 0;
		for( int j=0; j<h1cNnz.length; j++ )
			fp += (long)h1cNnz[j] * h2rNnz[j];
		return fp;
	}
}
