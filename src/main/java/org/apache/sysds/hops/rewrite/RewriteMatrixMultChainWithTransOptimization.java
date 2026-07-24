/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
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
import java.util.List;

import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.HopsException;

/**
 * Advanced Matrix Multiplication Chain Optimizer using Dynamic Programming.
 * <p>
 * This rewrite optimizes matrix multiplication chains by simultaneously exploring
 * standard parenthesization and the transpose property: (A %*% B)^T = B^T %*% A^T.
 * It uses a DP algorithm to find the execution plan with the minimal
 * computational cost (FLOPs), inserting physical transposes only when mathematically cheaper.
 * In comparison to RewriteMatrixMultChainOptimization.java this builds complete new HOP DAG and returns it
 */
public class RewriteMatrixMultChainWithTransOptimization extends HopRewriteRule {

	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state)
	{
		if( roots == null )
			return null;

		// Find the optimal order for the chain whose result is the current HOP
		for( Hop h : roots )
			ruleOptimizeMMChains(h, state);

		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state)
	{
		if( root == null )
			return null;

		// Find the optimal order for the chain whose result is the current HOP
		ruleOptimizeMMChains(root, state);

		return root;
	}

	/**
	 * Recursively traverses the HOP DAG to identify matrix multiplication chains.
	 * Looks for either direct AggBinaryOps (%*%) or TransposeOps (t()) wrapping an AggBinaryOp.
	 *
	 * @param hop The current high-level operator node.
	 * @param state The rewrite status.
	 */
	private void ruleOptimizeMMChains(Hop hop, ProgramRewriteStatus state) {
		if (hop.isVisited()) return;

		boolean isMatrixMult = HopRewriteUtils.isMatrixMultiply(hop) && !((AggBinaryOp) hop).hasLeftPMInput();

		boolean isTranspose = HopRewriteUtils.isTransposeOperation(hop) // hop is a t() operator.
			&& HopRewriteUtils.isMatrixMultiply(hop.getInput().get(0))  // HOP's only child is (..) %*% (...)
			&& !((AggBinaryOp) hop.getInput().get(0)).hasLeftPMInput();

		Hop currentHop = hop;

		if (isMatrixMult || isTranspose) {
			// Try to find and optimize the chain in which current Hop is the
			// last operator
			currentHop = prepAndOptimizeMMChain(hop, state);
		}

		currentHop.setVisited();

		// .toArray(new Hop[0]) this prevents ConcurrentModificationException because the optimizer
		// may replace or modify parts of the HOP DAG during recursion
		for( Hop i : currentHop.getInput().toArray(new Hop[0]) ) {
			ruleOptimizeMMChains(i, state);
		}
	}


	private Hop prepAndOptimizeMMChain(Hop hop, ProgramRewriteStatus state) {
		if( LOG.isTraceEnabled() ) {
			LOG.trace("MM Chain Optimization for HOP: (" + hop.getClass().getSimpleName()
				+ ", " + hop.getHopID() + ", " + hop.getName() + ")");
		}

		List<Hop> mmChain = new ArrayList<>();
		List<Boolean> isTransposedChain = new ArrayList<>();

		boolean isRootTranspose = HopRewriteUtils.isTransposeOperation(hop);

		// if top node is a transpose, then we look at children nodes, eitherweise not
		Hop currentRoot = isRootTranspose ? hop.getInput().get(0) : hop;

		if( isRootTranspose ) {
			// if e.g: t(A %*% B) then we store it like t(B) %*% t(A) in other order
			mmChain.add(currentRoot.getInput().get(1));
			mmChain.add(currentRoot.getInput().get(0));
		}
		else {
			// no transpose: store in normal order matrices that are multiplied
			mmChain.add(currentRoot.getInput().get(0));
			mmChain.add(currentRoot.getInput().get(1));
		}

		// store, wether matrices need to be transposed
		isTransposedChain.add(isRootTranspose);
		isTransposedChain.add(isRootTranspose);

		int i = 0;
		while (i < mmChain.size()) {
			Hop currentHop = mmChain.get(i);
			boolean currentIsTransposed = isTransposedChain.get(i);

			Hop matrixMultHop = currentHop;

			// does current HOP contain a transpose underneath?
			boolean hasTranspose = HopRewriteUtils.isTransposeOperation(currentHop);

			// if yes, take the child node as matrixMultHop
			matrixMultHop = hasTranspose ? currentHop.getInput().get(0) : currentHop;

			// default assumption: cannot expand this node
			boolean expandable = false;

			// only try to expand if standard matrix multiply
			if (HopRewriteUtils.isMatrixMultiply(matrixMultHop)) {

				// how many other HOPs are using matrixMultHop as input
				// excluding current position in our flattening process and original root
				long externalParents = matrixMultHop.getParent().stream()
					.filter(p -> (p != currentHop) && (p != currentRoot) && !p.isVisited())
					.count();

				// if current node is wrapped in t(..) also need to check parent nodes
				// of transpose node itself
				if (hasTranspose) {
					externalParents += currentHop.getParent().stream()
						.filter(p -> p != currentRoot && !p.isVisited())
						.count();
				}

				expandable = (externalParents == 0);
			}

			// Decision
			// 1. Not expandable
			if (!expandable) {
				mmChain.set(i, currentHop);
				isTransposedChain.set(i, currentIsTransposed);
				i++;
			}
			else {
				// 2. node is expandable
				matrixMultHop.setVisited();
				if (hasTranspose) {
					currentHop.setVisited();
					currentIsTransposed = !currentIsTransposed;
				}

				List<Hop> children = matrixMultHop.getInput();

				if (currentIsTransposed) {
					mmChain.set(i, children.get(1));
					mmChain.add(i+1, children.get(0));
				}
				else {
					mmChain.set(i, children.get(0));
					mmChain.add(i+1, children.get(1));
				}

				isTransposedChain.set(i, currentIsTransposed);
				isTransposedChain.add(i+1, currentIsTransposed);
			}
		}
		// only invoke if chain longer than 2 matrices
		if (mmChain.size() > 2) {
			return optimizeMMChain(hop, mmChain, isTransposedChain, isRootTranspose);
		}
		return hop;
	}

	protected Hop optimizeMMChain(Hop hop, List<Hop> mmChain, List<Boolean> isTransposedChain, boolean isRootTranspose) {
		// Step 2: construct dims array
		double[] dimsArray = new double[mmChain.size() + 1];
		boolean dimsKnown = getDimsArray( hop, mmChain, isTransposedChain, dimsArray );

		if (dimsKnown) {
			// Find the optimal ordering via dynamic programming.
			// Step 3: Invoke Dynamic Programming
			MemoTable memo = mmChainDP(dimsArray, mmChain.size(), mmChain, isTransposedChain);

			LOG.trace("Optimal MM Chain: ");
			// Step 4: read optimal ordering and construct new tree from that
			Hop newRoot = mmChainBuildTree(0, mmChain.size() - 1, mmChain, memo, isRootTranspose, hop);

			// swap pointers to new tree if new tree was built
			if (newRoot != hop) {
				List<Hop> parents = new ArrayList<>(hop.getParent());

				for (Hop parent : parents) {
					HopRewriteUtils.replaceChildReference(parent,hop, newRoot);
				}
				HopRewriteUtils.removeAllChildReferences(hop);
			}
			// return new tree
			return newRoot;
		}
		// no optimization happened
		return hop;
	}


	/**
	 * mmChainDP(): Core method to perform dynamic programming on a given array
	 * of matrix dimensions and additional array with transpose flags
	 */
	private static MemoTable mmChainDP(double[] dimArray, int size, List<Hop> mmChain, List<Boolean> isTransposeChain) {
		// create memo table
		MemoTable memo = new MemoTable(size);

		// 1.) THE BASE CASE
		// loop through every matrix in the chain
		for (int i = 0; i < size; i++) {
			// fetch and store rows, cols and transpose flag
			double rows = dimArray[i];
			double cols = dimArray[i+1];
			boolean isTransposed = isTransposeChain.get(i);

			// for standard matrix:
			if (!isTransposed) {
				// create the normal plan
				Plan normalPlan = new Plan();
				normalPlan.cost = 0; // no costs
				normalPlan.withTranspose = false;
				memo.setNormal(i, i, normalPlan);

				// create the transposed plan
				Plan transposedPlan = new Plan();
				transposedPlan.cost = rows * cols; // cost is FLOPs for transposing: rows * cols
				transposedPlan.withTranspose = true;
				memo.setTransposed(i, i, transposedPlan);
			}
			// opposite for transposed matrix
			else {
				// since matrix is transposed, normal plan requires transposing it again
				Plan normalPlan = new Plan();
				normalPlan.cost = rows * cols;
				normalPlan.withTranspose = true;
				memo.setNormal(i, i, normalPlan);

				// already transposed, so no costs
				Plan transposedPlan = new Plan();
				transposedPlan.cost = 0;
				transposedPlan.withTranspose = false;
				memo.setTransposed(i, i, transposedPlan);
			}
		}

		// 2. COMBINATIONS OF BLOCKS
		for (int subchainSize = 2; subchainSize <= size; subchainSize++) {
			for (int i = 0; i < size - subchainSize + 1; i++ ) {
				int j = i + subchainSize - 1;

				// final dimensions of subchain if multiplied normally
				double normalOutRows = dimArray[i];
				double normalOutCols = dimArray[j+1];

				Plan bestNormalPlan = new Plan();
				Plan bestTransposedPlan = new Plan();

				// evaluate and compare every split point of the chain A %*% (B %*% C) or (A %*% B) %*% C
				for (int k = i; k < j; k++) {
					// evaluate normal plan
					// The index where the line between Left and Right chain is splitted -> k
					Plan normalLeft = memo.getNormal(i, k);
					Plan normalRight = memo.getNormal(k+1, j);

					double costMatMult = normalOutRows * dimArray[k+1] * normalOutCols;
					double costNormal = normalLeft.cost + normalRight.cost + costMatMult;

					if (costNormal < bestNormalPlan.cost) {
						bestNormalPlan.cost = costNormal;
						bestNormalPlan.splitIndex = k;
						bestNormalPlan.withTranspose = false;
					}

					// evaluate transposed plan
					Plan transposedLeft = memo.getTransposed(i, k);
					Plan transposedRight = memo.getTransposed(k+1, j);
					double costTransposed = transposedLeft.cost + transposedRight.cost + costMatMult;

					if (costTransposed < bestTransposedPlan.cost) {
						bestTransposedPlan.cost = costTransposed;
						bestTransposedPlan.splitIndex = k;
						bestTransposedPlan.withTranspose = false;
					}
				}
				// costs, after the full subchain is calculated and then transposed
				double transposeCost = normalOutRows * normalOutCols;

				// check if t(A %*% B) cheaper than t(B) %*% t(A)
				if (bestNormalPlan.cost + transposeCost < bestTransposedPlan.cost) {
					bestTransposedPlan.cost = bestNormalPlan.cost + transposeCost;
					bestTransposedPlan.splitIndex = bestNormalPlan.splitIndex;
					bestTransposedPlan.withTranspose = true;
				}

				// check if t(t(B) %*% t(A)) cheaper than A %*% B
				if (bestTransposedPlan.cost + transposeCost < bestNormalPlan.cost) {
					bestNormalPlan.cost = bestTransposedPlan.cost + transposeCost;
					bestNormalPlan.splitIndex = bestTransposedPlan.splitIndex;
					bestNormalPlan.withTranspose = true;
				}
				memo.setNormal(i, j, bestNormalPlan);
				memo.setTransposed(i, j, bestTransposedPlan);
			}
		}
		return memo;
	}


	private Hop mmChainBuildTree(int i, int j, List<Hop> mmChain, MemoTable memo, boolean isTransposed, Hop rootHop) {
		Plan plan = isTransposed ? memo.getTransposed(i, j) : memo.getNormal(i, j);

		// Base Case with one matrix
		if (i == j) {
			Hop leaf = mmChain.get(i);
			if (plan.withTranspose) {
				Hop t = HopRewriteUtils.createTranspose(leaf);
				t.setExecType(rootHop.getExecType());
				t.refreshSizeInformation();
				t.setBlocksize(rootHop.getBlocksize());
				t.setVisited();
				return t;
			}
			return leaf;
		}
		if (plan.withTranspose) {
			Hop child = mmChainBuildTree(i, j, mmChain, memo, !isTransposed, rootHop);
			Hop t = HopRewriteUtils.createTranspose(child);
			t.setExecType(rootHop.getExecType());
			t.refreshSizeInformation();
			t.setBlocksize(rootHop.getBlocksize());
			t.setVisited();
			return t;
		}

		Hop leftChild, rightChild;
		if (isTransposed) {
			leftChild = mmChainBuildTree(plan.splitIndex + 1, j, mmChain, memo, true, rootHop);
			rightChild = mmChainBuildTree(i, plan.splitIndex, mmChain, memo, true, rootHop);
		}
		else {
			leftChild = mmChainBuildTree(i, plan.splitIndex, mmChain, memo, false, rootHop);
			rightChild = mmChainBuildTree(plan.splitIndex + 1, j, mmChain, memo, false, rootHop);
		}
		Hop multOp = HopRewriteUtils.createMatrixMultiply(leftChild, rightChild);
		multOp.setExecType(rootHop.getExecType());
		multOp.refreshSizeInformation();
		multOp.setBlocksize(rootHop.getBlocksize());
		multOp.setVisited();
		return multOp;
	}


	/**
	 * Obtains all dimension information of the chain and constructs the dimArray.
	 *
	 * If all dimensions are known it returns true; othrewise the mmchain rewrite
	 * should be ended without modifications.
	 *
	 * @param hop high-level operator
	 * @param chain list of high-level operators
	 * @param isTransposeChain Parallel list of boolean flags indicating if a matrix is transposed
	 * @param dimsArray dimension array
	 * @return true if all dimensions known
	 */
	protected static boolean getDimsArray(Hop hop, List<Hop> chain, List<Boolean> isTransposeChain, double[] dimsArray) {
		boolean dimsKnown = true;

		// Build the array containing dimensions from all matrices in the chain
		// check the dimensions in the matrix chain to ensure all dimensions are known
		for (int i = 0; i < chain.size(); i++) {
			Hop leaf = chain.get(i);
			// fetching dimensions
			long dim1 = leaf.getDim1();
			long dim2 = leaf.getDim2();


			if( chain.get(i).getDim1() <= 0 || chain.get(i).getDim2() <= 0 ) {
				dimsKnown = false;
				break;
			}

			if (isTransposeChain.get(i)) {
				long temp = dim1;
				dim1 = dim2;
				dim2 = temp;
			}

			if (i == 0) {
				dimsArray[i] = dim1;
			}
			else if (dimsArray[i] != dim1) {
				throw new HopsException( hop.printErrorLocation() +
					"Hops::optimizeMMChain() : Matrix Dimension Mismatch: " +
					dimsArray[i] +" != "+ dim1);
			}
			dimsArray[i + 1] = dim2;
		}
		return dimsKnown;
	}

	/**
	 * A blueprint object tracking the cheapest cost and split-point for a sub-problem.
	 */
	private static class Plan {
		double cost = Double.MAX_VALUE;
		int splitIndex = -1;
		boolean withTranspose;
	}

	/**
	 * Dual-state 2D array matrix holding the memoized sub-problems.
	 */
	private static class MemoTable {
		private final Plan[][] normalPlans;
		private final Plan[][] transposedPlans;
		public MemoTable(int size) {
			normalPlans = new Plan[size][size];
			transposedPlans = new Plan[size][size];
		}
		public Plan getNormal(int i, int j) { return normalPlans[i][j]; }
		public Plan getTransposed(int i, int j) { return transposedPlans[i][j]; }
		public void setNormal(int i, int j, Plan plan) { normalPlans[i][j] = plan; }
		public void setTransposed(int i, int j, Plan plan) { transposedPlans[i][j] = plan; }
	}
}
