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
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;

/**
 * Prerequisite: RewriteCommonSubexpressionElimination must run before this rule.
 *
 * Rewrite a chain of element-wise multiply hops that contain identical elements.
 * For example `(B * A) * B` is rewritten to `A * (B^2)` (or `(B^2) * A`), where `^` is element-wise power.
 * The order of the multiplicands depends on their data types, dimensions (matrix or vector), and sparsity.
 *
 * Does not rewrite in the presence of foreign parents in the middle of the e-wise multiply chain,
 * since foreign parents may rely on the individual results.
 * Does not perform rewrites on an element-wise multiply if its dimensions are unknown.
 *
 * The new order of element-wise multiply chains is as follows:
 * <pre>
 *     (((unknown * object * frame) * ([least-nnz-matrix * matrix] * most-nnz-matrix))
 *      * ([least-nnz-row-vector * row-vector] * most-nnz-row-vector))
 *     * ([[scalars * least-nnz-col-vector] * col-vector] * most-nnz-col-vector)
 * </pre>
 * Identical elements are replaced with powers.
 */
public class RewriteElementwiseMultChainOptimization extends HopRewriteRule {
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) {
		if( roots == null )
			return null;
		for( int i=0; i<roots.size(); i++ ) {
			Hop h = roots.get(i);
			roots.set(i, rule_RewriteEMult(h));
		}
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) {
		if( root == null )
			return null;
		return rule_RewriteEMult(root);
	}

	private static boolean isBinaryMult(final Hop hop) {
		return hop instanceof BinaryOp && ((BinaryOp)hop).getOp() == OpOp2.MULT;
	}

	private static Hop rule_RewriteEMult(final Hop root) {
		if (root.isVisited())
			return root;
		root.setVisited();

		// 1. Find immediate subtree of EMults. Check dimsKnown.
		if (isBinaryMult(root) && root.dimsKnown()) {
			final Hop left = root.getInput().get(0), right = root.getInput().get(1);
			// The set of BinaryOp element-wise multiply hops in the emult chain.
			final Set<BinaryOp> emults = new HashSet<>();
			// The multiset of multiplicands in the emult chain.
			final Map<Hop, Integer> leaves = new HashMap<>(); // poor man's HashMultiset
			findEMultsAndLeaves((BinaryOp)root, emults, leaves);

			// 2. Ensure it is profitable to do a rewrite.
			// Only optimize a subtree of emults if there are at least two emults (meaning, at least 3 multiplicands).
			if (emults.size() >= 2) {
				// 3. Check for foreign parents.
				// A foreign parent is a parent of some EMult that is not in the set.
				// Foreign parents destroy correctness of this rewrite.
				final boolean okay = (!isBinaryMult(left) || checkForeignParent(emults, (BinaryOp)left)) &&
						(!isBinaryMult(right) || checkForeignParent(emults, (BinaryOp)right));
				if (okay) {
					// 4. Construct replacement EMults for the leaves
					final Hop replacement = constructReplacement(emults, leaves);
					if (LOG.isDebugEnabled())
						LOG.debug(String.format(
								"Element-wise multiply chain rewrite of %d e-mults at sub-dag %d to new sub-dag %d",
								emults.size(), root.getHopID(), replacement.getHopID()));

					// 5. Replace root with replacement
					final Hop newRoot = HopRewriteUtils.rewireAllParentChildReferences(root, replacement);

					// 6. Recurse at leaves (no need to repeat the interior emults)
					for (final Hop leaf : leaves.keySet()) {
						recurseInputs(leaf);
					}
					return newRoot;
				}
			}
		}
		// This rewrite is not applicable to the current root.
		// Try the root's children.
		recurseInputs(root);
		return root;
	}

	private static void recurseInputs(final Hop parent) {
		final ArrayList<Hop> inputs = parent.getInput();
		for (int i = 0; i < inputs.size(); i++) {
			final Hop input = inputs.get(i);
			final Hop newInput = rule_RewriteEMult(input);
			inputs.set(i, newInput);
		}
	}

	private static Hop constructReplacement(final Set<BinaryOp> emults, final Map<Hop, Integer> leaves) {
		// Sort by data type
		final SortedSet<Hop> sorted = new TreeSet<>(compareByDataType);
		for (final Map.Entry<Hop, Integer> entry : leaves.entrySet()) {
			final Hop h = entry.getKey();
			// unlink parents that are in the emult set(we are throwing them away)
			// keep other parents
			h.getParent().removeIf(parent -> parent instanceof BinaryOp && emults.contains(parent));
			sorted.add(constructPower(h, entry.getValue()));
		}
		// sorted contains all leaves, sorted by data type, stripped from their parents

		// Construct right-deep EMult tree
		final Iterator<Hop> iterator = sorted.iterator();

		Hop next = iterator.hasNext() ? iterator.next() : null;
		Hop colVectorsScalars = null;
		while(next != null &&
				(next.getDataType() == DataType.SCALAR
						|| next.getDataType() == DataType.MATRIX && next.getDim2() == 1))
		{
			if( colVectorsScalars == null )
				colVectorsScalars = next;
			else {
				colVectorsScalars = HopRewriteUtils.createBinary(next, colVectorsScalars, OpOp2.MULT);
				colVectorsScalars.setVisited();
			}
			next = iterator.hasNext() ? iterator.next() : null;
		}
		// next is not processed and is either null or past col vectors

		Hop rowVectors = null;
		while(next != null &&
				(next.getDataType() == DataType.MATRIX && next.getDim1() == 1))
		{
			if( rowVectors == null )
				rowVectors = next;
			else {
				rowVectors = HopRewriteUtils.createBinary(rowVectors, next, OpOp2.MULT);
				rowVectors.setVisited();
			}
			next = iterator.hasNext() ? iterator.next() : null;
		}
		// next is not processed and is either null or past row vectors

		Hop matrices = null;
		while(next != null &&
				(next.getDataType() == DataType.MATRIX))
		{
			if( matrices == null )
				matrices = next;
			else {
				matrices = HopRewriteUtils.createBinary(matrices, next, OpOp2.MULT);
				matrices.setVisited();
			}
			next = iterator.hasNext() ? iterator.next() : null;
		}
		// next is not processed and is either null or past matrices

		Hop other = null;
		while(next != null)
		{
			if( other == null )
				other = next;
			else {
				other = HopRewriteUtils.createBinary(other, next, OpOp2.MULT);
				other.setVisited();
			}
			next = iterator.hasNext() ? iterator.next() : null;
		}
		// finished

		// ((other * matrices) * rowVectors) * colVectorsScalars
		Hop top = null;
		if( other == null && matrices != null )
			top = matrices;
		else if( other != null && matrices == null )
			top = other;
		else if( other != null ) { //matrices != null
			top = HopRewriteUtils.createBinary(other, matrices, OpOp2.MULT);
			top.setVisited();
		}

		if( top == null && rowVectors != null )
			top = rowVectors;
		else if( rowVectors != null ) { //top != null
			top = HopRewriteUtils.createBinary(top, rowVectors, OpOp2.MULT);
			top.setVisited();
		}

		if( top == null && colVectorsScalars != null )
			top = colVectorsScalars;
		else if( colVectorsScalars != null ) { //top != null
			top = HopRewriteUtils.createBinary(top, colVectorsScalars, OpOp2.MULT);
			top.setVisited();
		}

		return top;
	}

	private static Hop constructPower(final Hop hop, final int cnt) {
		assert(cnt >= 1);
		hop.setVisited(); // we will visit the leaves' children next
		if (cnt == 1)
			return hop;
		final Hop pow = HopRewriteUtils.createBinary(hop, new LiteralOp(cnt), OpOp2.POW);
		pow.setVisited();
		return pow;
	}

	/**
	 * A Comparator that orders Hops by their data type, dimension, and sparsity.
	 * The order is as follows:
	 * 		scalars < col vectors < row vectors <
	 *      non-vector matrices ordered by sparsity (higher nnz last, unknown sparsity last) >
	 *      other data types.
	 * Disambiguate by Hop ID.
	 */
	//TODO replace by ComparableHop wrapper around hop that implements equals and compareTo
	//in order to ensure comparisons that are 'consistent with equals'
	private static final Comparator<Hop> compareByDataType = new Comparator<Hop>() {
		private final int[] orderDataType = new int[DataType.values().length];
		{
			for (int i = 0, valuesLength = DataType.values().length; i < valuesLength; i++)
				switch(DataType.values()[i]) {
					case SCALAR: orderDataType[i] = 0; break;
					case MATRIX: orderDataType[i] = 1; break;
					case TENSOR: orderDataType[i] = 2; break;
					case FRAME:  orderDataType[i] = 3; break;
					case UNKNOWN:orderDataType[i] = 4; break;
					case LIST:   orderDataType[i] = 5; break;
				}
		}

		@Override
		public final int compare(final Hop o1, final Hop o2) {
			final int c = Integer.compare(orderDataType[o1.getDataType().ordinal()], orderDataType[o2.getDataType().ordinal()]);
			if (c != 0) return c;

			// o1 and o2 have the same data type
			switch (o1.getDataType()) {
			case MATRIX:
				// two matrices; check for vectors
				if (o1.getDim2() == 1) { // col vector
						if (o2.getDim2() != 1) return -1; // col vectors are greatest of matrices
						return compareBySparsityThenId(o1, o2); // both col vectors
				} else if (o2.getDim2() == 1) { // 2 is col vector; 1 is not
						return 1; // col vectors are the greatest matrices
				} else if (o1.getDim1() == 1) { // row vector
						if (o2.getDim1() != 1) return -1; // row vectors greater than non-vectors
						return compareBySparsityThenId(o1, o2); // both row vectors
				} else if (o2.getDim1() == 1) { // 2 is row vector; 1 is not
						return 1; // row vectors greater than non-vectors
				} else { // both non-vectors
					return compareBySparsityThenId(o1, o2);
				}
			default:
				return Long.compare(o1.getHopID(), o2.getHopID());
			}
		}
		private int compareBySparsityThenId(final Hop o1, final Hop o2) {
			// the hop with more nnz is first; unknown nnz (-1) last
			final int c = Long.compare(o1.getNnz(), o2.getNnz());
			if (c != 0) return -c;
			return Long.compare(o1.getHopID(), o2.getHopID());
		}
	};

	/**
	 * Check if a node has a parent that is not in the set of emults. Recursively check children who are also emults.
	 * @param emults The set of BinaryOp element-wise multiply hops in the emult chain.
	 * @param child An interior emult hop in the emult chain dag.
	 * @return Whether this interior emult or any child emult has a foreign parent.
	 */
	private static boolean checkForeignParent(final Set<BinaryOp> emults, final BinaryOp child) {
		final ArrayList<Hop> parents = child.getParent();
		if (parents.size() > 1)
			for (final Hop parent : parents)
				if (!(parent instanceof BinaryOp) || !emults.contains(parent))
					return false;
		// child does not have foreign parents

		final ArrayList<Hop> inputs = child.getInput();
		final Hop left = inputs.get(0), right = inputs.get(1);
		return  (!isBinaryMult(left) || checkForeignParent(emults, (BinaryOp)left)) &&
				(!isBinaryMult(right) || checkForeignParent(emults, (BinaryOp)right));
	}

	/**
	 * Create a set of the counts of all BinaryOp MULTs in the immediate subtree, starting with root, recursively.
	 * @param root Root of sub-dag
	 * @param emults Out parameter. The set of BinaryOp element-wise multiply hops in the emult chain (including root).
	 * @param leaves Out parameter. The multiset of multiplicands in the emult chain.
	 */
	private static void findEMultsAndLeaves(final BinaryOp root, final Set<BinaryOp> emults, final Map<Hop, Integer> leaves) {
		// Because RewriteCommonSubexpressionElimination already ran, it is safe to compare by equality.
		emults.add(root);
		
		// TODO proper handling of DAGs (avoid collecting the same leaf multiple times)
		// TODO exclude hops with unknown dimensions and move rewrites to dynamic rewrites 
		
		final ArrayList<Hop> inputs = root.getInput();
		final Hop left = inputs.get(0), right = inputs.get(1);

		if (isBinaryMult(left))
			findEMultsAndLeaves((BinaryOp) left, emults, leaves);
		else
			addMultiset(leaves, left);

		if (isBinaryMult(right))
			findEMultsAndLeaves((BinaryOp) right, emults, leaves);
		else
			addMultiset(leaves, right);
	}

	private static <K> void addMultiset(final Map<K,Integer> map, final K k) {
		map.put(k, map.getOrDefault(k, 0) + 1);
	}

}
