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

package org.apache.sysml.hops.rewrite;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;

import org.apache.sysml.hops.BinaryOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.LiteralOp;
import org.apache.sysml.parser.Expression;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;

/**
 * Prerequisite: RewriteCommonSubexpressionElimination must run before this rule.
 *
 * Rewrite a chain of element-wise multiply hops that contain identical elements.
 * For example `(B * A) * B` is rewritten to `A * (B^2)` (or `(B^2) * A`), where `^` is element-wise power.
 *
 * Does not rewrite in the presence of foreign parents in the middle of the e-wise multiply chain,
 * since foreign parents may rely on the individual results.
 */
public class RewriteEMult extends HopRewriteRule {
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) throws HopsException {
		if( roots == null )
			return null;
		for( int i=0; i<roots.size(); i++ ) {
			Hop h = roots.get(i);
			roots.set(i, rule_RewriteEMult(h));
		}
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) throws HopsException {
		if( root == null )
			return null;
		return rule_RewriteEMult(root);
	}

	private static boolean isBinaryMult(final Hop hop) {
		return hop instanceof BinaryOp && ((BinaryOp)hop).getOp() == Hop.OpOp2.MULT;
	}

	private static Hop rule_RewriteEMult(final Hop root) {
		if (root.isVisited())
			return root;
		root.setVisited();

		final ArrayList<Hop> rootInputs = root.getInput();

		// 1. Find immediate subtree of EMults.
		if (isBinaryMult(root)) {
			final Hop left = rootInputs.get(0), right = rootInputs.get(1);
			final BinaryOp r = (BinaryOp)root;
			final Set<BinaryOp> emults = new HashSet<>();
			final Multiset<Hop> leaves = HashMultiset.create();
			findEMultsAndLeaves(r, emults, leaves);

			// 2. Ensure it is profitable to do a rewrite.
			if (isOptimizable(leaves)) {
				// 3. Check for foreign parents.
				// A foreign parent is a parent of some EMult that is not in the set.
				// Foreign parents destroy correctness of this rewrite.
				final boolean okay = (!isBinaryMult(left) || checkForeignParent(emults, (BinaryOp)left)) &&
						(!isBinaryMult(right) || checkForeignParent(emults, (BinaryOp)right));
				if (okay) {
					// 4. Construct replacement EMults for the leaves
					final Hop replacement = constructReplacement(leaves);
					// 5. Replace root with replacement
					if (LOG.isDebugEnabled())
						LOG.debug(String.format(
								"Element-wise multiply chain rewrite of %d e-mults at sub-dag %d to new sub-dag %d",
								emults.size(), root.getHopID(), replacement.getHopID()));
					replacement.setVisited();
					return HopRewriteUtils.replaceHop(root, replacement);
				}
			}
		}

		// This rewrite is not applicable to the current root.
		// Try the root's children.
		for (int i = 0; i < rootInputs.size(); i++) {
			final Hop input = rootInputs.get(i);
			final Hop newInput = rule_RewriteEMult(input);
			rootInputs.set(i, newInput);
		}
		return root;
	}

	private static Hop constructReplacement(final Multiset<Hop> leaves) {
		// Sort by data type
		final SortedMap<Hop,Integer> sorted = new TreeMap<>(compareByDataType);
		for (final Multiset.Entry<Hop> entry : leaves.entrySet()) {
			final Hop h = entry.getElement();
			// unlink parents (the EMults, which we are throwing away)
			h.getParent().clear();
			sorted.put(h, entry.getCount());
		}
		// sorted contains all leaves, sorted by data type, stripped from their parents

		// Construct right-deep EMult tree
		final Iterator<Map.Entry<Hop, Integer>> iterator = sorted.entrySet().iterator();
		Hop first = constructPower(iterator.next());

		for (int i = 1; i < sorted.size(); i++) {
			final Hop second = constructPower(iterator.next());
			first = HopRewriteUtils.createBinary(second, first, Hop.OpOp2.MULT);
		}
		return first;
	}

	private static Hop constructPower(Map.Entry<Hop, Integer> entry) {
		final Hop hop = entry.getKey();
		final int cnt = entry.getValue();
		assert(cnt >= 1);
		if (cnt == 1) return hop;
		return HopRewriteUtils.createBinary(hop, new LiteralOp(cnt), Hop.OpOp2.POW);
	}



	// Order: scalars > row vectors > col vectors >
	//        non-vector matrices ordered by sparsity (higher nnz first, unknown sparsity last) >
	//        other data types
	// disambiguate by Hop ID
	private static final Comparator<Hop> compareByDataType = new Comparator<Hop>() {
		@Override
		public final int compare(Hop o1, Hop o2) {
			int c = Integer.compare(orderDataType[o1.getDataType().ordinal()], orderDataType[o2.getDataType().ordinal()]);
			if (c != 0) return c;

			// o1 and o2 have the same data type
			switch (o1.getDataType()) {
			case SCALAR: return Long.compare(o1.getHopID(), o2.getHopID());
			case MATRIX:
				// two matrices; check for vectors
				if (o1.getDim1() == 1) { // row vector
						if (o2.getDim1() != 1) return 1; // row vectors are greatest of matrices
						return compareBySparsityThenId(o1, o2); // both row vectors
				} else if (o2.getDim1() == 1) { // 2 is row vector; 1 is not
						return -1; // row vectors are the greatest matrices
				} else if (o1.getDim2() == 1) { // col vector
						if (o2.getDim2() != 1) return 1; // col vectors greater than non-vectors
						return compareBySparsityThenId(o1, o2); // both col vectors
				} else if (o2.getDim2() == 1) { // 2 is col vector; 1 is not
						return 1; // col vectors greater than non-vectors
				} else { // both non-vectors
						return compareBySparsityThenId(o1, o2);
				}
			default:
				return Long.compare(o1.getHopID(), o2.getHopID());
			}
		}
		private int compareBySparsityThenId(Hop o1, Hop o2) {
			// the hop with more nnz is first; unknown nnz (-1) last
			int c = Long.compare(o1.getNnz(), o2.getNnz());
			if (c != 0) return c;
			return Long.compare(o1.getHopID(), o2.getHopID());
		}
		private final int[] orderDataType;
		{
			Expression.DataType[] dtValues = Expression.DataType.values();
			orderDataType = new int[dtValues.length];
			for (int i = 0, valuesLength = dtValues.length; i < valuesLength; i++) {
				switch(dtValues[i]) {
				case SCALAR:
					orderDataType[i] = 4;
					break;
				case MATRIX:
					orderDataType[i] = 3;
					break;
				case FRAME:
					orderDataType[i] = 2;
					break;
				case OBJECT:
					orderDataType[i] = 1;
					break;
				case UNKNOWN:
					orderDataType[i] = 0;
					break;
				}
			}
		}
	};

	private static boolean checkForeignParent(final Set<BinaryOp> emults, final BinaryOp child) {
		final ArrayList<Hop> parents = child.getParent();
		if (parents.size() > 1)
			for (final Hop parent : parents)
				//noinspection SuspiciousMethodCalls
				if (!emults.contains(parent))
					return false;
		// child does not have foreign parents

		final ArrayList<Hop> inputs = child.getInput();
		final Hop left = inputs.get(0), right = inputs.get(1);
		return  (!isBinaryMult(left) || checkForeignParent(emults, (BinaryOp)left)) &&
				(!isBinaryMult(right) || checkForeignParent(emults, (BinaryOp)right));
	}

	/**
	 * Create a set of the counts of all BinaryOp MULTs in the immediate subtree, starting with root.
	 */
	private static void findEMultsAndLeaves(final BinaryOp root, final Set<BinaryOp> emults, final Multiset<Hop> leaves) {
		// Because RewriteCommonSubexpressionElimination already ran, it is safe to compare by equality.
		emults.add(root);

		final ArrayList<Hop> inputs = root.getInput();
		final Hop left = inputs.get(0), right = inputs.get(1);

		if (isBinaryMult(left)) findEMultsAndLeaves((BinaryOp) left, emults, leaves);
		else leaves.add(left);

		if (isBinaryMult(right)) findEMultsAndLeaves((BinaryOp) right, emults, leaves);
		else leaves.add(right);
	}

	/** Only optimize a subtree of EMults if at least one leaf occurs more than once. */
	private static boolean isOptimizable(final Multiset<Hop> set) {
		for (Multiset.Entry<Hop> hopEntry : set.entrySet()) {
			if (hopEntry.getCount() > 1)
				return true;
		}
		return false;
	}
}
