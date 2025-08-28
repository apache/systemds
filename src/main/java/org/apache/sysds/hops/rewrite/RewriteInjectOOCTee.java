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

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * This Rewrite rule injects a Tee Operator for specific Out-Of-Core (OOC) patterns
 * where a value or an intermediate result is shared twice. Since for OOC we data streams
 * can only be consumed once.
 *
 * <p>
 *   Pattern identified {@code t(X) %*% X}, where the data {@code X} will be shared by
 *   {@code t(X)} and {@code %*%} multiplication.
 * </p>
 *
 * The rewrite uses a stable two-pass approach:
 * 1. <b>Find candidates (Read-Only):</b> Traverse the entire HOP DAG to identify candidates
 * the fit the target pattern.
 * 2. <b>Apply Rewrites (Modification):</b> Iterate over the collected candidate and put
 * {@code TeeOp}, and safely rewire the graph.
 */
public class RewriteInjectOOCTee extends HopRewriteRule {

	private static final Set<Long> rewrittenHops = new HashSet<>();
	private static final Map<Long, Hop> handledHop = new HashMap<>();

	// Maintain a list of candidates to rewrite in the second pass
	private final List<Hop> rewriteCandidates = new ArrayList<>();

	/**
	 * Handle a generic (last-level) hop DAG with multiple roots.
	 *
	 * @param roots high-level operator roots
	 * @param state program rewrite status
	 * @return list of high-level operators
	 */
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) {
		if (roots == null) {
			return null;
		}

		// Clear candidates for this pass
		rewriteCandidates.clear();

		// PASS 1: Identify candidates without modifying the graph
		for (Hop root : roots) {
			root.resetVisitStatus();
			findRewriteCandidates(root);
		}

		// PASS 2: Apply rewrites to identified candidates
		for (Hop candidate : rewriteCandidates) {
			applyTopDownTeeRewrite(candidate);
		}

		return roots;
	}

	/**
	 * Handle a predicate hop DAG with exactly one root.
	 *
	 * @param root  high-level operator root
	 * @param state program rewrite status
	 * @return high-level operator
	 */
	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) {
		if (root == null) {
			return null;
		}

		// Clear candidates for this pass
		rewriteCandidates.clear();

		// PASS 1: Identify candidates without modifying the graph
		root.resetVisitStatus();
		findRewriteCandidates(root);

		// PASS 2: Apply rewrites to identified candidates
		for (Hop candidate : rewriteCandidates) {
			applyTopDownTeeRewrite(candidate);
		}

		return root;
	}

	/**
	 * First pass: Find candidates for rewrite without modifying the graph.
	 * This method traverses the graph and identifies nodes that need to be
	 * rewritten based on the transpose-matrix multiply pattern.
	 *
	 * @param hop current hop being examined
	 */
	private void findRewriteCandidates(Hop hop) {
		if (hop.isVisited()) {
			return;
		}

		// Mark as visited to avoid processing the same hop multiple times
		hop.setVisited(true);

		// Recursively traverse the graph (depth-first)
		for (Hop input : hop.getInput()) {
			findRewriteCandidates(input);
		}

		// Check if this hop is a candidate for OOC Tee injection
		if (isRewriteCandidate(hop)) {
			rewriteCandidates.add(hop);
		}
	}

	/**
	 * Check if a hop should be considered for rewrite.
	 *
	 * @param hop the hop to check
	 * @return true if the hop meets all criteria for rewrite
	 */
	private boolean isRewriteCandidate(Hop hop) {
		// Skip if already handled
		if (rewrittenHops.contains(hop.getHopID()) || handledHop.containsKey(hop.getHopID())) {
			return false;
		}

		boolean multipleConsumers = hop.getParent().size() > 1;
		boolean isNotAlreadyTee = isNotAlreadyTee(hop);
		boolean isOOCEnabled = DMLScript.USE_OOC;
		boolean isTransposeMM = isTranposePattern(hop);
		boolean isMatrix = hop.getDataType() == Types.DataType.MATRIX;

		return isOOCEnabled && multipleConsumers && isNotAlreadyTee && isTransposeMM && isMatrix;
	}

	/**
	 * Second pass: Apply the TeeOp transformation to a candidate hop.
	 * This safely rewires the graph by creating a TeeOp node and placeholders.
	 *
	 * @param sharedInput the hop to be rewritten
	 */
	private void applyTopDownTeeRewrite(Hop sharedInput) {
		// Only process if not already handled
		if (handledHop.containsKey(sharedInput.getHopID())) {
			return;
		}

		// Take a defensive copy of consumers before modifying the graph
		ArrayList<Hop> consumers = new ArrayList<>(sharedInput.getParent());

		// Create the new TeeOp with the original hop as input
		TeeOp teeOp = new TeeOp(sharedInput);

		// Rewire the graph: replace original connections with TeeOp outputs
		int i = 0;
		for (Hop consumer : consumers) {
			Hop placeholder = new DataOp("tee_out_" + sharedInput.getName() + "_" + i,
							sharedInput.getDataType(),
							sharedInput.getValueType(),
							Types.OpOpData.TRANSIENTWRITE,
							null,
							sharedInput.getDim1(),
							sharedInput.getDim2(),
							sharedInput.getNnz(),
							sharedInput.getBlocksize()
			);

			// Copy metadata
			placeholder.setBeginLine(sharedInput.getBeginLine());
			placeholder.setBeginColumn(sharedInput.getBeginColumn());
			placeholder.setEndLine(sharedInput.getEndLine());
			placeholder.setEndColumn(sharedInput.getEndColumn());

			// Connect placeholder to TeeOp and consumer
			HopRewriteUtils.addChildReference(placeholder, teeOp);
			HopRewriteUtils.replaceChildReference(consumer, sharedInput, placeholder);

			i++;
		}

		// Record that we've handled this hop
		handledHop.put(sharedInput.getHopID(), teeOp);
		rewrittenHops.add(sharedInput.getHopID());
	}

	private boolean isNotAlreadyTee(Hop hop) {
		if (hop.getParent().size() > 1) {
			for (Hop consumer : hop.getParent()) {
				if (consumer instanceof TeeOp) {
					return false;
				}
			}
		}
		return true;
	}

	private boolean isTranposePattern (Hop hop) {
		boolean hasTransposeConsumer = false; // t(X)
		boolean hasMatrixMultiplyConsumer = false; // %*%

		for (Hop parent: hop.getParent()) {
			String opString = parent.getOpString();
			if (parent instanceof ReorgOp) {
				if (opString.contains("r'") || opString.contains("transpose")) {
					hasTransposeConsumer = true;
				}
			}
			else if (parent instanceof AggBinaryOp)
				if (opString.contains("*") || opString.contains("ba+*")) {
					hasMatrixMultiplyConsumer = true;
				}
		}
		return hasTransposeConsumer &&  hasMatrixMultiplyConsumer;
	}
}
