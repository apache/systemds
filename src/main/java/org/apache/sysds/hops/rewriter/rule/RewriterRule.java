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

package org.apache.sysds.hops.rewriter.rule;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.mutable.MutableObject;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.hops.rewriter.assertions.RewriterAssertionUtils;
import org.apache.sysds.hops.rewriter.assertions.RewriterAssertions;
import org.apache.sysds.hops.rewriter.estimators.RewriterCostEstimator;
import org.apache.sysds.hops.rewriter.estimators.RewriterSparsityEstimator;
import scala.Tuple2;
import scala.Tuple3;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class RewriterRule {

	private final RuleContext ctx;
	private final String name;
	private final RewriterStatement fromRoot;
	private final RewriterStatement toRoot;
	private List<RewriterStatement> toRoots;
	private final HashMap<RewriterStatement, LinkObject> linksStmt1ToStmt2; // Contains the explicit links a transformation has (like instructions, (a+b)-c = a+(b-c), but '+' and '-' are the same instruction still [important if instructions have metadata])
	private final HashMap<RewriterStatement, LinkObject> linksStmt2ToStmt1;
	private final List<Tuple2<RewriterStatement, BiConsumer<RewriterStatement, RewriterStatement.MatchingSubexpression>>> applyStmt1ToStmt2;
	private final List<Tuple2<RewriterStatement, BiConsumer<RewriterStatement, RewriterStatement.MatchingSubexpression>>> applyStmt2ToStmt1;
	private final Function<RewriterStatement.MatchingSubexpression, Boolean> iff1to2;
	private final Function<RewriterStatement.MatchingSubexpression, Boolean> iff2to1;
	private final boolean unidirectional;
	private final Consumer<RewriterStatement> postProcessor;
	private Set<RewriterStatement> allowedMultiReferences = Collections.emptySet();
	private RewriterAssertions combinedAssertions;
	private boolean allowCombinations = false;
	private boolean requireCostCheck = false;
	private RewriterStatement fromCost = null;
	private List<RewriterStatement> toCosts = null;

	public RewriterRule(final RuleContext ctx, String name, RewriterStatement fromRoot, RewriterStatement toRoot, boolean unidirectional, HashMap<RewriterStatement, LinkObject> linksStmt1ToStmt2, HashMap<RewriterStatement, LinkObject> linksStmt2ToStmt1) {
		this(ctx, name, fromRoot, toRoot, unidirectional, linksStmt1ToStmt2, linksStmt2ToStmt1, null, null, null, null, null);
	}

	public RewriterRule(final RuleContext ctx, String name, RewriterStatement fromRoot, RewriterStatement toRoot, boolean unidirectional, HashMap<RewriterStatement, LinkObject> linksStmt1ToStmt2, HashMap<RewriterStatement, LinkObject> linksStmt2ToStmt1, Function<RewriterStatement.MatchingSubexpression, Boolean> iff1to2, Function<RewriterStatement.MatchingSubexpression, Boolean> iff2to1, List<Tuple2<RewriterStatement, BiConsumer<RewriterStatement, RewriterStatement.MatchingSubexpression>>> apply1To2, List<Tuple2<RewriterStatement, BiConsumer<RewriterStatement, RewriterStatement.MatchingSubexpression>>> apply2To1) {
		this(ctx, name, fromRoot, toRoot, unidirectional, linksStmt1ToStmt2, linksStmt2ToStmt1, iff1to2, iff2to1, apply1To2, apply2To1, null);
	}

	public RewriterRule(final RuleContext ctx, String name, RewriterStatement fromRoot, RewriterStatement toRoot, boolean unidirectional, HashMap<RewriterStatement, LinkObject> linksStmt1ToStmt2, HashMap<RewriterStatement, LinkObject> linksStmt2ToStmt1, Function<RewriterStatement.MatchingSubexpression, Boolean> iff1to2, Function<RewriterStatement.MatchingSubexpression, Boolean> iff2to1, List<Tuple2<RewriterStatement, BiConsumer<RewriterStatement, RewriterStatement.MatchingSubexpression>>> apply1To2, List<Tuple2<RewriterStatement, BiConsumer<RewriterStatement, RewriterStatement.MatchingSubexpression>>> apply2To1, Consumer<RewriterStatement> postProcessor) {
		this.ctx = ctx;
		this.name = name;
		this.fromRoot = fromRoot;
		this.toRoot = toRoot;
		this.unidirectional = unidirectional;
		this.linksStmt1ToStmt2 = linksStmt1ToStmt2;
		this.linksStmt2ToStmt1 = linksStmt2ToStmt1;
		this.iff1to2 = iff1to2;
		this.iff2to1 = iff2to1;
		this.applyStmt1ToStmt2 = apply1To2;
		this.applyStmt2ToStmt1 = apply2To1;
		this.postProcessor = postProcessor;
	}

	// Determine if this rule can universally be applied or only in some conditions (e.g. certain dimensions / sparsity)
	public boolean determineConditionalApplicability() {
		RewriterAssertions assertions = new RewriterAssertions(ctx);
		RewriterAssertionUtils.buildImplicitAssertion(fromRoot, assertions, fromRoot, ctx);
		for (RewriterStatement root : getStmt2AsList())
			RewriterAssertionUtils.buildImplicitAssertion(root, assertions, root, ctx);

		List<Tuple3<List<Number>, Long, Long>> costs = RewriterCostEstimator.compareCosts(fromRoot, getStmt2(), assertions, ctx, false, -1, false);

		requireCostCheck = isConditionalMultiRule() || RewriterCostEstimator.doesHaveAnImpactOnOptimalExpression(costs, false, true, 20);

		if (!requireCostCheck)
			return false;

		List<RewriterStatement> roots = toRoots == null ? List.of(toRoot) : toRoots;

		boolean integrateSparsityInCosts = isConditionalMultiRule() || RewriterCostEstimator.doesHaveAnImpactOnOptimalExpression(costs, true, false, 20);

		MutableObject<RewriterAssertions> assertionRef = new MutableObject<>(assertions);
		fromCost = RewriterCostEstimator.getRawCostFunction(fromRoot, ctx, assertionRef, !integrateSparsityInCosts);
		toCosts = getStmt2AsList().stream().map(root -> RewriterCostEstimator.getRawCostFunction(root, ctx, assertionRef, !integrateSparsityInCosts)).collect(Collectors.toList());

		fromCost = RewriterSparsityEstimator.rollupSparsities(fromCost, RewriterSparsityEstimator.estimateAllNNZ(fromRoot, ctx), ctx);
		toCosts = IntStream.range(0, toCosts.size()).mapToObj(i -> RewriterSparsityEstimator.rollupSparsities(toCosts.get(i), RewriterSparsityEstimator.estimateAllNNZ(roots.get(i), ctx), ctx)).collect(Collectors.toList());

		return requireCostCheck;
	}

	public boolean requiresCostCheck() {
		return requireCostCheck;
	}

	public RewriterStatement getStmt1Cost() {
		return fromCost;
	}

	public RewriterStatement getStmt2Cost() {
		return toCosts.get(0);
	}

	public List<RewriterStatement> getStmt2Costs() {
		return toCosts;
	}

	public void buildCombinedAssertions() {
		combinedAssertions = RewriterAssertionUtils.buildImplicitAssertions(fromRoot, ctx);
		if (toRoot != null)
			RewriterAssertionUtils.buildImplicitAssertions(toRoot, combinedAssertions, ctx);
		else {
			for (RewriterStatement root : toRoots)
				RewriterAssertionUtils.buildImplicitAssertions(root, combinedAssertions, ctx);
		}
	}

	public RewriterAssertions getCombinedAssertions() {
		if (combinedAssertions == null)
			buildCombinedAssertions();

		return combinedAssertions;
	}

	public void setAllowedMultiReferences(Set<RewriterStatement> allowed, boolean allowCombinations) {
		this.allowedMultiReferences = allowed;
		this.allowCombinations = allowCombinations;
	}

	/**
	 *  Overwrites the rule as a conditional rule
	 * @param targets all possible target statements
	 */
	public void setConditional(List<RewriterStatement> targets) {
		toRoots = targets;
	}

	public boolean isConditionalMultiRule() {
		return toRoots != null;
	}

	public List<RewriterStatement> getConditionalMultiRuleTargets() {
		return toRoots;
	}

	public String getName() {
		return name;
	}

	public RewriterStatement getStmt1() {
		return fromRoot;
	}

	/**
	 * Returns the target statement.
	 * @return the target statement; in case of a multi-rule, this will return the first option
	 */
	public RewriterStatement getStmt2() {
		return toRoot != null ? toRoot : toRoots.get(0);
	}

	public List<RewriterStatement> getStmt2AsList() {
		return toRoot != null ? List.of(toRoot) : toRoots;
	}

	public boolean isUnidirectional() {
		return unidirectional;
	}

	public HashMap<RewriterStatement, LinkObject> getForwardLinks() {
		return linksStmt1ToStmt2;
	}

	public HashMap<RewriterStatement, LinkObject> getBackwardLinks() {
		return linksStmt2ToStmt1;
	}

	public RewriterStatement apply(RewriterStatement.MatchingSubexpression match, RewriterStatement rootNode, boolean forward, boolean inplace) {
		return apply(match, rootNode, forward, inplace, false);
	}

	public RewriterStatement apply(RewriterStatement.MatchingSubexpression match, RewriterStatement rootNode, boolean forward, boolean inplace, boolean updateTypes) {
		return forward ? applyForward(match, rootNode, inplace, updateTypes) : applyBackward(match, rootNode, inplace, updateTypes);
	}

	public RewriterStatement applyForward(RewriterStatement.MatchingSubexpression match, RewriterStatement rootNode, boolean inplace, boolean updateTypes) {
		return applyForward(match, rootNode, inplace, updateTypes, new MutableObject<>(null));
	}

	public RewriterStatement applyForward(RewriterStatement.MatchingSubexpression match, RewriterStatement rootNode, boolean inplace, boolean updateTypes, MutableObject<Tuple3<RewriterStatement, RewriterStatement, Integer>> modificationHandle) {
		if (inplace)
			throw new NotImplementedException("Inplace operations have been removed");
		RewriterStatement out = apply(match, rootNode, toRoot, modificationHandle, applyStmt1ToStmt2 == null ? Collections.emptyList() : applyStmt1ToStmt2);
		if (updateTypes)
			updateTypes(out, ctx);
		return out;
	}

	public RewriterStatement applyBackward(RewriterStatement.MatchingSubexpression match, RewriterStatement rootNode, boolean inplace, boolean updateTypes) {
		return applyBackward(match, rootNode, inplace, updateTypes, new MutableObject<>(null));
	}

	public RewriterStatement applyBackward(RewriterStatement.MatchingSubexpression match, RewriterStatement rootNode, boolean inplace, boolean updateTypes, MutableObject<Tuple3<RewriterStatement, RewriterStatement, Integer>> modificationHandle) {
		if (inplace)
			throw new NotImplementedException("Inplace operations have been removed");
		RewriterStatement out = apply(match, rootNode, fromRoot, modificationHandle, applyStmt2ToStmt1 == null ? Collections.emptyList() : applyStmt2ToStmt1);
		if (updateTypes)
			updateTypes(out, ctx);
		return out;
	}

	public RewriterStatement.MatchingSubexpression matchSingleStmt1(RewriterStatement exprRoot, RewriterStatement.RewriterPredecessor pred, RewriterStatement stmt, boolean allowImplicitTypeConversions) {
		RewriterStatement.MatcherContext mCtx = new RewriterStatement.MatcherContext(ctx, stmt, pred, exprRoot, getStmt1(), true, true, false, true, true, false, true, false, false, allowImplicitTypeConversions, linksStmt1ToStmt2);
		mCtx.currentStatement = stmt;
		boolean match = getStmt1().match(mCtx);

		if (match) {
			RewriterStatement.MatchingSubexpression matchExpr = mCtx.toMatch();

			if (iff1to2 == null || iff1to2.apply(matchExpr))
				return matchExpr;
		}

		return null;
	}

	public RewriterStatement.MatchingSubexpression matchSingleStmt2(RewriterStatement exprRoot, RewriterStatement.RewriterPredecessor pred, RewriterStatement stmt, boolean allowImplicitTypeConversions) {
		RewriterStatement.MatcherContext mCtx = new RewriterStatement.MatcherContext(ctx, stmt, pred, exprRoot, getStmt2(), true, true, false, true, true, false, true, false, false, allowImplicitTypeConversions, linksStmt2ToStmt1);
		mCtx.currentStatement = stmt;
		boolean match = getStmt2().match(mCtx);

		if (match) {
			RewriterStatement.MatchingSubexpression matchExpr = mCtx.toMatch();

			if (iff2to1 == null || iff2to1.apply(matchExpr))
				return matchExpr;
		}

		return null;
	}

	public void updateTypes(RewriterStatement root, final RuleContext ctx) {
		root.forEachPostOrder((cur, pred) -> {
			cur.refreshReturnType(ctx);
		}, true);
	}

	private RewriterStatement apply(RewriterStatement.MatchingSubexpression match, RewriterStatement rootInstruction, RewriterStatement dest, MutableObject<Tuple3<RewriterStatement, RewriterStatement, Integer>> modificationHandle, List<Tuple2<RewriterStatement, BiConsumer<RewriterStatement, RewriterStatement.MatchingSubexpression>>> applyFunction) {
		if (match.getPredecessor().isRoot()) {
			final Map<RewriterStatement, RewriterStatement> createdObjects = new HashMap<>();
			RewriterStatement cpy = dest.nestedCopyOrInject(createdObjects, obj -> {
				RewriterStatement assoc = match.getAssocs().get(obj);
				if (assoc != null) {
					RewriterStatement assocCpy = createdObjects.get(assoc);
					if (assocCpy == null) {
						assocCpy = assoc.nestedCopyOrInject(createdObjects, obj2 -> null);
						createdObjects.put(assoc, assocCpy);
					}

					return assocCpy;
				}

				return null;
			});

			RewriterStatement tmp = cpy.simplify(ctx);
			if (tmp != null)
				cpy = tmp;

			match.setNewExprRoot(cpy);

			RewriterStatement oldRootCpy = createdObjects.get(match.getExpressionRoot());
			RewriterAssertions assertions = null;

			if (oldRootCpy != null) {
				assertions = (RewriterAssertions) oldRootCpy.getMeta("_assertions");
				oldRootCpy.unsafeRemoveMeta("_assertions");
			} else if (match.getExpressionRoot().getMeta("_assertions") != null) {
				assertions = ((RewriterAssertions) match.getExpressionRoot().getMeta("_assertions")).nestedCopyOrInject(createdObjects, (obj, p, pIdx) -> {
					RewriterStatement assoc = match.getAssocs().get(obj);
					if (assoc != null) {
						RewriterStatement assocCpy = createdObjects.get(assoc);
						if (assocCpy == null) {
							assocCpy = assoc.nestedCopyOrInject(createdObjects, obj2 -> null);
							createdObjects.put(assoc, assocCpy);
						}

						return assocCpy;
					}

					return null;
				}, match.getNewExprRoot());
				match.getExpressionRoot().unsafeRemoveMeta("_assertions");
			}

			if (assertions != null) {
				if (!cpy.isLiteral())
					cpy.unsafePutMeta("_assertions", assertions);
			}

			match.getLinks().forEach(lnk -> lnk.newStmt.replaceAll(createdObjects::get));
			match.getLinks().forEach(lnk -> lnk.transferFunction.accept(lnk));
			applyFunction.forEach(t -> t._2.accept(createdObjects.get(t._1), match));

			if (postProcessor != null)
				postProcessor.accept(cpy);

			if (ctx.metaPropagator != null) {
				RewriterStatement mNew = ctx.metaPropagator.apply(cpy);

				if (mNew != cpy) {
					mNew.unsafePutMeta("_assertions", cpy.getMeta("_assertions"));
					cpy.unsafeRemoveMeta("_assertions");
					cpy = mNew;
				}
			}

			cpy.prepareForHashing();
			cpy.recomputeHashCodes(ctx);

			modificationHandle.setValue(new Tuple3<>(cpy, null, -1));

			return cpy;
		}

		final Map<RewriterStatement, RewriterStatement> createdObjects = new HashMap<>();
		RewriterStatement cpy2 = rootInstruction.nestedCopyOrInject(createdObjects, (obj2, parent, pIdx) -> {
			if (obj2.equals(match.getMatchRoot())) {
				RewriterStatement cpy = dest.nestedCopyOrInject(createdObjects, obj -> {
					RewriterStatement assoc = match.getAssocs().get(obj);
					if (assoc != null) {
						RewriterStatement assocCpy = createdObjects.get(assoc);
						if (assocCpy == null) {
							assocCpy = assoc.nestedCopyOrInject(createdObjects, obj3 -> null);
							createdObjects.put(assoc, assocCpy);
						}
						return assocCpy;
					}
					return null;
				});
				createdObjects.put(obj2, cpy);
				modificationHandle.setValue(new Tuple3<>(cpy, parent, pIdx));
				return cpy;
			}
			return null;
		});
		RewriterStatement tmp = cpy2.simplify(ctx);
		if (tmp != null)
			cpy2 = tmp;

		match.setNewExprRoot(cpy2);

		match.getLinks().forEach(lnk -> lnk.newStmt.replaceAll(createdObjects::get));
		cpy2.prepareForHashing();
		match.getLinks().forEach(lnk -> lnk.transferFunction.accept(lnk));
		applyFunction.forEach(t -> t._2.accept(createdObjects.get(t._1), match));

		if (postProcessor != null)
			postProcessor.accept(cpy2);

		if (ctx.metaPropagator != null) {
			RewriterStatement mNew = ctx.metaPropagator.apply(cpy2);

			if (mNew != cpy2) {
				mNew.unsafePutMeta("_assertions", cpy2.getMeta("_assertions"));
				cpy2.unsafeRemoveMeta("_assertions");
				cpy2 = mNew;
			}
		}

		cpy2.prepareForHashing();
		cpy2.recomputeHashCodes(ctx);

		return cpy2;
	}

	public String toString() {
		if (isUnidirectional())
			if (isConditionalMultiRule())
				return fromRoot.toParsableString(ctx) + " => {" + toRoots.stream().map(stmt -> stmt.toParsableString(ctx)).collect(Collectors.joining("; ")) + "}";
			else
				return fromRoot.toParsableString(ctx) + " => " + toRoot.toParsableString(ctx);
		else
			return fromRoot.toParsableString(ctx) + " <=> " + toRoot.toParsableString(ctx);
	}

	public String toParsableString(final RuleContext ctx) {
		Map<String, Set<String>> varDefs = new HashMap<>();
		StringBuilder sb = new StringBuilder();
		Map<RewriterStatement, Integer> refs = new HashMap<>();
		int refIdx = fromRoot.toParsableString(sb, refs, 0, varDefs, allowedMultiReferences, ctx);
		String stmt1 = sb.toString();
		sb = new StringBuilder();
		if (toRoot != null) {
			toRoot.toParsableString(sb, refs, refIdx, varDefs, allowedMultiReferences, ctx);
		} else {
			for (RewriterStatement mToRoot : toRoots) {
				mToRoot.toParsableString(sb, refs, refIdx, varDefs, allowedMultiReferences, ctx);
				sb.append('\n');
			}
		}
		String stmt2 = sb.toString();
		String multiRefDefs = "";

		if (!allowedMultiReferences.isEmpty()) {
			multiRefDefs = "AllowedMultiRefs:" + allowedMultiReferences.stream().map(stmt -> "$" + refs.get(stmt)).collect(Collectors.joining(",")) + "\nAllowCombinations:" + allowCombinations + "\n";
		}

		String defs = RewriterStatement.parsableDefinitions(varDefs);

		if (toRoot != null)
			return multiRefDefs + defs + "\n" + stmt1 + "\n=>\n" + stmt2;
		else
			return multiRefDefs + defs + "\n" + stmt1 + "\n=>\n{\n" + stmt2 + "}";
	}

	public static class LinkObject {
		public List<RewriterStatement> stmt;
		public Consumer<ExplicitLink> transferFunction;

		public LinkObject() {
			stmt = new ArrayList<>(2);
		}

		public LinkObject(List<RewriterStatement> stmt, Consumer<ExplicitLink> transferFunction) {
			this.stmt = stmt;
			this.transferFunction = transferFunction;
		}

		@Override
		public String toString() {
			StringBuilder sb = new StringBuilder();
			for (int i = 0; i < stmt.size(); i++) {
				if (i != 0)
					sb.append(", ");
				sb.append(stmt.get(i));
			}
			return sb.toString();
		}

		@Override
		public boolean equals(Object o) {
			return o instanceof LinkObject && ((LinkObject)o).stmt == stmt;
		}

		@Override
		public int hashCode() {
			return stmt.hashCode();
		}
	}

	public static class ExplicitLink {
		public final RewriterStatement oldStmt;
		public List<RewriterStatement> newStmt;
		public final Consumer<ExplicitLink> transferFunction;

		public ExplicitLink(RewriterStatement oldStmt, List<RewriterStatement> newStmt, Consumer<ExplicitLink> transferFunction) {
			this.oldStmt = oldStmt;
			this.newStmt = new ArrayList<>(newStmt);
			this.transferFunction = transferFunction;
		}
	}
}
