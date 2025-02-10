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

package org.apache.sysds.hops.rewriter;

import org.apache.commons.collections4.bidimap.DualHashBidiMap;
import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.function.TriFunction;
import org.apache.commons.lang3.mutable.MutableInt;
import org.apache.logging.log4j.util.TriConsumer;
import org.apache.sysds.hops.rewriter.assertions.RewriterAssertions;
import org.apache.sysds.hops.rewriter.estimators.RewriterCostEstimator;
import org.apache.sysds.hops.rewriter.rule.RewriterRule;
import org.apache.sysds.hops.rewriter.utils.StatementUtils;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public abstract class RewriterStatement {
	public static final String META_VARNAME = "_varName";


	protected int rid = 0;
	public int refCtr = 0;
	protected long cost = -2;

	protected HashMap<String, Object> meta = null;


	public static class MatchingSubexpression {
		private final RewriterStatement expressionRoot;
		private final RewriterStatement matchRoot;
		private final RewriterPredecessor pred;
		private final Map<RewriterStatement, RewriterStatement> assocs;
		private final List<RewriterRule.ExplicitLink> links;
		public RewriterStatement newExprRoot;

		public MatchingSubexpression(RewriterStatement expressionRoot, RewriterStatement matchRoot, RewriterPredecessor pred, Map<RewriterStatement, RewriterStatement> assocs, List<RewriterRule.ExplicitLink> links) {
			this.expressionRoot = expressionRoot;
			this.matchRoot = matchRoot;
			this.pred = pred;
			this.assocs = assocs;
			this.links = links;
		}

		public RewriterStatement getExpressionRoot() {
			return expressionRoot;
		}

		public RewriterStatement getMatchRoot() {
			return matchRoot;
		}

		public RewriterPredecessor getPredecessor() {
			return pred;
		}

		public Map<RewriterStatement, RewriterStatement> getAssocs() {
			return assocs;
		}

		public List<RewriterRule.ExplicitLink> getLinks() {
			return links;
		}

		public RewriterStatement getNewExprRoot() {
			return newExprRoot;
		}

		public void setNewExprRoot(RewriterStatement exprRoot) {
			newExprRoot = exprRoot;
		}
	}

	public static class MatcherContext {
		final RuleContext ctx;
		final boolean statementsCanBeVariables;
		final boolean literalsCanBeVariables;
		final boolean ignoreLiteralValues;
		final boolean allowDuplicatePointers;
		final boolean allowPropertyScan;
		final boolean allowTypeHierarchy;
		final boolean terminateOnFirstMatch;
		final boolean findMinimalMismatchRoot;
		final boolean traceVariableEliminations;
		final boolean allowImplicitTypeConversions;
		final Map<RewriterStatement, RewriterRule.LinkObject> ruleLinks;
		final RewriterStatement expressionRoot;
		final RewriterStatement thisExpressionRoot;
		RewriterStatement matchRoot;
		RewriterPredecessor pred;

		public RewriterStatement currentStatement;

		private Map<RewriterStatement, RewriterStatement> dependencyMap;
		private List<RewriterRule.ExplicitLink> links;
		private DualHashBidiMap<RewriterStatement, RewriterStatement> internalReferences;

		private List<MatcherContext> subMatches;
		private Tuple2<RewriterStatement, RewriterStatement> firstMismatch;
		private boolean debug;
		private boolean assertionsFetched = false;
		private RewriterAssertions assertionsThat;
		private RewriterAssertions assertionsThis;
		private Set<RewriterStatement> dontVisitAgain;

		public MatcherContext(final RuleContext ctx, RewriterStatement matchRoot, RewriterStatement expressionRoot, RewriterStatement thisExpressionRoot) {
			this(ctx, matchRoot, expressionRoot, thisExpressionRoot, false, false, false, false, false, false, false, false, false, Collections.emptyMap());
		}

		public MatcherContext(final RuleContext ctx, RewriterStatement matchRoot, RewriterStatement expressionRoot, RewriterStatement thisExpressionRoot, final boolean statementsCanBeVariables, final boolean literalsCanBeVariables, final boolean ignoreLiteralValues, final boolean allowDuplicatePointers, final boolean allowPropertyScan, final boolean allowTypeHierarchy, final boolean terminateOnFirstMatch, final boolean findMinimalMismatchRoot, boolean traceVariableEliminations, final Map<RewriterStatement, RewriterRule.LinkObject> ruleLinks) {
			this.ctx = ctx;
			this.matchRoot = matchRoot;
			this.pred = new RewriterPredecessor();
			this.expressionRoot = expressionRoot;
			this.thisExpressionRoot = thisExpressionRoot;
			this.statementsCanBeVariables = statementsCanBeVariables;
			this.currentStatement = matchRoot;
			this.literalsCanBeVariables = literalsCanBeVariables;
			this.ignoreLiteralValues = ignoreLiteralValues;
			this.allowDuplicatePointers = allowDuplicatePointers;
			this.allowPropertyScan = allowPropertyScan;
			this.allowTypeHierarchy = allowTypeHierarchy;
			this.terminateOnFirstMatch = terminateOnFirstMatch;
			this.ruleLinks = ruleLinks;
			this.findMinimalMismatchRoot = findMinimalMismatchRoot;
			this.traceVariableEliminations = traceVariableEliminations;
			this.allowImplicitTypeConversions = false;
			this.debug = false;
		}

		public MatcherContext(final RuleContext ctx, RewriterStatement matchRoot, RewriterPredecessor pred, RewriterStatement expressionRoot, RewriterStatement thisExprRoot, final boolean statementsCanBeVariables, final boolean literalsCanBeVariables, final boolean ignoreLiteralValues, final boolean allowDuplicatePointers, final boolean allowPropertyScan, final boolean allowTypeHierarchy, final boolean terminateOnFirstMatch, final boolean findMinimalMismatchRoot, boolean traceVariableEliminations, boolean allowImplicitTypeConversions, final Map<RewriterStatement, RewriterRule.LinkObject> ruleLinks) {
			this.ctx = ctx;
			this.matchRoot = matchRoot;
			this.pred = pred;
			this.expressionRoot = expressionRoot;
			this.thisExpressionRoot = thisExprRoot;
			this.currentStatement = matchRoot;
			this.statementsCanBeVariables = statementsCanBeVariables;
			this.literalsCanBeVariables = literalsCanBeVariables;
			this.ignoreLiteralValues = ignoreLiteralValues;
			this.allowDuplicatePointers = allowDuplicatePointers;
			this.allowPropertyScan = allowPropertyScan;
			this.allowTypeHierarchy = allowTypeHierarchy;
			this.terminateOnFirstMatch = terminateOnFirstMatch;
			this.ruleLinks = ruleLinks;
			this.findMinimalMismatchRoot = findMinimalMismatchRoot;
			this.traceVariableEliminations = traceVariableEliminations;
			this.allowImplicitTypeConversions = allowImplicitTypeConversions;
			this.debug = false;
		}

		private void fetchAssertions() {
			if (!assertionsFetched) {
				assertionsThat = (RewriterAssertions) expressionRoot.getMeta("_assertions");
				assertionsThis = (RewriterAssertions) thisExpressionRoot.getMeta("_assertions");
				assertionsFetched = true;
			}
		}

		public boolean allowsImplicitTypeConversions() {
			return allowImplicitTypeConversions;
		}

		public void dontVisitAgain(RewriterStatement stmt) {
			if (dontVisitAgain == null) {
				dontVisitAgain = new HashSet<>();
			}

			dontVisitAgain.add(stmt);
		}

		public boolean wasVisited(RewriterStatement stmt) {
			if (dontVisitAgain == null)
				return false;

			return dontVisitAgain.contains(stmt);
		}

		public RewriterAssertions getOldAssertionsThat() {
			fetchAssertions();

			return assertionsThat;
		}

		public RewriterAssertions getOldAssertionsThis() {
			fetchAssertions();

			return assertionsThis;
		}

		public Map<RewriterStatement, RewriterStatement> getDependencyMap() {
			if (dependencyMap == null)
				if (allowDuplicatePointers)
					dependencyMap = new HashMap<>();
				else
					dependencyMap = new DualHashBidiMap<RewriterStatement, RewriterStatement>();
			return dependencyMap;
		}

		public List<RewriterRule.ExplicitLink> getLinks() {
			if (links == null)
				links = new ArrayList<>();
			return links;
		}

		public RewriterStatement findInternalReference(RewriterStatement stmt) {
			if (internalReferences == null)
				return null;
			return internalReferences.get(stmt);
		}

		public RewriterStatement findReverseInternalReference(RewriterStatement stmt) {
			if (internalReferences == null)
				return null;
			return internalReferences.getKey(stmt);
		}

		public Map<RewriterStatement, RewriterStatement> getInternalReferences() {
			if (internalReferences == null)
				internalReferences = new DualHashBidiMap<>();
			return internalReferences;
		}

		public List<MatcherContext> getSubMatches() {
			if (subMatches == null)
				return Collections.emptyList();
			return subMatches;
		}

		public boolean hasSubMatches() {
			return subMatches != null && !subMatches.isEmpty();
		}

		public void addSubMatch(MatcherContext matcherContext) {
			if (subMatches == null)
				subMatches = new ArrayList<>();
			subMatches.addAll(matcherContext.getFlattenedSubMatches());
		}

		public List<MatcherContext> getFlattenedSubMatches() {
			if (hasSubMatches())
				return subMatches.stream().flatMap(mCtx -> mCtx.getFlattenedSubMatches().stream()).collect(Collectors.toList());
			return Collections.emptyList();
		}

		public MatchingSubexpression toMatch() {
			return new MatchingSubexpression(expressionRoot, matchRoot, pred, getDependencyMap(), getLinks());
		}

		public void reset() {
			if (dependencyMap != null)
				dependencyMap.clear();
			if (links != null)
				links.clear();
			if (internalReferences != null)
				internalReferences.clear();
		}

		public void setFirstMismatch(RewriterStatement stmt1, RewriterStatement stmt2) {
			firstMismatch = new Tuple2<>(stmt1, stmt2);
		}

		public Tuple2<RewriterStatement, RewriterStatement> getFirstMismatch() {
			return firstMismatch;
		}

		public MatcherContext debug(boolean debug) {
			this.debug = debug;
			return this;
		}

		public boolean match() {
			return thisExpressionRoot.match(this);
		}

		public boolean isDebug() {
			return debug;
		}

		public static MatcherContext exactMatch(final RuleContext ctx, RewriterStatement stmt, RewriterStatement thisExprRoot) {
			return new MatcherContext(ctx, stmt, stmt, thisExprRoot);
		}

		public static MatcherContext exactMatchWithDifferentLiteralValues(final RuleContext ctx, RewriterStatement stmt, RewriterStatement thisExprRoot) {
			return new MatcherContext(ctx, stmt, stmt, thisExprRoot, false, false, true, false, false, false, false, false, false, Collections.emptyMap());
		}

		public static MatcherContext findMinimalDifference(final RuleContext ctx, RewriterStatement stmt, RewriterStatement thisExpressionRoot) {
			return new MatcherContext(ctx, stmt, stmt, thisExpressionRoot, false, false, true, false, false, false, false, true, false, Collections.emptyMap());
		}
	}

	public static final class RewriterPredecessor {
		private final Object obj;
		private final Object meta;

		// Use iff the element is already the root
		public RewriterPredecessor() {
			obj = null;
			meta = null;
		}

		public RewriterPredecessor(RewriterStatement parent, Integer idx) {
			obj = parent;
			meta = idx;
		}

		// Use iff the element is a meta object
		public RewriterPredecessor(RewriterStatement parent, String meta) {
			obj = parent;
			this.meta = meta;
		}

		public RewriterPredecessor(RewriterAssertions assertions, RewriterAssertions.RewriterAssertion assertion) {
			obj = assertions;
			meta = assertion;
		}

		public boolean isOperand() {
			return obj instanceof RewriterStatement && meta instanceof Integer;
		}

		public boolean isRoot() {
			return obj == null && meta == null;
		}

		public boolean isMetaObject() {
			return obj instanceof RewriterStatement && meta instanceof String;
		}

		public boolean isAssertionObject() {
			return obj instanceof RewriterAssertions && meta instanceof RewriterAssertions.RewriterAssertion;
		}

		public RewriterStatement getParent() {
			return (RewriterStatement) obj;
		}

		public RewriterAssertions getAssertions() {
			return (RewriterAssertions) obj;
		}

		public RewriterAssertions.RewriterAssertion getAssertion() {
			return (RewriterAssertions.RewriterAssertion) meta;
		}

		public String getMetaKey() {
			return (String) meta;
		}

		public int getIndex() {
			return (Integer) meta;
		}
	}

	public static enum ReferenceType {
		ROOT, OPERAND, NCOL, NROW, BACKREF, ASSERTION
	}

	public static class RewriterStatementReference {
		public final ReferenceType referenceType;
		public final RewriterStatement stmt;
		public final Object parentRef;
		public final Object ref;

		// TODO: What about root?
		public RewriterStatementReference(ReferenceType type, RewriterStatement stmt, RewriterStatement parentRef) {
			this.referenceType = type;
			this.stmt = stmt;
			this.parentRef = parentRef;
			this.ref = null;
		}

		public RewriterStatementReference(RewriterStatement stmt, RewriterStatement parentRef, int idx) {
			this.referenceType = parentRef == null ? ReferenceType.ROOT : ReferenceType.OPERAND;
			this.stmt = stmt;
			this.parentRef = parentRef;
			this.ref = idx;
		}

		public RewriterStatementReference(RewriterStatement stmt, RewriterAssertions assertions, RewriterAssertions.RewriterAssertion assertion) {
			this.referenceType = ReferenceType.ASSERTION;
			this.stmt = stmt;
			this.parentRef = assertions;
			this.ref = assertion;
		}

		public void replace(RewriterStatement newStmt) {
			switch (referenceType) {
				case ROOT:
					throw new NotImplementedException();
				case OPERAND:
					((RewriterStatement) parentRef).getOperands().set((Integer)ref, newStmt);
					break;
				case NCOL:
					((RewriterStatement) parentRef).unsafePutMeta("ncol", newStmt);
					break;
				case NROW:
					((RewriterStatement) parentRef).unsafePutMeta("nrow", newStmt);
					break;
				case BACKREF:
					((RewriterStatement) parentRef).unsafePutMeta("backRef", newStmt);
					break;
				case ASSERTION:
					((RewriterAssertions) parentRef).replaceAssertionContent(stmt, newStmt, (RewriterAssertions.RewriterAssertion) ref);
					break;
			}
		}
	}

	public abstract String getId();
	public abstract String getResultingDataType(final RuleContext ctx);
	public abstract boolean isLiteral();
	public abstract Object getLiteral();
	public abstract RewriterStatement getLiteralStatement();
	public long intLiteral() {
		return intLiteral(false);
	}
	public abstract long intLiteral(boolean cast);
	public abstract double floatLiteral();
	public abstract boolean boolLiteral();

	public void setLiteral(Object literal) {
		throw new IllegalArgumentException("This class does not support setting literals");
	}
	public abstract RewriterStatement consolidate(final RuleContext ctx);
	public abstract boolean isConsolidated();
	@Deprecated
	public abstract RewriterStatement clone();
	public abstract RewriterStatement copyNode();
	// Performs a nested copy until a condition is met
	public abstract RewriterStatement nestedCopyOrInject(Map<RewriterStatement, RewriterStatement> copiedObjects, TriFunction<RewriterStatement, RewriterStatement, Integer, RewriterStatement> injector, RewriterStatement parent, int pIdx);
	// Returns the new maxRefId
	public abstract int toParsableString(StringBuilder builder, Map<RewriterStatement, Integer> refs, int maxRefId, Map<String, Set<String>> vars, Set<RewriterStatement> forceCreateRefs, final RuleContext ctx);
	public abstract void refreshReturnType(final RuleContext ctx);
	protected abstract void compress(RewriterAssertions assertions);

	public static String parsableDefinitions(Map<String, Set<String>> defs) {
		StringBuilder sb = new StringBuilder();
		defs.forEach((k, v) -> {
			sb.append(k);
			sb.append(':');

			int i = 0;
			for (String varName : v) {
				if (i > 0)
					sb.append(',');

				sb.append(varName);
				i++;
			}

			sb.append('\n');
		});

		return sb.toString();
	}

	public String toParsableString(final RuleContext ctx, Map<String, Set<String>> defs) {
		return toParsableString(ctx, defs, Collections.emptySet());
	}

	public String toParsableString(final RuleContext ctx, Map<String, Set<String>> defs, Set<RewriterStatement> forceCreateRefs) {
		StringBuilder sb = new StringBuilder();
		toParsableString(sb, new HashMap<>(), 0, defs, forceCreateRefs, ctx);
		return sb.toString();
	}

	public String toParsableString(final RuleContext ctx, boolean includeDefinitions) {
		return toParsableString(ctx, includeDefinitions, Collections.emptySet());
	}

	public String toParsableString(final RuleContext ctx, boolean includeDefinitions, Set<RewriterStatement> forceCreateRefs) {
		StringBuilder sb = new StringBuilder();
		HashMap<String, Set<String>> defs = new HashMap<>();
		toParsableString(sb, new HashMap<>(), 0, defs, forceCreateRefs, ctx);

		if (includeDefinitions)
			return parsableDefinitions(defs) + sb;

		return sb.toString();
	}

	public String toParsableString(final RuleContext ctx) {
		return toParsableString(ctx, false);
	}

	public RewriterStatement nestedCopyOrInject(Map<RewriterStatement, RewriterStatement> copiedObjects, TriFunction<RewriterStatement, RewriterStatement, Integer, RewriterStatement> injector) {
		return nestedCopyOrInject(copiedObjects, injector, null, -1);
	}

	public RewriterStatement nestedCopyOrInject(Map<RewriterStatement, RewriterStatement> copiedObjects, Function<RewriterStatement, RewriterStatement> injector) {
		return nestedCopyOrInject(copiedObjects, (el, parent, pIdx) -> injector.apply(el), null, -1);
	}

	public RewriterStatement nestedCopy(boolean copyAssertions) {
		return nestedCopy(copyAssertions, new HashMap<>());
	}

	public RewriterStatement nestedCopy(boolean copyAssertions, Map<RewriterStatement, RewriterStatement> createdObjects) {
		RewriterStatement cpy = nestedCopyOrInject(createdObjects, el -> null);

		if (copyAssertions) {
			RewriterAssertions assertions = (RewriterAssertions) getMeta("_assertions");

			if (assertions != null) {
				cpy.unsafePutMeta("_assertions", RewriterAssertions.copy(assertions, createdObjects, true));
			}
		} else {
			cpy.unsafeRemoveMeta("_assertions");
		}

		return cpy;
	}

	// Returns the root of the matching sub-statement, null if there is no match
	public abstract boolean match(MatcherContext matcherContext);

	public abstract int recomputeHashCodes(boolean recursively, final RuleContext ctx);
	public abstract RewriterStatement simplify(final RuleContext ctx);
	public abstract RewriterStatement as(String id);
	public abstract String toString(final RuleContext ctx);
	public abstract boolean isArgumentList();
	public abstract List<RewriterStatement> getArgumentList();
	public abstract boolean isInstruction();
	public abstract boolean isEClass();
	public abstract String trueInstruction();
	public abstract String trueTypedInstruction(final RuleContext ctx);
	public abstract String trueTypedInstruction(boolean allowImplicitConversions, final RuleContext ctx);
	public abstract int structuralHashCode();
	public abstract RewriterStatement rename(String id);
	public void prepareDefinitions(final RuleContext ctx, final List<String> strDefs, final Set<String> varDefs) {
		if (getMeta(META_VARNAME) != null)
			return;

		if (getOperands() != null)
			getOperands().forEach(op -> op.prepareDefinitions(ctx, strDefs, varDefs));

		if (this instanceof RewriterInstruction) {
			RewriterInstruction self = ((RewriterInstruction) this);
			// Check if it is necessary to define variables
			if (refCtr > 1 || self.trueInstruction().equals("_asVar")) {
				Pattern pattern = Pattern.compile("[a-zA-Z0-9_]+");
				String instr = pattern.matcher(self.getInstr()).matches() ? self.getInstr() : "tmp";
				instr = instr.replace("_", "");
				String varName = "var_" + instr + "_";

				int ctr = 1;
				while (varDefs.contains(varName + ctr))
					ctr++;

				strDefs.add(varName + ctr + " = " + toString(ctx));
				varDefs.add(varName + ctr);
				unsafePutMeta(META_VARNAME, varName + ctr);
			}
		}
	}

	public void eraseDefinitions() {
		unsafeRemoveMeta(META_VARNAME);

		if (getOperands() != null)
			getOperands().forEach(RewriterStatement::eraseDefinitions);
	}

	public List<RewriterStatement> getOperands() {
		return Collections.emptyList();
	}

	public int recomputeHashCodes(final RuleContext ctx) {
		return recomputeHashCodes(true, ctx);
	}

	public void prepareForHashing() {
		resetRefCtrs();
		computeRefCtrs();
		resetIds();
		computeIds(1);
	}

	protected void resetRefCtrs() {
		refCtr = 0;
		if (getOperands() != null)
			getOperands().forEach(RewriterStatement::resetRefCtrs);
	}

	protected void computeRefCtrs() {
		refCtr++;
		if (refCtr < 2 && getOperands() != null)
			getOperands().forEach(RewriterStatement::computeRefCtrs);
	}

	protected void resetIds() {
		rid = 0;
		if (getOperands() != null)
			getOperands().forEach(RewriterStatement::resetIds);
	}

	protected int computeIds(int id) {
		rid = id++;

		if (getOperands() != null) {
			for (RewriterStatement stmt : getOperands())
				id = stmt.computeIds(id);
		}

		return id;
	}

	/**
	 * Traverses the DAG in-order. If nodes with multiple parents exist, those are visited multiple times.
	 * If the function returns false, the sub-DAG of the current node will not be traversed.
	 * @param function test
	 */
	@Deprecated
	public void forEachPreOrderWithDuplicates(Function<RewriterStatement, Boolean> function) {
		if (function.apply(this) && getOperands() != null)
			for (int i = 0; i < getOperands().size(); i++)
				getOperands().get(i).forEachPreOrderWithDuplicates(function);
	}

	public void forEachPreOrder(Function<RewriterStatement, Boolean> function, boolean includeMeta) {
		forEachPreOrder((el, pred) -> function.apply(el), includeMeta);
	}

	public void forEachPreOrder(BiFunction<RewriterStatement, RewriterPredecessor, Boolean> function, boolean includeMeta) {
		forEachPreOrder(function, new HashSet<>(), new RewriterPredecessor(), includeMeta);
	}

	// We will also include metadata
	private void forEachPreOrder(BiFunction<RewriterStatement, RewriterPredecessor, Boolean> function, Set<RewriterStatement> visited, RewriterPredecessor pred, boolean includeMeta) {
		if (!visited.add(this))
			return;

		if (function.apply(this, pred)) {
			for (int i = 0; i < getOperands().size(); i++)
				getOperands().get(i).forEachPreOrder(function, visited, new RewriterPredecessor(this, i), includeMeta);

			if (includeMeta)
				forEachMetaObject((stmt, mPred) -> stmt.forEachPreOrder(function, visited, mPred, includeMeta));
		}
	}

	public void forEachPostOrder(BiConsumer<RewriterStatement, RewriterPredecessor> consumer, boolean includeMeta) {
		forEachPostOrder(consumer, new HashSet<>(), new RewriterPredecessor(), includeMeta);
	}

	private void forEachPostOrder(BiConsumer<RewriterStatement, RewriterPredecessor> consumer, Set<RewriterStatement> visited, RewriterPredecessor pred, boolean includeMeta) {
		if (!visited.add(this))
			return;

		if (getOperands() != null)
			for (int i = 0; i < getOperands().size(); i++)
				getOperands().get(i).forEachPostOrder(consumer, visited, new RewriterPredecessor(this, i), includeMeta);

		if (includeMeta)
			forEachMetaObject((stmt, mPred) -> stmt.forEachPostOrder(consumer, visited, mPred, includeMeta));

		consumer.accept(this, pred);
	}

	@Deprecated
	public void forEachPostOrderWithDuplicates(TriConsumer<RewriterStatement, RewriterStatement, Integer> consumer) {
		forEachPostOrderWithDuplicates(consumer, null, -1);
	}

	@Deprecated
	private void forEachPostOrderWithDuplicates(TriConsumer<RewriterStatement, RewriterStatement, Integer> consumer, RewriterStatement parent, int pIdx) {
		for (int i = 0; i < getOperands().size(); i++)
			getOperands().get(i).forEachPostOrderWithDuplicates(consumer, this, i);

		consumer.accept(this, parent, pIdx);
	}

	public void putMeta(String key, Object value) {
		if (isConsolidated())
			throw new IllegalArgumentException("An instruction cannot be modified after consolidation");

		if (meta == null)
			meta = new HashMap<>();

		meta.put(key, value);
	}

	public void unsafePutMeta(String key, Object value) {
		if (isLiteral())
			throw new UnsupportedOperationException("Cannot put meta for literals");

		if (meta == null)
			meta = new HashMap<>();

		meta.put(key, value);
	}

	public void unsafeRemoveMeta(String key) {
		if (meta == null)
			return;

		meta.remove(key);

		if (meta.isEmpty())
			meta = null;
	}

	public Object getMeta(String key) {
		if (meta == null)
			return null;

		return meta.get(key);
	}

	public long getCost() {
		if (!isInstruction())
			return 0;

		return cost;
	}

	public RewriterAssertions getAssertions(final RuleContext ctx) {
		RewriterAssertions assertions = (RewriterAssertions) getMeta("_assertions");
		if (assertions == null) {
			assertions = new RewriterAssertions(ctx);
			if (!isLiteral()) // Otherwise the assertion object will just be temporary
				unsafePutMeta("_assertions", assertions);
		}

		return assertions;
	}

	public RewriterStatement getNCol() {
		return (RewriterStatement) getMeta("ncol");
	}

	public RewriterStatement getNRow() {
		return (RewriterStatement) getMeta("nrow");
	}

	public RewriterStatement getBackRef() {
		return (RewriterStatement) getMeta("_backRef");
	}

	public RewriterStatement getChild(int index) {
		return getOperands().get(index);
	}

	public RewriterStatement getChild(int... indices) {
		RewriterStatement current = this;

		for (int i = 0; i < indices.length; i++)
			current = current.getOperands().get(indices[i]);

		return current;
	}

	// This can only be called from the root expression to add a new assertion manually
	public RewriterStatement givenThatEqualDimensions(RewriterStatement stmt1, RewriterStatement stmt2, final RuleContext ctx) {
		getAssertions(ctx).addEqualityAssertion(stmt1.getNRow(), stmt2.getNRow(), this);
		getAssertions(ctx).addEqualityAssertion(stmt1.getNCol(), stmt2.getNCol(), this);
		return this;
	}

	// This can only be called from the root expression to add a new assertion manually
	public RewriterStatement givenThatEqual(RewriterStatement stmt1, RewriterStatement stmt2, final RuleContext ctx) {
		return givenThatEqual(stmt1, stmt2, this, ctx);
	}

	public RewriterStatement givenThatEqual(RewriterStatement stmt1, RewriterStatement stmt2, RewriterStatement exprRoot, final RuleContext ctx) {
		getAssertions(ctx).addEqualityAssertion(stmt1, stmt2, exprRoot);
		return this;
	}

	public RewriterStatement recomputeAssertions() {
		RewriterAssertions assertions = (RewriterAssertions) getMeta("_assertions");

		if (assertions != null)
			return assertions.update(this);

		return this;
	}

	public static void transferMeta(RewriterRule.ExplicitLink link) {
		if (link.oldStmt instanceof RewriterInstruction) {
			for (RewriterStatement mNew : link.newStmt) {
				if (mNew instanceof RewriterInstruction &&
						!((RewriterInstruction)mNew).trueInstruction().equals(((RewriterInstruction)link.oldStmt).trueInstruction())) {
					((RewriterInstruction) mNew).unsafeSetInstructionName(((RewriterInstruction)link.oldStmt).trueInstruction());
				}
			}
		}

		if (link.oldStmt.meta != null) {
			link.newStmt.forEach(stmt -> {
				HashMap<String, Object> newMap = new HashMap<>(link.oldStmt.meta);
				stmt.overwriteImplicitMetaObjects(newMap);
				stmt.meta = newMap;
			});
		}
		else
			link.newStmt.forEach(RewriterStatement::cleanupMeta/*stmt.meta = null*/);
	}

	public void moveRootTo(RewriterStatement newRoot) {
		RewriterAssertions assertions = (RewriterAssertions) getMeta("_assertions");

		if (assertions != null && !newRoot.isLiteral())
			newRoot.unsafePutMeta("_assertions", assertions);
	}

	private void overwriteImplicitMetaObjects(Map<String, Object> map) {
		RewriterAssertions assertions = (RewriterAssertions) getMeta("_assertions");
		RewriterStatement ncol = getNCol();
		RewriterStatement nrow = getNRow();
		RewriterStatement backref = getBackRef();

		if (assertions != null)
			map.put("_assertions", assertions);

		if (ncol != null)
			map.put("ncol", ncol);

		if (nrow != null)
			map.put("nrow", nrow);

		if (backref != null)
			map.put("_backRef", backref);
	}

	private void cleanupMeta() {
		if (meta == null)
			return;

		RewriterAssertions assertions = (RewriterAssertions) getMeta("_assertions");
		RewriterStatement ncol = getNCol();
		RewriterStatement nrow = getNRow();
		RewriterStatement backref = getBackRef();

		if (assertions == null && ncol == null && nrow == null && backref == null)
			return;

		meta = new HashMap<>();

		if (assertions != null)
			meta.put("_assertions", assertions);

		if (ncol != null)
			meta.put("ncol", ncol);

		if (nrow != null)
			meta.put("nrow", nrow);

		if (backref != null)
			meta.put("_backRef", ncol);
	}

	@Override
	public String toString() {
		return toString(RuleContext.currentContext);
	}

	public boolean isColVector() {
		RewriterStatement nrow = getNRow();

		if (nrow == null)
			return false;

		if (nrow.isLiteral() && nrow.getLiteral().equals(1L))
			return true;

		if (nrow.isEClass() && nrow.getChild(0).getOperands().stream().anyMatch(el -> el.isLiteral() && el.getLiteral().equals(1L)))
			return true;

		return false;
	}

	public boolean isRowVector() {
		RewriterStatement ncol = getNCol();

		if (ncol == null)
			return false;

		if (ncol.isLiteral() && ncol.getLiteral().equals(1L))
			return true;

		if (ncol.isEClass() && ncol.getChild(0).getOperands().stream().anyMatch(el -> el.isLiteral() && el.getLiteral().equals(1L)))
			return true;

		return false;
	}

	public List<String> toExecutableString(final RuleContext ctx) {
		ArrayList<String> defList = new ArrayList<>();
		prepareDefinitions(ctx, defList, new HashSet<>());
		defList.add(toString(ctx));
		eraseDefinitions();

		return defList;
	}

	public void compress() {
		RewriterAssertions assertions = (RewriterAssertions) getMeta("_assertions");
		this.forEachPostOrder((cur, pred) -> {
			cur.compress(assertions);
		}, true);
	}

	public long getCost(final RuleContext ctx) {
		if (!this.isInstruction())
			return 0;

		if (cost != -2)
			return cost;

		try {
			cost = RewriterCostEstimator.estimateCost(this, ctx);
		} catch (Exception e) {
			cost = -1L;
		}

		return cost;
	}

	// This may create cycles if visited objects are not tracked
	public void forEachMetaObject(BiConsumer<RewriterStatement, RewriterPredecessor> consumer) {
		RewriterStatement backref = getBackRef();
		RewriterAssertions assertions = (RewriterAssertions) getMeta("_assertions");

		if (backref != null)
			consumer.accept(backref, new RewriterPredecessor(this, "_backRef"));
		if (assertions != null)
			assertions.forEachAssertionContents(consumer);
	}

	public void updateMetaObjects(Function<RewriterStatement, RewriterStatement> f) {
		RewriterStatement backref = getBackRef();

		RewriterStatement mNew;

		if (backref != null) {
			mNew = f.apply(backref);

			if (backref != mNew)
				unsafePutMeta("_backRef", backref);
		}

		RewriterAssertions assertions = (RewriterAssertions) getMeta("_assertions");

		if (assertions != null)
			assertions.updateAssertionContents(f);
	}

	protected void nestedCopyOrInjectMetaStatements(Map<RewriterStatement, RewriterStatement> copiedObjects, TriFunction<RewriterStatement, RewriterStatement, Integer, RewriterStatement> injector) {
		if (getNCol() != null) {
			unsafePutMeta("ncol", getNCol().nestedCopyOrInject(copiedObjects, injector, this, -1));
		}

		if (getNRow() != null)
			unsafePutMeta("nrow", getNRow().nestedCopyOrInject(copiedObjects, injector, this, -1));

		RewriterStatement backRef = (RewriterStatement) getMeta("_backRef");

		if (backRef != null)
			unsafePutMeta("_backRef", backRef.nestedCopyOrInject(copiedObjects, injector, this, -1));

		RewriterAssertions assertions = (RewriterAssertions) getMeta("_assertions");

		if (assertions != null) {
			assertions = assertions.nestedCopyOrInject(copiedObjects, injector, this);
			unsafePutMeta("_assertions", assertions);
		}
	}

	// This returns a stream of all children including metadata and assertions if available
	// This may contain loops in case of back references
	public Stream<Tuple2<RewriterStatement, RewriterPredecessor>> allChildren() {
		Stream<Tuple2<RewriterStatement, RewriterPredecessor>> stream = IntStream.range(0, getOperands().size()).mapToObj(i -> new Tuple2<>(getOperands().get(i), new RewriterPredecessor(this, i)));
		RewriterStatement ncol = getNCol();
		RewriterStatement nrow = getNRow();
		RewriterStatement backRef = getBackRef();

		if (ncol != null)
			stream = Stream.concat(stream, Stream.of(new Tuple2<>(ncol, new RewriterPredecessor(this, "ncol"))));
		if (nrow != null)
			stream = Stream.concat(stream, Stream.of(new Tuple2<>(nrow, new RewriterPredecessor(this, "nrow"))));
		if (backRef != null)
			stream = Stream.concat(stream, Stream.of(new Tuple2<>(backRef, new RewriterPredecessor(this, "_backRef"))));

		RewriterAssertions assertions = (RewriterAssertions) getMeta("_assertions");

		if (assertions != null)
			stream = Stream.concat(stream, assertions.streamOfContents());

		return stream;
	}

	public boolean isDataOrigin() {
		if (!isInstruction())
			return true;

		switch (trueInstruction()) {
			case "rowVec":
			case "colVec":
			case "const":
				return true;
		}

		return false;
	}

	public int countInstructions() {
		MutableInt i = new MutableInt();
		forEachPreOrder(cur -> {
			if (!cur.isDataOrigin() || cur.isLiteral()) {
				i.add(1 + cur.getOperands().size());
			}
			return true;
		}, false);
		return i.getAndIncrement();
	}

	public static RewriterStatement argList(final RuleContext ctx, RewriterStatement... args) {
		return new RewriterInstruction().as(UUID.randomUUID().toString()).withInstruction("argList").withOps(args).consolidate(ctx);
	}

	public static RewriterStatement argList(final RuleContext ctx, List<RewriterStatement> args) {
		return argList(ctx, args.toArray(RewriterStatement[]::new));
	}

	public static RewriterStatement castFloat(final RuleContext ctx, RewriterStatement stmt) {
		return new RewriterInstruction().as(UUID.randomUUID().toString()).withInstruction("cast.FLOAT").withOps(stmt).consolidate(ctx);
	}

	public static RewriterStatement nnz(RewriterStatement of, final RuleContext ctx) {
		return nnz(of, ctx, false);
	}

	public static RewriterStatement nnz(RewriterStatement of, final RuleContext ctx, boolean treatAsDense) {
		if (treatAsDense)
			return StatementUtils.length(ctx, of);
		return new RewriterInstruction().as(UUID.randomUUID().toString()).withInstruction("_nnz").withOps(of).consolidate(ctx);
	}

	public static RewriterStatement literal(final RuleContext ctx, Object literal) {
		if (literal == null)
			throw new IllegalArgumentException();

		if (literal instanceof Double) { // We need to differentiate between -0.0 and 0.0 because otherwise this may leed to bugs
			return new RewriterDataType().as(literal.toString()).ofType("FLOAT").asLiteral(((Double) literal).doubleValue() == -0.0 ? 0.0 : literal).consolidate(ctx);
		} else if (literal instanceof Long) {
			return new RewriterDataType().as(literal.toString()).ofType("INT").asLiteral(literal).consolidate(ctx);
		} else if (literal instanceof Boolean)  {
			return new RewriterDataType().as(literal.toString()).ofType("BOOL").asLiteral(literal).consolidate(ctx);
		}

		throw new IllegalArgumentException();
	}

	public static RewriterStatement multiArgInstr(final RuleContext ctx, String instrName, RewriterStatement... ops) {
		RewriterStatement argList = new RewriterInstruction().as(UUID.randomUUID().toString()).withInstruction("argList").withOps(ops).consolidate(ctx);
		return new RewriterInstruction().as(UUID.randomUUID().toString()).withInstruction(instrName).withOps(argList).consolidate(ctx);
	}
}
