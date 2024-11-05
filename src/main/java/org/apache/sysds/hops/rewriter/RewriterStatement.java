package org.apache.sysds.hops.rewriter;

import org.apache.commons.collections4.bidimap.DualHashBidiMap;
import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.function.TriFunction;
import org.apache.commons.lang3.mutable.MutableObject;
import org.apache.logging.log4j.util.TriConsumer;
import org.jetbrains.annotations.NotNull;
import scala.Tuple2;
import spire.macros.CheckedRewriter;

import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Stack;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public abstract class RewriterStatement implements Comparable<RewriterStatement> {
	public static final String META_VARNAME = "_varName";


	protected int rid = 0;
	protected int refCtr = 0;

	protected HashMap<String, Object> meta = null;

	static RewriterStatementLink resolveNode(RewriterStatementLink link, DualHashBidiMap<RewriterStatementLink, RewriterStatementLink> links) {
		if (links == null)
			return link;

		RewriterStatementLink next = links.getOrDefault(link, link);
		while (!next.equals(link)) {
			link = next;
			next = links.getOrDefault(next, next);
		}
		return next;
	}

	static void insertLinks(DualHashBidiMap<RewriterStatementLink, RewriterStatementLink> links, Map<RewriterStatementLink, RewriterStatementLink> inserts) {
		inserts.forEach((key, value) -> insertLink(links, key, value));
	}

	static void insertLink(DualHashBidiMap<RewriterStatementLink, RewriterStatementLink> links, RewriterStatementLink key, RewriterStatementLink value) {
		RewriterStatementLink origin = links.removeValue(key);
		RewriterStatementLink dest = links.remove(value);
		origin = origin != null ? origin : key;
		dest = dest != null ? dest : value;

		//System.out.println(" + " + origin.stmt.toStringWithLinking(links) + " -> " + dest.stmt.toStringWithLinking(links));

		if (origin != dest)
			links.put(origin, dest);
	}

	/*private static final Map<Object, RewriterStatement> allLiterals = new ConcurrentHashMap<>();

	public static RewriterStatement newLiteral(Object literal, final RuleContext ctx) {
		RewriterStatement mLiteral = allLiterals.get(literal);
		if (mLiteral != null)
			return mLiteral;

		String type;
		if (literal instanceof Long)
			type = "INT";
		else if (literal instanceof Double)
			type = "FLOAT";
		else if (literal instanceof Boolean)
			type = "BOOL";
		else
			throw new IllegalArgumentException();

		RewriterStatement stmt = new RewriterDataType().as(UUID.randomUUID().toString()).ofType(type).asLiteral(literal).consolidate(ctx);
		allLiterals.put(literal, stmt);

		return stmt;
	}*/


	public static class MatchingSubexpression {
		private final RewriterStatement expressionRoot;
		private final RewriterStatement matchRoot;
		private final RewriterStatement matchParent;
		private final int rootIndex;
		private final Map<RewriterStatement, RewriterStatement> assocs;
		private final List<RewriterRule.ExplicitLink> links;
		public RewriterStatement newExprRoot;

		public MatchingSubexpression(RewriterStatement expressionRoot, RewriterStatement matchRoot, RewriterStatement matchParent, int rootIndex, Map<RewriterStatement, RewriterStatement> assocs, List<RewriterRule.ExplicitLink> links) {
			this.expressionRoot = expressionRoot;
			this.matchRoot = matchRoot;
			this.matchParent = matchParent;
			this.assocs = assocs;
			this.rootIndex = rootIndex;
			this.links = links;
		}

		public boolean isRootInstruction() {
			return matchParent == null || matchParent == matchRoot;
		}

		public RewriterStatement getExpressionRoot() {
			return expressionRoot;
		}

		public RewriterStatement getMatchRoot() {
			return matchRoot;
		}

		public RewriterStatement getMatchParent() {
			return matchParent;
		}

		public int getRootIndex() {
			return rootIndex;
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
		final Map<RewriterStatement, RewriterRule.LinkObject> ruleLinks;
		final RewriterStatement expressionRoot;
		RewriterStatement matchRoot;
		RewriterStatement matchParent;
		int matchParentIndex;

		public RewriterStatement currentStatement;

		private Map<RewriterStatement, RewriterStatement> dependencyMap;
		private List<RewriterRule.ExplicitLink> links;
		private DualHashBidiMap<RewriterStatement, RewriterStatement> internalReferences;

		private List<MatcherContext> subMatches;
		private Tuple2<RewriterStatement, RewriterStatement> firstMismatch;
		private boolean debug;

		public MatcherContext(final RuleContext ctx, RewriterStatement matchRoot, RewriterStatement expressionRoot) {
			this(ctx, matchRoot, expressionRoot, false, false, false, false, false, false, false, false, Collections.emptyMap());
		}

		public MatcherContext(final RuleContext ctx, RewriterStatement matchRoot, RewriterStatement expressionRoot, final boolean statementsCanBeVariables, final boolean literalsCanBeVariables, final boolean ignoreLiteralValues, final boolean allowDuplicatePointers, final boolean allowPropertyScan, final boolean allowTypeHierarchy, final boolean terminateOnFirstMatch, final boolean findMinimalMismatchRoot, final Map<RewriterStatement, RewriterRule.LinkObject> ruleLinks) {
			this.ctx = ctx;
			this.matchRoot = matchRoot;
			this.expressionRoot = expressionRoot;
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
			this.debug = false;
		}

		public MatcherContext(final RuleContext ctx, RewriterStatement matchRoot, RewriterStatement matchParent, int rootIndex, RewriterStatement expressionRoot, final boolean statementsCanBeVariables, final boolean literalsCanBeVariables, final boolean ignoreLiteralValues, final boolean allowDuplicatePointers, final boolean allowPropertyScan, final boolean allowTypeHierarchy, final boolean terminateOnFirstMatch, final boolean findMinimalMismatchRoot, final Map<RewriterStatement, RewriterRule.LinkObject> ruleLinks) {
			this.ctx = ctx;
			this.matchRoot = matchRoot;
			this.matchParent = matchParent;
			this.matchParentIndex = rootIndex;
			this.expressionRoot = expressionRoot;
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
			this.debug = false;
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
			return new MatchingSubexpression(expressionRoot, matchRoot, matchParent, matchParentIndex, getDependencyMap(), getLinks());
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

		/*public MatcherContext createCheckpoint() {
			MatcherContext checkpoint = new MatcherContext(ctx, matchRoot, statementsCanBeVariables, literalsCanBeVariables, ignoreLiteralValues, allowDuplicatePointers, allowPropertyScan, allowTypeHierarchy, terminateOnFirstMatch, ruleLinks);
			checkpoint.matchParent = matchParent;
			checkpoint.matchParentIndex = matchParentIndex;
			if (dependencyMap != null)
				checkpoint.dependencyMap = new HashMap<>(dependencyMap);
			if (links != null)
				checkpoint.links = new ArrayList<>(links);
			if (internalReferences != null)
				checkpoint.internalReferences = new HashMap<>(internalReferences);
			if (subMatches != null)
				checkpoint.subMatches = new ArrayList<>(subMatches);
			return checkpoint;
		}*/

		public MatcherContext debug(boolean debug) {
			this.debug = debug;
			return this;
		}

		public boolean isDebug() {
			return debug;
		}

		public static MatcherContext exactMatch(final RuleContext ctx, RewriterStatement stmt) {
			return new MatcherContext(ctx, stmt, stmt);
		}

		public static MatcherContext exactMatchWithDifferentLiteralValues(final RuleContext ctx, RewriterStatement stmt) {
			return new MatcherContext(ctx, stmt, stmt, false, false, true, false, false, false, false, false, Collections.emptyMap());
		}

		public static MatcherContext findMinimalDifference(final RuleContext ctx, RewriterStatement stmt) {
			return new MatcherContext(ctx, stmt, stmt, false, false, true, false, false, false, false, true, Collections.emptyMap());
		}
	}

	public abstract String getId();
	public abstract String getResultingDataType(final RuleContext ctx);
	public abstract boolean isLiteral();
	public abstract Object getLiteral();

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
	abstract int toParsableString(StringBuilder builder, Map<RewriterRule.IdentityRewriterStatement, Integer> refs, int maxRefId, Map<String, Set<String>> vars, final RuleContext ctx);
	abstract void refreshReturnType(final RuleContext ctx);

	public String toParsableString(final RuleContext ctx, boolean includeDefinitions) {
		StringBuilder sb = new StringBuilder();
		HashMap<String, Set<String>> defs = new HashMap<>();
		toParsableString(sb, new HashMap<>(), 0, defs, ctx);

		if (includeDefinitions) {
			StringBuilder newSB = new StringBuilder();
			defs.forEach((k, v) -> {
				newSB.append(k);
				newSB.append(':');

				int i = 0;
				for (String varName : v) {
					if (i > 0)
						newSB.append(',');

					newSB.append(varName);
					i++;
				}

				newSB.append('\n');
			});

			newSB.append(sb);
			return newSB.toString();
		}

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

	public RewriterStatement nestedCopy() {
		return nestedCopyOrInject(new HashMap<>(), el -> null);
	}
	//String toStringWithLinking(int dagId, DualHashBidiMap<RewriterStatementLink, RewriterStatementLink> links);

	// Returns the root of the matching sub-statement, null if there is no match
	public abstract boolean match(MatcherContext matcherContext);

	/*public boolean match(final RuleContext ctx, RewriterStatement stmt, HashMap<RewriterStatement, RewriterStatement> dependencyMap, boolean literalsCanBeVariables, boolean ignoreLiteralValues, List<RewriterRule.ExplicitLink> links, final Map<RewriterStatement, RewriterRule.LinkObject> ruleLinks, boolean allowDuplicatePointers, boolean allowPropertyScan, boolean allowTypeHierarchy) {
		return match(new MatcherContext(ctx, stmt, dependencyMap, literalsCanBeVariables, ignoreLiteralValues, links, ruleLinks, allowDuplicatePointers, allowPropertyScan, allowTypeHierarchy, new HashMap<>()));
	}*/

	public abstract int recomputeHashCodes(boolean recursively, final RuleContext ctx);
	public abstract long getCost();
	public abstract RewriterStatement simplify(final RuleContext ctx);
	public abstract RewriterStatement as(String id);
	public abstract String toString(final RuleContext ctx);
	public abstract boolean isArgumentList();
	public abstract List<RewriterStatement> getArgumentList();
	public abstract boolean isInstruction();
	public abstract String trueInstruction();
	public abstract String trueTypedInstruction(final RuleContext ctx);
	public abstract int structuralHashCode();
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

	// TODO: Rework if necessary
	public boolean matchSubexpr(MatcherContext ctx, List<MatchingSubexpression> matches, Function<MatchingSubexpression, Boolean> iff) {
		/*

		ctx.reset();
		boolean foundMatch = match(ctx);
		//boolean foundMatch = match(ctx, root, dependencyMap, literalsCanBeVariables, ignoreLiteralValues, links, ruleLinks, allowDuplicatePointers, allowPropertyScan, allowTypeHierarchy);

		if (foundMatch) {
			MatchingSubexpression match = ctx.toMatch();
			if (iff == null || iff.apply(match)) {
				matches.add(match);

				if (ctx.terminateOnFirstMatch)
					return true;
			} else {
				foundMatch = false;
			}
		}

		int idx = 0;

		if (ctx.matchRoot.getOperands() != null && ctx.matchRoot instanceof RewriterInstruction) {
			for (RewriterStatement stmt : ctx.matchRoot.getOperands()) {
				ctx.matchRoot = stmt;
				if (matchSubexpr(ctx, matches, iff)) {
					//TODO
					foundMatch = true;

					if (findFirst)
						return true;
				}
				idx++;
			}
		}

		return foundMatch;*/
		throw new NotImplementedException();
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
		/*if (isArgumentList())
			return;*/
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
		/*if (rid != 0 || isArgumentList())
			return id;*/

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
	public void forEachPreOrderWithDuplicates(Function<RewriterStatement, Boolean> function) {
		if (function.apply(this) && getOperands() != null)
			for (int i = 0; i < getOperands().size(); i++)
				getOperands().get(i).forEachPreOrderWithDuplicates(function);
	}

	public void forEachPreOrder(Function<RewriterStatement, Boolean> function) {
		forEachPreOrder((el, p, pIdx) -> function.apply(el));
	}

	public void forEachPreOrder(TriFunction<RewriterStatement, RewriterStatement, Integer, Boolean> function) {
		forEachPreOrder(function, new HashSet<>(), null, -1);
	}

	private void forEachPreOrder(TriFunction<RewriterStatement, RewriterStatement, Integer, Boolean> function, Set<RewriterRule.IdentityRewriterStatement> visited, RewriterStatement parent, int rootIdx) {
		if (!visited.add(new RewriterRule.IdentityRewriterStatement(this)))
			return;

		if (function.apply(this, parent, rootIdx) && getOperands() != null)
			for (int i = 0; i < getOperands().size(); i++)
				getOperands().get(i).forEachPreOrder(function, visited, this, i);
	}

	public void forEachPostOrder(TriConsumer<RewriterStatement, RewriterStatement, Integer> consumer) {
		forEachPostOrder(consumer, new HashSet<>(), null, -1);
	}

	private void forEachPostOrder(TriConsumer<RewriterStatement, RewriterStatement, Integer> consumer, Set<RewriterRule.IdentityRewriterStatement> visited, RewriterStatement parent, int rootIdx) {
		if (!visited.add(new RewriterRule.IdentityRewriterStatement(this)))
			return;

		if (getOperands() != null)
			for (int i = 0; i < getOperands().size(); i++)
				getOperands().get(i).forEachPostOrder(consumer, visited, this, i);

		consumer.accept(this, parent, rootIdx);
	}

	public void forEachPostOrderWithDuplicates(TriConsumer<RewriterStatement, RewriterStatement, Integer> consumer) {
		forEachPostOrderWithDuplicates(consumer, null, -1);
	}

	private void forEachPostOrderWithDuplicates(TriConsumer<RewriterStatement, RewriterStatement, Integer> consumer, RewriterStatement parent, int pIdx) {
		for (int i = 0; i < getOperands().size(); i++)
			getOperands().get(i).forEachPostOrderWithDuplicates(consumer, this, i);

		consumer.accept(this, parent, pIdx);
	}

	@Override
	public int compareTo(@NotNull RewriterStatement o) {
		return Long.compare(getCost(), o.getCost());
	}

	public void putMeta(String key, Object value) {
		if (isConsolidated())
			throw new IllegalArgumentException("An instruction cannot be modified after consolidation");

		if (meta == null)
			meta = new HashMap<>();

		meta.put(key, value);
	}

	public void unsafePutMeta(String key, Object value) {
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

	public RewriterAssertions getAssertions(final RuleContext ctx) {
		RewriterAssertions assertions = (RewriterAssertions) getMeta("_assertions");
		if (assertions == null) {
			assertions = new RewriterAssertions(ctx);
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
	public RewriterStatement givenThatEqual(RewriterStatement stmt1, RewriterStatement stmt2, final RuleContext ctx) {
		getAssertions(ctx).addEqualityAssertion(stmt1, stmt2);
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

		if (link.oldStmt.meta != null)
			link.newStmt.forEach(stmt -> stmt.meta = new HashMap<>(link.oldStmt.meta));
		else
			link.newStmt.forEach(stmt -> stmt.meta = null);
	}

	@Override
	public String toString() {
		String str = toString(RuleContext.currentContext);
		return str;
	}

	public List<String> toExecutableString(final RuleContext ctx) {
		ArrayList<String> defList = new ArrayList<>();
		prepareDefinitions(ctx, defList, new HashSet<>());
		defList.add(toString(ctx));
		eraseDefinitions();

		return defList;
	}

	protected void nestedCopyOrInjectMetaStatements(Map<RewriterStatement, RewriterStatement> copiedObjects, TriFunction<RewriterStatement, RewriterStatement, Integer, RewriterStatement> injector) {
		if (getNCol() != null) {
			//RewriterStatement oldNCol = getNCol();
			//RewriterStatement newNCol = oldNCol.nestedCopyOrInject(copiedObjects, injector, this, -1);
			unsafePutMeta("ncol", getNCol().nestedCopyOrInject(copiedObjects, injector, this, -1));
			//System.out.println("Copied meta: " + oldNCol + " => " + getNCol().toString() + " (from " + this + ")");
		}

		if (getNRow() != null)
			unsafePutMeta("nrow", getNRow().nestedCopyOrInject(copiedObjects, injector, this, -1));

		RewriterStatement backRef = (RewriterStatement) getMeta("_backRef");

		if (backRef != null)
			unsafePutMeta("_backRef", backRef.nestedCopyOrInject(copiedObjects, injector, this, -1));
	}
}
