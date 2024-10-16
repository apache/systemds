package org.apache.sysds.hops.rewriter;

import org.apache.commons.collections4.bidimap.DualHashBidiMap;
import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.function.TriFunction;
import org.apache.commons.lang3.mutable.MutableObject;
import org.apache.logging.log4j.util.TriConsumer;
import org.jetbrains.annotations.NotNull;
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


	public static class MatchingSubexpression {
		private final RewriterStatement matchRoot;
		private final RewriterStatement matchParent;
		private final int rootIndex;
		private final Map<RewriterStatement, RewriterStatement> assocs;
		private final List<RewriterRule.ExplicitLink> links;
		public Object shared_data = null;

		public MatchingSubexpression(RewriterStatement matchRoot, RewriterStatement matchParent, int rootIndex, Map<RewriterStatement, RewriterStatement> assocs, List<RewriterRule.ExplicitLink> links) {
			this.matchRoot = matchRoot;
			this.matchParent = matchParent;
			this.assocs = assocs;
			this.rootIndex = rootIndex;
			this.links = links;
		}

		public boolean isRootInstruction() {
			return matchParent == null || matchParent == matchRoot;
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
	}

	public static class MatcherContext {
		final RuleContext ctx;
		final boolean literalsCanBeVariables;
		final boolean ignoreLiteralValues;
		final boolean allowDuplicatePointers;
		final boolean allowPropertyScan;
		final boolean allowTypeHierarchy;
		final boolean terminateOnFirstMatch;
		final Map<RewriterStatement, RewriterRule.LinkObject> ruleLinks;
		RewriterStatement matchRoot;
		RewriterStatement matchParent;
		int matchParentIndex;

		public RewriterStatement currentStatement;

		private HashMap<RewriterStatement, RewriterStatement> dependencyMap;
		private List<RewriterRule.ExplicitLink> links;
		private HashMap<RewriterRule.IdentityRewriterStatement, RewriterStatement> internalReferences;

		private List<MatcherContext> subMatches;

		public MatcherContext(final RuleContext ctx, RewriterStatement matchRoot) {
			this(ctx, matchRoot, false, false, false, false, false, false, Collections.emptyMap());
		}

		public MatcherContext(final RuleContext ctx, RewriterStatement matchRoot, final boolean literalsCanBeVariables, final boolean ignoreLiteralValues, final boolean allowDuplicatePointers, final boolean allowPropertyScan, final boolean allowTypeHierarchy, final boolean terminateOnFirstMatch, final Map<RewriterStatement, RewriterRule.LinkObject> ruleLinks) {
			this.ctx = ctx;
			this.matchRoot = matchRoot;
			this.currentStatement = matchRoot;
			this.literalsCanBeVariables = literalsCanBeVariables;
			this.ignoreLiteralValues = ignoreLiteralValues;
			this.allowDuplicatePointers = allowDuplicatePointers;
			this.allowPropertyScan = allowPropertyScan;
			this.allowTypeHierarchy = allowTypeHierarchy;
			this.terminateOnFirstMatch = terminateOnFirstMatch;
			this.ruleLinks = ruleLinks;
		}

		public MatcherContext(final RuleContext ctx, RewriterStatement matchRoot, RewriterStatement matchParent, int rootIndex, final boolean literalsCanBeVariables, final boolean ignoreLiteralValues, final boolean allowDuplicatePointers, final boolean allowPropertyScan, final boolean allowTypeHierarchy, final boolean terminateOnFirstMatch, final Map<RewriterStatement, RewriterRule.LinkObject> ruleLinks) {
			this.ctx = ctx;
			this.matchRoot = matchRoot;
			this.matchParent = matchParent;
			this.matchParentIndex = rootIndex;
			this.currentStatement = matchRoot;
			this.literalsCanBeVariables = literalsCanBeVariables;
			this.ignoreLiteralValues = ignoreLiteralValues;
			this.allowDuplicatePointers = allowDuplicatePointers;
			this.allowPropertyScan = allowPropertyScan;
			this.allowTypeHierarchy = allowTypeHierarchy;
			this.terminateOnFirstMatch = terminateOnFirstMatch;
			this.ruleLinks = ruleLinks;
		}

		public Map<RewriterStatement, RewriterStatement> getDependencyMap() {
			if (dependencyMap == null)
				dependencyMap = new HashMap<>();
			return dependencyMap;
		}

		public List<RewriterRule.ExplicitLink> getLinks() {
			if (links == null)
				links = new ArrayList<>();
			return links;
		}

		public RewriterStatement findInternalReference(RewriterRule.IdentityRewriterStatement stmt) {
			if (internalReferences == null)
				return null;
			return internalReferences.get(stmt);
		}

		public Map<RewriterRule.IdentityRewriterStatement, RewriterStatement> getInternalReferences() {
			if (internalReferences == null)
				internalReferences = new HashMap<>();
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
			return new MatchingSubexpression(matchRoot, matchParent, matchParentIndex, getDependencyMap(), getLinks());
		}

		public void reset() {
			if (dependencyMap != null)
				dependencyMap.clear();
			if (links != null)
				links.clear();
			if (internalReferences != null)
				internalReferences.clear();
		}

		public MatcherContext createCheckpoint() {
			MatcherContext checkpoint = new MatcherContext(ctx, matchRoot, literalsCanBeVariables, ignoreLiteralValues, allowDuplicatePointers, allowPropertyScan, allowTypeHierarchy, terminateOnFirstMatch, ruleLinks);
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

	public RewriterStatement nestedCopyOrInject(Map<RewriterStatement, RewriterStatement> copiedObjects, TriFunction<RewriterStatement, RewriterStatement, Integer, RewriterStatement> injector) {
		return nestedCopyOrInject(copiedObjects, injector, null, -1);
	}

	public RewriterStatement nestedCopyOrInject(Map<RewriterStatement, RewriterStatement> copiedObjects, Function<RewriterStatement, RewriterStatement> injector) {
		return nestedCopyOrInject(copiedObjects, (el, parent, pIdx) -> injector.apply(el), null, -1);
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
		if (getOperands() != null)
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
	}

	public Object getMeta(String key) {
		if (meta == null)
			return null;

		return meta.get(key);
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
		return toString(RuleContext.currentContext);
	}

	public List<String> toExecutableString(final RuleContext ctx) {
		ArrayList<String> defList = new ArrayList<>();
		prepareDefinitions(ctx, defList, new HashSet<>());
		defList.add(toString(ctx));
		eraseDefinitions();

		return defList;
	}
}
