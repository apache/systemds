package org.apache.sysds.hops.rewriter;

import com.google.common.collect.Sets;
import org.apache.commons.collections4.bidimap.DualHashBidiMap;
import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.mutable.MutableObject;
import org.apache.hadoop.yarn.webapp.hamlet2.Hamlet;
import org.apache.spark.sql.catalyst.expressions.Exp;
import scala.Tuple2;
import scala.Tuple3;
import scala.reflect.internal.Trees;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;

public class RewriterRule extends AbstractRewriterRule {

	private final RuleContext ctx;
	private final String name;
	private final RewriterStatement fromRoot;
	private final RewriterStatement toRoot;
	private final HashMap<RewriterStatement, LinkObject> linksStmt1ToStmt2; // Contains the explicit links a transformation has (like instructions, (a+b)-c = a+(b-c), but '+' and '-' are the same instruction still [important if instructions have metadata])
	private final HashMap<RewriterStatement, LinkObject> linksStmt2ToStmt1;
	private final List<Tuple2<RewriterStatement, BiConsumer<RewriterStatement, RewriterStatement.MatchingSubexpression>>> applyStmt1ToStmt2;
	private final List<Tuple2<RewriterStatement, BiConsumer<RewriterStatement, RewriterStatement.MatchingSubexpression>>> applyStmt2ToStmt1;
	private final Function<RewriterStatement.MatchingSubexpression, Boolean> iff1to2;
	private final Function<RewriterStatement.MatchingSubexpression, Boolean> iff2to1;
	private final boolean unidirectional;
	private final Consumer<RewriterStatement> postProcessor;

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

	public String getName() {
		return name;
	}

	public RewriterStatement getStmt1() {
		return fromRoot;
	}

	public RewriterStatement getStmt2() {
		return toRoot;
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

	public RewriterStatement apply(RewriterStatement.MatchingSubexpression match, RewriterInstruction rootNode, boolean forward, boolean inplace) {
		return forward ? applyForward(match, rootNode, inplace) : applyBackward(match, rootNode, inplace);
	}

	public RewriterStatement applyForward(RewriterStatement.MatchingSubexpression match, RewriterStatement rootNode, boolean inplace) {
		return applyForward(match, rootNode, inplace, new MutableObject<>(null));
	}

	public RewriterStatement applyForward(RewriterStatement.MatchingSubexpression match, RewriterStatement rootNode, boolean inplace, MutableObject<Tuple3<RewriterStatement, RewriterStatement, Integer>> modificationHandle) {
		return inplace ? applyInplace(match, rootNode, toRoot, applyStmt1ToStmt2 == null ? Collections.emptyList() : applyStmt1ToStmt2) : apply(match, rootNode, toRoot, modificationHandle, applyStmt1ToStmt2 == null ? Collections.emptyList() : applyStmt1ToStmt2);
	}

	public RewriterStatement applyBackward(RewriterStatement.MatchingSubexpression match, RewriterStatement rootNode, boolean inplace) {
		return applyBackward(match, rootNode, inplace, new MutableObject<>(null));
	}

	public RewriterStatement applyBackward(RewriterStatement.MatchingSubexpression match, RewriterStatement rootNode, boolean inplace, MutableObject<Tuple3<RewriterStatement, RewriterStatement, Integer>> modificationHandle) {
		return inplace ? applyInplace(match, rootNode, fromRoot, applyStmt2ToStmt1 == null ? Collections.emptyList() : applyStmt2ToStmt1) : apply(match, rootNode, fromRoot, modificationHandle, applyStmt2ToStmt1 == null ? Collections.emptyList() : applyStmt2ToStmt1);
	}

	/*@Override
	public boolean matchStmt1(RewriterStatement stmt, ArrayList<RewriterStatement.MatchingSubexpression> arr, boolean findFirst) {
		return getStmt1().matchSubexpr(ctx, stmt, null, -1, arr, new HashMap<>(), true, false, findFirst, null, linksStmt1ToStmt2, true, true, false, iff1to2);
	}*/

	public RewriterStatement.MatchingSubexpression matchSingleStmt1(RewriterInstruction parent, int rootIndex, RewriterStatement stmt, HashMap<RewriterStatement, RewriterStatement> dependencyMap, List<ExplicitLink> links, Map<RewriterStatement, LinkObject> ruleLinks) {
		RewriterStatement.MatcherContext mCtx = new RewriterStatement.MatcherContext(ctx, stmt, parent, rootIndex, true, false, true, true, false, true, linksStmt1ToStmt2);
		mCtx.currentStatement = stmt;
		boolean match = getStmt1().match(mCtx);

		if (match) {
			RewriterStatement.MatchingSubexpression matchExpr = mCtx.toMatch();

			if (iff1to2 == null || iff1to2.apply(matchExpr))
				return matchExpr;
		}

		return null;
	}

	/*@Override
	public boolean matchStmt2(RewriterStatement stmt, ArrayList<RewriterStatement.MatchingSubexpression> arr, boolean findFirst) {
		return getStmt2().matchSubexpr(ctx, stmt, null, -1, arr, new HashMap<>(), true, false, findFirst, null, linksStmt2ToStmt1, true, true, false, iff2to1);
	}*/

	public RewriterStatement.MatchingSubexpression matchSingleStmt2(RewriterInstruction parent, int rootIndex, RewriterStatement stmt, HashMap<RewriterStatement, RewriterStatement> dependencyMap, List<ExplicitLink> links, Map<RewriterStatement, LinkObject> ruleLinks) {
		RewriterStatement.MatcherContext mCtx = new RewriterStatement.MatcherContext(ctx, stmt, parent, rootIndex, true, false, true, true, false, true, linksStmt2ToStmt1);
		mCtx.currentStatement = stmt;
		boolean match = getStmt2().match(mCtx);

		if (match) {
			RewriterStatement.MatchingSubexpression matchExpr = mCtx.toMatch();

			if (iff2to1 == null || iff2to1.apply(matchExpr))
				return matchExpr;
		}

		return null;
	}

	// TODO: Give the possibility to get a handle to the parent and root of the replaced sub-DAG
	private RewriterStatement apply(RewriterStatement.MatchingSubexpression match, RewriterStatement rootInstruction, RewriterStatement dest, MutableObject<Tuple3<RewriterStatement, RewriterStatement, Integer>> modificationHandle, List<Tuple2<RewriterStatement, BiConsumer<RewriterStatement, RewriterStatement.MatchingSubexpression>>> applyFunction) {
		if (match.getMatchParent() == null || match.getMatchParent() == match.getMatchRoot()) {
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

			match.getLinks().forEach(lnk -> lnk.newStmt.replaceAll(createdObjects::get));
			match.getLinks().forEach(lnk -> lnk.transferFunction.accept(lnk));
			applyFunction.forEach(t -> t._2.accept(createdObjects.get(t._1), match));

			if (postProcessor != null)
				postProcessor.accept(cpy);

			if (ctx.metaPropagator != null)
				cpy = ctx.metaPropagator.apply(cpy);

			cpy.prepareForHashing();
			cpy.recomputeHashCodes();

			modificationHandle.setValue(new Tuple3<>(cpy, null, -1));

			return cpy;
		}

		final Map<RewriterStatement, RewriterStatement> createdObjects = new HashMap<>();
		RewriterStatement cpy2 = rootInstruction.nestedCopyOrInject(createdObjects, (obj2, parent, pIdx) -> {
			if (obj2 == match.getMatchRoot()) {
				RewriterStatement cpy = dest.nestedCopyOrInject(createdObjects, obj -> {
					RewriterStatement assoc = match.getAssocs().get(obj);
					/*for (Map.Entry<RewriterStatement, RewriterStatement> mAssoc : match.getAssocs().entrySet())
						System.out.println(mAssoc.getKey() + " -> " + mAssoc.getValue());*/
					if (assoc != null) {
						RewriterStatement assocCpy = createdObjects.get(assoc);
						if (assocCpy == null) {
							assocCpy = assoc.nestedCopyOrInject(createdObjects, obj3 -> null);
							createdObjects.put(assoc, assocCpy);
						}
						return assocCpy;
					}
					//System.out.println("ObjInner: " + obj);
					return null;
				});
				createdObjects.put(obj2, cpy);
				modificationHandle.setValue(new Tuple3<>(cpy, parent, pIdx));
				return cpy;
			}
			//System.out.println("Obj: " + obj2);
			return null;
		});
		RewriterStatement tmp = cpy2.simplify(ctx);
		if (tmp != null)
			cpy2 = tmp;

		match.getLinks().forEach(lnk -> lnk.newStmt.replaceAll(createdObjects::get));
		match.getLinks().forEach(lnk -> lnk.transferFunction.accept(lnk));
		applyFunction.forEach(t -> t._2.accept(createdObjects.get(t._1), match));

		if (postProcessor != null)
			postProcessor.accept(cpy2);

		if (ctx.metaPropagator != null)
			cpy2 = ctx.metaPropagator.apply(cpy2);

		cpy2.prepareForHashing();
		cpy2.recomputeHashCodes();
		return cpy2;
	}

	private RewriterStatement applyInplace(RewriterStatement.MatchingSubexpression match, RewriterStatement rootInstruction, RewriterStatement dest, List<Tuple2<RewriterStatement, BiConsumer<RewriterStatement, RewriterStatement.MatchingSubexpression>>> applyFunction) {
		if (match.getMatchParent() == null || match.getMatchParent() == match.getMatchRoot()) {
			final Map<RewriterStatement, RewriterStatement> createdObjects = new HashMap<>();
			RewriterStatement cpy = dest.nestedCopyOrInject(createdObjects, obj -> match.getAssocs().get(obj));
			RewriterStatement cpy2 = cpy.simplify(ctx);
			if (cpy2 != null)
				cpy = cpy2;

			match.getLinks().forEach(lnk -> lnk.newStmt.replaceAll(createdObjects::get));
			match.getLinks().forEach(lnk -> lnk.transferFunction.accept(lnk));
			applyFunction.forEach(t -> t._2.accept(createdObjects.get(t._1), match));

			if (postProcessor != null)
				postProcessor.accept(cpy);

			if (ctx.metaPropagator != null)
				cpy = ctx.metaPropagator.apply(cpy);

			cpy.prepareForHashing();
			cpy.recomputeHashCodes();
			return cpy;
		}

		final Map<RewriterStatement, RewriterStatement> createdObjects = new HashMap<>();
		match.getMatchParent().getOperands().set(match.getRootIndex(), dest.nestedCopyOrInject(createdObjects, obj -> match.getAssocs().get(obj)));
		RewriterStatement out = rootInstruction.simplify(ctx);
		if (out != null)
			out = rootInstruction;

		match.getLinks().forEach(lnk -> lnk.newStmt.replaceAll(createdObjects::get));
		match.getLinks().forEach(lnk -> lnk.transferFunction.accept(lnk));
		applyFunction.forEach(t -> t._2.accept(createdObjects.get(t._1), match));

		if (postProcessor != null)
			postProcessor.accept(rootInstruction);

		if (ctx.metaPropagator != null)
			rootInstruction = ctx.metaPropagator.apply(rootInstruction);

		rootInstruction.prepareForHashing();
		rootInstruction.recomputeHashCodes();
		return rootInstruction;
	}

	public String toString() {
		if (isUnidirectional())
			return fromRoot.toString() + " => " + toRoot.toString();
		else
			return fromRoot.toString() + " <=> " + toRoot.toString();
	}

	// TODO: Rework
	public List<RewriterRule> createNonGenericRules(Map<String, Set<String>> funcMappings) {
		/*Set<IdentityRewriterStatement> visited = new HashSet<>();
		List<Tuple2<RewriterStatement, Set<String>>> matches = new ArrayList<>();

		RewriterStatement from = fromRoot.nestedCopyOrInject(new HashMap<>(), stmt -> null);

		from.forEachPreOrderWithDuplicates(stmt -> {
			IdentityRewriterStatement identity = new IdentityRewriterStatement(stmt);
			if (!visited.add(identity))
				return false;

			if (!(stmt instanceof RewriterInstruction))
				return true;


			Set<String> implementations = funcMappings.get(((RewriterInstruction)stmt).trueTypedInstruction(ctx));

			if (implementations != null && !implementations.isEmpty())
				matches.add(new Tuple2<>(stmt, implementations));

			return true;
		});

		Set<List<String>> permutations = Sets.cartesianProduct(matches.stream().map(t -> t._2).collect(Collectors.toList()));

		List<RewriterRule> rules = new ArrayList<>();

		for (List<String> permutation : permutations) {
			for (int i = 0; i < permutation.size(); i++) {
				((RewriterInstruction)matches.get(i)._1).unsafeSetInstructionName(permutation.get(i));
			}
			RewriterStatement cpy = from.nestedCopyOrInject(new HashMap<>(), stmt -> null);
			ArrayList<RewriterStatement.MatchingSubexpression> mmatches = new ArrayList<>();

			this.matchStmt1((RewriterInstruction)cpy, mmatches, true);
			if (mmatches.isEmpty()) {
				System.out.println("Skipping rule: " + cpy);
				continue;
			}
			rules.add(new RewriterRule(ctx, name, cpy, this.apply(mmatches.get(0), (RewriterInstruction) cpy, true, true), true, new HashMap<>(), new HashMap<>()));
		}

		return rules;*/
		throw new NotImplementedException();
	}

	static class IdentityRewriterStatement {
		public RewriterStatement stmt;

		public IdentityRewriterStatement(RewriterStatement stmt) {
			this.stmt = stmt;
		}

		@Override
		public int hashCode() {
			return System.identityHashCode(stmt);
		}

		@Override
		public boolean equals(Object obj) {
			return (obj instanceof IdentityRewriterStatement && ((IdentityRewriterStatement)obj).stmt == stmt);
		}
	}

	static class LinkObject {
		List<RewriterStatement> stmt;
		Consumer<ExplicitLink> transferFunction;

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

		// TODO: Change
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
		final RewriterStatement oldStmt;
		List<RewriterStatement> newStmt;
		final Consumer<ExplicitLink> transferFunction;

		public ExplicitLink(RewriterStatement oldStmt, List<RewriterStatement> newStmt, Consumer<ExplicitLink> transferFunction) {
			this.oldStmt = oldStmt;
			this.newStmt = new ArrayList<>(newStmt);
			this.transferFunction = transferFunction;
		}
	}
}
