package org.apache.sysds.hops.rewriter;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.mutable.MutableObject;
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
	private Set<RewriterStatement> allowedMultiReferences = Collections.emptySet();
	private RewriterAssertions combinedAssertions;
	private boolean allowCombinations = false;
	private boolean requireCostCheck = false;
	private RewriterStatement fromCost = null;
	private RewriterStatement toCost = null;

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
		RewriterAssertionUtils.buildImplicitAssertion(toRoot, assertions, toRoot, ctx);

		List<Tuple3<List<Number>, Long, Long>> costs = RewriterCostEstimator.compareCosts(fromRoot, toRoot, assertions, ctx, false, -1, false);

		requireCostCheck = RewriterCostEstimator.doesHaveAnImpactOnOptimalExpression(costs, false, true, 20);

		if (!requireCostCheck)
			return false;

		boolean integrateSparsityInCosts = RewriterCostEstimator.doesHaveAnImpactOnOptimalExpression(costs, true, false, 20);

		System.out.println("Require cost check (sparsity >> " + integrateSparsityInCosts + "): " + this);

		MutableObject<RewriterAssertions> assertionRef = new MutableObject<>(assertions);
		fromCost = RewriterCostEstimator.getRawCostFunction(fromRoot, ctx, assertionRef, !integrateSparsityInCosts);
		toCost = RewriterCostEstimator.getRawCostFunction(toRoot, ctx, assertionRef, !integrateSparsityInCosts);

		fromCost = RewriterSparsityEstimator.rollupSparsities(fromCost, RewriterSparsityEstimator.estimateAllNNZ(fromRoot, ctx), ctx);
		toCost = RewriterSparsityEstimator.rollupSparsities(toCost, RewriterSparsityEstimator.estimateAllNNZ(toRoot, ctx), ctx);

		return requireCostCheck;
	}

	public boolean requiresCostCheck() {
		return requireCostCheck;
	}

	public RewriterStatement getStmt1Cost() {
		return fromCost;
	}

	public RewriterStatement getStmt2Cost() {
		return toCost;
	}

	public void buildCombinedAssertions() {
		combinedAssertions = RewriterAssertionUtils.buildImplicitAssertions(fromRoot, ctx);
		RewriterAssertionUtils.buildImplicitAssertions(toRoot, combinedAssertions, ctx);
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

	public RewriterStatement apply(RewriterStatement.MatchingSubexpression match, RewriterStatement rootNode, boolean forward, boolean inplace) {
		return forward ? applyForward(match, rootNode, inplace) : applyBackward(match, rootNode, inplace);
	}

	public RewriterStatement applyForward(RewriterStatement.MatchingSubexpression match, RewriterStatement rootNode, boolean inplace) {
		return applyForward(match, rootNode, inplace, new MutableObject<>(null));
	}

	public RewriterStatement applyForward(RewriterStatement.MatchingSubexpression match, RewriterStatement rootNode, boolean inplace, MutableObject<Tuple3<RewriterStatement, RewriterStatement, Integer>> modificationHandle) {
		if (inplace)
			throw new NotImplementedException("Inplace operations are currently not working");
		return inplace ? applyInplace(match, rootNode, toRoot, applyStmt1ToStmt2 == null ? Collections.emptyList() : applyStmt1ToStmt2) : apply(match, rootNode, toRoot, modificationHandle, applyStmt1ToStmt2 == null ? Collections.emptyList() : applyStmt1ToStmt2);
	}

	public RewriterStatement applyBackward(RewriterStatement.MatchingSubexpression match, RewriterStatement rootNode, boolean inplace) {
		return applyBackward(match, rootNode, inplace, new MutableObject<>(null));
	}

	public RewriterStatement applyBackward(RewriterStatement.MatchingSubexpression match, RewriterStatement rootNode, boolean inplace, MutableObject<Tuple3<RewriterStatement, RewriterStatement, Integer>> modificationHandle) {
		if (inplace)
			throw new NotImplementedException("Inplace operations are currently not working");
		return inplace ? applyInplace(match, rootNode, fromRoot, applyStmt2ToStmt1 == null ? Collections.emptyList() : applyStmt2ToStmt1) : apply(match, rootNode, fromRoot, modificationHandle, applyStmt2ToStmt1 == null ? Collections.emptyList() : applyStmt2ToStmt1);
	}

	/*@Override
	public boolean matchStmt1(RewriterStatement stmt, ArrayList<RewriterStatement.MatchingSubexpression> arr, boolean findFirst) {
		return getStmt1().matchSubexpr(ctx, stmt, null, -1, arr, new HashMap<>(), true, false, findFirst, null, linksStmt1ToStmt2, true, true, false, iff1to2);
	}*/

	public RewriterStatement.MatchingSubexpression matchSingleStmt1(RewriterStatement exprRoot, RewriterStatement.RewriterPredecessor pred, RewriterStatement stmt, HashMap<RewriterStatement, RewriterStatement> dependencyMap, List<ExplicitLink> links, Map<RewriterStatement, LinkObject> ruleLinks) {
		RewriterStatement.MatcherContext mCtx = new RewriterStatement.MatcherContext(ctx, stmt, pred, exprRoot, getStmt1(), true, true, false, true, true, false, true, false, false, linksStmt1ToStmt2);
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

	public RewriterStatement.MatchingSubexpression matchSingleStmt2(RewriterStatement exprRoot, RewriterStatement.RewriterPredecessor pred, RewriterStatement stmt, HashMap<RewriterStatement, RewriterStatement> dependencyMap, List<ExplicitLink> links, Map<RewriterStatement, LinkObject> ruleLinks) {
		RewriterStatement.MatcherContext mCtx = new RewriterStatement.MatcherContext(ctx, stmt, pred, exprRoot, getStmt2(), true, true, false, true, true, false, true, false, false, linksStmt2ToStmt1);
		mCtx.currentStatement = stmt;
		boolean match = getStmt2().match(mCtx);

		if (match) {
			RewriterStatement.MatchingSubexpression matchExpr = mCtx.toMatch();

			if (iff2to1 == null || iff2to1.apply(matchExpr))
				return matchExpr;
		}

		return null;
	}

	private RewriterStatement apply(RewriterStatement.MatchingSubexpression match, RewriterStatement rootInstruction, RewriterStatement dest, MutableObject<Tuple3<RewriterStatement, RewriterStatement, Integer>> modificationHandle, List<Tuple2<RewriterStatement, BiConsumer<RewriterStatement, RewriterStatement.MatchingSubexpression>>> applyFunction) {
		if (match.getPredecessor().isRoot() /*|| match.getMatchParent() == match.getMatchRoot()*/) {
			//System.out.println("As root");
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
				// TODO: Maybe there is a better way?
				if (!cpy.isLiteral())
					cpy.unsafePutMeta("_assertions", assertions);
				//System.out.println("Put: " + assertions);
			}

			/*RewriterAssertions assertions = (RewriterAssertions) match.getExpressionRoot().getMeta("_assertions");

			if (assertions != null) {
				//assertions = RewriterAssertions.copy(assertions, createdObjects, true);
				cpy.unsafePutMeta("_assertions", assertions);
			}*/

			match.getLinks().forEach(lnk -> lnk.newStmt.replaceAll(createdObjects::get));
			match.getLinks().forEach(lnk -> lnk.transferFunction.accept(lnk));
			//RewriterAssertions assertions = RewriterAssertions.ofExpression(cpy, ctx);
			//cpy.unsafePutMeta("_assertions", assertions);
			applyFunction.forEach(t -> t._2.accept(createdObjects.get(t._1), match));

			if (postProcessor != null)
				postProcessor.accept(cpy);

			//cpy = assertions.buildEquivalences(cpy);

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

			//cpy.unsafePutMeta("_assertions", match.getExpressionRoot().getMeta("_assertions"));

			return cpy;
		}

		final Map<RewriterStatement, RewriterStatement> createdObjects = new HashMap<>();
		RewriterStatement cpy2 = rootInstruction.nestedCopyOrInject(createdObjects, (obj2, parent, pIdx) -> {
			if (obj2.equals(match.getMatchRoot())) {
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

		match.setNewExprRoot(cpy2);

		//System.out.println("NEWASS: " + cpy2.getMeta("_assertions"));

		/*RewriterAssertions assertions = (RewriterAssertions) match.getExpressionRoot().getMeta("_assertions");

		if (assertions != null) {
			assertions = RewriterAssertions.copy(assertions, createdObjects, true);
			cpy2.unsafePutMeta("_assertions", assertions);
		}*/

		match.getLinks().forEach(lnk -> lnk.newStmt.replaceAll(createdObjects::get));
		match.getLinks().forEach(lnk -> lnk.transferFunction.accept(lnk));
		//RewriterAssertions assertions = RewriterAssertions.ofExpression(cpy2, ctx);
		//cpy2.unsafePutMeta("_assertions", assertions);
		applyFunction.forEach(t -> t._2.accept(createdObjects.get(t._1), match));

		if (postProcessor != null)
			postProcessor.accept(cpy2);

		//cpy2 = assertions.buildEquivalences(cpy2);

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

	// TODO: ApplyInplace is currently not working
	private RewriterStatement applyInplace(RewriterStatement.MatchingSubexpression match, RewriterStatement rootInstruction, RewriterStatement dest, List<Tuple2<RewriterStatement, BiConsumer<RewriterStatement, RewriterStatement.MatchingSubexpression>>> applyFunction) {
		if (match.getPredecessor().isRoot() /*|| match.getMatchParent() == match.getMatchRoot()*/) {
			final Map<RewriterStatement, RewriterStatement> createdObjects = new HashMap<>();
			RewriterStatement cpy = dest.nestedCopyOrInject(createdObjects, obj -> match.getAssocs().get(obj));
			RewriterStatement cpy2 = cpy.simplify(ctx);
			if (cpy2 != null)
				cpy = cpy2;

			match.getLinks().forEach(lnk -> lnk.newStmt.replaceAll(createdObjects::get));
			match.getLinks().forEach(lnk -> lnk.transferFunction.accept(lnk));
			//RewriterAssertions assertions = RewriterAssertions.ofExpression(cpy, ctx);
			//cpy.unsafePutMeta("_assertions", assertions);
			applyFunction.forEach(t -> t._2.accept(createdObjects.get(t._1), match));

			if (postProcessor != null)
				postProcessor.accept(cpy);

			//cpy = assertions.buildEquivalences(cpy);

			if (ctx.metaPropagator != null)
				cpy = ctx.metaPropagator.apply(cpy);

			cpy.prepareForHashing();
			cpy.recomputeHashCodes(ctx);

			//if (match.getExpressionRoot() == match.getMatchRoot())
			//	cpy.unsafePutMeta("_assertions", rootInstruction.getMeta("_assertions"));
			return cpy;
		}

		final Map<RewriterStatement, RewriterStatement> createdObjects = new HashMap<>();
		// TODO
		//match.getMatchParent().getOperands().set(match.getRootIndex(), dest.nestedCopyOrInject(createdObjects, obj -> match.getAssocs().get(obj)));
		/*RewriterStatement out = rootInstruction.simplify(ctx);
		if (out != null)
			out = rootInstruction;*/

		match.getLinks().forEach(lnk -> lnk.newStmt.replaceAll(createdObjects::get));
		match.getLinks().forEach(lnk -> lnk.transferFunction.accept(lnk));
		//RewriterAssertions assertions = RewriterAssertions.ofExpression(rootInstruction, ctx);
		//rootInstruction.unsafePutMeta("_assertions", assertions);
		applyFunction.forEach(t -> t._2.accept(createdObjects.get(t._1), match));

		if (postProcessor != null)
			postProcessor.accept(rootInstruction);

		//rootInstruction = assertions.buildEquivalences(rootInstruction);

		if (ctx.metaPropagator != null)
			rootInstruction = ctx.metaPropagator.apply(rootInstruction);

		rootInstruction.prepareForHashing();
		rootInstruction.recomputeHashCodes(ctx);
		return rootInstruction;
	}

	public String toString() {
		if (isUnidirectional())
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
		toRoot.toParsableString(sb, refs, refIdx, varDefs, allowedMultiReferences, ctx);
		String stmt2 = sb.toString();
		//String stmt1 = fromRoot.toParsableString(ctx, varDefs, allowedMultiReferences);
		//String stmt2 = toRoot.toParsableString(ctx, varDefs, allowedMultiReferences);
		String multiRefDefs = "";

		if (!allowedMultiReferences.isEmpty()) {
			multiRefDefs = "AllowedMultiRefs:" + allowedMultiReferences.stream().map(stmt -> "$" + refs.get(stmt)).collect(Collectors.joining(",")) + "\nAllowCombinations:" + allowCombinations + "\n";
		}

		String defs = RewriterStatement.parsableDefinitions(varDefs);
		return multiRefDefs + defs + "\n" + stmt1 + "\n=>\n" + stmt2;
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

	/*static class IdentityRewriterStatement {
		public RewriterStatement stmt;

		@Deprecated
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

		@Override
		public String toString() {
			return stmt.toString() + "[" + hashCode() + "]";
		}
	}*/

	public static class LinkObject {
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
