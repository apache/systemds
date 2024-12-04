package org.apache.sysds.test.component.codegen.rewrite.functions;

import org.apache.commons.lang3.mutable.MutableObject;
import org.apache.sysds.hops.rewriter.RewriterRule;
import org.apache.sysds.hops.rewriter.RewriterRuleCreator;
import org.apache.sysds.hops.rewriter.RewriterRuleSet;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.hops.rewriter.assertions.RewriterAssertionUtils;
import org.apache.sysds.hops.rewriter.assertions.RewriterAssertions;
import org.apache.sysds.hops.rewriter.estimators.RewriterCostEstimator;
import org.apache.sysds.hops.rewriter.estimators.RewriterSparsityEstimator;
import org.junit.BeforeClass;
import org.junit.Test;
import scala.Tuple2;
import scala.Tuple3;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

public class SparsityEstimationTest {
	private static RuleContext ctx;
	private static Function<RewriterStatement, RewriterStatement> canonicalConverter;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
		canonicalConverter = RewriterUtils.buildCanonicalFormConverter(ctx, false);
	}

	@Test
	public void test1() {
		RewriterStatement stmt = RewriterUtils.parse("+*(A, 0.0, B)", ctx, "MATRIX:A,B", "LITERAL_FLOAT:0.0");
		System.out.println(RewriterSparsityEstimator.estimateNNZ(stmt, ctx).toParsableString(ctx));
	}

	@Test
	public void test2() {
		RewriterStatement stmt = RewriterUtils.parse("+*(A, a, B)", ctx, "MATRIX:A,B", "FLOAT:a");
		System.out.println(RewriterSparsityEstimator.estimateNNZ(stmt, ctx).toParsableString(ctx));
	}

	@Test
	public void test3() {
		RewriterStatement stmt = RewriterUtils.parse("%*%(A, -(B, A))", ctx, "MATRIX:A,B", "FLOAT:a");
		RewriterAssertionUtils.buildImplicitAssertions(stmt, stmt.getAssertions(ctx), ctx);

		Map<RewriterStatement, RewriterStatement> estimates = RewriterSparsityEstimator.estimateAllNNZ(stmt, ctx);

		estimates.forEach((k, v) -> {
			stmt.getAssertions(ctx).update(v);
			System.out.println("K: " + k.toParsableString(ctx));
			System.out.println("NNZ: " + v.toParsableString(ctx));
		});

		System.out.println("Rollup: " + RewriterSparsityEstimator.rollupSparsities(estimates.get(stmt), estimates, ctx).toParsableString(ctx));

		Map<RewriterStatement, Long> nnzs = new HashMap<>();
		nnzs.put(stmt.getChild(0), 3000L);
		nnzs.put(stmt.getChild(1, 0), 50000L);

		MutableObject<RewriterAssertions> assertionRef = new MutableObject<>();
		RewriterStatement costFunction = RewriterCostEstimator.getRawCostFunction(stmt, ctx, assertionRef, false);
		costFunction = RewriterSparsityEstimator.rollupSparsities(costFunction, estimates, ctx);

		System.out.println(costFunction.toParsableString(ctx));

		System.out.println("Dense cost:  " + RewriterCostEstimator.estimateCost(stmt, ctx));
		System.out.println("Sparse cost: " + RewriterCostEstimator.computeCostFunction(costFunction, RewriterCostEstimator.DEFAULT_COST_FN, (el, tpl) -> nnzs.get(el.getChild(0)), assertionRef.getValue(), ctx));

		/*System.out.println(RewriterCostEstimator.estimateCost(stmt, RewriterCostEstimator.DEFAULT_COST_FN, el -> {
			System.out.println(el.getChild(0));
			return nnzs.get(el.getChild(0));
		}, ctx, null));*/
	}

	@Test
	public void test4() {
		RewriterStatement from = RewriterUtils.parse("+(*(A, B), *(A, C))", ctx, "MATRIX:A,B,C");
		RewriterStatement to = RewriterUtils.parse("*(A, +(B, C))", ctx, "MATRIX:A,B,C");
		RewriterStatement canonicalForm1 = canonicalConverter.apply(from);
		RewriterStatement canonicalForm2 = canonicalConverter.apply(to);

		System.out.println("==========");
		System.out.println(canonicalForm1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(canonicalForm2.toParsableString(ctx, true));
		assert canonicalForm1.match(RewriterStatement.MatcherContext.exactMatch(ctx, canonicalForm2, canonicalForm1));

		RewriterRule rule = RewriterRuleCreator.createRule(from, to, canonicalForm1, canonicalForm2, ctx);
		System.out.println(rule);

		RewriterAssertionUtils.buildImplicitAssertion(rule.getStmt1(), rule.getStmt1().getAssertions(ctx), ctx);
		RewriterAssertionUtils.buildImplicitAssertion(rule.getStmt2(), rule.getStmt1().getAssertions(ctx), ctx);
		//rule.getStmt2().unsafePutMeta("_assertions", rule.getStmt1().getAssertions(ctx));

		RewriterCostEstimator.compareCosts(rule.getStmt1(), rule.getStmt2(), rule.getStmt1().getAssertions(ctx), ctx, true, 5, false);
	}

	@Test
	public void test5() {
		RewriterStatement from = RewriterUtils.parse("t(%*%(t(A), B))", ctx, "MATRIX:A,B,C");
		RewriterStatement to = RewriterUtils.parse("%*%(t(B), A)", ctx, "MATRIX:A,B,C");
		RewriterStatement canonicalForm1 = canonicalConverter.apply(from);
		RewriterStatement canonicalForm2 = canonicalConverter.apply(to);

		System.out.println("==========");
		System.out.println(canonicalForm1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(canonicalForm2.toParsableString(ctx, true));
		assert canonicalForm1.match(RewriterStatement.MatcherContext.exactMatch(ctx, canonicalForm2, canonicalForm1));

		RewriterRule rule = RewriterRuleCreator.createRule(from, to, canonicalForm1, canonicalForm2, ctx);
		System.out.println(rule);

		RewriterAssertionUtils.buildImplicitAssertion(rule.getStmt1(), rule.getStmt1().getAssertions(ctx), ctx);
		RewriterAssertionUtils.buildImplicitAssertion(rule.getStmt2(), rule.getStmt1().getAssertions(ctx), ctx);
		//rule.getStmt2().unsafePutMeta("_assertions", rule.getStmt1().getAssertions(ctx));

		List<Tuple3<List<Number>, Long, Long>> costs = RewriterCostEstimator.compareCosts(rule.getStmt1(), rule.getStmt2(), rule.getStmt1().getAssertions(ctx), ctx, false, 5, false);
		System.out.println(costs);
		System.out.println("Does sparsity have an impact on optimal expression? >> " + RewriterCostEstimator.doesHaveAnImpactOnOptimalExpression(costs, true, true, 0));
	}

	@Test
	public void test6() {
		RewriterStatement from = RewriterUtils.parse("t(+(A, B))", ctx, "MATRIX:A,B,C");
		RewriterStatement to = RewriterUtils.parse("+(t(A), t(B))", ctx, "MATRIX:A,B,C");
		RewriterStatement canonicalForm1 = canonicalConverter.apply(from);
		RewriterStatement canonicalForm2 = canonicalConverter.apply(to);

		System.out.println("==========");
		System.out.println(canonicalForm1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(canonicalForm2.toParsableString(ctx, true));
		assert canonicalForm1.match(RewriterStatement.MatcherContext.exactMatch(ctx, canonicalForm2, canonicalForm1));

		RewriterRule rule = RewriterRuleCreator.createRule(from, to, canonicalForm1, canonicalForm2, ctx);
		System.out.println(rule);

		RewriterAssertionUtils.buildImplicitAssertion(rule.getStmt1(), rule.getStmt1().getAssertions(ctx), ctx);
		RewriterAssertionUtils.buildImplicitAssertion(rule.getStmt2(), rule.getStmt1().getAssertions(ctx), ctx);
		//rule.getStmt2().unsafePutMeta("_assertions", rule.getStmt1().getAssertions(ctx));

		List<Tuple3<List<Number>, Long, Long>> costs = RewriterCostEstimator.compareCosts(rule.getStmt1(), rule.getStmt2(), rule.getStmt1().getAssertions(ctx), ctx, false, 5, false);
		System.out.println(costs);
		System.out.println("Does sparsity have an impact on optimal expression? >> " + RewriterCostEstimator.doesHaveAnImpactOnOptimalExpression(costs, true, true, 0));
		System.out.println("Does anything have an impact on optimal expression? >> " + RewriterCostEstimator.doesHaveAnImpactOnOptimalExpression(costs, true, false, 0));
	}
}
