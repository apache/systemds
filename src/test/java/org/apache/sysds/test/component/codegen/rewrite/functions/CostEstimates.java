package org.apache.sysds.test.component.codegen.rewrite.functions;

import org.apache.sysds.hops.rewriter.RewriterCostEstimator;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.function.Function;

public class CostEstimates {
	private static RuleContext ctx;
	private static Function<RewriterStatement, RewriterStatement> canonicalConverter;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
		canonicalConverter = RewriterUtils.buildCanonicalFormConverter(ctx, false);
	}

	@Test
	public void test1() {
		RewriterStatement stmt = RewriterUtils.parse("%*%(+(A,B), C)", ctx, "MATRIX:A,B,C");
		long cost = RewriterCostEstimator.estimateCost(stmt, el -> 2000L, ctx);
		System.out.println(cost);
	}

	@Test
	public void test2() {
		RewriterStatement stmt = RewriterUtils.parse("*(+(1, 1), 2)", ctx, "LITERAL_INT:1,2");
		System.out.println(canonicalConverter.apply(stmt));
	}

	@Test
	public void test3() {
		RewriterStatement stmt = RewriterUtils.parse("_EClass(argList(1, ncol(X)))", ctx, "LITERAL_INT:1", "MATRIX:X");
		System.out.println(canonicalConverter.apply(stmt));
	}

	@Test
	public void test4() {
		RewriterStatement stmt1 = RewriterUtils.parse("t(%*%(+(A,B), C))", ctx, "MATRIX:A,B,C");
		RewriterStatement stmt2 = RewriterUtils.parse("%*%(t(C), t(+(A,B)))", ctx, "MATRIX:A,B,C");
		long cost1 = RewriterCostEstimator.estimateCost(stmt1, el -> 2000L, ctx);
		long cost2 = RewriterCostEstimator.estimateCost(stmt2, el -> 2000L, ctx);
		System.out.println("Cost1: " + cost1);
		System.out.println("Cost2: " + cost2);
		System.out.println("Ratio: " + ((double)cost1)/cost2);
		assert cost1 < cost2;
	}

	@Test
	public void test5() {
		RewriterStatement stmt1 = RewriterUtils.parse("t(/(*(A, B), C))", ctx, "MATRIX:A,B,C");
		RewriterStatement stmt2 = RewriterUtils.parse("/(*(t(A), t(B)), t(C))", ctx, "MATRIX:A,B,C");
		long cost1 = RewriterCostEstimator.estimateCost(stmt1, el -> 2000L, ctx);
		long cost2 = RewriterCostEstimator.estimateCost(stmt2, el -> 2000L, ctx);
		System.out.println("Cost1: " + cost1);
		System.out.println("Cost2: " + cost2);
		System.out.println("Ratio: " + ((double)cost1)/cost2);
		assert cost1 < cost2;

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);
		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2));
	}

	@Test
	public void test6() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(+(A, B))", ctx, "MATRIX:A,B,C");
		RewriterStatement stmt2 = RewriterUtils.parse("+(sum(A), sum(B))", ctx, "MATRIX:A,B,C");
		long cost1 = RewriterCostEstimator.estimateCost(stmt1, el -> 2000L, ctx);
		long cost2 = RewriterCostEstimator.estimateCost(stmt2, el -> 2000L, ctx);
		System.out.println("Cost1: " + cost1);
		System.out.println("Cost2: " + cost2);
		System.out.println("Ratio: " + ((double)cost2)/cost1);
		assert cost2 < cost1;

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);
		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2));
	}

	@Test
	public void test7() {
		RewriterStatement stmt1 = RewriterUtils.parse("cast.MATRIX(sum(A))", ctx, "MATRIX:A,B,C");
		RewriterStatement stmt2 = RewriterUtils.parse("rowSums(colSums(A))", ctx, "MATRIX:A,B,C");
		long cost1 = RewriterCostEstimator.estimateCost(stmt1, el -> 2000L, ctx);
		long cost2 = RewriterCostEstimator.estimateCost(stmt2, el -> 2000L, ctx);
		System.out.println("Cost1: " + cost1);
		System.out.println("Cost2: " + cost2);
		System.out.println("Ratio: " + ((double)cost1)/cost2);
		assert cost1 < cost2;

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2));
	}

	@Test
	public void test8() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(*(diag(A), diag(B)))", ctx, "MATRIX:A,B,C");
		RewriterStatement stmt2 = RewriterUtils.parse("trace(*(A, B))", ctx, "MATRIX:A,B,C");

		long cost1 = RewriterCostEstimator.estimateCost(stmt1, el -> 2000L, ctx);
		long cost2 = RewriterCostEstimator.estimateCost(stmt2, el -> 2000L, ctx);
		System.out.println("Cost1: " + cost1);
		System.out.println("Cost2: " + cost2);
		System.out.println("Ratio: " + ((double)cost1)/cost2);
		assert cost1 < cost2;

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2));
	}
}
