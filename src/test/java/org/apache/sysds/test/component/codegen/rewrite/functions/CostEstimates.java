package org.apache.sysds.test.component.codegen.rewrite.functions;

import org.apache.sysds.hops.rewriter.RewriterCostEstimator;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.Random;
import java.util.function.Function;

public class CostEstimates {
	private static RuleContext ctx;
	private static Function<RewriterStatement, RewriterStatement> canonicalConverter;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
		canonicalConverter = RewriterUtils.buildCanonicalFormConverter(ctx, true);
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
		assert cost1 < cost2;
	}
}
