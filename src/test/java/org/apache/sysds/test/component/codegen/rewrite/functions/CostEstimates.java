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
		canonicalConverter = RewriterUtils.buildCanonicalFormConverter(ctx, true);
	}

	@Test
	public void test1() {
		RewriterStatement stmt = RewriterUtils.parse("%*%(+(A,B), C)", ctx, "MATRIX:A,B,C");
		RewriterCostEstimator.estimateCost(stmt, null, ctx);
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
}
