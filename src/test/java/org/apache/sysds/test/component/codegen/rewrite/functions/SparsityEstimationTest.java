package org.apache.sysds.test.component.codegen.rewrite.functions;

import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.hops.rewriter.estimators.RewriterSparsityEstimator;
import org.junit.BeforeClass;
import org.junit.Test;

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
		RewriterStatement stmt = RewriterUtils.parse("+(A, -(B, A))", ctx, "MATRIX:A,B", "FLOAT:a");
		Map<RewriterStatement, RewriterStatement> estimates = RewriterSparsityEstimator.estimateAllNNZ(stmt, ctx);

		estimates.forEach((k, v) -> {
			System.out.println("K: " + k.toParsableString(ctx));
			System.out.println("Sparsity: " + v.toParsableString(ctx));
		});

		System.out.println("Rollup: " + RewriterSparsityEstimator.rollupSparsities(estimates.get(stmt), estimates, ctx).toParsableString(ctx));
	}
}
