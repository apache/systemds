package org.apache.sysds.test.component.codegen.rewrite.functions;

import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.function.Function;

public class MinimalDifference {

	private static RuleContext ctx;
	private static Function<RewriterStatement, RewriterStatement> canonicalConverter;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
		canonicalConverter = RewriterUtils.buildCanonicalFormConverter(ctx, true);
	}

	@Test
	public void test1() {
		RewriterStatement stmt1 = RewriterUtils.parse("t(t(A))", ctx, "MATRIX:A");
		RewriterStatement stmt2 = RewriterUtils.parse("t(A)", ctx, "MATRIX:A");

		RewriterStatement.MatcherContext mCtx = RewriterStatement.MatcherContext.findMinimalDifference(ctx, stmt2);
		stmt1.match(mCtx);
		System.out.println("Minimal Difference: ");
		System.out.println(mCtx.getFirstMismatch()._1.toParsableString(ctx));
		System.out.println(mCtx.getFirstMismatch()._2.toParsableString(ctx));
	}
}
