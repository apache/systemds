package org.apache.sysds.test.component.codegen.rewrite.functions;

import org.apache.sysds.hops.rewriter.RewriterRule;
import org.apache.sysds.hops.rewriter.RewriterRuleBuilder;
import org.apache.sysds.hops.rewriter.RewriterRuleCreator;
import org.apache.sysds.hops.rewriter.RewriterRuleSet;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.List;
import java.util.function.Function;

public class TestRuleSet {
	private static RuleContext ctx;
	private static Function<RewriterStatement, RewriterStatement> canonicalConverter;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
		canonicalConverter = RewriterUtils.buildCanonicalFormConverter(ctx, false);
	}

	@Test
	public void test1() {
		RewriterRule rule = new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.withParsedStatement("sum(%*%(A, t(B)))")
				.toParsedStatement("sum(*(A, B))")
				.build();

		RewriterRuleSet rs = new RewriterRuleSet(ctx, List.of(rule));

		RewriterStatement stmt = RewriterUtils.parse("sum(%*%(colVec(A), t(colVec(B))))", ctx, "MATRIX:A,B");

		RewriterRuleSet.ApplicableRule ar = rs.acceleratedFindFirst(stmt);

		assert ar != null;

		stmt = ar.rule.apply(ar.matches.get(0), stmt, ar.forward, false);
		System.out.println(stmt.toParsableString(ctx));
	}

	@Test
	public void test2() {
		RewriterRule rule = new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.withParsedStatement("as.matrix(sum(colVec(A)))")
				.toParsedStatement("rowSums(colVec(A))")
				.build();

		RewriterRuleSet rs = new RewriterRuleSet(ctx, List.of(rule));

		RewriterStatement stmt = RewriterUtils.parse("as.matrix(sum(t(colVec(A))))", ctx, "MATRIX:A,B");

		RewriterRuleSet.ApplicableRule ar = rs.acceleratedFindFirst(stmt);

		assert ar != null;

		stmt = ar.rule.apply(ar.matches.get(0), stmt, ar.forward, false);
		System.out.println(stmt.toParsableString(ctx));
	}
}
