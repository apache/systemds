package org.apache.sysds.test.component.codegen.rewrite.functions;

import org.apache.commons.lang3.mutable.MutableObject;
import org.apache.sysds.hops.rewriter.RewriterAssertions;
import org.apache.sysds.hops.rewriter.RewriterCostEstimator;
import org.apache.sysds.hops.rewriter.RewriterRule;
import org.apache.sysds.hops.rewriter.RewriterRuleBuilder;
import org.apache.sysds.hops.rewriter.RewriterRuleCreator;
import org.apache.sysds.hops.rewriter.RewriterRuleSet;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.junit.BeforeClass;
import org.junit.Test;
import scala.Tuple2;

import java.util.List;
import java.util.Set;
import java.util.function.Function;

public class RuleCreationTests {
	private static RuleContext ctx;
	private static Function<RewriterStatement, RewriterStatement> canonicalConverter;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
		canonicalConverter = RewriterUtils.buildCanonicalFormConverter(ctx, false);
	}

	@Test
	public void test1() {
		RewriterStatement from = RewriterUtils.parse("t(%*%(t(U),V))", ctx, "MATRIX:U,V");
		RewriterStatement to = RewriterUtils.parse("%*%(t(U), V)", ctx, "MATRIX:U,V");
		RewriterStatement canonicalForm1 = canonicalConverter.apply(from);
		RewriterStatement canonicalForm2 = canonicalConverter.apply(to);

		System.out.println("==========");
		System.out.println(canonicalForm1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(canonicalForm2.toParsableString(ctx, true));
		assert canonicalForm1.match(RewriterStatement.MatcherContext.exactMatch(ctx, canonicalForm2, canonicalForm1));

		RewriterRule rule = RewriterRuleCreator.createRule(from, to, canonicalForm1, canonicalForm2, ctx);
		System.out.println(rule);
	}

	@Test
	public void test2() {
		RewriterStatement from = RewriterUtils.parse("t(t(A))", ctx, "MATRIX:A");
		RewriterStatement to = RewriterUtils.parse("A", ctx, "MATRIX:A");
		RewriterStatement canonicalForm1 = canonicalConverter.apply(from);
		RewriterStatement canonicalForm2 = canonicalConverter.apply(to);

		System.out.println("==========");
		System.out.println(canonicalForm1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(canonicalForm2.toParsableString(ctx, true));
		assert canonicalForm1.match(RewriterStatement.MatcherContext.exactMatch(ctx, canonicalForm2, canonicalForm1));

		RewriterRule rule = RewriterRuleCreator.createRule(from, to, canonicalForm1, canonicalForm2, ctx);
		System.out.println(rule);

		RewriterRuleSet rs = new RewriterRuleSet(ctx, List.of(rule));

		RewriterStatement testStmt = RewriterUtils.parse("t(t([](A, 1, ncol(A), 1, 1)))", ctx, "MATRIX:A", "LITERAL_INT:1");

		RewriterRuleSet.ApplicableRule ar = rs.acceleratedFindFirst(testStmt);

		assert ar != null;
	}

	@Test
	public void validationTest1() {
		RewriterRule rule = new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A")
				.parseGlobalVars("FLOAT:b")
				.withParsedStatement("sum(/(A, b))")
				.toParsedStatement("/(sum(A), b)")
				.build();

		assert RewriterRuleCreator.validateRuleCorrectnessAndGains(rule, ctx);
	}

	@Test
	public void validationTest2() {
		RewriterRule rule = new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("FLOAT:b")
				.withParsedStatement("rowSums(colSums(%*%(A, B)))")
				.toParsedStatement("%*%(colSums(A), rowSums(B))")
				.build();

		assert RewriterRuleCreator.validateRuleCorrectness(rule, ctx);
		assert !RewriterRuleCreator.validateRuleApplicability(rule, ctx);
	}

	@Test
	public void validationTest3() {
		RewriterRule rule = new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("*(sum([](A,1,1,1,ncol(A))),colSums(B))")
				.toParsedStatement("%*%([](A,1,1,1,ncol(A)),colSums(B))")
				.build();

		assert RewriterRuleCreator.validateRuleCorrectness(rule, ctx);
		assert !RewriterRuleCreator.validateRuleApplicability(rule, ctx);
	}
}
