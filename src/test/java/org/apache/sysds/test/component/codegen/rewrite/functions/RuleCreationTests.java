package org.apache.sysds.test.component.codegen.rewrite.functions;

import org.apache.commons.lang3.mutable.MutableObject;
import org.apache.sysds.hops.rewriter.RewriterAssertions;
import org.apache.sysds.hops.rewriter.RewriterCostEstimator;
import org.apache.sysds.hops.rewriter.RewriterRule;
import org.apache.sysds.hops.rewriter.RewriterRuleCreator;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.junit.BeforeClass;
import org.junit.Test;
import scala.Tuple2;

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
		assert canonicalForm1.match(RewriterStatement.MatcherContext.exactMatch(ctx, canonicalForm2));

		RewriterRule rule = RewriterRuleCreator.createRule(from, to, canonicalForm1, canonicalForm2, ctx);
		System.out.println(rule);
	}
}
