package org.apache.sysds.test.component.codegen.rewrite.functions;

import org.apache.commons.lang3.mutable.MutableObject;
import org.apache.sysds.hops.rewriter.assertions.RewriterAssertions;
import org.apache.sysds.hops.rewriter.estimators.RewriterCostEstimator;
import org.apache.sysds.hops.rewriter.RewriterRule;
import org.apache.sysds.hops.rewriter.RewriterRuleCreator;
import org.apache.sysds.hops.rewriter.RewriterRuleSet;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.junit.BeforeClass;
import org.junit.Test;
import scala.Tuple2;

import java.util.List;
import java.util.Set;
import java.util.function.Function;

public class RuleSerializationTest {

	private static RuleContext ctx;
	private static Function<RewriterStatement, RewriterStatement> canonicalConverter;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
		canonicalConverter = RewriterUtils.buildCanonicalFormConverter(ctx, false);
	}

	@Test
	public void test1() {
		String ruleStr1 = "MATRIX:A\nt(t(A))\n=>\nA";
		String ruleStr2 = "MATRIX:A\nrowSums(t(A))\n=>\nt(colSums(A))";
		RewriterRule rule1 = RewriterUtils.parseRule(ruleStr1, ctx);
		RewriterRule rule2 = RewriterUtils.parseRule(ruleStr2, ctx);

		RewriterRuleSet ruleSet = new RewriterRuleSet(ctx, List.of(rule1, rule2));
		String serialized = ruleSet.serialize(ctx);

		System.out.println(serialized);

		RewriterRuleSet newRuleSet = RewriterRuleSet.deserialize(serialized, ctx);
		String newSerialized = newRuleSet.serialize(ctx);

		System.out.println(newSerialized);

		assert serialized.equals(newSerialized);
	}

	@Test
	public void test2() {
		RewriterStatement from = RewriterUtils.parse("t(t(U))", ctx, "MATRIX:U,V");
		RewriterStatement to = RewriterUtils.parse("U", ctx, "MATRIX:U,V");
		RewriterStatement canonicalForm1 = canonicalConverter.apply(from);
		RewriterStatement canonicalForm2 = canonicalConverter.apply(to);

		System.out.println("==========");
		System.out.println(canonicalForm1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(canonicalForm2.toParsableString(ctx, true));
		assert canonicalForm1.match(RewriterStatement.MatcherContext.exactMatch(ctx, canonicalForm2, canonicalForm1));

		RewriterRule rule = RewriterRuleCreator.createRule(from, to, canonicalForm1, canonicalForm2, ctx);
		from = rule.getStmt1();
		to = rule.getStmt2();

		MutableObject<RewriterAssertions> assertionRef = new MutableObject<>();
		long fullCost = RewriterCostEstimator.estimateCost(to, ctx);
		long maxCost = RewriterCostEstimator.estimateCost(from, ctx, assertionRef);
		Tuple2<Set<RewriterStatement>, Boolean> result = RewriterCostEstimator.determineSingleReferenceRequirement(from, RewriterCostEstimator.DEFAULT_COST_FN, assertionRef.getValue(), fullCost, maxCost, ctx);

		assert result._1.size() == 1 && result._2;

		rule.setAllowedMultiReferences(result._1, result._2);

		String serialized = rule.toParsableString(ctx);

		System.out.println("::RULE");
		System.out.println(serialized);
		System.out.println();

		RewriterRule newRule = RewriterUtils.parseRule(serialized, ctx);
		String newSerialized = newRule.toParsableString(ctx);

		System.out.println(newSerialized);

		assert serialized.equals(newSerialized);
	}

	@Test
	public void test3() {
		RewriterStatement from = RewriterUtils.parse("sum(t(U))", ctx, "MATRIX:U,V");
		RewriterStatement to = RewriterUtils.parse("sum(U)", ctx, "MATRIX:U,V");
		RewriterStatement canonicalForm1 = canonicalConverter.apply(from);
		RewriterStatement canonicalForm2 = canonicalConverter.apply(to);

		System.out.println("==========");
		System.out.println(canonicalForm1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(canonicalForm2.toParsableString(ctx, true));
		assert canonicalForm1.match(RewriterStatement.MatcherContext.exactMatch(ctx, canonicalForm2, canonicalForm1));

		RewriterRule rule = RewriterRuleCreator.createRule(from, to, canonicalForm1, canonicalForm2, ctx);
		from = rule.getStmt1();
		to = rule.getStmt2();

		MutableObject<RewriterAssertions> assertionRef = new MutableObject<>();
		long fullCost = RewriterCostEstimator.estimateCost(to, ctx);
		long maxCost = RewriterCostEstimator.estimateCost(from, ctx, assertionRef);
		Tuple2<Set<RewriterStatement>, Boolean> result = RewriterCostEstimator.determineSingleReferenceRequirement(from, RewriterCostEstimator.DEFAULT_COST_FN, assertionRef.getValue(), fullCost, maxCost, ctx);

		assert result._1.size() == 1 && result._2;

		rule.setAllowedMultiReferences(result._1, result._2);

		String serialized = rule.toParsableString(ctx);

		System.out.println("::RULE");
		System.out.println(serialized);
		System.out.println();

		RewriterRule newRule = RewriterUtils.parseRule(serialized, ctx);
		String newSerialized = newRule.toParsableString(ctx);

		System.out.println(newSerialized);

		assert serialized.equals(newSerialized);
	}
}
