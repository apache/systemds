package org.apache.sysds.test.component.codegen.rewrite.functions;

import org.apache.sysds.hops.rewriter.RewriterRule;
import org.apache.sysds.hops.rewriter.RewriterRuleSet;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.List;
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
}
