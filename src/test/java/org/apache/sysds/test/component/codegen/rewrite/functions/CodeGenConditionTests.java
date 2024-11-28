package org.apache.sysds.test.component.codegen.rewrite.functions;

import org.apache.sysds.hops.rewriter.RewriterRule;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.hops.rewriter.codegen.CodeGenCondition;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.List;
import java.util.function.Function;

public class CodeGenConditionTests {

	private static RuleContext ctx;
	private static Function<RewriterStatement, RewriterStatement> canonicalConverter;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
		canonicalConverter = RewriterUtils.buildCanonicalFormConverter(ctx, false);
	}

	@Test
	public void test1() {
		String ruleStr = "MATRIX:A\n" +
				"\n" +
				"t(t(A))\n" +
				"=>\n" +
				"A";

		RewriterRule rule = RewriterUtils.parseRule(ruleStr, ctx);

		List<CodeGenCondition> cgcs = CodeGenCondition.buildCondition(List.of(rule), ctx);
		System.out.println(cgcs);
	}

}
