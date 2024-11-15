package org.apache.sysds.test.component.codegen.rewrite.functions;

import org.apache.commons.lang3.mutable.MutableBoolean;
import org.apache.sysds.hops.rewriter.DMLCodeGenerator;
import org.apache.sysds.hops.rewriter.DMLExecutor;
import org.apache.sysds.hops.rewriter.RewriterRule;
import org.apache.sysds.hops.rewriter.RewriterRuleCreator;
import org.apache.sysds.hops.rewriter.RewriterRuleSet;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.List;
import java.util.UUID;
import java.util.function.Function;

public class DMLCodeGenTest {

	private static RuleContext ctx;
	private static Function<RewriterStatement, RewriterStatement> canonicalConverter;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
		canonicalConverter = RewriterUtils.buildCanonicalFormConverter(ctx, false);
	}

	@Test
	public void test1() {
		RewriterStatement stmt = RewriterUtils.parse("trace(+(A, t(B)))", ctx, "MATRIX:A,B");
		System.out.println(DMLCodeGenerator.generateDML(stmt));
	}

	@Test
	public void test2() {
		String ruleStr1 = "MATRIX:A\nt(t(A))\n=>\nA";
		String ruleStr2 = "MATRIX:A\nrowSums(t(A))\n=>\nt(colSums(A))";
		RewriterRule rule1 = RewriterUtils.parseRule(ruleStr1, ctx);
		RewriterRule rule2 = RewriterUtils.parseRule(ruleStr2, ctx);

		//RewriterRuleSet ruleSet = new RewriterRuleSet(ctx, List.of(rule1, rule2));
		String sessionId = UUID.randomUUID().toString();
		String validationScript = DMLCodeGenerator.generateRuleValidationDML(rule2, DMLCodeGenerator.EPS, sessionId);
		System.out.println("Validation script:");
		System.out.println(validationScript);
		MutableBoolean valid = new MutableBoolean(true);
		DMLExecutor.executeCode(validationScript, line -> {
			if (!line.startsWith(sessionId))
				return;

			if (!line.endsWith("valid: TRUE")) {
				DMLExecutor.println("An invalid rule was found!");
				valid.setValue(false);
			}
		});

		System.out.println("Exiting...");
		assert valid.booleanValue();
	}

	@Test
	public void test3() {
		String ruleStr2 = "MATRIX:A,B\nt(+(A,t(B)))\n=>\n+(t(A),B)";
		RewriterRule rule2 = RewriterUtils.parseRule(ruleStr2, ctx);

		RewriterRuleCreator.validateRuleCorrectnessAndGains(rule2, ctx);
	}
}
