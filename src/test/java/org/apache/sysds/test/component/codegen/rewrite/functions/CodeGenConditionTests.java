package org.apache.sysds.test.component.codegen.rewrite.functions;

import org.apache.sysds.hops.rewriter.RewriterRule;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.hops.rewriter.codegen.CodeGenCondition;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

public class CodeGenConditionTests {

	private static RuleContext ctx;
	private static Function<RewriterStatement, RewriterStatement> canonicalConverter;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
		canonicalConverter = RewriterUtils.buildCanonicalFormConverter(ctx, false);
	}

	//@Test
	public void test1() {
		String ruleStr = "MATRIX:A\n" +
				"\n" +
				"t(t(A))\n" +
				"=>\n" +
				"A";

		RewriterRule rule = RewriterUtils.parseRule(ruleStr, ctx);

		List<CodeGenCondition> cgcs = CodeGenCondition.buildCondition(List.of(rule), 1, ctx);
		System.out.println(cgcs);
	}

	//@Test
	public void test2() {
		String ruleStr = "MATRIX:A\n" +
				"\n" +
				"t(t(A))\n" +
				"=>\n" +
				"A";

		RewriterRule rule = RewriterUtils.parseRule(ruleStr, ctx);

		String ruleStr2 = "MATRIX:A,B\n" +
				"\n" +
				"+(t(A), t(B))\n" +
				"=>\n" +
				"t(+(A, B))";

		RewriterRule rule2 = RewriterUtils.parseRule(ruleStr2, ctx);

		String ruleStr3 = "MATRIX:A,B\n" +
				"\n" +
				"%*%(t(A), t(B))\n" +
				"=>\n" +
				"t(%*%(B, A))";

		RewriterRule rule3 = RewriterUtils.parseRule(ruleStr3, ctx);

		Map<RewriterRule, String> fNames = new HashMap<>();
		fNames.put(rule, "rule1");
		fNames.put(rule2, "rule2");
		fNames.put(rule3, "rule3");

		List<CodeGenCondition> cgcs = CodeGenCondition.buildCondition(List.of(rule, rule2, rule3), 1, ctx);
		System.out.println(cgcs);
		System.out.println(CodeGenCondition.getSelectionString(cgcs, 0, fNames, ctx));
	}

	@Test
	public void test3() {
		String ruleStr = "MATRIX:A\nFLOAT:b\n" +
				"\n" +
				"!=(-(b,rev(A)),A)\n" +
				"=>\n" +
				"!=(A,-(b,A))";

		RewriterRule rule = RewriterUtils.parseRule(ruleStr, ctx);

		String ruleStr2 = "MATRIX:A,B\n" +
				"\n" +
				"!=(-(B,rev(A)),A)\n" +
				"=>\n" +
				"!=(A,-(B,A))";

		RewriterRule rule2 = RewriterUtils.parseRule(ruleStr2, ctx);

		String ruleStr3 = "MATRIX:A,B,C\n" +
				"\n" +
				"+(*(A,C),*(A,B))\n" +
				"=>\n" +
				"*(A,+(B,C))";

		RewriterRule rule3 = RewriterUtils.parseRule(ruleStr3, ctx);

		String ruleStr4 = "MATRIX:A,B,C\n" +
				"\n" +
				"+(*(A,C),*(B,A))\n" +
				"=>\n" +
				"*(A,+(B,C))";

		RewriterRule rule4 = RewriterUtils.parseRule(ruleStr4, ctx);

		/*String ruleStr5 = "FLOAT:A,B,C\n" +
				"\n" +
				"+(cast.MATRIX(A), B)\n" +
				"=>\n" +
				"cast.MATRIX(+(A,B))";*/
		String ruleStr5 = "MATRIX:B,C\nFLOAT:a\n" +
				"\n" +
				"+(*(a,C),*(B,a))\n" +
				"=>\n" +
				"*(a,+(B,C))";

		RewriterRule rule5 = RewriterUtils.parseRule(ruleStr5, ctx);

		Map<RewriterRule, String> fNames = new HashMap<>();
		fNames.put(rule, "rule1");
		fNames.put(rule2, "rule2");
		fNames.put(rule3, "rule3");
		fNames.put(rule4, "rule4");
		fNames.put(rule5, "rule5");

		List<CodeGenCondition> cgcs = CodeGenCondition.buildCondition(List.of(rule, rule2, rule3, rule4, rule5), 1, ctx);
		System.out.println(cgcs);
		System.out.println(CodeGenCondition.getSelectionString(cgcs, 0, fNames, ctx));
	}



	/*@Test
	public void codeGen() {
		try {
			List<String> lines = Files.readAllLines(Paths.get(RewriteAutomaticallyGenerated.FILE_PATH));
			RewriterRuleSet ruleSet = RewriterRuleSet.deserialize(lines, ctx);
			Map<RewriterRule, String> fNames = new HashMap<>();
			int ruleCtr = 1;
			for (RewriterRule rr : ruleSet.getRules())
				fNames.put(rr, "rule_" + (ruleCtr++));

			List<CodeGenCondition> cgcs = CodeGenCondition.buildCondition(ruleSet.getRules(), 5, ctx);
			System.out.println(CodeGenCondition.getSelectionString(cgcs, 0, fNames, ctx));
			//System.out.println(ruleSet.toJavaCode("GeneratedRewriteClass", true));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}*/

}
