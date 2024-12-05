package org.apache.sysds.test.component.codegen.rewrite;

import org.apache.sysds.hops.rewriter.RewriteAutomaticallyGenerated;
import org.apache.sysds.hops.rewriter.estimators.RewriterCostEstimator;
import org.apache.sysds.hops.rewriter.RewriterRule;
import org.apache.sysds.hops.rewriter.RewriterRuleCreator;
import org.apache.sysds.hops.rewriter.RewriterRuleSet;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.function.Function;

public class RewriterRuleValidationTest {

	private static RuleContext ctx;
	private static Function<RewriterStatement, RewriterStatement> canonicalConverter;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
		canonicalConverter = RewriterUtils.buildCanonicalFormConverter(ctx, false);
	}


	@Test
	public void test() {
		try {
			List<String> lines = Files.readAllLines(Paths.get(RewriteAutomaticallyGenerated.RAW_FILE_PATH));
			RewriterRuleSet ruleSet = RewriterRuleSet.deserialize(lines, ctx);
			RewriterRuleCreator ruleCreator = new RewriterRuleCreator(ctx);

			int ctr = 0;
			for (RewriterRule rule : ruleSet.getRules()) {
				if (ctr % 10 == 0)
					System.out.println("Done: " + ctr + " / " + ruleSet.getRules().size());

				ctr++;
				try {
					System.out.println(rule.getStmt1().toParsableString(ctx) + " => " + rule.getStmt2().toParsableString(ctx));
					long preCost = RewriterCostEstimator.estimateCost(rule.getStmt1(), ctx);
					long postCost = RewriterCostEstimator.estimateCost(rule.getStmt2(), ctx);
					System.out.println(ruleCreator.registerRule(rule, preCost, postCost, true, canonicalConverter));
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
			//System.out.println(ruleSet.toJavaCode("GeneratedRewriteClass", false));
			String serialized = ruleCreator.getRuleSet().serialize(ctx);
			//System.out.println(serialized);

			try (FileWriter writer = new FileWriter(RewriteAutomaticallyGenerated.FILE_PATH)) {
				writer.write(serialized);
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
