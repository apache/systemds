/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.test.component.codegen.rewrite;

import org.apache.sysds.hops.rewriter.generated.RewriteAutomaticallyGenerated;
import org.apache.sysds.hops.rewriter.estimators.RewriterCostEstimator;
import org.apache.sysds.hops.rewriter.rule.RewriterRule;
import org.apache.sysds.hops.rewriter.rule.RewriterRuleCreator;
import org.apache.sysds.hops.rewriter.rule.RewriterRuleSet;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.junit.BeforeClass;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.function.Function;

public class RewriterRuleValidationTest {

	public static String RAW_FILE_PATH; // Must be specified
	public static String FILE_PATH; // Must be specified

	private static RuleContext ctx;
	private static Function<RewriterStatement, RewriterStatement> canonicalConverter;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
		canonicalConverter = RewriterUtils.buildCanonicalFormConverter(ctx, false);
	}


	//@Test
	public void test() {
		try {
			List<String> lines = Files.readAllLines(Paths.get(RAW_FILE_PATH));
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
			String serialized = ruleCreator.getRuleSet().serialize();
			//System.out.println(serialized);

			try (FileWriter writer = new FileWriter(FILE_PATH)) {
				writer.write(serialized);
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
