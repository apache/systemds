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

package org.apache.sysds.test.component.codegen.rewrite.functions;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.hops.rewriter.rule.RewriterRule;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.hops.rewriter.codegen.CodeGenCondition;
import org.apache.sysds.test.component.codegen.rewrite.RewriterTopologySortTests;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

public class CodeGenConditionTests {
	protected static final Log LOG = LogFactory.getLog(CodeGenConditionTests.class.getName());

	private static RuleContext ctx;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
	}

	@Test
	public void test1() {
		String ruleStr = "MATRIX:A\n" +
				"\n" +
				"t(t(A))\n" +
				"=>\n" +
				"A";

		RewriterRule rule = RewriterUtils.parseRule(ruleStr, ctx);

		List<CodeGenCondition> cgcs = CodeGenCondition.buildCondition(List.of(rule), 1, ctx);
	}

	@Test
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
		LOG.info(CodeGenCondition.getSelectionString(cgcs, 0, fNames, ctx));
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
		LOG.info(cgcs);
		LOG.info(CodeGenCondition.getSelectionString(cgcs, 0, fNames, ctx));
	}
}
