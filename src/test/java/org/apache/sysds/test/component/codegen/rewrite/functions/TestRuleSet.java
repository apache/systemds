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

import org.apache.sysds.hops.rewriter.rule.RewriterRule;
import org.apache.sysds.hops.rewriter.rule.RewriterRuleBuilder;
import org.apache.sysds.hops.rewriter.rule.RewriterRuleSet;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.List;

public class TestRuleSet {
	private static RuleContext ctx;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
	}

	@Test
	public void test1() {
		RewriterRule rule = new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.withParsedStatement("sum(%*%(A, t(B)))")
				.toParsedStatement("sum(*(A, B))")
				.build();

		RewriterRuleSet rs = new RewriterRuleSet(ctx, List.of(rule));

		RewriterStatement stmt = RewriterUtils.parse("sum(%*%(colVec(A), t(colVec(B))))", ctx, "MATRIX:A,B");

		RewriterRuleSet.ApplicableRule ar = rs.acceleratedFindFirst(stmt);

		assert ar != null;

		stmt = ar.rule.apply(ar.matches.get(0), stmt, ar.forward, false);
	}

	@Test
	public void test2() {
		RewriterRule rule = new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.withParsedStatement("as.matrix(sum(colVec(A)))")
				.toParsedStatement("rowSums(rowVec(A))")
				.build();

		RewriterRuleSet rs = new RewriterRuleSet(ctx, List.of(rule));

		RewriterStatement stmt = RewriterUtils.parse("as.matrix(sum(t(rowVec(A))))", ctx, "MATRIX:A,B");

		RewriterRuleSet.ApplicableRule ar = rs.acceleratedFindFirst(stmt);

		assert ar != null;

		stmt = ar.rule.apply(ar.matches.get(0), stmt, ar.forward, false);
	}
}
