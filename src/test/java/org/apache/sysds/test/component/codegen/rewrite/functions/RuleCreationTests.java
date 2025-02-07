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
import org.apache.sysds.hops.rewriter.rule.RewriterRuleBuilder;
import org.apache.sysds.hops.rewriter.rule.RewriterRuleCreator;
import org.apache.sysds.hops.rewriter.rule.RewriterRuleSet;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.test.component.codegen.rewrite.RewriterTopologySortTests;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.List;
import java.util.function.Function;

public class RuleCreationTests {
	protected static final Log LOG = LogFactory.getLog(RuleCreationTests.class.getName());

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

		LOG.info("==========");
		LOG.info(canonicalForm1.toParsableString(ctx, true));
		LOG.info("==========");
		LOG.info(canonicalForm2.toParsableString(ctx, true));
		assert canonicalForm1.match(RewriterStatement.MatcherContext.exactMatch(ctx, canonicalForm2, canonicalForm1));

		RewriterRule rule = RewriterRuleCreator.createRule(from, to, canonicalForm1, canonicalForm2, ctx);
		LOG.info(rule);
	}

	@Test
	public void test2() {
		RewriterStatement from = RewriterUtils.parse("t(t(A))", ctx, "MATRIX:A");
		RewriterStatement to = RewriterUtils.parse("A", ctx, "MATRIX:A");
		RewriterStatement canonicalForm1 = canonicalConverter.apply(from);
		RewriterStatement canonicalForm2 = canonicalConverter.apply(to);

		LOG.info("==========");
		LOG.info(canonicalForm1.toParsableString(ctx, true));
		LOG.info("==========");
		LOG.info(canonicalForm2.toParsableString(ctx, true));
		assert canonicalForm1.match(RewriterStatement.MatcherContext.exactMatch(ctx, canonicalForm2, canonicalForm1));

		RewriterRule rule = RewriterRuleCreator.createRule(from, to, canonicalForm1, canonicalForm2, ctx);
		LOG.info(rule);

		RewriterRuleSet rs = new RewriterRuleSet(ctx, List.of(rule));

		RewriterStatement testStmt = RewriterUtils.parse("t(t([](A, 1, ncol(A), 1, 1)))", ctx, "MATRIX:A", "LITERAL_INT:1");

		RewriterRuleSet.ApplicableRule ar = rs.acceleratedFindFirst(testStmt);

		assert ar != null;
	}

	@Test
	public void validationTest1() {
		RewriterRule rule = new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A")
				.parseGlobalVars("FLOAT:b")
				.withParsedStatement("sum(/(A, b))")
				.toParsedStatement("/(sum(A), b)")
				.build();

		assert RewriterRuleCreator.validateRuleCorrectnessAndGains(rule, ctx);
	}

	@Test
	public void validationTest2() {
		RewriterRule rule = new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("FLOAT:b")
				.withParsedStatement("rowSums(colSums(%*%(A, B)))")
				.toParsedStatement("%*%(colSums(A), rowSums(B))")
				.build();

		assert RewriterRuleCreator.validateRuleCorrectness(rule, ctx);
		assert !RewriterRuleCreator.validateRuleApplicability(rule, ctx);
	}

	@Test
	public void validationTest3() {
		RewriterRule rule = new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.withParsedStatement("cast.MATRIX(sum(rowVec(A)))")
				.toParsedStatement("rowSums(rowVec(A))")
				.build();

		assert RewriterRuleCreator.validateRuleCorrectness(rule, ctx);
		assert !RewriterRuleCreator.validateRuleApplicability(rule, ctx);
	}

	@Test
	public void test3() {
		RewriterStatement from = RewriterUtils.parse("%*%(A,%*%(B,rowVec(C)))", ctx, "MATRIX:A,B,C");
		RewriterStatement to = RewriterUtils.parse("%*%(%*%(A,B),rowVec(C))", ctx, "MATRIX:A,B,C");
		RewriterStatement canonicalForm1 = canonicalConverter.apply(from);
		RewriterStatement canonicalForm2 = canonicalConverter.apply(to);

		LOG.info("==========");
		LOG.info(canonicalForm1.toParsableString(ctx, true));
		LOG.info("==========");
		LOG.info(canonicalForm2.toParsableString(ctx, true));

		assert canonicalForm1.match(RewriterStatement.MatcherContext.exactMatch(ctx, canonicalForm2, canonicalForm1));
	}

	@Test
	public void test4() {
		RewriterStatement from = RewriterUtils.parse("*(a,0.0)", ctx, "FLOAT:a", "LITERAL_FLOAT:0.0");
		RewriterStatement to = RewriterUtils.parse("0.0", ctx, "LITERAL_FLOAT:0.0");
		RewriterStatement canonicalForm1 = canonicalConverter.apply(from);
		RewriterStatement canonicalForm2 = canonicalConverter.apply(to);

		LOG.info("==========");
		LOG.info(canonicalForm1.toParsableString(ctx, true));
		LOG.info("==========");
		LOG.info(canonicalForm2.toParsableString(ctx, true));

		assert canonicalForm1.match(RewriterStatement.MatcherContext.exactMatch(ctx, canonicalForm2, canonicalForm1));

		RewriterRule rule = RewriterRuleCreator.createRule(from, to, canonicalForm1, canonicalForm2, ctx);
		LOG.info(rule);

		RewriterStatement from2 = RewriterUtils.parse("/(0.0,a)", ctx, "FLOAT:a", "LITERAL_FLOAT:0.0");
		RewriterStatement to2 = RewriterUtils.parse("0.0", ctx, "LITERAL_FLOAT:0.0");
		RewriterStatement canonicalForm12 = canonicalConverter.apply(from2);
		RewriterStatement canonicalForm22 = canonicalConverter.apply(to2);

		LOG.info("==========");
		LOG.info(canonicalForm12.toParsableString(ctx, true));
		LOG.info("==========");
		LOG.info(canonicalForm22.toParsableString(ctx, true));

		assert canonicalForm12.match(RewriterStatement.MatcherContext.exactMatch(ctx, canonicalForm22, canonicalForm12));

		RewriterRule rule2 = RewriterRuleCreator.createRule(from2, to2, canonicalForm12, canonicalForm22, ctx);
		LOG.info(rule2);

		RewriterRuleSet rs = new RewriterRuleSet(ctx, List.of(rule, rule2));

		RewriterStatement testStmt = RewriterUtils.parse("/(*(a,0.0), b)", ctx, "FLOAT:a,b", "LITERAL_FLOAT:0.0");

		RewriterRuleSet.ApplicableRule ar = rs.acceleratedFindFirst(testStmt);

		assert ar != null;

		testStmt = ar.rule.apply(ar.matches.get(0), testStmt, true, false);

		LOG.info("HERE");
		LOG.info(testStmt.toParsableString(ctx));

		ar = rs.acceleratedFindFirst(testStmt);

		assert ar != null;

		testStmt = ar.rule.apply(ar.matches.get(0), testStmt, true, false);

		LOG.info(testStmt);
	}

	@Test
	public void test5() {
		RewriterRule rule1 = RewriterUtils.parseRule("FLOAT:a\nLITERAL_FLOAT:0.0\n*(a, 0.0)\n=>\n0.0", ctx);
		RewriterRule rule2 = RewriterUtils.parseRule("FLOAT:a\nLITERAL_FLOAT:0.0\n/(0.0, a)\n=>\n0.0", ctx);
		RewriterRule rule3 = RewriterUtils.parseRule("FLOAT:a,b\nLITERAL_FLOAT:0.0\n/(*(a, 0.0), b)\n=>\n0.0", ctx);
		RewriterRuleCreator rc = new RewriterRuleCreator(ctx);
		rc.registerRule(rule3, rule3.getStmt1().getCost(ctx), rule3.getStmt2().getCost(ctx), false, canonicalConverter);
		rc.registerRule(rule2, rule2.getStmt1().getCost(ctx), rule2.getStmt2().getCost(ctx), false, canonicalConverter);
		rc.registerRule(rule1, rule1.getStmt1().getCost(ctx), rule1.getStmt2().getCost(ctx), false, canonicalConverter);

		LOG.info(rc.getRuleSet().serialize());
	}

	@Test
	public void test6() {
		RewriterStatement from = RewriterUtils.parse("%*%(const(colVec(A),0.0),log_nz(B))", ctx, "MATRIX:A,B", "LITERAL_FLOAT:0.0");
		RewriterStatement to = RewriterUtils.parse("%*%(colVec(A),const(B,0.0))", ctx, "MATRIX:A,B", "LITERAL_FLOAT:0.0");
		RewriterStatement canonicalForm1 = canonicalConverter.apply(from);
		RewriterStatement canonicalForm2 = canonicalConverter.apply(to);

		LOG.info("==========");
		LOG.info(canonicalForm1.toParsableString(ctx, true));
		LOG.info("==========");
		LOG.info(canonicalForm2.toParsableString(ctx, true));
		/*LOG.info(canonicalForm1.getChild(1, 1, 0));
		LOG.info(canonicalForm1.getChild(1, 1, 0).getNCol());
		LOG.info(canonicalForm1.getChild(1, 1, 0).getNRow());
		LOG.info(canonicalForm2.getChild(1, 1, 0));
		LOG.info(canonicalForm2.getChild(1, 1, 0).getNCol());
		LOG.info(canonicalForm2.getChild(1, 1, 0).getNRow());*/
		RewriterStatement.MatcherContext mCtx = RewriterStatement.MatcherContext.exactMatch(ctx, canonicalForm2, canonicalForm1);
		if (!canonicalForm1.match(mCtx)) {
			LOG.info(mCtx.getFirstMismatch()._1);
			LOG.info(mCtx.getFirstMismatch()._2);
			assert false;
		}

		RewriterRule rule = RewriterRuleCreator.createRule(from, to, canonicalForm1, canonicalForm2, ctx);
		LOG.info(rule);
	}

	@Test
	public void testTypeInvariantRuleRegistration() {
		RewriterRule rule1 = RewriterUtils.parseRule("FLOAT:a\nLITERAL_FLOAT:0\n*(a,0)\n=>\na", ctx);
		RewriterRule rule2 = RewriterUtils.parseRule("INT:a\nLITERAL_INT:0\n*(a,0)\n=>\na", ctx);
		RewriterRuleCreator ruleCreator = new RewriterRuleCreator(ctx);
		ruleCreator.registerRule(rule1, canonicalConverter, ctx);

		assert !ruleCreator.registerRule(rule2, canonicalConverter, ctx);
	}

	@Test
	public void testRuleElimination() {
		String rs1 =
				"MATRIX:tmp34827,tmp40318\n" +
				"LITERAL_FLOAT:0.0\n" +
				"\n" +
				"+(%*%(tmp34827,tmp40318),0.0)\n" +
				"=>\n" +
				"%*%(tmp34827,tmp40318)";
		String rs2 =
				"MATRIX:tmp34827,tmp40318\n" +
						"LITERAL_FLOAT:0.0\n" +
						"\n" +
						"+(tmp34827,0.0)\n" +
						"=>\n" +
						"tmp34827";

		RewriterRule rule1 = RewriterUtils.parseRule(rs1, ctx);
		RewriterRule rule2 = RewriterUtils.parseRule(rs2, ctx);

		RewriterRuleCreator ruleCreator = new RewriterRuleCreator(ctx);
		ruleCreator.registerRule(rule1, canonicalConverter, ctx);

		assert ruleCreator.registerRule(rule2, canonicalConverter, ctx);
		LOG.info(ruleCreator.getRuleSet().getRules());
		assert ruleCreator.getRuleSet().getRules().size() == 1;
	}

	@Test
	public void testExpansiveRule() {
		String rs1 =
				"MATRIX:A,B\n" +
						"LITERAL_FLOAT:0.0\n" +
						"\n" +
						"+*(A,0.0,B)\n" +
						"=>\n" +
						"+*(A,0.0,!=(B,B))";

		RewriterRule rule1 = RewriterUtils.parseRule(rs1, ctx);

		RewriterRuleCreator ruleCreator = new RewriterRuleCreator(ctx);
		assert !ruleCreator.registerRule(rule1, canonicalConverter, ctx);
	}
}
