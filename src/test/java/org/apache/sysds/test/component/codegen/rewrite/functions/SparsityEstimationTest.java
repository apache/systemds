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

import org.apache.commons.lang3.mutable.MutableObject;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.hops.rewriter.rule.RewriterRule;
import org.apache.sysds.hops.rewriter.rule.RewriterRuleCreator;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.hops.rewriter.assertions.RewriterAssertionUtils;
import org.apache.sysds.hops.rewriter.assertions.RewriterAssertions;
import org.apache.sysds.hops.rewriter.estimators.RewriterCostEstimator;
import org.apache.sysds.hops.rewriter.estimators.RewriterSparsityEstimator;
import org.apache.sysds.test.component.codegen.rewrite.RewriterTopologySortTests;
import org.junit.BeforeClass;
import org.junit.Test;
import scala.Tuple3;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

public class SparsityEstimationTest {
	protected static final Log LOG = LogFactory.getLog(SparsityEstimationTest.class.getName());

	private static RuleContext ctx;
	private static Function<RewriterStatement, RewriterStatement> canonicalConverter;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
		canonicalConverter = RewriterUtils.buildCanonicalFormConverter(ctx, false);
	}

	@Test
	public void test1() {
		RewriterStatement stmt = RewriterUtils.parse("+*(A, 0.0, B)", ctx, "MATRIX:A,B", "LITERAL_FLOAT:0.0");
		LOG.info(RewriterSparsityEstimator.estimateNNZ(stmt, ctx).toParsableString(ctx));
	}

	@Test
	public void test2() {
		RewriterStatement stmt = RewriterUtils.parse("+*(A, a, B)", ctx, "MATRIX:A,B", "FLOAT:a");
		LOG.info(RewriterSparsityEstimator.estimateNNZ(stmt, ctx).toParsableString(ctx));
	}

	@Test
	public void test3() {
		RewriterStatement stmt = RewriterUtils.parse("%*%(A, -(B, A))", ctx, "MATRIX:A,B", "FLOAT:a");
		RewriterAssertionUtils.buildImplicitAssertions(stmt, stmt.getAssertions(ctx), ctx);

		Map<RewriterStatement, RewriterStatement> estimates = RewriterSparsityEstimator.estimateAllNNZ(stmt, ctx);

		estimates.forEach((k, v) -> {
			stmt.getAssertions(ctx).update(v);
			LOG.info("K: " + k.toParsableString(ctx));
			LOG.info("NNZ: " + v.toParsableString(ctx));
		});

		LOG.info("Rollup: " + RewriterSparsityEstimator.rollupSparsities(estimates.get(stmt), estimates, ctx).toParsableString(ctx));

		Map<RewriterStatement, Long> nnzs = new HashMap<>();
		nnzs.put(stmt.getChild(0), 3000L);
		nnzs.put(stmt.getChild(1, 0), 50000L);

		MutableObject<RewriterAssertions> assertionRef = new MutableObject<>();
		RewriterStatement costFunction = RewriterCostEstimator.getRawCostFunction(stmt, ctx, assertionRef, false);
		costFunction = RewriterSparsityEstimator.rollupSparsities(costFunction, estimates, ctx);

		LOG.info(costFunction.toParsableString(ctx));

		LOG.info("Dense cost:  " + RewriterCostEstimator.estimateCost(stmt, ctx));
		LOG.info("Sparse cost: " + RewriterCostEstimator.computeCostFunction(costFunction, RewriterCostEstimator.DEFAULT_COST_FN, (el, tpl) -> nnzs.get(el.getChild(0)), assertionRef.getValue(), ctx));
	}

	@Test
	public void test4() {
		RewriterStatement from = RewriterUtils.parse("+(*(A, B), *(A, C))", ctx, "MATRIX:A,B,C");
		RewriterStatement to = RewriterUtils.parse("*(A, +(B, C))", ctx, "MATRIX:A,B,C");
		RewriterStatement canonicalForm1 = canonicalConverter.apply(from);
		RewriterStatement canonicalForm2 = canonicalConverter.apply(to);

		LOG.info("==========");
		LOG.info(canonicalForm1.toParsableString(ctx, true));
		LOG.info("==========");
		LOG.info(canonicalForm2.toParsableString(ctx, true));
		assert canonicalForm1.match(RewriterStatement.MatcherContext.exactMatch(ctx, canonicalForm2, canonicalForm1));

		RewriterRule rule = RewriterRuleCreator.createRule(from, to, canonicalForm1, canonicalForm2, ctx);
		LOG.info(rule);

		RewriterAssertionUtils.buildImplicitAssertion(rule.getStmt1(), rule.getStmt1().getAssertions(ctx), rule.getStmt1(), ctx);
		RewriterAssertionUtils.buildImplicitAssertion(rule.getStmt2(), rule.getStmt1().getAssertions(ctx), rule.getStmt2(), ctx);

		RewriterCostEstimator.compareCosts(rule.getStmt1(), rule.getStmt2(), rule.getStmt1().getAssertions(ctx), ctx, true, 5, false);
	}

	@Test
	public void test5() {
		RewriterStatement from = RewriterUtils.parse("t(%*%(t(A), B))", ctx, "MATRIX:A,B,C");
		RewriterStatement to = RewriterUtils.parse("%*%(t(B), A)", ctx, "MATRIX:A,B,C");
		RewriterStatement canonicalForm1 = canonicalConverter.apply(from);
		RewriterStatement canonicalForm2 = canonicalConverter.apply(to);

		LOG.info("==========");
		LOG.info(canonicalForm1.toParsableString(ctx, true));
		LOG.info("==========");
		LOG.info(canonicalForm2.toParsableString(ctx, true));
		assert canonicalForm1.match(RewriterStatement.MatcherContext.exactMatch(ctx, canonicalForm2, canonicalForm1));

		RewriterRule rule = RewriterRuleCreator.createRule(from, to, canonicalForm1, canonicalForm2, ctx);
		LOG.info(rule);

		RewriterAssertionUtils.buildImplicitAssertion(rule.getStmt1(), rule.getStmt1().getAssertions(ctx), rule.getStmt1(), ctx);
		RewriterAssertionUtils.buildImplicitAssertion(rule.getStmt2(), rule.getStmt1().getAssertions(ctx), rule.getStmt2(), ctx);
		//rule.getStmt2().unsafePutMeta("_assertions", rule.getStmt1().getAssertions(ctx));

		List<Tuple3<List<Number>, Long, Long>> costs = RewriterCostEstimator.compareCosts(rule.getStmt1(), rule.getStmt2(), rule.getStmt1().getAssertions(ctx), ctx, false, 5, false);
		LOG.info(costs);
		LOG.info("Does sparsity have an impact on optimal expression? >> " + RewriterCostEstimator.doesHaveAnImpactOnOptimalExpression(costs, true, true, 0));
	}

	@Test
	public void test6() {
		RewriterStatement from = RewriterUtils.parse("t(+(A, B))", ctx, "MATRIX:A,B,C");
		RewriterStatement to = RewriterUtils.parse("+(t(A), t(B))", ctx, "MATRIX:A,B,C");
		RewriterStatement canonicalForm1 = canonicalConverter.apply(from);
		RewriterStatement canonicalForm2 = canonicalConverter.apply(to);

		LOG.info("==========");
		LOG.info(canonicalForm1.toParsableString(ctx, true));
		LOG.info("==========");
		LOG.info(canonicalForm2.toParsableString(ctx, true));
		assert canonicalForm1.match(RewriterStatement.MatcherContext.exactMatch(ctx, canonicalForm2, canonicalForm1));

		RewriterRule rule = RewriterRuleCreator.createRule(from, to, canonicalForm1, canonicalForm2, ctx);
		LOG.info(rule);

		RewriterAssertionUtils.buildImplicitAssertion(rule.getStmt1(), rule.getStmt1().getAssertions(ctx), rule.getStmt1(), ctx);
		RewriterAssertionUtils.buildImplicitAssertion(rule.getStmt2(), rule.getStmt1().getAssertions(ctx), rule.getStmt2(), ctx);

		List<Tuple3<List<Number>, Long, Long>> costs = RewriterCostEstimator.compareCosts(rule.getStmt1(), rule.getStmt2(), rule.getStmt1().getAssertions(ctx), ctx, false, 5, false);
		LOG.info(costs);
		LOG.info("Does sparsity have an impact on optimal expression? >> " + RewriterCostEstimator.doesHaveAnImpactOnOptimalExpression(costs, true, true, 0));
		LOG.info("Does anything have an impact on optimal expression? >> " + RewriterCostEstimator.doesHaveAnImpactOnOptimalExpression(costs, true, false, 0));
	}
}
