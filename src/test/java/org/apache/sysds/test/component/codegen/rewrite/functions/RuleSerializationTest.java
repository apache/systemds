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
import org.apache.sysds.hops.rewriter.assertions.RewriterAssertions;
import org.apache.sysds.hops.rewriter.estimators.RewriterCostEstimator;
import org.apache.sysds.hops.rewriter.rule.RewriterRule;
import org.apache.sysds.hops.rewriter.rule.RewriterRuleCreator;
import org.apache.sysds.hops.rewriter.rule.RewriterRuleSet;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.junit.BeforeClass;
import org.junit.Test;
import scala.Tuple2;

import java.util.List;
import java.util.Set;
import java.util.function.Function;

public class RuleSerializationTest {
	protected static final Log LOG = LogFactory.getLog(RuleSerializationTest.class.getName());

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
		String serialized = ruleSet.serialize();

		LOG.info(serialized);

		RewriterRuleSet newRuleSet = RewriterRuleSet.deserialize(serialized, ctx);
		String newSerialized = newRuleSet.serialize();

		LOG.info(newSerialized);

		assert serialized.equals(newSerialized);
	}

	@Test
	public void test2() {
		RewriterStatement from = RewriterUtils.parse("t(t(U))", ctx, "MATRIX:U,V");
		RewriterStatement to = RewriterUtils.parse("U", ctx, "MATRIX:U,V");
		RewriterStatement canonicalForm1 = canonicalConverter.apply(from);
		RewriterStatement canonicalForm2 = canonicalConverter.apply(to);

		LOG.info("==========");
		LOG.info(canonicalForm1.toParsableString(ctx, true));
		LOG.info("==========");
		LOG.info(canonicalForm2.toParsableString(ctx, true));
		assert canonicalForm1.match(RewriterStatement.MatcherContext.exactMatch(ctx, canonicalForm2, canonicalForm1));

		RewriterRule rule = RewriterRuleCreator.createRule(from, to, canonicalForm1, canonicalForm2, ctx);
		from = rule.getStmt1();
		to = rule.getStmt2();

		MutableObject<RewriterAssertions> assertionRef = new MutableObject<>();
		long fullCost = RewriterCostEstimator.estimateCost(to, ctx);
		long maxCost = RewriterCostEstimator.estimateCost(from, ctx, assertionRef);
		Tuple2<Set<RewriterStatement>, Boolean> result = RewriterCostEstimator.determineSingleReferenceRequirement(from, RewriterCostEstimator.DEFAULT_COST_FN, assertionRef.getValue(), fullCost, maxCost, ctx);

		assert result._1.size() == 1 && result._2;

		rule.setAllowedMultiReferences(result._1, result._2);

		String serialized = rule.toParsableString(ctx);

		LOG.info("::RULE");
		LOG.info(serialized);
		LOG.info("");

		RewriterRule newRule = RewriterUtils.parseRule(serialized, ctx);
		String newSerialized = newRule.toParsableString(ctx);

		LOG.info(newSerialized);

		assert serialized.equals(newSerialized);
	}

	@Test
	public void test3() {
		RewriterStatement from = RewriterUtils.parse("sum(t(U))", ctx, "MATRIX:U,V");
		RewriterStatement to = RewriterUtils.parse("sum(U)", ctx, "MATRIX:U,V");
		RewriterStatement canonicalForm1 = canonicalConverter.apply(from);
		RewriterStatement canonicalForm2 = canonicalConverter.apply(to);

		LOG.info("==========");
		LOG.info(canonicalForm1.toParsableString(ctx, true));
		LOG.info("==========");
		LOG.info(canonicalForm2.toParsableString(ctx, true));
		assert canonicalForm1.match(RewriterStatement.MatcherContext.exactMatch(ctx, canonicalForm2, canonicalForm1));

		RewriterRule rule = RewriterRuleCreator.createRule(from, to, canonicalForm1, canonicalForm2, ctx);
		from = rule.getStmt1();
		to = rule.getStmt2();

		MutableObject<RewriterAssertions> assertionRef = new MutableObject<>();
		long fullCost = RewriterCostEstimator.estimateCost(to, ctx);
		long maxCost = RewriterCostEstimator.estimateCost(from, ctx, assertionRef);
		Tuple2<Set<RewriterStatement>, Boolean> result = RewriterCostEstimator.determineSingleReferenceRequirement(from, RewriterCostEstimator.DEFAULT_COST_FN, assertionRef.getValue(), fullCost, maxCost, ctx);

		assert result._1.size() == 1 && result._2;

		rule.setAllowedMultiReferences(result._1, result._2);

		String serialized = rule.toParsableString(ctx);

		LOG.info("::RULE");
		LOG.info(serialized);
		LOG.info("");

		RewriterRule newRule = RewriterUtils.parseRule(serialized, ctx);
		String newSerialized = newRule.toParsableString(ctx);

		LOG.info(newSerialized);

		assert serialized.equals(newSerialized);
	}

	@Test
	public void test4() {
		String ruleStr1 = "MATRIX:W1_rand,tmp29911\n" +
				"FLOAT:tmp65095\n" +
				"\n" +
				"*(tmp65095,%*%(W1_rand,t(tmp29911)))\n" +
				"=>\n" +
				"{\n" +
				"t(%*%(*(tmp65095,tmp29911),t(W1_rand)))\n" +
				"%*%(*(tmp65095,W1_rand),t(tmp29911))\n" +
				"*(tmp65095,t(%*%(tmp29911,t(W1_rand))))\n" +
				"}";
		RewriterRule rule1 = RewriterUtils.parseRule(ruleStr1, ctx);
		LOG.info(rule1.toString());
	}
}
