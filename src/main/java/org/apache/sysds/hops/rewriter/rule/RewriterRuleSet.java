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

package org.apache.sysds.hops.rewriter.rule;

import org.apache.commons.collections4.bidimap.DualHashBidiMap;
import org.apache.commons.lang3.mutable.MutableObject;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.rewriter.RewriterInstruction;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.hops.rewriter.codegen.RewriterCodeGen;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;
import scala.Tuple2;
import scala.Tuple3;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class RewriterRuleSet {

	public static class ApplicableRule {
		public final ArrayList<RewriterStatement.MatchingSubexpression> matches;
		public final RewriterRule rule;
		public final boolean forward;

		public ApplicableRule(ArrayList<RewriterStatement.MatchingSubexpression> matches, RewriterRule rule, boolean forward) {
			this.matches = matches;
			this.rule = rule;
			this.forward = forward;
		}

		public String toString(final RuleContext ctx) {
			StringBuilder builder = new StringBuilder();
			builder.append("Rule: " + rule + "\n\n");
			int ctr = 1;
			for (RewriterStatement.MatchingSubexpression match : matches) {
				builder.append("Match " + ctr++ + ": \n");
				builder.append(" " + match.getMatchRoot() + " = " + (forward ? rule.getStmt1() : rule.getStmt2())  + "\n\n");
				for (Map.Entry<RewriterStatement, RewriterStatement> entry : match.getAssocs().entrySet()) {
					builder.append(" - " + entry.getKey() + "::" + (ctx == null ? "?" : entry.getKey().getResultingDataType(ctx)) + " -> " + entry.getValue().getId() + "::" + (ctx == null ? "?" : entry.getValue().getResultingDataType(ctx)) + "\n");
				}
				builder.append("\n");
			}

			return builder.toString();
		}

		@Override
		public String toString() {
			return toString(null);
		}
	}

	private RuleContext ctx;
	private List<RewriterRule> rules;
	private Map<String, List<Tuple2<RewriterRule, Boolean>>> accelerator;

	public RewriterRuleSet(RuleContext ctx, List<RewriterRule> rules) {
		this.ctx = ctx;
		this.rules = rules;
		accelerate();
	}

	public RuleContext getContext() {
		return ctx;
	}

	public void determineConditionalApplicability() {
		rules.forEach(RewriterRule::determineConditionalApplicability);
	}

	public void forEachRule(BiConsumer<RewriterRule, RuleContext> consumer) {
		rules.forEach(r -> consumer.accept(r, ctx));
	}

	public List<RewriterRule> getRules() {
		return rules;
	}

	public ApplicableRule acceleratedFindFirst(RewriterStatement root) {
		return acceleratedFindFirst(root, false);
	}

	public ApplicableRule acceleratedFindFirst(RewriterStatement root, boolean allowImplicitTypeConversions) {
		List<ApplicableRule> match = acceleratedRecursiveMatch(root, true, allowImplicitTypeConversions);
		if (match.isEmpty())
			return null;
		else
			return match.get(0);
	}

	public List<ApplicableRule> acceleratedRecursiveMatch(RewriterStatement root, boolean findFirst, boolean allowImplicitTypeConversions) {
		List<Tuple3<RewriterRule, Boolean, RewriterStatement.MatchingSubexpression>> matches = new ArrayList<>();
		MutableObject<HashMap<RewriterStatement, RewriterStatement>> dependencyMap = new MutableObject<>(new HashMap<>());
		MutableObject<List<RewriterRule.ExplicitLink>> links = new MutableObject<>(new ArrayList<>());
		MutableObject<Map<RewriterStatement, RewriterRule.LinkObject>> linkObjects = new MutableObject<>(new HashMap<>());

		root.forEachPreOrder((el, pred) -> {
			String typedStr = el.isInstruction() ? el.trueTypedInstruction(allowImplicitTypeConversions, ctx) : RewriterUtils.convertImplicitly(el.getResultingDataType(ctx), allowImplicitTypeConversions);
			Set<String> props = el instanceof RewriterInstruction ? ((RewriterInstruction)el).getProperties(ctx) : Collections.emptySet();
			boolean found = acceleratedMatch(root, el, matches, typedStr, RewriterUtils.convertImplicitly(el.getResultingDataType(ctx), allowImplicitTypeConversions), props, pred, dependencyMap, links, linkObjects, findFirst, allowImplicitTypeConversions);
			return !findFirst || !found;
		}, true);

		Map<Tuple2<RewriterRule, Boolean>, ApplicableRule> uniqueRules = new HashMap<>();

		for (Tuple3<RewriterRule, Boolean, RewriterStatement.MatchingSubexpression> match : matches) {
			Tuple2<RewriterRule, Boolean> t = new Tuple2<>(match._1(), match._2());

			if (uniqueRules.containsKey(t))
				uniqueRules.get(t).matches.add(match._3());
			else {
				ArrayList<RewriterStatement.MatchingSubexpression> list = new ArrayList<>();
				list.add(match._3());
				uniqueRules.put(t, new ApplicableRule(list, match._1(), match._2()));
			}
		}

		return new ArrayList<>(uniqueRules.values());
	}

	public boolean acceleratedMatch(RewriterStatement exprRoot, RewriterStatement stmt, List<Tuple3<RewriterRule, Boolean, RewriterStatement.MatchingSubexpression>> appRules, String realTypedInstr, String realType, Set<String> properties, RewriterStatement.RewriterPredecessor pred, MutableObject<HashMap<RewriterStatement, RewriterStatement>> dependencyMap, MutableObject<List<RewriterRule.ExplicitLink>> links, MutableObject<Map<RewriterStatement, RewriterRule.LinkObject>> linkObjects, boolean findFirst, boolean allowImplicitTypeConversions) {
		List<Tuple2<RewriterRule, Boolean>> potentialMatches;
		boolean foundMatch = false;

		if (realTypedInstr != null) {
			potentialMatches = accelerator.get(realTypedInstr);
			if (potentialMatches != null) {
				foundMatch |= checkPotentialMatches(stmt, potentialMatches, appRules, pred, dependencyMap, links, linkObjects, exprRoot, findFirst, allowImplicitTypeConversions);

				if (foundMatch && findFirst)
					return true;
			}
		}

		potentialMatches = accelerator.get(realType);
		if (potentialMatches != null) {
			foundMatch |= checkPotentialMatches(stmt, potentialMatches, appRules, pred, dependencyMap, links, linkObjects, exprRoot, findFirst, allowImplicitTypeConversions);

			if (foundMatch && findFirst)
				return true;
		}

		if (properties != null) {
			for (String props : properties) {
				potentialMatches = accelerator.get(props);
				if (potentialMatches != null) {
					foundMatch |= checkPotentialMatches(stmt, potentialMatches, appRules, pred, dependencyMap, links, linkObjects, exprRoot, findFirst, allowImplicitTypeConversions);

					if (foundMatch && findFirst)
						return true;
				}
			}
		}

		return foundMatch;
	}

	private boolean checkPotentialMatches(RewriterStatement stmt, List<Tuple2<RewriterRule, Boolean>> potentialMatches, List<Tuple3<RewriterRule, Boolean, RewriterStatement.MatchingSubexpression>> appRules, RewriterStatement.RewriterPredecessor pred, MutableObject<HashMap<RewriterStatement, RewriterStatement>> dependencyMap, MutableObject<List<RewriterRule.ExplicitLink>> links, MutableObject<Map<RewriterStatement, RewriterRule.LinkObject>> linkObjects, RewriterStatement exprRoot, boolean findFirst, boolean allowImplicitTypeConversions) {
		boolean anyMatch = false;
		for (Tuple2<RewriterRule, Boolean> m : potentialMatches) {
			RewriterStatement.MatchingSubexpression match;

			if (m._2()) {
				match = m._1().matchSingleStmt1(exprRoot, pred, stmt, allowImplicitTypeConversions);
			} else {
				match = m._1().matchSingleStmt2(exprRoot, pred, stmt, allowImplicitTypeConversions);
			}

			if (match != null) {
				appRules.add(new Tuple3<>(m._1(), m._2(), match));
				dependencyMap.setValue(new HashMap<>());
				links.setValue(new ArrayList<>());
				linkObjects.setValue(new HashMap<>());

				if (findFirst)
					return true;

				anyMatch = true;
			} else {
				dependencyMap.getValue().clear();
				links.getValue().clear();
				linkObjects.getValue().clear();
			}
		}

		return anyMatch;
	}

	// Look for intersecting roots and try to find them once
	public void accelerate() {
		accelerator = new HashMap<>();
		for (RewriterRule rule : rules) {
			accelerate(rule, true);
			if (!rule.isUnidirectional())
				accelerate(rule, false);
		}
	}

	private void accelerate(RewriterRule rule, boolean forward) {
		RewriterStatement stmt = forward ? rule.getStmt1() : rule.getStmt2();
		String t = stmt.isInstruction() ? stmt.trueTypedInstruction(ctx) : stmt.getResultingDataType(ctx);
		List<Tuple2<RewriterRule, Boolean>> l = accelerator.get(t);

		if (l == null) {
			l = new ArrayList<>();
			accelerator.put(t, l);
		}

		l.add(new Tuple2<>(rule, forward));
	}

	@Override
	public String toString() {
		return serialize();
	}

	public String serialize() {
		StringBuilder sb = new StringBuilder();

		for (RewriterRule rule : rules) {
			try {
				sb.append("::RULE\n");
				sb.append(rule.toParsableString(ctx));
				sb.append("\n\n");
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		return sb.toString();
	}

	public Set<RewriterRule> generateCodeAndTest(boolean optimize, boolean print) {
		String javaCode = toJavaCode("MGeneratedRewriteClass", optimize, false, true, true);
		Function<Hop, Hop> f = RewriterCodeGen.compile(javaCode, "MGeneratedRewriteClass");

		if (f == null)
			return null; // Then, the code could not compile

		Set<RewriterRule> removed = new HashSet<>();

		for (int i = 0; i < rules.size(); i++) {
			if (!RewriterRuleCreator.validateRuleApplicability(rules.get(i), ctx, print, f)) {
				System.out.println("Faulty rule: " + rules.get(i));
				removed.add(rules.get(i));
			}
		}

		return removed;
	}

	public static RewriterRuleSet deserialize(String data, final RuleContext ctx) {
		return deserialize(data.split("\n"), ctx);
	}

	public static RewriterRuleSet deserialize(List<String> data, final RuleContext ctx) {
		return deserialize(data.toArray(String[]::new), ctx);
	}

	public static RewriterRuleSet deserialize(String[] data, final RuleContext ctx) {
		List<String> currentLines = new ArrayList<>();
		List<RewriterRule> rules = new ArrayList<>();

		for (int i = 0; i < data.length; i++) {
			if (data[i].equals("::RULE")) {
				if (!currentLines.isEmpty()) {
					try {
						rules.add(RewriterUtils.parseRule(String.join("\n", currentLines), ctx));
					} catch (Exception e) {
						System.err.println("An error occurred while parsing the rule:\n" + String.join("\n", currentLines));
						e.printStackTrace();
					}
					currentLines.clear();
				}
			} else {
				currentLines.add(data[i]);
			}
		}

		if (!currentLines.isEmpty()) {
			rules.add(RewriterUtils.parseRule(String.join("\n", currentLines), ctx));
			currentLines.clear();
		}

		for (RewriterRule rule : rules) {
			try {
				rule.determineConditionalApplicability();
			} catch (Exception e) {
				System.err.println("Error while determining the conditional ability of " + rule.toString());
				e.printStackTrace();
			}
		}

		return new RewriterRuleSet(ctx, rules);
	}

	public String toJavaCode(String className, boolean optimize, boolean includePackageInfo, boolean printErrors, boolean maintainStatistics) {
		List<Tuple2<String, RewriterRule>> mRules = IntStream.range(0, rules.size()).mapToObj(i -> new Tuple2<>("_applyRewrite" + i, rules.get(i))).collect(Collectors.toList());
		return RewriterCodeGen.generateClass(className, mRules, optimize, 2, includePackageInfo, ctx, true, printErrors, maintainStatistics);
	}

	public String toJavaCode(String className, boolean optimize) {
		List<Tuple2<String, RewriterRule>> mRules = IntStream.range(0, rules.size()).mapToObj(i -> new Tuple2<>("_applyRewrite" + i, rules.get(i))).collect(Collectors.toList());
		return RewriterCodeGen.generateClass(className, mRules, optimize, 2, true, ctx, true, true, false);
	}

	public String toJavaCode(String className, boolean optimize, int maxOptimizationDepth, boolean includePackageInfo, boolean printErrors, boolean maintainStatistics) {
		List<Tuple2<String, RewriterRule>> mRules = IntStream.range(0, rules.size()).mapToObj(i -> new Tuple2<>("_applyRewrite" + i, rules.get(i))).collect(Collectors.toList());
		return RewriterCodeGen.generateClass(className, mRules, optimize, maxOptimizationDepth, includePackageInfo, ctx, true, printErrors, maintainStatistics);
	}

	public Function<Hop, Hop> compile(String className, boolean printErrors) {
		try {
			List<Tuple2<String, RewriterRule>> mRules = IntStream.range(0, rules.size()).mapToObj(i -> new Tuple2<>("_applyRewrite" + i, rules.get(i))).collect(Collectors.toList());
			return RewriterCodeGen.compileRewrites(className, mRules, ctx, true, printErrors);
		} catch (Exception e) {
			if (printErrors)
				e.printStackTrace();

			return null;
		}
	}
}
