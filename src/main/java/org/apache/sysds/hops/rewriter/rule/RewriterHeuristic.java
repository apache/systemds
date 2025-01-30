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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.mutable.MutableBoolean;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.hops.rewriter.RewriterInstruction;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RuleContext;

import javax.annotation.Nullable;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;

public class RewriterHeuristic implements RewriterHeuristicTransformation {
	private final RewriterRuleSet ruleSet;
	private final Function<RewriterStatement, RewriterStatement> f;
	private final boolean accelerated;

	public RewriterHeuristic(RewriterRuleSet ruleSet) {
		this(ruleSet, true);
	}

	public RewriterHeuristic(RewriterRuleSet ruleSet, boolean accelerated) {
		this.ruleSet = ruleSet;
		this.accelerated = accelerated;
		this.f = null;
	}

	public RewriterHeuristic(Function<RewriterStatement, RewriterStatement> f) {
		this.ruleSet = null;
		this.accelerated = false;
		this.f = f;
	}

	public void forEachRuleSet(Consumer<RewriterRuleSet> consumer, boolean printNames) {
		consumer.accept(ruleSet);
	}

	public RewriterStatement apply(RewriterStatement current) {
		return apply(current, null);
	}

	public RewriterStatement apply(RewriterStatement current, @Nullable BiFunction<RewriterStatement, RewriterRule, Boolean> handler) {
		return apply(current, handler, new MutableBoolean(false), true);
	}

	public RewriterStatement apply(RewriterStatement currentStmt, @Nullable BiFunction<RewriterStatement, RewriterRule, Boolean> handler, MutableBoolean foundRewrite, boolean print) {
		if (f != null)
			return f.apply(currentStmt);

		RuleContext.currentContext = ruleSet.getContext();

		if (handler != null && !handler.apply(currentStmt, null))
			return currentStmt;

		RewriterRuleSet.ApplicableRule rule;
		if (accelerated)
			rule = ruleSet.acceleratedFindFirst(currentStmt);
		else
			throw new NotImplementedException("Must use accelerated mode");

		if (rule != null)
			foundRewrite.setValue(true);

		for (int i = 0; i < 500 && rule != null; i++) {
			currentStmt = rule.rule.apply(rule.matches.get(0), currentStmt, rule.forward, false);

			if (handler != null && !handler.apply(currentStmt, rule.rule)) {
				rule = null;
				break;
			}

			if (!(currentStmt instanceof RewriterInstruction)) {
				rule = null;
				break;
			}

			if (accelerated)
				rule = ruleSet.acceleratedFindFirst(currentStmt);
			else
				throw new IllegalArgumentException("Must use accelerated mode!");
		}

		if (rule != null)
			throw new IllegalArgumentException("Expression did not converge:\n" + currentStmt.toParsableString(ruleSet.getContext(), true) + "\nRule: " + rule);

		return currentStmt;
	}

	@Override
	public String toString() {
		return ruleSet.toString();
	}
}
