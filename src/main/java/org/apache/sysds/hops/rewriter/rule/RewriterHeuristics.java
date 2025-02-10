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

import org.apache.commons.lang3.mutable.MutableBoolean;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.jetbrains.annotations.Nullable;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Consumer;

public class RewriterHeuristics implements RewriterHeuristicTransformation {
	protected static final Log LOG = LogFactory.getLog(RewriterHeuristic.class.getName());
	List<HeuristicEntry> heuristics = new ArrayList<>();

	public void forEachRuleSet(Consumer<RewriterRuleSet> consumer, boolean printNames) {
		heuristics.forEach(entry -> {
			if (printNames) {
				LOG.info("\n");
				LOG.info("> " + entry.name + " <");
				LOG.info("\n");
			}
			entry.heuristics.forEachRuleSet(consumer, printNames);
		});
	}

	public void add(String name, RewriterHeuristicTransformation heur) {
		heuristics.add(new HeuristicEntry(name, heur));
	}

	public void addRepeated(String name, RewriterHeuristicTransformation heur) {
		heuristics.add(new HeuristicEntry(name, new RepeatedHeuristics(heur)));
	}

	@Override
	public RewriterStatement apply(RewriterStatement stmt, @Nullable BiFunction<RewriterStatement, RewriterRule, Boolean> func, MutableBoolean bool, boolean print) {
		for (HeuristicEntry entry : heuristics) {
			if (print) {
				System.out.println("\n");
				System.out.println("> " + entry.name + " <");
				System.out.println("\n");
			}

			stmt = entry.heuristics.apply(stmt, func, bool, print);
		}

		return stmt;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();

		for (HeuristicEntry entry : heuristics) {
			sb.append("\n> ");
			sb.append(entry.name);
			sb.append(" <\n");

			sb.append(entry.heuristics.toString());
		}

		return sb.toString();
	}

	class RepeatedHeuristics implements RewriterHeuristicTransformation {
		RewriterHeuristicTransformation heuristic;

		public RepeatedHeuristics(RewriterHeuristicTransformation heuristic) {
			this.heuristic = heuristic;
		}

		@Override
		public RewriterStatement apply(RewriterStatement stmt, @Nullable BiFunction<RewriterStatement, RewriterRule, Boolean> func, MutableBoolean bool, boolean print) {
			bool.setValue(true);

			while (bool.getValue()) {
				bool.setValue(false);
				stmt = heuristic.apply(stmt, func, bool, print);
			}

			return stmt;
		}

		@Override
		public void forEachRuleSet(Consumer<RewriterRuleSet> consumer, boolean printNames) {
			heuristic.forEachRuleSet(consumer, printNames);
		}

		@Override
		public String toString() {
			StringBuilder sb = new StringBuilder();

			sb.append("\n===== REPEAT =====\n");

			for (HeuristicEntry entry : heuristics) {
				sb.append("\n> ");
				sb.append(entry.name);
				sb.append(" <\n");

				sb.append(entry.heuristics.toString());
			}

			sb.append("\n===== END REPEAT =====");

			return sb.toString();
		}
	}


	class HeuristicEntry {
		String name;
		RewriterHeuristicTransformation heuristics;

		public HeuristicEntry(String name, RewriterHeuristicTransformation heuristics) {
			this.name = name;
			this.heuristics = heuristics;
		}
	}
}
