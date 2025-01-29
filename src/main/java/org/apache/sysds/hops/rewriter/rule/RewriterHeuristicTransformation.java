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
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.rule.RewriterRule;
import org.apache.sysds.hops.rewriter.rule.RewriterRuleSet;

import javax.annotation.Nullable;
import java.util.function.BiFunction;
import java.util.function.Consumer;

public interface RewriterHeuristicTransformation {
	RewriterStatement apply(RewriterStatement stmt, @Nullable BiFunction<RewriterStatement, RewriterRule, Boolean> func, MutableBoolean bool, boolean print);

	void forEachRuleSet(Consumer<RewriterRuleSet> consumer, boolean printNames);

	default RewriterStatement apply(RewriterStatement stmt, @Nullable BiFunction<RewriterStatement, RewriterRule, Boolean> func) {
		return apply(stmt, func, new MutableBoolean(false), true);
	}

	default RewriterStatement apply(RewriterStatement stmt, @Nullable BiFunction<RewriterStatement, RewriterRule, Boolean> func, boolean print) {
		return apply(stmt, func, new MutableBoolean(false), print);
	}

	default RewriterStatement apply(RewriterStatement stmt) {
		return apply(stmt, null, new MutableBoolean(false), true);
	}
}
