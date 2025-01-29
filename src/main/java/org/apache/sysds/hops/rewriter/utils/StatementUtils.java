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

package org.apache.sysds.hops.rewriter.utils;

import org.apache.sysds.hops.rewriter.RewriterInstruction;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RuleContext;

public class StatementUtils {
	public static RewriterStatement max(final RuleContext ctx, RewriterStatement... of) {
		if (of.length == 1)
			return of[0];

		if (of.length == 2)
			return new RewriterInstruction("max", ctx, of);

		throw new UnsupportedOperationException();
	}

	public static RewriterStatement min(final RuleContext ctx, RewriterStatement... of) {
		if (of.length == 1)
			return of[0];

		if (of.length == 2)
			return new RewriterInstruction("min", ctx, of);

		throw new UnsupportedOperationException();
	}

	public static RewriterStatement length(final RuleContext ctx, RewriterStatement matrix) {
		if (!matrix.getResultingDataType(ctx).equals("MATRIX"))
			throw new IllegalArgumentException(matrix.toParsableString(ctx));

		return new RewriterInstruction("*", ctx, matrix.getNRow(), matrix.getNCol());
	}

	public static RewriterStatement add(final RuleContext ctx, RewriterStatement... terms) {
		if (terms.length == 1)
			return terms[0];

		return new RewriterInstruction("+", ctx, new RewriterInstruction("argList", ctx, terms));
	}
}
