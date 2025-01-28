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

package org.apache.sysds.hops.rewriter.assertions;

import org.apache.sysds.hops.rewriter.RewriterInstruction;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RuleContext;

public class RewriterAssertionUtils {
	public static RewriterAssertions buildImplicitAssertions(RewriterStatement root, final RuleContext ctx) {
		RewriterAssertions assertions = new RewriterAssertions(ctx);
		buildImplicitAssertions(root, assertions, ctx);
		return assertions;
	}

	public static void buildImplicitAssertions(RewriterStatement root, RewriterAssertions assertions, final RuleContext ctx) {
		root.forEachPreOrder(cur -> {
			buildImplicitAssertion(cur, assertions, root, ctx);
			return true;
		}, false);
	}

	public static boolean buildImplicitAssertion(RewriterStatement stmt, RewriterAssertions assertions, RewriterStatement exprRoot, final RuleContext ctx) {
		if (!stmt.isInstruction())
			return false;

		switch (stmt.trueInstruction()) {
			case "%*%":
				assertions.addEqualityAssertion(stmt.getChild(0).getNCol(), stmt.getChild(1).getNRow(), exprRoot);
				return true;
			case "diag":
				assertions.addEqualityAssertion(stmt.getChild(0).getNCol(), stmt.getChild(0).getNRow(), exprRoot);
				return true;
			case "RBind":
				assertions.addEqualityAssertion(stmt.getChild(0).getNCol(), stmt.getChild(1).getNCol(), exprRoot);
				return true;
			case "CBind":
				assertions.addEqualityAssertion(stmt.getChild(0).getNRow(), stmt.getChild(1).getNRow(), exprRoot);
				return true;
			case "1-*":
				assertions.addEqualityAssertion(stmt.getChild(0).getNCol(), stmt.getChild(1).getNCol(), exprRoot);
				assertions.addEqualityAssertion(stmt.getChild(0).getNRow(), stmt.getChild(1).getNRow(), exprRoot);
				return true;
			case "+*":
			case "-*":
				assertions.addEqualityAssertion(stmt.getChild(0).getNCol(), stmt.getChild(2).getNCol(), exprRoot);
				assertions.addEqualityAssertion(stmt.getChild(0).getNRow(), stmt.getChild(2).getNRow(), exprRoot);
				return true;
		}

		switch (stmt.trueTypedInstruction(ctx)) {
			case "trace(MATRIX)":
				assertions.addEqualityAssertion(stmt.getChild(0).getNRow(), stmt.getChild(0).getNCol(), exprRoot);
				return true;
			case "cast.FLOAT(MATRIX)":
				assertions.addEqualityAssertion(stmt.getChild(0).getNRow(), stmt.getChild(0).getNCol(), exprRoot);
				assertions.addEqualityAssertion(stmt.getChild(0).getNRow(), RewriterStatement.literal(ctx, 1L), exprRoot);
				return true;
		}

		if (((RewriterInstruction) stmt).hasProperty("ElementWiseInstruction", ctx)) {
			if (stmt.getChild(0).getResultingDataType(ctx).equals("MATRIX")
				&& stmt.getChild(1).getResultingDataType(ctx).equals("MATRIX")) {
				assertions.addEqualityAssertion(stmt.getChild(0).getNCol(), stmt.getChild(1).getNCol(), exprRoot);
				assertions.addEqualityAssertion(stmt.getChild(0).getNRow(), stmt.getChild(1).getNRow(), exprRoot);
				return true;
			}
		}

		return false;
	}
}
