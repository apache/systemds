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

import org.apache.sysds.hops.rewriter.RewriterInstruction;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.hops.rewriter.assertions.RewriterAssertions;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;

import java.util.HashMap;
import java.util.List;
import java.util.UUID;

import static org.apache.sysds.hops.rewriter.RewriterContextSettings.ALL_TYPES;
import static org.apache.sysds.hops.rewriter.RewriterContextSettings.SCALARS;

public class RewriterRuleCollection {
	public static void substituteEquivalentStatements(final List<RewriterRule> rules, final RuleContext ctx) {
		HashMap<Integer, RewriterStatement> hooks = new HashMap<>();

		rules.add(new RewriterRuleBuilder(ctx, "as.scalar(A) => cast.FLOAT(A)")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A")
				.withParsedStatement("as.scalar(A)")
				.toParsedStatement("cast.FLOAT(A)")
				.build()
		);

		SCALARS.forEach(t -> {
			rules.add(new RewriterRuleBuilder(ctx, "as.matrix(a) => cast.MATRIX(a)")
					.setUnidirectional(true)
					.parseGlobalVars(t + ":a")
					.withParsedStatement("as.matrix(a)")
					.toParsedStatement("cast.MATRIX(a)")
					.build()
			);
		});

		// Some meta operators
		rules.add(new RewriterRuleBuilder(ctx, "rowVec(A) => [](A, ...)")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("rowVec(A)")
				.toParsedStatement("[]($1:A, 1, 1, 1, ncol(A))", hooks)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "colVec(A) => [](A, ...)")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("colVec(A)")
				.toParsedStatement("[](A, 1, nrow(A), 1, 1)")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "cellMat(A) => [](A, ...)")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("cellMat(A)")
				.toParsedStatement("[](A, 1, 1, 1, 1)")
				.build()
		);

		substituteFusedOps(rules, ctx);
	}

	public static void substituteFusedOps(final List<RewriterRule> rules, final RuleContext ctx) {
		// Now resolve fused operators
		rules.add(new RewriterRuleBuilder(ctx, "1-*(A,B) => -(1, *(A, B))")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("LITERAL_FLOAT:1.0") // We take a float as this framework is optimized for floats
				.withParsedStatement("1-*(A, B)")
				.toParsedStatement("-(1.0, *(A, B))")
				.build()
		);
		rules.add(new RewriterRuleBuilder(ctx, "log_nz(A) => *(!=(A, 0.0), log(A))")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A")
				.parseGlobalVars("LITERAL_FLOAT:0.0") // We take a float as this framework is optimized for floats
				.withParsedStatement("log_nz(A)")
				.toParsedStatement("*(!=(A, 0.0), log(A))")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "sumSq(A) => sum(*(A,A))")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A")
				.parseGlobalVars("LITERAL_FLOAT:0.0")
				.withParsedStatement("sumSq(A)")
				.toParsedStatement("sum(*(A,A))")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "+*(A,s,Y) => +(A, *(s, Y))")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,Y")
				.parseGlobalVars("FLOAT:s")
				.withParsedStatement("+*(A,s,Y)")
				.toParsedStatement("+(A, *(s, Y))")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "-*(A,s,Y) => -(A, *(s, Y))")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,Y")
				.parseGlobalVars("FLOAT:s")
				.withParsedStatement("-*(A,s,Y)")
				.toParsedStatement("-(A, *(s, Y))")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "sq(A) => *(A,A)")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A")
				.withParsedStatement("sq(A)")
				.toParsedStatement("*(A, A)")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "_nnz(A) => sum(!=(A,0.0))")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A")
				.parseGlobalVars("LITERAL_FLOAT:0.0")
				.withParsedStatement("_nnz(A)")
				.toParsedStatement("sum(!=(A,0.0))")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "*2(A) => +(A,A)")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A")
				.withParsedStatement("*2(A)")
				.toParsedStatement("+(A,A)")
				.build()
		);

		SCALARS.forEach(t -> {
			rules.add(new RewriterRuleBuilder(ctx, "log_nz(A, a) => *(!=(A, 0.0), *(log(A), inv(log(a)))")
					.setUnidirectional(true)
					.parseGlobalVars("MATRIX:A")
					.parseGlobalVars("FLOAT:a") // We take a float as this framework is optimized for floats
					.parseGlobalVars("LITERAL_FLOAT:0.0")
					.withParsedStatement("log_nz(A, a)")
					.toParsedStatement("*(!=(A, 0.0), *(log(A), inv(log(a))))")
					.build()
			);

			rules.add(new RewriterRuleBuilder(ctx, "log(A, a) => *(log(A), inv(log(a)))")
					.setUnidirectional(true)
					.parseGlobalVars("MATRIX:A")
					.parseGlobalVars("FLOAT:a")
					.withParsedStatement("log(A, a)")
					.toParsedStatement("*(log(A), inv(log(a)))")
					.build()
			);
		});
	}

	public static void eliminateMultipleCasts(final List<RewriterRule> rules, final RuleContext ctx) {
		SCALARS.forEach(t -> {
			rules.add(new RewriterRuleBuilder(ctx, "cast.TYPE(cast.TYPE(A)) => cast.TYPE(A)")
					.setUnidirectional(true)
					.parseGlobalVars(t + ":a")
					.withParsedStatement("cast.MATRIX(cast.MATRIX(a))")
					.toParsedStatement("cast.MATRIX(a)")
					.build()
			);

			rules.add(new RewriterRuleBuilder(ctx, "cast.TYPE(a::TYPE) => a")
					.setUnidirectional(true)
					.parseGlobalVars(t + ":a")
					.withParsedStatement("cast." + t + "(a)")
					.toParsedStatement("a")
					.build()
			);

			SCALARS.forEach(t2 -> {
				SCALARS.forEach(t3 -> {
					rules.add(new RewriterRuleBuilder(ctx, "cast.TYPE(+(a, b)) => ...")
							.setUnidirectional(true)
							.parseGlobalVars(t2 + ":a")
							.parseGlobalVars(t3 + ":b")
							.withParsedStatement("cast." + t + "(+(a,b))")
							.toParsedStatement("+(cast." + t + "(a), cast." + t + "(b))")
							.build()
					);

					rules.add(new RewriterRuleBuilder(ctx, "cast.TYPE(*(a, b)) => ...")
							.setUnidirectional(true)
							.parseGlobalVars(t2 + ":a")
							.parseGlobalVars(t3 + ":b")
							.withParsedStatement("cast." + t + "(*(a,b))")
							.toParsedStatement("*(cast." + t + "(a), cast." + t + "(b))")
							.build()
					);
				});

				rules.add(new RewriterRuleBuilder(ctx, "cast.TYPE(cast.TYPE(A)) => cast.TYPE(A)")
						.setUnidirectional(true)
						.parseGlobalVars(t + ":a")
						.withParsedStatement("cast." + t2 + "(cast." + t2 + "(a))")
						.toParsedStatement("cast." + t2 + "(a)")
						.build()
				);

				rules.add(new RewriterRuleBuilder(ctx, "cast.SCALAR(cast.MATRIX(a)) => a")
						.setUnidirectional(true)
						.parseGlobalVars(t2 + ":a")
						.withParsedStatement("cast." + t + "(cast.MATRIX(a))")
						.toParsedStatement("cast." + t + "(a)")
						.build()
				);
			});
		});
	}

	public static void canonicalizeAlgebraicStatements(final List<RewriterRule> rules, boolean allowInversionCanonicalization, final RuleContext ctx) {
		HashMap<Integer, RewriterStatement> hooks = new HashMap<>();

		RewriterUtils.buildBinaryPermutations(ALL_TYPES, (t1, t2) -> {
			rules.add(new RewriterRuleBuilder(ctx, "-(a,b) => +(a,-(b))")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.withParsedStatement("-(a, b)", hooks)
					.toParsedStatement("+(a, -(b))", hooks)
					.build()
			);

			if (allowInversionCanonicalization) {
				rules.add(new RewriterRuleBuilder(ctx, "/(a,b) => *(a, inv(b))")
						.setUnidirectional(true)
						.parseGlobalVars(t1 + ":a")
						.parseGlobalVars(t2 + ":b")
						.withParsedStatement("/(a, b)", hooks)
						.toParsedStatement("*(a, inv(b))", hooks)
						.build()
				);
			}

			rules.add(new RewriterRuleBuilder(ctx, "-(+(a, b)) => +(-(a), -(b))")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.withParsedStatement("-(+(a, b))", hooks)
					.toParsedStatement("$1:+(-(a), -(b))", hooks)
					.build()
			);

			rules.add(new RewriterRuleBuilder(ctx, "-(-(a)) => a")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.withParsedStatement("-(-(a))", hooks)
					.toParsedStatement("a", hooks)
					.build()
			);
		});

		rules.add(new RewriterRuleBuilder(ctx, "length(A) => nrow(A) * ncol(A)")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A")
				.withParsedStatement("length(A)", hooks)
				.toParsedStatement("*(nrow(A), ncol(A))", hooks)
				.build()
		);

		for (String t : ALL_TYPES) {
			rules.add(new RewriterRuleBuilder(ctx, "-(inv(a)) => inv(-(a))")
					.setUnidirectional(true)
					.parseGlobalVars(t + ":A")
					.withParsedStatement("-(inv(A))", hooks)
					.toParsedStatement("inv(-(A))", hooks)
					.build()
			);
		}

		rules.add(new RewriterRuleBuilder(ctx, "-(sum(A)) => sum(-(A))")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A")
				.withParsedStatement("-(sum(A))", hooks)
				.toParsedStatement("sum(-(A))", hooks)
				.build()
		);
	}

	public static void canonicalizeBooleanStatements(final List<RewriterRule> rules, final RuleContext ctx) {
		HashMap<Integer, RewriterStatement> hooks = new HashMap<>();

		RewriterUtils.buildBinaryPermutations(ALL_TYPES, (t1, t2) -> {
			rules.add(new RewriterRuleBuilder(ctx, ">(a, b) => <(b, a)")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.withParsedStatement(">(a, b)", hooks)
					.toParsedStatement("<(b, a)", hooks)
					.build()
			);

			// These hold only for boolean expressions
			/*rules.add(new RewriterRuleBuilder(ctx, "!(!(a)) = a")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.withParsedStatement("!(!(a))", hooks)
					.toParsedStatement("a", hooks)
					.build()
			);*/

			rules.add(new RewriterRuleBuilder(ctx, "<=(a, b) => |(<(a, b), ==(a, b))")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.withParsedStatement("<=(a, b)", hooks)
					.toParsedStatement("|(<(a, b), ==(a, b))", hooks)
					.build()
			);

			rules.add(new RewriterRuleBuilder(ctx, ">=(a, b) => |(<(b, a), ==(b, a))")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.withParsedStatement(">=(a, b)", hooks)
					.toParsedStatement("|(<(b, a), ==(b, a))", hooks)
					.build()
			);

			rules.add(new RewriterRuleBuilder(ctx, "!(&(a, b)) => |(!(a), !(b))")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.withParsedStatement("!(&(a, b))", hooks)
					.toParsedStatement("|(!(a), !(b))", hooks)
					.build()
			);

			List.of("&(a, b)", "&(b, a)").forEach(exp -> {
				List.of("|(" + exp + ", a)", "|(a, " + exp + ")").forEach(tExpr -> {
					rules.add(new RewriterRuleBuilder(ctx, tExpr + " => a")
							.setUnidirectional(true)
							.parseGlobalVars(t1 + ":a")
							.parseGlobalVars(t2 + ":b")
							.withParsedStatement(tExpr, hooks)
							.toParsedStatement("a", hooks)
							.build()
					);
				});
			});

			List.of("|(a, b)", "|(b, a)").forEach(exp -> {
				List.of("&(" + exp + ", a)", "&(a, " + exp + ")").forEach(tExpr -> {
					rules.add(new RewriterRuleBuilder(ctx, tExpr + " => a")
							.setUnidirectional(true)
							.parseGlobalVars(t1 + ":a")
							.parseGlobalVars(t2 + ":b")
							.withParsedStatement(tExpr, hooks)
							.toParsedStatement("a", hooks)
							.build()
					);
				});
			});

			rules.add(new RewriterRuleBuilder(ctx,  "|(<(b, a), <(a, b)) => b != a")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.withParsedStatement("|(<(b, a), <(a, b))", hooks)
					.toParsedStatement("!=(b, a)", hooks)
					.build()
			);

			rules.add(new RewriterRuleBuilder(ctx,  "&(<(b, a), <(a, b)) => FALSE")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.parseGlobalVars("LITERAL_BOOL:FALSE")
					.withParsedStatement("&(<(b, a), <(a, b))", hooks)
					.toParsedStatement("FALSE", hooks)
					.build()
			);

			rules.add(new RewriterRuleBuilder(ctx,  "!(!=(a, b)) => ==(a, b)")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.parseGlobalVars("LITERAL_BOOL:FALSE")
					.withParsedStatement("!(!=(a, b))", hooks)
					.toParsedStatement("==(a, b)", hooks)
					.build()
			);

			/*rules.add(new RewriterRuleBuilder(ctx,  "==(a, b) => isZero(+(a, -(b)))")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.parseGlobalVars("LITERAL_BOOL:FALSE")
					.withParsedStatement("!(!=(a, b))", hooks)
					.toParsedStatement("==(a, b)", hooks)
					.build()
			);*/
		});
	}

	// E.g. expand A * B -> _m($1:_idx(), 1, nrow(A), _m($2:_idx(), 1, nrow(B), A[$1, $2] * B[$1, $2]))
	public static void expandStreamingExpressions(final List<RewriterRule> rules, final RuleContext ctx) {
		HashMap<Integer, RewriterStatement> hooks = new HashMap<>();

		// cast.MATRIX
		rules.add(new RewriterRuleBuilder(ctx, "Expand const matrix")
				.setUnidirectional(true)
				.parseGlobalVars("FLOAT:a")
				.parseGlobalVars("MATRIX:A")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("cast.MATRIX(a)", hooks)
				.toParsedStatement("$4:_m(1, 1, a)", hooks)
				.build()
		);

		// cast.FLOAT
		rules.add(new RewriterRuleBuilder(ctx, "")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:a")
				.parseGlobalVars("MATRIX:A")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("cast.FLOAT(A)", hooks)
				.toParsedStatement("[](A, 1, 1)", hooks)
				.build()
		);

		// Const
		rules.add(new RewriterRuleBuilder(ctx, "Expand const matrix")
				.setUnidirectional(true)
				.parseGlobalVars("FLOAT:a")
				.parseGlobalVars("MATRIX:A")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("const(A, a)", hooks)
				.toParsedStatement("$4:_m($1:_idx(1, nrow(A)), $2:_idx(1, ncol(A)), a)", hooks)
				.apply(hooks.get(1).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(2).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(4).getId(), (stmt, match) -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getChild(0).unsafePutMeta("ownerId", id);
				}, true) // Assumes it will never collide
				.build()
		);

		// Diag
		rules.add(new RewriterRuleBuilder(ctx, "Expand diag matrix")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A")
				.parseGlobalVars("LITERAL_INT:1")
				.parseGlobalVars("LITERAL_FLOAT:0.0")
				.withParsedStatement("diag(A)", hooks)
				.toParsedStatement("$4:_m($1:_idx(1, nrow(A)), $2:_idx(1, ncol(A)), $5:ifelse(==($1,$2), [](A, $1, $2), 0.0))", hooks)
				.apply(hooks.get(1).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(2).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(4).getId(), (stmt, match) -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getChild(0).unsafePutMeta("ownerId", id);
					RewriterStatement aRef = stmt.getChild(0, 1, 0);
					match.getNewExprRoot().getAssertions(ctx).addEqualityAssertion(aRef.getNCol(), aRef.getNRow(), match.getNewExprRoot());
				}, true) // Assumes it will never collide
				.build()
		);


		// Matrix Multiplication
		rules.add(new RewriterRuleBuilder(ctx, "Expand matrix product")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("%*%(A, B)", hooks)
				.toParsedStatement("$4:_m($1:_idx(1, nrow(A)), $2:_idx(1, ncol(B)), sum($5:_m($3:_idx(1, ncol(A)), 1, *([](A, $1, $3), [](B, $3, $2)))))", hooks)
				.apply(hooks.get(1).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(2).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(3).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(4).getId(), (stmt, match) -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
					stmt.getOperands().get(1).unsafePutMeta("ownerId", id);

					RewriterStatement aRef = stmt.getChild(0, 1, 0);
					RewriterStatement bRef = stmt.getChild(1, 1, 0);
					RewriterAssertions assertions = match.getNewExprRoot().getAssertions(ctx);
					assertions.addEqualityAssertion(aRef.getNCol(), bRef.getNRow(), match.getNewExprRoot());
					assertions.update(match.getNewExprRoot());
				}, true) // Assumes it will never collide
				.apply(hooks.get(5).getId(), stmt -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
				}, true) // Assumes it will never collide
				.build()
		);

		// E.g. A + B
		rules.add(new RewriterRuleBuilder(ctx, "Expand Element Wise Instruction")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("$1:ElementWiseInstruction(A,B)", hooks)
				.toParsedStatement("$7:_m($2:_idx(1, $5:nrow(A)), $3:_idx(1, $6:ncol(A)), $4:ElementWiseInstruction([](A, $2, $3), [](B, $2, $3)))", hooks)
				.link(hooks.get(1).getId(), hooks.get(4).getId(), RewriterStatement::transferMeta)
				.apply(hooks.get(2).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(3).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true)
				.apply(hooks.get(7).getId(), (stmt, match) -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
					stmt.getOperands().get(1).unsafePutMeta("ownerId", id);

					// Now we assert that nrow(A) = nrow(B) and ncol(A) = ncol(B)
					RewriterStatement aRef = stmt.getChild(2, 0, 0);
					RewriterStatement bRef = stmt.getChild(2, 1, 0);
					match.getNewExprRoot().getAssertions(ctx).addEqualityAssertion(aRef.getNRow(), bRef.getNRow(), match.getNewExprRoot());
					match.getNewExprRoot().getAssertions(ctx).addEqualityAssertion(aRef.getNCol(), bRef.getNCol(), match.getNewExprRoot());
				}, true) // Assumes it will never collide
				.build()
		);

		List.of("$2:_m(i, j, v1), v2", "v1, $2:_m(i, j, v2)").forEach(s -> {
			rules.add(new RewriterRuleBuilder(ctx)
					.setUnidirectional(true)
					.parseGlobalVars("MATRIX:A,B")
					.parseGlobalVars("LITERAL_INT:1")
					.parseGlobalVars("INT:i,j")
					.parseGlobalVars("FLOAT:v1,v2")
					.withParsedStatement("$1:ElementWiseInstruction(" + s + ")", hooks)
					.toParsedStatement("$3:_m(i, j, $4:ElementWiseInstruction(v1, v2))", hooks)
					.link(hooks.get(1).getId(), hooks.get(4).getId(), RewriterStatement::transferMeta)
					.link(hooks.get(2).getId(), hooks.get(3).getId(), RewriterStatement::transferMeta)
					.build()
			);
		});

		// Trace(A)
		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("trace(A)", hooks)
				.toParsedStatement("sum($3:_m($1:_idx(1, $2:nrow(A)), 1, [](A, $1, $1)))", hooks)
				.apply(hooks.get(1).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(2).getId(), stmt -> stmt.unsafePutMeta("dontExpand", true), true)
				.apply(hooks.get(3).getId(), (stmt, match) -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);

					// Assert that the matrix is squared
					RewriterStatement aRef = stmt.getChild(0, 1, 0);
					match.getNewExprRoot().getAssertions(ctx).addEqualityAssertion(aRef.getNRow(), aRef.getNCol(), match.getNewExprRoot());
				}, true)
				.build()
		);

		// t(A)
		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("t(A)", hooks)
				.toParsedStatement("$3:_m($1:_idx(1, ncol(A)), $2:_idx(1, nrow(A)), [](A, $2, $1))", hooks)
				.apply(hooks.get(1).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(2).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true)
				.apply(hooks.get(3).getId(), stmt -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
					stmt.getOperands().get(1).unsafePutMeta("ownerId", id);
				}, true)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("rev(A)", hooks)
				.toParsedStatement("$3:_m($1:_idx(1, nrow(A)), $2:_idx(1, ncol(A)), [](A, -(+(ncol(A), 1), $1), $2))", hooks)
				.apply(hooks.get(1).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(2).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true)
				.apply(hooks.get(3).getId(), stmt -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
					stmt.getOperands().get(1).unsafePutMeta("ownerId", id);
				}, true)
				.build()
		);

		// rand(rows, cols, min, max)
		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("LITERAL_INT:1")
				.parseGlobalVars("INT:n,m")
				.parseGlobalVars("FLOAT:a,b")
				.withParsedStatement("rand(n, m, a, b)", hooks)
				.toParsedStatement("$3:_m($1:_idx(1, n), $2:_idx(1, m), +(a, $4:*(+(b, -(a)), rand(argList($1,$2)))))", hooks)
				.apply(hooks.get(1).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(2).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true)
				.apply(hooks.get(3).getId(), stmt -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
					stmt.getOperands().get(1).unsafePutMeta("ownerId", id);
				}, true)
				.build()
		);

		// sum(A) = sum(_m($1:_idx(1, nrow(A)), 1, sum(_m($2:_idx(1, ncol(A)), 1, [](A, $1, $2)))))
		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("sum(A)", hooks)
				.toParsedStatement("sum($3:_m($1:_idx(1, nrow(A)), 1, sum($4:_m($2:_idx(1, ncol(A)), 1, [](A, $1, $2)))))", hooks)
				.iff(match -> {
					RewriterStatement meta = (RewriterStatement) match.getMatchRoot().getOperands().get(0).getMeta("ncol");

					if (meta == null)
						throw new IllegalArgumentException("Column meta should not be null: " + match.getMatchRoot().getOperands().get(0).toString(ctx));

					return !meta.isLiteral() || ((long)meta.getLiteral()) != 1;
				}, true)
				.apply(hooks.get(1).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(2).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true)
				.apply(hooks.get(3).getId(), stmt -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
				}, true)
				.apply(hooks.get(4).getId(), stmt -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
				}, true)
				.build()
		);

		// rowSums(A) -> _m($1:_idx(1, nrow(A)), 1, sum(_m($2:_idx(1, ncol(A)), 1, [](A, $1, $2)))
		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("rowSums(A)", hooks)
				.toParsedStatement("$3:_m($1:_idx(1, nrow(A)), 1, sum($4:_m($2:_idx(1, ncol(A)), 1, [](A, $1, $2))))", hooks)
				.apply(hooks.get(1).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(2).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true)
				.apply(hooks.get(3).getId(), stmt -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
				}, true)
				.apply(hooks.get(4).getId(), stmt -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
				}, true)
				.build()
		);

		// colSums(A) -> _m($1:_idx(1, ncol(A)), 1, sum(_m($2:_idx(1, nrow(A)), 1, [](A, $2, $1)))
		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("colSums(A)", hooks)
				.toParsedStatement("$3:_m(1, $1:_idx(1, ncol(A)), sum($4:_m($2:_idx(1, nrow(A)), 1, [](A, $2, $1))))", hooks)
				.apply(hooks.get(1).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(2).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true)
				.apply(hooks.get(3).getId(), stmt -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(1).unsafePutMeta("ownerId", id);
				}, true)
				.apply(hooks.get(4).getId(), stmt -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
				}, true)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("INT:l")
				.withParsedStatement("_idx(l, l)", hooks)
				.toParsedStatement("l", hooks)
				.build()
		);

		// Scalars dependent on matrix to index streams
		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("sum(A)", hooks)
				.toParsedStatement("sum($3:_idxExpr($1:_idx(1, nrow(A)), $4:_idxExpr($2:_idx(1, ncol(A)), [](A, $1, $2))))", hooks)
				.apply(hooks.get(1).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(2).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true)
				.apply(hooks.get(3).getId(), stmt -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
				}, true)
				.apply(hooks.get(4).getId(), stmt -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
				}, true)
				.build()
		);

		// diag(A) -> _m($1:_idx(1, nrow(A)), 1, [](A, $1, $1))
		/*rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("diag(A)", hooks)
				.toParsedStatement("$2:_m($1:_idx(1, nrow(A)), 1, [](A, $1, $1))", hooks)
				.apply(hooks.get(1).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(2).getId(), (stmt, match) -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);

					RewriterStatement aRef = stmt.getChild(0, 1, 0);
					match.getNewExprRoot().getAssertions(ctx).addEqualityAssertion(aRef.getNRow(), aRef.getNCol(), match.getNewExprRoot());
				}, true)
				.build()
		);*/

		// cast.MATRIX(a) => _m(1, 1, a)
		for (String t : List.of("INT", "BOOL", "FLOAT")) {
			rules.add(new RewriterRuleBuilder(ctx)
					.setUnidirectional(true)
					.parseGlobalVars(t + ":a")
					.parseGlobalVars("LITERAL_INT:1")
					.withParsedStatement("cast.MATRIX(a)", hooks)
					.toParsedStatement("$2:_m(1, 1, a)", hooks)
					.apply(hooks.get(2).getId(), (stmt, match) -> stmt.unsafePutMeta("ownerId", UUID.randomUUID()), true)
					.build()
			);
		}
	}

	public static void expandArbitraryMatrices(final List<RewriterRule> rules, final RuleContext ctx) {
		HashMap<Integer, RewriterStatement> hooks = new HashMap<>();
		// This must be the last rule in the heuristic as it handles any matrix that has not been written as a stream
		// A -> _m()
		rules.add(new RewriterRuleBuilder(ctx, "Expand arbitrary matrix expression")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("A", hooks)
				.toParsedStatement("$3:_m($1:_idx(1, nrow(A)), $2:_idx(1, ncol(A)), [](A, $1, $2))", hooks)
				.iff(match -> match.getMatchRoot().getMeta("dontExpand") == null && !(match.getMatchRoot().isInstruction() && match.getMatchRoot().trueInstruction().equals("_m")), true)
				.apply(hooks.get(1).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(2).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true)
				.apply(hooks.get(3).getId(), stmt -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
					stmt.getOperands().get(1).unsafePutMeta("ownerId", id);
					RewriterStatement A = stmt.getChild(0, 1, 0);
					A.unsafePutMeta("dontExpand", true);
					if (A.getNRow().isInstruction() && A.getNRow().trueInstruction().equals("nrow") && A.getNRow().getChild(0) == stmt)
						A.getNRow().getOperands().set(0, A);
					if (A.getNCol().isInstruction() && A.getNCol().trueInstruction().equals("ncol") && A.getNCol().getChild(0) == stmt)
						A.getNCol().getOperands().set(0, A);
				}, true)
				.build()
		);
	}

	// TODO: Big issue when having multiple references to the same sub-dag
	public static void pushdownStreamSelections(final List<RewriterRule> rules, final RuleContext ctx) {
		HashMap<Integer, RewriterStatement> hooks = new HashMap<>();

		// ifelse merging
		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("FLOAT:a,b,c,d")
				.parseGlobalVars("INT:l1,l2")
				.withParsedStatement("$1:ElementWiseInstruction(ifelse(==(l1, l2), a, b), ifelse(==(l1, l2), c, d))", hooks)
				.toParsedStatement("ifelse(==(l1, l2), $2:ElementWiseInstruction(a, c), $3:ElementWiseInstruction(b, d))", hooks)
				.linkManyUnidirectional(hooks.get(1).getId(), List.of(hooks.get(2).getId(), hooks.get(3).getId()), RewriterStatement::transferMeta, true)
				.build()
		);

		SCALARS.forEach(t -> {
			SCALARS.forEach(t2 -> {
				// redundant ifelse elimination
				rules.add(new RewriterRuleBuilder(ctx, "Remove redundant ifelse")
						.setUnidirectional(true)
						.parseGlobalVars(t2 + ":c,d,e")
						.parseGlobalVars(t + ":a,b")
						.withParsedStatement("ifelse(==(a, b), ifelse(==(a, b), c, e), d)", hooks)
						.toParsedStatement("ifelse(==(a, b), c, d)", hooks)
						.build()
				);
				rules.add(new RewriterRuleBuilder(ctx, "Remove redundant ifelse")
						.setUnidirectional(true)
						.parseGlobalVars(t2 + ":c,d,e")
						.parseGlobalVars(t + ":a,b")
						.withParsedStatement("ifelse(==(a, b), d, ifelse(==(a, b), c, e))", hooks)
						.toParsedStatement("ifelse(==(a, b), d, e)", hooks)
						.build()
				);

				// ifelse expression pullup
				rules.add(new RewriterRuleBuilder(ctx, "Ifelse expression pullup")
						.setUnidirectional(true)
						.parseGlobalVars(t + ":a,c")
						.parseGlobalVars(t2 + ":d")
						.parseGlobalVars("BOOL:b")
						.withParsedStatement("$1:ElementWiseInstruction(ifelse(b, a, c), d)", hooks)
						.toParsedStatement("ifelse(b, $2:ElementWiseInstruction(a, d), $3:ElementWiseInstruction(c, d))", hooks)
						.linkManyUnidirectional(hooks.get(1).getId(), List.of(hooks.get(2).getId(), hooks.get(3).getId()), RewriterStatement::transferMeta, true)
						.build()
				);
				rules.add(new RewriterRuleBuilder(ctx, "Ifelse expression pullup")
						.setUnidirectional(true)
						.parseGlobalVars(t + ":a,c")
						.parseGlobalVars(t2 + ":d")
						.parseGlobalVars("BOOL:b")
						.withParsedStatement("$1:ElementWiseInstruction(d, ifelse(b, a, c))", hooks)
						.toParsedStatement("ifelse(b, $2:ElementWiseInstruction(d, a), $3:ElementWiseInstruction(d, c))", hooks)
						.linkManyUnidirectional(hooks.get(1).getId(), List.of(hooks.get(2).getId(), hooks.get(3).getId()), RewriterStatement::transferMeta, true)
						.build()
				);
			});

			rules.add(new RewriterRuleBuilder(ctx, "Ifelse branch merge")
					.setUnidirectional(true)
					.parseGlobalVars(t + ":a,c,d")
					.parseGlobalVars("BOOL:b")
					.withParsedStatement("ifelse(b, a, a)", hooks)
					.toParsedStatement("a", hooks)
					.build()
			);
		});

		SCALARS.forEach(t -> {
			rules.add(new RewriterRuleBuilder(ctx, "Fold true statement")
					.setUnidirectional(true)
					.parseGlobalVars(t  + ":a")
					.parseGlobalVars("LITERAL_BOOL:TRUE")
					.withParsedStatement("==(a,a)", hooks)
					.toParsedStatement("TRUE", hooks)
					.build()
			);
		});

		rules.add(new RewriterRuleBuilder(ctx, "Eliminate unnecessary branches")
				.setUnidirectional(true)
				.parseGlobalVars("FLOAT:a,b")
				.parseGlobalVars("LITERAL_BOOL:TRUE")
				.withParsedStatement("ifelse(TRUE, a, b)", hooks)
				.toParsedStatement("a", hooks)
				.build()
		);
		rules.add(new RewriterRuleBuilder(ctx, "Eliminate unnecessary branches")
				.setUnidirectional(true)
				.parseGlobalVars("FLOAT:a,b")
				.parseGlobalVars("LITERAL_BOOL:FALSE")
				.withParsedStatement("ifelse(FALSE, a, b)", hooks)
				.toParsedStatement("b", hooks)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("INT:l")
				.withParsedStatement("_idx(l, l)", hooks)
				.toParsedStatement("l", hooks)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "Eliminate scalar matrices")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("INT:i,j")
				.parseGlobalVars("FLOAT:v")
				.withParsedStatement("as.scalar(v)", hooks)
				.toParsedStatement("v", hooks)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "Element selection pushdown")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("INT:h,i,j,k,l,m")
				.parseGlobalVars("FLOAT:v")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("[]($1:_m(h, i, v), l, m)", hooks)
				.toParsedStatement("$3:as.scalar($2:_m(l, m, v))", hooks)
				.iff(match -> {
					List<RewriterStatement> ops = match.getMatchRoot().getOperands().get(0).getOperands();
					return (ops.get(0).isInstruction()
							&& ops.get(0).trueTypedInstruction(ctx).equals("_idx(INT,INT)"))
							|| (ops.get(1).isInstruction()
							&& ops.get(1).trueTypedInstruction(ctx).equals("_idx(INT,INT)"));
				}, true)
				.linkUnidirectional(hooks.get(1).getId(), hooks.get(2).getId(), lnk -> {
					RewriterStatement.transferMeta(lnk);

					for (int idx = 0; idx < 2; idx++) {
						RewriterStatement oldRef = lnk.oldStmt.getChild(idx);

						if (!oldRef.isInstruction() || !oldRef.trueTypedInstruction(ctx).equals("_idx(INT,INT)"))
							continue;

						UUID oldRefId = (UUID)oldRef.getMeta("idxId");

						RewriterStatement newRef = lnk.newStmt.get(0).getChild(idx);

						RewriterStatement newOne = RewriterUtils.replaceReferenceAware(lnk.newStmt.get(0).getChild(2), stmt -> {
							UUID idxId = (UUID) stmt.getMeta("idxId");
							if (idxId != null) {
								if (idxId.equals(oldRefId))
									return newRef;
							}

							return null;
						});

						if (newOne != null)
							lnk.newStmt.get(0).getOperands().set(2, newOne);
					}
				}, true)
				.apply(hooks.get(3).getId(), stmt -> {
					stmt.getOperands().set(0, stmt.getChild(0, 2));
				}, true)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "Scalar matrix selection pushdown")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("INT:h,i,j,k,l,m")
				.parseGlobalVars("FLOAT:v")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("[]($1:_m(1, 1, v), j, k)", hooks)
				.toParsedStatement("v", hooks)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "Selection pushdown")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("INT:h,i,j,k,l,m")
				.parseGlobalVars("FLOAT:v")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("[]($1:_m(h, i, v), j, k, l, m)", hooks)
				.toParsedStatement("$2:_m(_idx(1, +(+(k, 1), -(j))), _idx(1, +(+(m, 1), -(l))), v)", hooks) // Assuming that selections are valid
				.linkUnidirectional(hooks.get(1).getId(), hooks.get(2).getId(), lnk -> {
					RewriterStatement.transferMeta(lnk);

					for (int idx = 0; idx < 2; idx++) {
						RewriterStatement oldRef = lnk.oldStmt.getOperands().get(idx);
						RewriterStatement newRef = lnk.newStmt.get(0).getChild(idx);
						RewriterStatement mStmtC = new RewriterInstruction().as(UUID.randomUUID().toString()).withInstruction("+").withOps(newRef.getChild(1, 1, 0), RewriterStatement.literal(ctx, -1L)).consolidate(ctx);
						RewriterStatement mStmt = new RewriterInstruction().as(UUID.randomUUID().toString()).withInstruction("+").withOps(newRef, mStmtC).consolidate(ctx);
						final RewriterStatement newStmt = RewriterUtils.foldConstants(mStmt, ctx);

						UUID oldRefId = (UUID)oldRef.getMeta("idxId");

						RewriterStatement newOne = RewriterUtils.replaceReferenceAware(lnk.newStmt.get(0).getChild(2), stmt -> {
							UUID idxId = (UUID) stmt.getMeta("idxId");
							if (idxId != null) {
								if (idxId.equals(oldRefId))
									return newStmt;
							}

							return null;
						});

						if (newOne != null)
							lnk.newStmt.get(0).getOperands().set(2, newOne);
					}
				}, true)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "_idx(a,a) => a")
				.setUnidirectional(true)
				.parseGlobalVars("INT:a")
				.withParsedStatement("_idx(a,a)", hooks)
				.toParsedStatement("a", hooks)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "_idxExpr(i::<const>, v) => v")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("INT:i,j")
				.parseGlobalVars("FLOAT:v")
				.withParsedStatement("_idxExpr(i, v)", hooks)
				.toParsedStatement("v", hooks)
				.iff(match -> {
					List<RewriterStatement> ops = match.getMatchRoot().getOperands();

					boolean matching = (!ops.get(0).isInstruction() || !ops.get(0).trueInstruction().equals("_idx") || ops.get(0).getMeta("ownerId") != match.getMatchRoot().getMeta("ownerId"))
							&& (!ops.get(1).isInstruction() || !ops.get(1).trueInstruction().equals("_idx") || ops.get(1).getMeta("ownerId") != match.getMatchRoot().getMeta("ownerId"));

					return matching;
				}, true)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "_idxExpr(i::<const>, v) => v")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("INT:i,j")
				.parseGlobalVars("FLOAT*:v")
				.withParsedStatement("_idxExpr(i, v)", hooks)
				.toParsedStatement("v", hooks)
				.iff(match -> {
					List<RewriterStatement> ops = match.getMatchRoot().getOperands();

					boolean matching = (!ops.get(0).isInstruction() || !ops.get(0).trueInstruction().equals("_idx") || ops.get(0).getMeta("ownerId") != match.getMatchRoot().getMeta("ownerId"))
							&& (!ops.get(1).isInstruction() || !ops.get(1).trueInstruction().equals("_idx") || ops.get(1).getMeta("ownerId") != match.getMatchRoot().getMeta("ownerId"));

					return matching;
				}, true)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "_idxExpr(i, sum(...)) => sum(_idxExpr(i, ...))")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("INT:i,j")
				.parseGlobalVars("FLOAT:v")
				.withParsedStatement("$1:_idxExpr(i, sum(v))", hooks)
				.toParsedStatement("sum($2:_idxExpr(i, v))", hooks)
				.link(hooks.get(1).getId(), hooks.get(2).getId(), RewriterStatement::transferMeta)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "_idxExpr(i, sum(...)) => sum(_idxExpr(i, ...))")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("INT:i,j")
				.parseGlobalVars("FLOAT*:v")
				.withParsedStatement("$1:_idxExpr(i, sum(v))", hooks)
				.toParsedStatement("sum($2:_idxExpr(i, v))", hooks)
				.link(hooks.get(1).getId(), hooks.get(2).getId(), RewriterStatement::transferMeta)
				.build()
		);

		RewriterUtils.buildBinaryPermutations(List.of("FLOAT"), (t1, t2) -> {
			rules.add(new RewriterRuleBuilder(ctx, "*(sum(_idxExpr(i, ...)), sum(_idxExpr(j, ...))) => _idxExpr(i, _idxExpr(j, sum(*(...)))")
					.setUnidirectional(true)
					.parseGlobalVars("MATRIX:A,B")
					.parseGlobalVars("INT:i,j")
					.parseGlobalVars(t1 + ":v1")
					.parseGlobalVars(t2 + ":v2")
					.withParsedStatement("$1:*(sum($2:_idxExpr(i, v1)), sum($3:_idxExpr(j, v2)))", hooks)
					.toParsedStatement("sum($4:_idxExpr(i, $5:_idxExpr(j, $6:*(v1, v2))))", hooks)
					.link(hooks.get(1).getId(), hooks.get(6).getId(), RewriterStatement::transferMeta)
					.link(hooks.get(2).getId(), hooks.get(4).getId(), RewriterStatement::transferMeta)
					.link(hooks.get(3).getId(), hooks.get(5).getId(), RewriterStatement::transferMeta)
					.build()
			);
		});

		rules.add(new RewriterRuleBuilder(ctx, "sum(sum(v)) => sum(v)")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("INT:i,j")
				.parseGlobalVars("FLOAT:v")
				.withParsedStatement("sum(sum(v))", hooks)
				.toParsedStatement("sum(v)", hooks)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "sum(sum(v)) => sum(v)")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("INT:i,j")
				.parseGlobalVars("FLOAT*:v")
				.withParsedStatement("sum(sum(v))", hooks)
				.toParsedStatement("sum(v)", hooks)
				.build()
		);

		SCALARS.forEach(t -> {
			rules.add(new RewriterRuleBuilder(ctx, "sum(v::" + t + ") => v::" + t)
					.setUnidirectional(true)
					.parseGlobalVars(t + ":v")
					.withParsedStatement("sum(v)", hooks)
					.toParsedStatement("v", hooks)
					.build()
			);
		});

		rules.add(new RewriterRuleBuilder(ctx, "[](UnaryElementWiseOperator(A), i, j) => UnaryElementWiseOperator([](A, i, j))")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A")
				.parseGlobalVars("INT:i,j")
				.withParsedStatement("[]($1:UnaryElementWiseOperator(A), i, j)", hooks)
				.toParsedStatement("$2:UnaryElementWiseOperator([](A, i, j))", hooks)
				.link(hooks.get(1).getId(), hooks.get(2).getId(), RewriterStatement::transferMeta)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "[](ElementWiseUnary.FLOAT(A), i, j) => ElementWiseUnary.FLOAT([](A, i, j))")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A")
				.parseGlobalVars("INT:i,j")
				.withParsedStatement("[]($1:ElementWiseUnary.FLOAT(A), i, j)", hooks)
				.toParsedStatement("$2:ElementWiseUnary.FLOAT([](A, i, j))", hooks)
				.link(hooks.get(1).getId(), hooks.get(2).getId(), RewriterStatement::transferMeta)
				.build()
		);

		for (String t : ALL_TYPES) {
			if (t.equals("MATRIX")) {
				rules.add(new RewriterRuleBuilder(ctx, "ElementWiseInstruction(_m(i, j, v), b) => _m(i, j, ElementWiseInstruction(v, b))")
						.setUnidirectional(true)
						.parseGlobalVars("FLOAT:v")
						.parseGlobalVars(t + ":B")
						.parseGlobalVars("INT:i,j")
						.withParsedStatement("$1:ElementWiseInstruction($2:_m(i, j, v), B)", hooks)
						.toParsedStatement("$3:_m(i, j, $4:ElementWiseInstruction(v, [](B, i, j)))", hooks)
						.link(hooks.get(1).getId(), hooks.get(4).getId(), RewriterStatement::transferMeta)
						.link(hooks.get(2).getId(), hooks.get(3).getId(), RewriterStatement::transferMeta)
						.apply(hooks.get(3).getId(), (stmt, match) -> {
							// Then we an infer that the two matrices have the same dimensions
							match.getNewExprRoot().getAssertions(ctx).addEqualityAssertion(stmt.getNCol(), stmt.getChild(2, 1, 0).getNCol(), match.getNewExprRoot());
							match.getNewExprRoot().getAssertions(ctx).addEqualityAssertion(stmt.getNRow(), stmt.getChild(2, 1, 0).getNRow(), match.getNewExprRoot());
						}, true)
						.build()
				);

				continue;
			}
			rules.add(new RewriterRuleBuilder(ctx, "ElementWiseInstruction(_m(i, j, A), b) => _m(i, j, ElementWiseInstruction(A, b))")
					.setUnidirectional(true)
					.parseGlobalVars("FLOAT:v")
					.parseGlobalVars(t + ":b")
					.parseGlobalVars("INT:i,j")
					.withParsedStatement("$1:ElementWiseInstruction($2:_m(i, j, v), b)", hooks)
					.toParsedStatement("$3:_m(i, j, $4:ElementWiseInstruction(v, b))", hooks)
					.link(hooks.get(1).getId(), hooks.get(4).getId(), RewriterStatement::transferMeta)
					.link(hooks.get(2).getId(), hooks.get(3).getId(), RewriterStatement::transferMeta)
					.build()
			);

			rules.add(new RewriterRuleBuilder(ctx, "[](ElementWiseInstruction(A, v), i, j) => ElementWiseInstruction(v, [](A, i, j))")
					.setUnidirectional(true)
					.parseGlobalVars("MATRIX:A")
					.parseGlobalVars(t + ":v")
					.parseGlobalVars("INT:i,j")
					.withParsedStatement("[]($1:ElementWiseInstruction(A, v), i, j)", hooks)
					.toParsedStatement("$2:ElementWiseInstruction([](A, i, j), v)", hooks)
					.link(hooks.get(1).getId(), hooks.get(2).getId(), RewriterStatement::transferMeta)
					.build()
			);

			rules.add(new RewriterRuleBuilder(ctx, "[](ElementWiseInstruction(v, A), i, j) => ElementWiseInstruction(v, [](A, i, j))")
					.setUnidirectional(true)
					.parseGlobalVars("MATRIX:A")
					.parseGlobalVars(t + ":v")
					.parseGlobalVars("INT:i,j")
					.withParsedStatement("[]($1:ElementWiseInstruction(v, A), i, j)", hooks)
					.toParsedStatement("$2:ElementWiseInstruction(v, [](A, i, j))", hooks)
					.link(hooks.get(1).getId(), hooks.get(2).getId(), RewriterStatement::transferMeta)
					.build()
			);
		}
	}

	// This expands the statements to a common canonical form
	public static void canonicalExpandAfterFlattening(final List<RewriterRule> rules, final RuleContext ctx) {
		HashMap<Integer, RewriterStatement> hooks = new HashMap<>();

		rules.add(new RewriterRuleBuilder(ctx, "sum($1:_idxExpr(indices, -(A))) => -(sum($2:_idxExpr(indices, A)))")
				.setUnidirectional(true)
				.parseGlobalVars("FLOAT:a")
				.parseGlobalVars("INT...:indices")
				.withParsedStatement("sum($1:_idxExpr(indices, -(a)))", hooks)
				.toParsedStatement("-(sum($2:_idxExpr(indices, a)))", hooks)
				.link(hooks.get(1).getId(), hooks.get(2).getId(), RewriterStatement::transferMeta)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "sum($1:_idxExpr(indices, -(a))) => -(sum($2:_idxExpr(indices, a)))")
				.setUnidirectional(true)
				.parseGlobalVars("INT:a")
				.parseGlobalVars("INT...:indices")
				.withParsedStatement("sum($1:_idxExpr(indices, -(a)))", hooks)
				.toParsedStatement("-(sum($2:_idxExpr(indices, a)))", hooks)
				.link(hooks.get(1).getId(), hooks.get(2).getId(), RewriterStatement::transferMeta)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "sum(_idxExpr(indices, +(ops))) => +(argList(sum(_idxExpr(indices, op1)), sum(_idxExpr(...)), ...))")
				.setUnidirectional(true)
				.parseGlobalVars("INT...:indices")
				.parseGlobalVars("FLOAT...:ops")
				.withParsedStatement("sum($1:_idxExpr(indices, +(ops)))", hooks)
				.toParsedStatement("$4:+($3:argList(sum($2:_idxExpr(indices, +(ops)))))", hooks) // The inner +(ops) is temporary and will be removed
				.link(hooks.get(1).getId(), hooks.get(2).getId(), RewriterStatement::transferMeta)
				.apply(hooks.get(3).getId(), newArgList -> {
					RewriterStatement oldArgList = newArgList.getChild(0, 0, 1, 0);
					newArgList.getChild(0, 0).getOperands().set(1, oldArgList.getChild(0));

					for (int i = 1; i < oldArgList.getOperands().size(); i++) {
						RewriterStatement newIdxExpr = newArgList.getChild(0, 0).copyNode();
						newIdxExpr.getOperands().set(1, oldArgList.getChild(i));
						RewriterStatement newSum = new RewriterInstruction()
								.as(UUID.randomUUID().toString())
								.withInstruction("sum")
								.withOps(newIdxExpr);
						RewriterUtils.copyIndexList(newIdxExpr);
						newIdxExpr.refreshReturnType(ctx);
						newSum.consolidate(ctx);
						newArgList.getOperands().add(newSum);
					}

					newArgList.refreshReturnType(ctx);
				}, true)
				.apply(hooks.get(4).getId(), stmt -> {
					stmt.refreshReturnType(ctx);
				}, true)
				.build()
		);
	}

	public static void flattenedAlgebraRewrites(final List<RewriterRule> rules, final RuleContext ctx) {
		HashMap<Integer, RewriterStatement> hooks = new HashMap<>();

		// Minus pushdown
		rules.add(new RewriterRuleBuilder(ctx, "-(+(...)) => +(-(el1), -(el2), ...)")
				.setUnidirectional(true)
				.parseGlobalVars("FLOAT...:ops")
				.withParsedStatement("-(+(ops))", hooks)
				.toParsedStatement("$1:+(ops)", hooks) // Temporary
				.apply(hooks.get(1).getId(), (stmt, match) -> {
					RewriterStatement argList = stmt.getChild(0);

					for (int i = 0; i < argList.getOperands().size(); i++) {
						RewriterInstruction newStmt = new RewriterInstruction().as(UUID.randomUUID().toString()).withInstruction("-").withOps(argList.getOperands().get(i));
						newStmt.consolidate(ctx);
						argList.getOperands().set(i, newStmt);
					}

					RewriterUtils.tryFlattenNestedOperatorPatterns(ctx, match.getNewExprRoot());
				}, true)
				.build()
		);
	}

	public static List<RewriterRule> buildElementWiseAlgebraicCanonicalization(final List<RewriterRule> rules, final RuleContext ctx) {
		RewriterUtils.buildTernaryPermutations(List.of("FLOAT", "INT", "BOOL"), (t1, t2, t3) -> {
			rules.add(new RewriterRuleBuilder(ctx, "*(+(a, b), c) => +(*(a, c), *(b, c))")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.parseGlobalVars(t3 + ":c")
					.withParsedStatement("*(+(a, b), c)")
					.toParsedStatement("+(*(a, c), *(b, c))")
					.build()
			);

			rules.add(new RewriterRuleBuilder(ctx, "*(c, +(a, b)) => +(*(c, a), *(c, b))")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.parseGlobalVars(t3 + ":c")
					.withParsedStatement("*(c, +(a, b))")
					.toParsedStatement("+(*(c, a), *(c, b))")
					.build()
			);
		});

		/*List.of("FLOAT", "INT").forEach(t -> {
			rules.add(new RewriterRuleBuilder(ctx, "-(a) => *(-1.0, a)")
					.setUnidirectional(true)
					.parseGlobalVars(t + ":a")
					.parseGlobalVars("LITERAL_" + t + ":-1")
					.withParsedStatement("-(a)")
					.toParsedStatement("*(-1, a)")
					.build()
			);
		});*/

		return rules;
	}

	public static List<RewriterRule> replaceNegation(final List<RewriterRule> rules, final RuleContext ctx) {
		List.of("FLOAT", "INT").forEach(t -> {
			rules.add(new RewriterRuleBuilder(ctx, "-(a) => *(-1.0, a)")
					.setUnidirectional(true)
					.parseGlobalVars(t + ":a")
					.parseGlobalVars("LITERAL_" + t + ":-1")
					.withParsedStatement("-(a)")
					.toParsedStatement("*(-1, a)")
					.build()
			);
		});

		return rules;
	}

	@Deprecated
	public static void streamifyExpressions(final List<RewriterRule> rules, final RuleContext ctx) {
		HashMap<Integer, RewriterStatement> hooks = new HashMap<>();

		ALL_TYPES.forEach(t -> {
			if (t.equals("MATRIX"))
				return;

			rules.add(new RewriterRuleBuilder(ctx)
					.setUnidirectional(true)
					.parseGlobalVars("MATRIX:A")
					.parseGlobalVars(t + ":b")
					.parseGlobalVars("INT:i,j")
					.parseGlobalVars("FLOAT:v")
					.withParsedStatement("$1:ElementWiseInstruction($3:_m(i, j, v), b)", hooks)
					.toParsedStatement("$4:_m(i, j, $2:ElementWiseInstruction(v, b))", hooks)
					.link(hooks.get(1).getId(), hooks.get(2).getId(), RewriterStatement::transferMeta)
					.link(hooks.get(3).getId(), hooks.get(4).getId(), RewriterStatement::transferMeta)
					.build());

			rules.add(new RewriterRuleBuilder(ctx)
					.setUnidirectional(true)
					.parseGlobalVars("MATRIX:A")
					.parseGlobalVars(t + ":b")
					.parseGlobalVars("INT:i,j")
					.parseGlobalVars("FLOAT:v")
					.withParsedStatement("$1:ElementWiseInstruction(b, $3:_m(i, j, v))", hooks)
					.toParsedStatement("$4:_m(i, j, $2:ElementWiseInstruction(b, v))", hooks)
					.link(hooks.get(1).getId(), hooks.get(2).getId(), RewriterStatement::transferMeta)
					.link(hooks.get(3).getId(), hooks.get(4).getId(), RewriterStatement::transferMeta)
					.build());
		});


	}

	public static void flattenOperations(final List<RewriterRule> rules, final RuleContext ctx) {
		HashMap<Integer, RewriterStatement> hooks = new HashMap<>();

		RewriterUtils.buildBinaryPermutations(List.of("INT", "INT..."), (t1, t2) -> {
			for (String t3 : List.of("FLOAT", "FLOAT*", "INT", "INT*", "BOOL", "BOOL*")) {
				rules.add(new RewriterRuleBuilder(ctx, "Flatten nested index expression")
						.setUnidirectional(true)
						.parseGlobalVars(t1 + ":i")
						.parseGlobalVars(t2 + ":j")
						.parseGlobalVars(t3 + ":v")
						.withParsedStatement("$1:_idxExpr(i, $2:_idxExpr(j, v))", hooks)
						.toParsedStatement("$3:_idxExpr(argList(i, j), v)", hooks)
						.link(hooks.get(1).getId(), hooks.get(3).getId(), RewriterStatement::transferMeta)
						.apply(hooks.get(3).getId(), (stmt, match) -> {
							UUID newOwnerId = (UUID) stmt.getMeta("ownerId");

							if (newOwnerId == null)
								throw new IllegalArgumentException();

							if (!stmt.getChild(0, 1).isLiteral())
								stmt.getOperands().get(0).getOperands().get(1).unsafePutMeta("ownerId", newOwnerId);
						}, true)
						.build());

				if (t1.equals("INT")) {
					// This must be executed after the rule above
					rules.add(new RewriterRuleBuilder(ctx, "Flatten nested index expression")
							.setUnidirectional(true)
							.parseGlobalVars(t1 + ":i")
							.parseGlobalVars(t3 + ":v")
							.withParsedStatement("$1:_idxExpr(i, v)", hooks)
							.toParsedStatement("$3:_idxExpr(argList(i), v)", hooks)
							.link(hooks.get(1).getId(), hooks.get(3).getId(), RewriterStatement::transferMeta)
							.build());
				}
			}
		});

		RewriterUtils.buildBinaryPermutations(List.of("MATRIX", "INT", "FLOAT", "BOOL"), (t1, t2) -> {
			rules.add(new RewriterRuleBuilder(ctx, "Flatten fusable binary operator")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":A")
					.parseGlobalVars(t2 + ":B")
					.withParsedStatement("$1:FusableBinaryOperator(A,B)", hooks)
					.toParsedStatement("$2:FusedOperator(argList(A,B))", hooks)
					.link(hooks.get(1).getId(), hooks.get(2).getId(), RewriterStatement::transferMeta)
					.build());

			rules.add(new RewriterRuleBuilder(ctx, "Flatten fusable binary operator")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + "...:A")
					.parseGlobalVars(t2 + ":B")
					.withParsedStatement("$1:FusableBinaryOperator($2:FusedOperator(A), B)", hooks)
					.toParsedStatement("$3:FusedOperator(argList(A, B))", hooks)
					.iff(match -> {
						return match.getMatchRoot().trueInstruction().equals(match.getMatchRoot().getOperands().get(0).trueInstruction());
					}, true)
					.link(hooks.get(2).getId(), hooks.get(3).getId(), RewriterStatement::transferMeta)
					.build());

			rules.add(new RewriterRuleBuilder(ctx, "Flatten fusable binary operator")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + "...:A")
					.parseGlobalVars(t2 + ":B")
					.withParsedStatement("$1:FusableBinaryOperator(B, $2:FusedOperator(A))", hooks)
					.toParsedStatement("$3:FusedOperator(argList(B, A))", hooks)
					.iff(match -> {
						return match.getMatchRoot().trueInstruction().equals(match.getMatchRoot().getOperands().get(0).trueInstruction());
					}, true)
					.link(hooks.get(2).getId(), hooks.get(3).getId(), RewriterStatement::transferMeta)
					.build());
		});

	}
}
