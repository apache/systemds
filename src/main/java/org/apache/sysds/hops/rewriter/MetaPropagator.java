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

package org.apache.sysds.hops.rewriter;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.mutable.MutableObject;
import org.apache.sysds.hops.rewriter.assertions.RewriterAssertions;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;

import java.util.HashMap;
import java.util.Optional;
import java.util.UUID;
import java.util.function.Function;

/**
 * This class is used to propagate dimension information.
 * Each instruction that produces a matrix must be implemented here.
 */
public class MetaPropagator implements Function<RewriterStatement, RewriterStatement> {
	private final RuleContext ctx;

	public MetaPropagator(RuleContext ctx) {
		this.ctx = ctx;
	}

	public RewriterStatement apply(RewriterStatement root) {
		RewriterAssertions assertions = root.getAssertions(ctx);
		MutableObject<RewriterStatement> out = new MutableObject<>(root);
		HashMap<Object, RewriterStatement> literalMap = new HashMap<>();

		root.forEachPostOrderWithDuplicates((el, parent, pIdx) -> {
			RewriterStatement toSet = propagateDims(el, parent, pIdx, assertions);

			if (toSet != null && toSet != el) {
				el = toSet;
				if (parent == null)
					out.setValue(toSet);
				else
					parent.getOperands().set(pIdx, toSet);
			}

			// Assert
			if (el.getResultingDataType(ctx).startsWith("MATRIX")
				&& (el.getNCol() == null || el.getNRow() == null))
				throw new IllegalArgumentException("Some properties have not been set by the meta propagator: " + el.toString(ctx) + " :: " + el.getResultingDataType(ctx));


			// Eliminate common literals
			if (el.isLiteral()) {
				RewriterStatement existingLiteral = literalMap.get(el.getLiteral());

				if (existingLiteral != null) {
					if (parent == null)
						out.setValue(existingLiteral);
					else
						parent.getOperands().set(pIdx, existingLiteral);
				} else {
					literalMap.put(el.getLiteral(), el);
				}
			}

			validate(el);
		});

		return out.getValue();
	}

	private RewriterStatement propagateDims(RewriterStatement root, RewriterStatement parent, int pIdx, RewriterAssertions assertions) {
		if (root.getResultingDataType(ctx) == null)
			throw new IllegalArgumentException("Null type: " + root.toParsableString(ctx));
		if (!root.getResultingDataType(ctx).startsWith("MATRIX")) {
			if (root.isInstruction()) {
				String ti = root.trueTypedInstruction(ctx);
				RewriterStatement ret = null;

				switch (ti) {
					case "ncol(MATRIX)":
						ret = (RewriterStatement)root.getOperands().get(0).getMeta("ncol");
						break;
					case "nrow(MATRIX)":
						ret = (RewriterStatement)root.getOperands().get(0).getMeta("nrow");
						break;
				}

				if (ret == null)
					return null;

				RewriterStatement asserted = assertions != null ? assertions.getAssertionStatement(ret, parent) : null;

				if (asserted == null)
					return ret;

				return asserted;
			}
			return null;
		}

		Object colAccess;
		Object rowAccess;

		if (root.getOperands() == null || root.getOperands().isEmpty()) {
			RewriterStatement ncol = root.getNCol();

			if (ncol == null) {
				root.unsafePutMeta("ncol", new RewriterInstruction().withInstruction("ncol").withOps(root).as(UUID.randomUUID().toString()).consolidate(ctx));
			}

			RewriterStatement nrow = root.getNRow();

			if (nrow == null) {
				root.unsafePutMeta("nrow", new RewriterInstruction().withInstruction("nrow").withOps(root).as(UUID.randomUUID().toString()).consolidate(ctx));
			}

			return null;
		}

		if (root.isInstruction()) {
			Optional<RewriterStatement> firstMatrixStatement = root.getOperands().stream().filter(el -> el.getResultingDataType(ctx).startsWith("MATRIX")).findFirst();
			switch(root.trueInstruction()) {
				// Handle generators
				case "rand":
					root.unsafePutMeta("nrow", root.getOperands().get(0));
					root.unsafePutMeta("ncol", root.getOperands().get(1));
					return null;
				case "as.matrix":
					root.unsafePutMeta("ncol", new RewriterDataType().ofType("INT").as("1").asLiteral(1L).consolidate(ctx));
					root.unsafePutMeta("nrow", new RewriterDataType().ofType("INT").as("1").asLiteral(1L).consolidate(ctx));
					return null;
				case "argList":
					// We assume argLists always occur if the matrix properties don't change
					root.unsafePutMeta("nrow", firstMatrixStatement.get().getMeta("nrow"));
					root.unsafePutMeta("ncol", firstMatrixStatement.get().getMeta("ncol"));
					return null;
				case "_map":
					root.unsafePutMeta("nrow", root.getOperands().get(1).getMeta("nrow"));
					root.unsafePutMeta("ncol", root.getOperands().get(1).getMeta("ncol"));
					return null;
				case "+":
				case "-":
				case "*":
				case "inv":
				case "==":
				case "!=":
				case "&":
				case "|":
				case "<":
				case ">":
				case "abs":
				case "round":
				case "exp":
				case "^":
					if (firstMatrixStatement.isEmpty())
						throw new IllegalArgumentException(root.toString(ctx) + " has empty args!");
					root.unsafePutMeta("nrow", firstMatrixStatement.get().getMeta("nrow"));
					root.unsafePutMeta("ncol", firstMatrixStatement.get().getMeta("ncol"));
					return null;
				case "cast.MATRIX":
					String mDT = root.getChild(0).getResultingDataType(ctx);
					if (mDT.equals("BOOL") || mDT.equals("INT") || mDT.equals("FLOAT")) {
						root.unsafePutMeta("ncol", new RewriterDataType().ofType("INT").as("1").asLiteral(1L).consolidate(ctx));
						root.unsafePutMeta("nrow", new RewriterDataType().ofType("INT").as("1").asLiteral(1L).consolidate(ctx));
						return null;
					}
				case "log_nz":
				case "log":
					root.unsafePutMeta("nrow", root.getOperands().get(0).getMeta("nrow"));
					root.unsafePutMeta("ncol", root.getOperands().get(0).getMeta("ncol"));
					return null;
			}

			switch(root.trueTypedInstruction(ctx)) {
				case "t(MATRIX)":
					colAccess = root.getOperands().get(0).getMeta("ncol");
					rowAccess = root.getOperands().get(0).getMeta("nrow");
					root.unsafePutMeta("ncol", rowAccess);
					root.unsafePutMeta("nrow", colAccess);
					return null;
				case "_m(INT,INT,FLOAT)":
				case "_m(INT,INT,BOOL)":
				case "_m(INT,INT,INT)":
					if (root.getOperands().get(0).isInstruction()
							&& root.getOperands().get(0).trueTypedInstruction(ctx).equals("_idx(INT,INT)")) {
						root.unsafePutMeta("nrow", root.getOperands().get(0).getOperands().get(1));
					} else {
						root.unsafePutMeta("nrow", new RewriterDataType().ofType("INT").as("1").asLiteral(1L).consolidate(ctx));
					}

					if (root.getOperands().get(1).isInstruction()
							&& root.getOperands().get(1).trueTypedInstruction(ctx).equals("_idx(INT,INT)")) {
						root.unsafePutMeta("ncol", root.getOperands().get(1).getOperands().get(1));
					} else {
						root.unsafePutMeta("ncol", new RewriterDataType().ofType("INT").as("1").asLiteral(1L).consolidate(ctx));
					}
					return null;
				case "%*%(MATRIX,MATRIX)":
					rowAccess = root.getOperands().get(0).getMeta("nrow");
					colAccess = root.getOperands().get(1).getMeta("ncol");
					root.unsafePutMeta("nrow", rowAccess);
					root.unsafePutMeta("ncol", colAccess);
					return null;
				case "diag(MATRIX)":
					root.unsafePutMeta("nrow", root.getOperands().get(0).getMeta("nrow"));
					root.unsafePutMeta("ncol", root.getOperands().get(0).getMeta("ncol"));
					return null;
				case "[](MATRIX,INT,INT,INT,INT)":
					Long[] ints = new Long[4];

					for (int i = 1; i < 5; i++)
						if (root.getChild(i).isLiteral())
							if (root.getChild(i).getLiteral() instanceof Integer)
								ints[i-1] = (Long)root.getChild(i).getLiteral();

					if (ints[0] != null && ints[1] != null) {
						String literalString = Long.toString(ints[1] - ints[0] + 1);
						root.unsafePutMeta("nrow", RewriterUtils.foldConstants(RewriterUtils.parse(literalString, ctx, "LITERAL_INT:" + literalString), ctx));
					} else {
						HashMap<String, RewriterStatement> subStmts = new HashMap<>();
						subStmts.put("i1", root.getOperands().get(2));
						subStmts.put("i0", root.getOperands().get(1));

						if (ints[0] != null) {
							root.unsafePutMeta("nrow", RewriterUtils.foldConstants(RewriterUtils.parse("+(argList(i1, " + (1 - ints[0]) + "))", ctx, subStmts, "LITERAL_INT:" + (1 - ints[0])), ctx));
						} else if (ints[1] != null) {
							root.unsafePutMeta("nrow", RewriterUtils.foldConstants(RewriterUtils.parse("+(argList(" + (ints[1] + 1) + ", -(i0)))", ctx, subStmts, "LITERAL_INT:" + (ints[1] + 1)), ctx));
						} else {
							root.unsafePutMeta("nrow", RewriterUtils.foldConstants(RewriterUtils.parse("+(argList(i1, -(i0), 1))", ctx, subStmts, "LITERAL_INT:1"), ctx));
						}
					}

					if (ints[2] != null && ints[3] != null) {
						root.unsafePutMeta("ncol", ints[3] - ints[2] + 1);
					} else {
						HashMap<String, RewriterStatement> subStmts = new HashMap<>();
						subStmts.put("i3", root.getOperands().get(4));
						subStmts.put("i2", root.getOperands().get(3));
						if (ints[2] != null) {
							root.unsafePutMeta("ncol", RewriterUtils.foldConstants(RewriterUtils.parse("+(argList(i3, " + (1 - ints[2]) + "))", ctx, subStmts, "LITERAL_INT:" + (1 - ints[2])), ctx));
						} else if (ints[3] != null) {
							root.unsafePutMeta("ncol", RewriterUtils.foldConstants(RewriterUtils.parse("+(argList(" + (ints[3] + 1) + ", -(i2)))", ctx, subStmts, "LITERAL_INT:" + (ints[3] + 1)), ctx));
						} else {
							root.unsafePutMeta("ncol", RewriterUtils.foldConstants(RewriterUtils.parse("+(argList(i3, -(i2), 1))", ctx, subStmts, "LITERAL_INT:1"), ctx));
						}
					}

					return null;
				case "rowSums(MATRIX)":
					root.unsafePutMeta("nrow", root.getOperands().get(0).getMeta("nrow"));
					root.unsafePutMeta("ncol", new RewriterDataType().ofType("INT").as("1").asLiteral(1L).consolidate(ctx));
					return null;
				case "colSums(MATRIX)":
					root.unsafePutMeta("ncol", root.getOperands().get(0).getMeta("ncol"));
					root.unsafePutMeta("nrow", new RewriterDataType().ofType("INT").as("1").asLiteral(1L).consolidate(ctx));
					return null;
				case "cast.MATRIX(MATRIX)":
					root.unsafePutMeta("nrow", root.getOperands().get(0).getMeta("nrow"));
					root.unsafePutMeta("ncol", root.getOperands().get(0).getMeta("ncol"));
					return null;
				case "RBind(MATRIX,MATRIX)":
					HashMap<String, RewriterStatement> mstmts = new HashMap<>();
					mstmts.put("row1", (RewriterStatement)root.getOperands().get(0).getMeta("nrow"));
					mstmts.put("row2", (RewriterStatement)root.getOperands().get(1).getMeta("nrow"));
					root.unsafePutMeta("nrow", RewriterUtils.parse("+(argList(row1, row2))", ctx, mstmts));
					root.unsafePutMeta("ncol", root.getOperands().get(0).getMeta("ncol"));
					return null;
				case "CBind(MATRIX,MATRIX)":
					mstmts = new HashMap<>();
					mstmts.put("col1", (RewriterStatement)root.getOperands().get(0).getMeta("ncol"));
					mstmts.put("col2", (RewriterStatement)root.getOperands().get(1).getMeta("ncol"));
					root.unsafePutMeta("nrow", root.getOperands().get(0).getMeta("nrow"));
					root.unsafePutMeta("ncol", RewriterUtils.parse("+(argList(col1, col2))", ctx, mstmts));
					return null;

				// Fused ops
				case "1-*(MATRIX,MATRIX)":
				case "log_nz(MATRIX)":
					root.unsafePutMeta("nrow", root.getOperands().get(0).getMeta("nrow"));
					root.unsafePutMeta("ncol", root.getOperands().get(0).getMeta("ncol"));
					return null;
				case "const(MATRIX,FLOAT)":
					root.unsafePutMeta("nrow", root.getOperands().get(0).getMeta("nrow"));
					root.unsafePutMeta("ncol", root.getOperands().get(0).getMeta("ncol"));
					return null;
				case "rowVec(MATRIX)":
					root.unsafePutMeta("nrow", root.getOperands().get(0).getMeta("nrow"));
					root.unsafePutMeta("ncol", RewriterStatement.literal(ctx, 1L));
					return null;
				case "colVec(MATRIX)":
					root.unsafePutMeta("ncol", root.getOperands().get(0).getMeta("ncol"));
					root.unsafePutMeta("nrow", RewriterStatement.literal(ctx, 1L));
					return null;
				case "cellMat(MATRIX)":
					root.unsafePutMeta("ncol", RewriterStatement.literal(ctx, 1L));
					root.unsafePutMeta("nrow", RewriterStatement.literal(ctx, 1L));
					return null;
				case "rev(MATRIX)":
				case "replace(MATRIX,FLOAT,FLOAT)":
				case "sumSq(MATRIX)":
				case "+*(MATRIX,FLOAT,MATRIX)":
				case "-*(MATRIX,FLOAT,MATRIX)":
				case "*2(MATRIX)":
				case "sq(MATRIX)":
				case "!(MATRIX)":
					root.unsafePutMeta("nrow", root.getChild(0).getMeta("nrow"));
					root.unsafePutMeta("ncol", root.getChild(0).getMeta("ncol"));
					return null;
			}

			RewriterInstruction instr = (RewriterInstruction) root;

			if (instr.getProperties(ctx).contains("ElementWiseInstruction")) {
				if (root.getOperands().get(0).getResultingDataType(ctx).startsWith("MATRIX")) {
					root.unsafePutMeta("nrow", root.getOperands().get(0).getMeta("nrow"));
					root.unsafePutMeta("ncol", root.getOperands().get(0).getMeta("ncol"));
				} else {
					root.unsafePutMeta("nrow", root.getOperands().get(1).getMeta("nrow"));
					root.unsafePutMeta("ncol", root.getOperands().get(1).getMeta("ncol"));
				}

				return null;
			}

			if (instr.getProperties(ctx).contains("ElementWiseUnary.FLOAT")) {
				if (root.getOperands().get(0).getResultingDataType(ctx).startsWith("MATRIX")) {
					root.unsafePutMeta("nrow", root.getOperands().get(0).getMeta("nrow"));
					root.unsafePutMeta("ncol", root.getOperands().get(0).getMeta("ncol"));
				} else {
					root.unsafePutMeta("nrow", root.getOperands().get(1).getMeta("nrow"));
					root.unsafePutMeta("ncol", root.getOperands().get(1).getMeta("ncol"));
				}

				return null;
			}

			throw new NotImplementedException("Unknown instruction: " + instr.trueTypedInstruction(ctx) + "\n" + instr.toParsableString(ctx));
		}

		return null;
	}

	private void validate(RewriterStatement stmt) {
		if (stmt.isInstruction()) {
			if (stmt.trueInstruction().equals("_idx") && (stmt.getMeta("ownerId") == null || stmt.getMeta("idxId") == null))
				throw new IllegalArgumentException(stmt.toString(ctx));

			if (stmt.trueInstruction().equals("_m") && stmt.getMeta("ownerId") == null)
				throw new IllegalArgumentException(stmt.toString(ctx));

			if (stmt.getResultingDataType(ctx) == null)
				throw new IllegalArgumentException(stmt.toString(ctx));
		}
	}
}
