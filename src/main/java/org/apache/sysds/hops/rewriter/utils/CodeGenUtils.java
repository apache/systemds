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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.hops.DataGenOp;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.hops.rewriter.assertions.RewriterAssertions;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RuleContext;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;

public class CodeGenUtils {
	// Function to access child statement (which are not neccessarily through .getInput(n))
	public static String getChildAccessor(String parentVar, RewriterStatement stmt, int childIdx) {
		switch (stmt.trueInstruction()) {
			case "const":
				if (childIdx != 1)
					return null;

				if (stmt.getChild(1).isLiteral() && Math.abs(stmt.getChild(1).floatLiteral()) == 0.0)
					return "new LiteralOp(0.0D)"; // as this might be nnz = 0 and not DataGenOp
				return "((DataGenOp)" + parentVar + ").getConstantValue()";
		}

		return parentVar + ".getInput(" + childIdx + ")";
	}

	public static String getSpecialOpCheck(RewriterStatement stmt, final RuleContext ctx, String hopVar) {
		if (!stmt.isInstruction())
			return null;
		switch (stmt.trueInstruction()) {
			case "%*%":
				return "HopRewriteUtils.isMatrixMultiply(" + hopVar + ")";
			case "const":
				if (stmt.getChild(1).isLiteral()) {
					if (Math.abs(stmt.getChild(1).floatLiteral()) == 0.0) // Then this also holds for nnz=0
						return "HopRewriteUtils.isDataGenOpWithConstantValue(" + hopVar + ", " + stmt.getChild(1).floatLiteral() + ") || " + hopVar + ".getNnz() == 0";
					return "HopRewriteUtils.isDataGenOpWithConstantValue(" + hopVar + ", " + stmt.getChild(1).floatLiteral() + ")";
				} else
					return "HopRewriteUtils.isDataGenOpWithConstantValue(" + hopVar + ")";
		}

		return null;
	}

	public static String getAdditionalCheck(RewriterStatement stmt, final RuleContext ctx, String hopVar) {
		if (!stmt.isInstruction())
			return null;

		switch (stmt.trueInstruction()) {
			case "rowSums":
				return hopVar + ".getDirection() == Types.Direction.Row";
			case "colSums":
				return hopVar + ".getDirection() == Types.Direction.Col";
			case "sum":
				return hopVar + ".getDirection() == Types.Direction.RowCol";
		}

		return null;
	}

	public static String getOpCode(RewriterStatement stmt, final RuleContext ctx) {
		if (stmt.getOperands().size() == 1) {
			// Handle unary ops
			switch (stmt.trueInstruction()) {
				case "t":
					return "Types.ReOrgOp.TRANS";
				case "rev":
					return "Types.ReOrgOp.REV";
				case "!":
					return "Types.OpOp1.NOT";
				case "sqrt":
					return "Types.OpOp1.SQRT";
				//case "sq":
				//	return "Types.OpOp1.POW2"; // POW2 does not seem to work in all cases when applying the rewrite (e.g., LinearLogRegTest)
				case "log":
					return "Types.OpOp1.LOG";
				case "log_nz":
					return "Types.OpOp1.LOG_NZ";
				case "abs":
					return "Types.OpOp1.ABS";
				case "round":
					return "Types.OpOp1.ROUND";
				case "exp":
					return "Types.OpOp1.EXP";
				case "rowSums":
				case "colSums":
				case "sum":
					return "Types.AggOp.SUM";
				case "sumSq":
					return "Types.AggOp.SUM_SQ";
				case "trace":
					return "Types.AggOp.TRACE";
				case "*2":
					return "Types.OpOp1.MULT2";
				case "cast.MATRIX":
					return "Types.OpOp1.CAST_AS_MATRIX";
				case "cast.FLOAT":
					return "Types.OpOp1.CAST_AS_SCALAR";
				case "const":
					return "Types.OpOpDG.RAND";
				case "nrow":
					return "Types.OpOp1.NROW";
				case "ncol":
					return "Types.OpOp1.NCOL";
				case "length":
					return "Types.OpOp1.LENGTH";
			}
		} else if (stmt.getOperands().size() == 2) {
			switch (stmt.trueInstruction()) {
				case "+":
					if (stmt.getOperands().size() != 2)
						throw new IllegalArgumentException();

					return "Types.OpOp2.PLUS";
				case "-":
					if (stmt.getOperands().size() != 2)
						throw new IllegalArgumentException();

					return "Types.OpOp2.MINUS";
				case "*":
					if (stmt.getOperands().size() != 2)
						throw new IllegalArgumentException();

					return "Types.OpOp2.MULT";
				case "/":
					if (stmt.getOperands().size() != 2)
						throw new IllegalArgumentException();

					return "Types.OpOp2.DIV";
				case "min":
					if (stmt.getOperands().size() != 2)
						throw new IllegalArgumentException();

					return "Types.OpOp2.MIN";
				case "max":
					if (stmt.getOperands().size() != 2)
						throw new IllegalArgumentException();

					return "Types.OpOp2.MAX";
				case "!=":
					if (stmt.getOperands().size() != 2)
						throw new IllegalArgumentException();

					return "Types.OpOp2.NOTEQUAL";
				case "==":
					if (stmt.getOperands().size() != 2)
						throw new IllegalArgumentException();

					return "Types.OpOp2.EQUAL";
				case ">":
					if (stmt.getOperands().size() != 2)
						throw new IllegalArgumentException();

					return "Types.OpOp2.GREATER";
				case ">=":
					if (stmt.getOperands().size() != 2)
						throw new IllegalArgumentException();

					return "Types.OpOp2.GREATEREQUAL";
				case "<":
					if (stmt.getOperands().size() != 2)
						throw new IllegalArgumentException();

					return "Types.OpOp2.LESS";
				case "<=":
					if (stmt.getOperands().size() != 2)
						throw new IllegalArgumentException();

					return "Types.OpOp2.LESSEQUAL";
				case "&":
					if (stmt.getOperands().size() != 2)
						throw new IllegalArgumentException();

					return "Types.OpOp2.AND";
				case "|":
					if (stmt.getOperands().size() != 2)
						throw new IllegalArgumentException();

					return "Types.OpOp2.OR";
				case "^":
					if (stmt.getOperands().size() != 2)
						throw new IllegalArgumentException();

					return "Types.OpOp2.POW";

				case "RBind":
					if (stmt.getOperands().size() != 2)
						throw new IllegalArgumentException();

					return "Types.OpOp2.RBIND";
				case "CBind":
					if (stmt.getOperands().size() != 2)
						throw new IllegalArgumentException();

					return "Types.OpOp2.CBIND";
				case "1-*":
					if (stmt.getOperands().size() != 2)
						throw new IllegalArgumentException();

					return "Types.OpOp2.MINUS1_MULT";
				case "log_nz":
					if (stmt.getOperands().size() != 1)
						throw new IllegalArgumentException();

					return "Types.OpOp1.LOG_NZ";

				case "%*%":
					return "true"; // This should be resolved by the custom handler function
			}
		} else {
			switch (stmt.trueInstruction()) {
				case "+*":
					if (stmt.getOperands().size() != 3)
						throw new IllegalArgumentException();

					return "Types.OpOp3.PLUS_MULT";
				case "-*":
					if (stmt.getOperands().size() != 3)
						throw new IllegalArgumentException();

					return "Types.OpOp3.MINUS_MULT";
				case "literal.FLOAT":
					return null; // There is no opcheck on literals
			}
		}

		throw new NotImplementedException(stmt.trueInstruction());
	}

	/**
	 *
	 * @param stmt the statement
	 * @param ctx the context
	 * @return a list of operand indices that must be matched
	 */
	public static List<Integer> matchingDimRequirement(RewriterStatement stmt, final RuleContext ctx) {
		switch (stmt.trueInstruction()) {
			case "1-*":
				return List.of(0, 1);
			case "+*":
			case "-*":
				return List.of(0, 2);
			default:
				return Collections.emptyList();
		}
	}

	public static boolean opRequiresBinaryBroadcastingMatch(RewriterStatement stmt, final RuleContext ctx) {
		return getOpClass(stmt, ctx).equals("BinaryOp") && stmt.getChild(0).getResultingDataType(ctx).equals("MATRIX") && stmt.getChild(1).getResultingDataType(ctx).equals("MATRIX");
	}

	public static String getOpClass(RewriterStatement stmt, final RuleContext ctx) {
		switch (stmt.trueInstruction()) {
			case "!":
			case "sqrt":
			case "log":
			case "log_nz":
			case "abs":
			case "round":
			case "*2":
			case "cast.MATRIX":
			case "cast.FLOAT":
			case "nrow":
			case "ncol":
			case "length":
			//case "sq": // SQ does not appear to work in some cases
			case "exp":
				return "UnaryOp";

			case "rowSums":
			case "colSums":
			case "sum":
			case "sumSq":
			case "trace":
				return "AggUnaryOp";

			case "+":
			case "-":
			case "*":
			case "/":
			case "min":
			case "max":
			case "!=":
			case "==":
			case ">":
			case ">=":
			case "<":
			case "<=":
			case "&":
			case "|":
			case "^":
			case "RBind":
			case "CBind":
			case "1-*":
				if (stmt.getOperands().size() != 2)
					throw new IllegalArgumentException();

				return "BinaryOp";

			case "%*%":
				if (stmt.getOperands().size() != 2)
					throw new IllegalArgumentException();

				return "AggBinaryOp";

			case "t":
			case "rev":
				return "ReorgOp";

			case "+*":
			case "-*":
				return "TernaryOp";

			case "const":
				return "DataGenOp";

			case "literal.FLOAT":
			case "literal.INT":
			case "literal.BOOL":
				return "LiteralOp";
		}

		throw new NotImplementedException(stmt.trueTypedInstruction(ctx));
	}

	public static String[] getReturnType(RewriterStatement stmt, final RuleContext ctx) {
		return getReturnType(stmt.getResultingDataType(ctx));
	}

	public static String[] getReturnType(String typeStr) {
		switch (typeStr) {
			case "FLOAT":
				return new String[] { "Types.DataType.SCALAR", "Types.ValueType.FP64", "Types.ValueType.FP32" };
			case "INT":
				return new String[] { "Types.DataType.SCALAR", "Types.ValueType.INT64", "Types.ValueType.INT32" };
			case "BOOL":
				return new String[] { "Types.DataType.SCALAR", "Types.ValueType.BOOLEAN" };
			case "MATRIX":
				return new String[] { "Types.DataType.MATRIX" };
		}

		throw new NotImplementedException(typeStr);
	}

	public static String literalGetterFunction(RewriterStatement stmt, final RuleContext ctx) {
		switch (stmt.getResultingDataType(ctx)) {
			case "INT":
				return "getLongValue()";
			case "FLOAT":
				return "getDoubleValue()";
			case "BOOL":
				return "getBooleanValue()";
		}

		throw new IllegalArgumentException();
	}

	public static String getHopConstructor(RewriterStatement cur, RewriterAssertions assertions, Map<RewriterStatement, String> varNameMapping, final RuleContext ctx, String... children) {
		String opClass = getOpClass(cur, ctx);
		String opCode = null;

		for (int i = 0; i < children.length; i++)
			if (children[i] == null)
				throw new IllegalArgumentException("The argument " + i + " is null: " + cur.toParsableString(ctx));

		// Special instructions
		switch (cur.trueInstruction()) {
			case "%*%":
				if (children.length != 2)
					throw new IllegalArgumentException();

				return "HopRewriteUtils.createMatrixMultiply(" + children[0] + ", " + children[1] + ")";

			case "t":
				if (children.length != 1)
					throw new IllegalArgumentException();
				return "HopRewriteUtils.createTranspose(" + children[0] + ")";

			case "rev":
				if (children.length != 1)
					throw new IllegalArgumentException();
				return "HopRewriteUtils.createReorg(" + children[0] + ", Types.ReOrgOp.REV)";

			case "rowSums":
				if (children.length != 1)
					throw new IllegalArgumentException();

				return "HopRewriteUtils.createAggUnaryOp(" + children[0] + ", Types.AggOp.SUM, Types.Direction.Row)";

			case "colSums":
				if (children.length != 1)
					throw new IllegalArgumentException();

				return "HopRewriteUtils.createAggUnaryOp(" + children[0] + ", Types.AggOp.SUM, Types.Direction.Col)";

			case "sum":
				if (children.length != 1)
					throw new IllegalArgumentException();

				return "HopRewriteUtils.createAggUnaryOp(" + children[0] + ", Types.AggOp.SUM, Types.Direction.RowCol)";

			case "sumSq":
				if (children.length != 1)
					throw new IllegalArgumentException();

				return "HopRewriteUtils.createAggUnaryOp(" + children[0] + ", Types.AggOp.SUM_SQ, Types.Direction.RowCol)";
			case "trace":
				if (children.length != 1)
					throw new IllegalArgumentException();

				return "HopRewriteUtils.createAggUnaryOp(" + children[0] + ", Types.AggOp.TRACE, Types.Direction.RowCol)";

			case "ncol":
				if (children.length != 1)
					throw new IllegalArgumentException();

				return "HopRewriteUtils.createUnary(" + children[0] + ", Types.OpOp1.NCOL)";

			case "nrow":
				if (children.length != 1)
					throw new IllegalArgumentException();

				return "HopRewriteUtils.createUnary(" + children[0] + ", Types.OpOp1.NROW)";

			case "const":
				String referredVarName = varNameMapping.get(cur.getChild(0));
				String nrowContent;
				String ncolContent;

				if (referredVarName == null) {
					Optional<RewriterStatement> nrowLiteral = cur.getNRow().isLiteral() ? Optional.of(cur.getNRow()) : Optional.empty();
					Optional<RewriterStatement> ncolLiteral = cur.getNCol().isLiteral() ? Optional.of(cur.getNCol()) : Optional.empty();

					RewriterAssertions.RewriterAssertion nrowAssertion = assertions.getAssertionObj(cur.getNRow());
					RewriterAssertions.RewriterAssertion ncolAssertion = assertions.getAssertionObj(cur.getNCol());

					nrowLiteral = nrowAssertion == null ? nrowLiteral : nrowAssertion.getLiteral();
					ncolLiteral = ncolAssertion == null ? ncolLiteral : ncolAssertion.getLiteral();


					if (nrowLiteral.isPresent()) {
						nrowContent = "new LiteralOp(" + nrowLiteral.get().getLiteral().toString() + ")";
					} else {
						// Find the first
						nrowContent = null;

						if (nrowAssertion == null)
							throw new IllegalArgumentException();

						for (RewriterStatement stmt : nrowAssertion.getEClass()) {
							String mappedName = varNameMapping.get(stmt);

							if (mappedName != null) {
								nrowContent = getHopConstructor(stmt, assertions, varNameMapping, ctx, mappedName);
								if (nrowContent != null)
									break;
							}
						}

						if (nrowContent == null)
							throw new IllegalArgumentException(nrowAssertion.toString());
					}

					if (ncolLiteral.isPresent()) {
						ncolContent = "new LiteralOp(" + ncolLiteral.get().getLiteral().toString() + ")";
					} else {
						// Find the first
						ncolContent = null;

						if (ncolAssertion == null)
							throw new IllegalArgumentException();

						for (RewriterStatement stmt : ncolAssertion.getEClass()) {
							String mappedName = varNameMapping.get(stmt);

							if (mappedName != null) {
								ncolContent = getHopConstructor(stmt, assertions, varNameMapping, ctx, mappedName);
								break;
							}
						}

						if (ncolContent == null)
							throw new IllegalArgumentException();
					}
				} else {
					nrowContent = getHopConstructor(cur.getChild(0).getNRow(), assertions, varNameMapping, ctx, referredVarName);
					ncolContent = getHopConstructor(cur.getChild(0).getNCol(), assertions, varNameMapping, ctx, referredVarName);
				}

				if (!cur.getChild(1).isLiteral())
					throw new IllegalArgumentException("Constant operator only supports literals!");

				return "((DataGenOp) HopRewriteUtils.createDataGenOpFromDims(" + nrowContent + "," + ncolContent + "," + cur.getChild(1).getLiteral() + "D))";
		}

		switch (opClass) {
			case "UnaryOp":
				if (children.length != 1)
					throw new IllegalArgumentException();

				opCode = getOpCode(cur, ctx);
				return "HopRewriteUtils.createUnary(" + children[0] + ", " + opCode + ")";
			case "BinaryOp":
				if (children.length != 2)
					throw new IllegalArgumentException();

				opCode = getOpCode(cur, ctx);
				return "HopRewriteUtils.createAutoGeneratedBinary(" + children[0] + ", " + children[1] + ", " + opCode + ")";
			case "TernaryOp":
				if (children.length != 3)
					throw new IllegalArgumentException();

				opCode = getOpCode(cur, ctx);
				return "HopRewriteUtils.createTernary(" + children[0] + ", " + children[1] + ", " + children[2] + "," + opCode + ")";
		}

		throw new NotImplementedException(cur.trueTypedInstruction(ctx));
	}

}
