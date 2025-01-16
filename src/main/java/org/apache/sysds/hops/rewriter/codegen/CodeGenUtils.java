package org.apache.sysds.hops.rewriter.codegen;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.hops.rewriter.assertions.RewriterAssertions;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RuleContext;

import javax.annotation.Nullable;
import java.util.Map;
import java.util.Optional;

public class CodeGenUtils {
	public static String getSpecialOpCheck(RewriterStatement stmt, final RuleContext ctx, String hopVar) {
		if (!stmt.isInstruction())
			return null;
		switch (stmt.trueInstruction()) {
			case "%*%":
				return "HopRewriteUtils.isMatrixMultiply(" + hopVar + ")";
			//case "literal.FLOAT":
			//	return "(" + hopVar + " instanceof LiteralOp && ((LiteralOp) " + hopVar + ")";
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
			// TODO: nrow, ncol, length
			switch (stmt.trueInstruction()) {
				case "t":
					return "Types.ReOrgOp.TRANS";
				case "rev":
					return "Types.ReOrgOp.REV";
				case "!":
					return "Types.OpOp1.NOT";
				case "sqrt":
					return "Types.OpOp1.SQRT";
				case "log":
					return "Types.OpOp1.LOG";
				case "abs":
					return "Types.OpOp1.ABS";
				case "round":
					return "Types.OpOp1.ROUND";
				case "rowSums":
				case "colSums":
				case "sum":
					return "Types.AggOp.SUM";
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

	public static String getOpClass(RewriterStatement stmt, final RuleContext ctx) {
		switch (stmt.trueInstruction()) {
			case "!":
			case "sqrt":
			case "log":
			case "abs":
			case "round":
			case "*2":
			case "cast.MATRIX":
			case "cast.FLOAT":
			case "nrow":
			case "ncol":
				return "UnaryOp";

			case "rowSums":
			case "colSums":
			case "sum":
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
								break;
							}
						}

						if (nrowContent == null)
							throw new IllegalArgumentException();
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
				return "HopRewriteUtils.createBinary(" + children[0] + ", " + children[1] + ", " + opCode + ")";
			case "TernaryOp":
				if (children.length != 3)
					throw new IllegalArgumentException();

				opCode = getOpCode(cur, ctx);
				return "HopRewriteUtils.createTernary(" + children[0] + ", " + children[1] + ", " + children[2] + "," + opCode + ")";
		}

		throw new NotImplementedException(cur.trueTypedInstruction(ctx));
	}

}
