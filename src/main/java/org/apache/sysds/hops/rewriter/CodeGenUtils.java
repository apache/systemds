package org.apache.sysds.hops.rewriter;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;

public class CodeGenUtils {
	public static String getSpecialOpCheck(RewriterStatement stmt, final RuleContext ctx, String hopVar) {
		//Types.AggOp.SUM
		//HopRewriteUtils.is
		switch (stmt.trueInstruction()) {
			case "%*%":
				return "HopRewriteUtils.isMatrixMultiply(" + hopVar + ")";
		}

		return null;
	}

	public static String getAdditionalCheck(RewriterStatement stmt, final RuleContext ctx, String hopVar) {
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
					return "Types.AggOp.SUM";
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
			}
		}

		throw new NotImplementedException();
	}

	public static String getOpClass(RewriterStatement stmt, final RuleContext ctx) {
		switch (stmt.trueInstruction()) {
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
				if (stmt.getOperands().size() != 2)
					throw new IllegalArgumentException();

				return "BinaryOp";

			case "%*%":
				if (stmt.getOperands().size() != 2)
					throw new IllegalArgumentException();

				return "AggBinaryOp";

			case "t":
				return "ReorgOp";

			case "rowSums":
			case "colSums":
				return "AggUnaryOp";
		}

		throw new NotImplementedException();
	}

	public static String[] getReturnType(RewriterStatement stmt, final RuleContext ctx) {
		switch (stmt.getResultingDataType(ctx)) {
			case "FLOAT":
				return new String[] { "Types.DataType.SCALAR", "Types.ValueType.FP64", "Types.ValueType.FP32" };
			case "INT":
				return new String[] { "Types.DataType.SCALAR", "Types.ValueType.INT64", "Types.ValueType.INT32" };
			case "BOOL":
				return new String[] { "Types.DataType.SCALAR", "Types.ValueType.BOOLEAN" };
			case "MATRIX":
				return new String[] { "Types.DataType.MATRIX" };
		}

		throw new NotImplementedException();
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

	public static String getHopConstructor(RewriterStatement cur, final RuleContext ctx, String... children) {
		String opClass = getOpClass(cur, ctx);

		switch (opClass) {
			case "BinaryOp":
				if (children.length != 2)
					throw new IllegalArgumentException();

				String opCode = getOpCode(cur, ctx);
				return "HopRewriteUtils.createBinary(" + children[0] + ", " + children[1] + ", " + opCode + ")";
		}

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
		}

		throw new NotImplementedException(cur.trueTypedInstruction(ctx));
	}

}
