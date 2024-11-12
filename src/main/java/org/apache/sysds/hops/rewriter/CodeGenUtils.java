package org.apache.sysds.hops.rewriter;

import org.apache.commons.lang3.NotImplementedException;

public class CodeGenUtils {
	public static String getOpCode(RewriterStatement stmt, final RuleContext ctx) {
		switch (stmt.trueInstruction()) {
			case "+":
				if (stmt.getOperands().size() != 2)
					throw new IllegalArgumentException();

				return "Types.OpOp2.PLUS";
		}

		throw new NotImplementedException();
	}

	public static String getOpClass(RewriterStatement stmt, final RuleContext ctx) {
		switch (stmt.trueInstruction()) {
			case "+":
				if (stmt.getOperands().size() != 2)
					throw new IllegalArgumentException();

				return "BinaryOp";
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

}
