package org.apache.sysds.hops.rewriter;


import java.util.function.BiFunction;

public class ConstantFoldingFunctions {
	public static BiFunction<Number, RewriterStatement, Number> foldingBiFunction(String op, String type) {
		switch (op) {
			case "+":
				if (type.equals("FLOAT"))
					return (num, stmt) -> foldSumFloat(num == null ? 0.0 : (double)num, stmt);
				else if (type.equals("INT"))
					return (num, stmt) -> foldSumInt(num == null ? 0L : (long)num, stmt);
				else
					throw new UnsupportedOperationException();
			case "*":
				if (type.equals("FLOAT"))
					return (num, stmt) -> foldMulFloat(num == null ? 1.0D : (double)num, stmt);
				else if (type.equals("INT"))
					return (num, stmt) -> foldMulInt(num == null ? 1L : (long)num, stmt);
				else
					throw new UnsupportedOperationException();
		}

		throw new UnsupportedOperationException();
	}

	public static boolean isNeutralElement(Object num, String op) {
		switch (op) {
			case "+":
				return num.equals(0L) || num.equals(0.0D);
			case "*":
				return num.equals(1L) || num.equals(1.0D);
		}

		return false;
	}

	public static double foldSumFloat(double num, RewriterStatement next) {
		return num + next.floatLiteral();
	}

	public static long foldSumInt(long num, RewriterStatement next) {
		return num + next.intLiteral();
	}

	public static double foldMulFloat(double num, RewriterStatement next) {
		return num * next.floatLiteral();
	}

	public static long foldMulInt(long num, RewriterStatement next) {
		return num * next.intLiteral();
	}
}
