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

import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RuleContext;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.BiFunction;

public class ConstantFoldingUtils {
	static final double EPS = 1e-20;

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
			case "min":
				if (type.equals("FLOAT"))
					return (num, stmt) -> num == null ? stmt.floatLiteral() : foldMinFloat((double)num, stmt);
				else if (type.equals("INT"))
					return (num, stmt) -> num == null ? stmt.intLiteral(false) : foldMinInt((long)num, stmt);
				break;
			case "max":
				if (type.equals("FLOAT"))
					return (num, stmt) -> num == null ? stmt.floatLiteral() : foldMaxFloat((double)num, stmt);
				else if (type.equals("INT"))
					return (num, stmt) -> num == null ? stmt.intLiteral(false) : foldMaxInt((long)num, stmt);
				break;
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

	public static boolean isNegNeutral(Object num, String op) {
		if (num == null)
			return false;

		switch (op) {
			case "*":
				return num.equals(-1L) || num.equals(-1.0D);
		}

		return false;
	}

	public static boolean cancelOutNary(String op, List<RewriterStatement> stmts) {
		Set<Integer> toRemove = new HashSet<>();
		switch (op) {
			case "+":
				for (int i = 0; i < stmts.size(); i++) {
					RewriterStatement stmt1 = stmts.get(i);
					for (int j = i+1; j < stmts.size(); j++) {
						RewriterStatement stmt2 = stmts.get(j);

						if (stmt1.isInstruction() && stmt1.trueInstruction().equals("-") && stmt1.getChild(0).equals(stmt2)
							|| (stmt2.isInstruction() && stmt2.trueInstruction().equals("-") && stmt2.getChild(0).equals(stmt1))) {
							if (!toRemove.contains(i) && !toRemove.contains(j)) {
								toRemove.add(i);
								toRemove.add(j);
							}
						}

					}
				}
			case "*":
				for (int i = 0; i < stmts.size(); i++) {
					RewriterStatement stmt1 = stmts.get(i);
					for (int j = i+1; j < stmts.size(); j++) {
						RewriterStatement stmt2 = stmts.get(j);

						if (stmt1.isInstruction() && stmt1.trueInstruction().equals("inv") && stmt1.getChild(0).equals(stmt2)
								|| (stmt2.isInstruction() && stmt2.trueInstruction().equals("inv") && stmt2.getChild(0).equals(stmt1))) {
							if (!toRemove.contains(i) && !toRemove.contains(j)) {
								toRemove.add(i);
								toRemove.add(j);
							}
						}

					}
				}
		}

		if (toRemove.isEmpty())
			return false;

		List<RewriterStatement> oldCpy = new ArrayList<>(stmts);
		stmts.clear();

		for (int i = 0; i < oldCpy.size(); i++) {
			if (!toRemove.contains(i))
				stmts.add(oldCpy.get(i));
		}

		return true;
	}

	// This function does not handle NaNs
	public static RewriterStatement overwritesLiteral(Number num, String op, final RuleContext ctx) {
		if (op.equals("*") && Math.abs(num.doubleValue()) < EPS) {
			if (num instanceof Double)
				return RewriterStatement.literal(ctx, 0.0);
			else
				return RewriterStatement.literal(ctx, 0L);
		}

		return null;
	}

	public static double foldSumFloat(double num, RewriterStatement next) {
		return num + next.floatLiteral();
	}

	public static long foldSumInt(long num, RewriterStatement next) {
		return num + next.intLiteral(false);
	}

	public static double foldMulFloat(double num, RewriterStatement next) {
		return num * next.floatLiteral();
	}

	public static long foldMulInt(long num, RewriterStatement next) {
		return num * next.intLiteral(false);
	}

	public static double foldMinFloat(double num, RewriterStatement next) {
		return Math.min(num, next.floatLiteral());
	}

	public static long foldMinInt(long num, RewriterStatement next) {
		return Math.min(num, next.intLiteral(false));
	}

	public static double foldMaxFloat(double num, RewriterStatement next) {
		return Math.max(num, next.floatLiteral());
	}

	public static long foldMaxInt(long num, RewriterStatement next) {
		return Math.max(num, next.intLiteral(false));
	}
}
