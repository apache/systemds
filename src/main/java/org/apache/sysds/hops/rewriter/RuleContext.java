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

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.function.Function;

public class RuleContext {
	public static RuleContext currentContext;

	public HashMap<String, Function<List<RewriterStatement>, Long>> instrCosts = new HashMap<>();

	public HashMap<String, String> instrTypes = new HashMap<>();

	public HashMap<String, Function<RewriterInstruction, RewriterStatement>> simplificationRules = new HashMap<>();

	public HashMap<String, HashSet<String>> instrProperties = new HashMap<>();

	public HashMap<String, HashSet<String>> typeHierarchy = new HashMap<>();

	public HashMap<String, BiFunction<RewriterStatement, RuleContext, String>> customStringRepr = new HashMap<>();

	public Function<RewriterStatement, RewriterStatement> metaPropagator = null;

	public static RuleContext floatArithmetic = new RuleContext();
	public static RuleContext selectionPushdownContext = new RuleContext();

	static {
		floatArithmetic.instrCosts.put("+(float,float)", d -> 1l);
		floatArithmetic.instrCosts.put("*(float,float)", d ->  1l);

		floatArithmetic.instrTypes.put("+(float,float)", "float");
		floatArithmetic.instrTypes.put("*(float,float)", "float");

		floatArithmetic.simplificationRules.put("+(float,float)", i -> {
			RewriterStatement op1 = i.getOperands().get(0);
			RewriterStatement op2 = i.getOperands().get(1);

			if (op1.isLiteral() && op2.isLiteral()) {
				op1.setLiteral(((Float)op1.getLiteral()) + ((Float)op2.getLiteral()));
				return op1;
			}

			return null;
		});
		floatArithmetic.simplificationRules.put("*(float, float)", i -> {
			RewriterStatement op1 = i.getOperands().get(0);
			RewriterStatement op2 = i.getOperands().get(1);

			if (op1.isLiteral() && op2.isLiteral()) {
				op1.setLiteral(((Float)op1.getLiteral()) * ((Float)op2.getLiteral()));
				return op1;
			}

			return null;
		});

		selectionPushdownContext.instrCosts.put("RowSelectPushableBinaryInstruction(MATRIX,MATRIX)", d -> 1l); // Just temporary costs
		selectionPushdownContext.instrTypes.put("RowSelectPushableBinaryInstruction(MATRIX,MATRIX)", "MATRIX");
		selectionPushdownContext.instrCosts.put("rowSelect(MATRIX,INT,INT)", d -> 1l);
		selectionPushdownContext.instrTypes.put("rowSelect(MATRIX,INT,INT)", "MATRIX");
		selectionPushdownContext.instrCosts.put("min(INT,INT)", d -> 1l);
		selectionPushdownContext.instrTypes.put("min(INT,INT)", "INT");
		selectionPushdownContext.instrCosts.put("max(INT,INT)", d -> 1l);
		selectionPushdownContext.instrTypes.put("max(INT,INT)", "INT");

		selectionPushdownContext.instrCosts.put("+(MATRIX,MATRIX)", d -> 1l);
		selectionPushdownContext.instrTypes.put("+(MATRIX,MATRIX)", "MATRIX");
	}

	public static RuleContext createContext(String contextString) {
		RuleContext ctx = new RuleContext();
		HashMap<String, String> instrTypes = ctx.instrTypes;
		HashMap<String, HashSet<String>> instrProps = ctx.instrProperties;
		String[] lines = contextString.split("\n");
		String fName = null;
		String fArgTypes = null;
		String fReturnType = null;
		for (String line : lines) {
			line = line.replaceFirst("^\\s+", "");
			if (line.isEmpty())
				continue;

			if (line.startsWith("impl")) {
				if (fArgTypes == null || fReturnType == null)
					throw new IllegalArgumentException();
				String newFName = line.substring(4).replace(" ", "");
				if (newFName.isEmpty())
					throw new IllegalArgumentException();

				instrTypes.put(newFName + fArgTypes, fReturnType);

				final String propertyFunction = fName + fArgTypes;

				if (instrProps.containsKey(newFName + fArgTypes)) {
					HashSet<String> props = instrProps.get(newFName + fArgTypes);
					props.add(propertyFunction);
					props.add(fName);
				} else {
					HashSet<String> mset = new HashSet<>();
					mset.add(propertyFunction);
					mset.add(fName);
					instrProps.put(newFName + fArgTypes, mset);
				}

				ctx.instrCosts.put(newFName + fArgTypes, d -> 1L);
			} else if (line.startsWith("dtype ")) {
				String[] dTypeStr = line.substring(6).split("::");
				if (dTypeStr.length > 1) {
					Set<String> mSet = ctx.typeHierarchy.compute(dTypeStr[0], (k, v) -> v == null ? new HashSet<>() : v);
					for (int i = 1; i < dTypeStr.length; i++)
						mSet.add(dTypeStr[i]);
				}

			} else {
				String[] keyVal = readFunctionDefinition(line);
				fName = keyVal[0];
				fArgTypes = keyVal[1];
				fReturnType = keyVal[2];
				instrTypes.put(fName + fArgTypes, fReturnType);
				ctx.instrCosts.put(fName + fArgTypes, d -> 1L);
			}
		}

		// Resolve transitive function properties
		boolean changed = true;
		while (changed) {
			changed = false;
			for (Map.Entry<String, HashSet<String>> pair : instrProps.entrySet()) {
				HashSet<String> toAdd = new HashSet<>();
				for (String propertyFunction : pair.getValue()) {
					if (instrProps.containsKey(propertyFunction))
						toAdd.addAll(instrProps.get(propertyFunction));
				}

				changed |= pair.getValue().addAll(toAdd);
			}
		}

		changed = true;
		while (changed) {
			changed = false;
			for (Map.Entry<String, HashSet<String>> pair : ctx.typeHierarchy.entrySet()) {
				HashSet<String> toAdd = new HashSet<>();
				for (String superTypes : pair.getValue()) {
					if (instrProps.containsKey(superTypes))
						toAdd.addAll(instrProps.get(superTypes));
				}

				changed |= pair.getValue().addAll(toAdd);
			}
		}

		return ctx;
	}

	public static String[] readFunctionDefinition(String line) {
		int leftParanthesisIdx = line.indexOf('(');

		if (leftParanthesisIdx == -1)
			throw new IllegalArgumentException();

		String fName = line.substring(0, leftParanthesisIdx).replace(" ", "");
		String rest = line.substring(leftParanthesisIdx+1);

		int parenthesisCloseIdx = rest.indexOf(')');

		if (parenthesisCloseIdx == -1)
			throw new IllegalArgumentException();

		String argsStr = rest.substring(0, parenthesisCloseIdx);
		String[] args = argsStr.split(",");

		args = Arrays.stream(args).map(arg -> arg.replace(" ", "")).toArray(String[]::new);

		if (args.length != 1 && Arrays.stream(args).anyMatch(String::isEmpty))
			throw new IllegalArgumentException();

		if (!rest.substring(parenthesisCloseIdx+1, parenthesisCloseIdx+3).equals("::"))
			throw new IllegalArgumentException();

		String returnDataType = rest.substring(parenthesisCloseIdx+3);
		return new String[] { fName, "(" + String.join(",", args) + ")", returnDataType };
	}
}
