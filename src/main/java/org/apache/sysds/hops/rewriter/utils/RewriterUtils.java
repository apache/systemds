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
import org.apache.commons.lang3.mutable.MutableBoolean;
import org.apache.commons.lang3.mutable.MutableInt;
import org.apache.commons.lang3.mutable.MutableObject;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.logging.log4j.util.TriConsumer;
import org.apache.sysds.hops.rewriter.MetaPropagator;
import org.apache.sysds.hops.rewriter.RewriterContextSettings;
import org.apache.sysds.hops.rewriter.RewriterDataType;
import org.apache.sysds.hops.rewriter.rule.RewriterHeuristic;
import org.apache.sysds.hops.rewriter.rule.RewriterHeuristics;
import org.apache.sysds.hops.rewriter.RewriterInstruction;
import org.apache.sysds.hops.rewriter.rule.RewriterRule;
import org.apache.sysds.hops.rewriter.rule.RewriterRuleBuilder;
import org.apache.sysds.hops.rewriter.rule.RewriterRuleCollection;
import org.apache.sysds.hops.rewriter.rule.RewriterRuleSet;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.hops.rewriter.TopologicalSort;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.UUID;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class RewriterUtils {
	protected static final Log LOG = LogFactory.getLog(RewriterUtils.class.getName());

	public static final Pattern LONG_PATTERN = Pattern.compile("-?\\d+");
	public static final Pattern DOUBLE_PATTERN = Pattern.compile("-?\\d*\\.\\d+([eE][+-]?\\d+)?");
	public static final Pattern SPECIAL_FLOAT_PATTERN = Pattern.compile("Infinity|NaN");

	public static String typedToUntypedInstruction(String instr) {
		return instr.substring(0, instr.indexOf('('));
	}

	public static BiFunction<RewriterStatement, RuleContext, String> binaryStringRepr(String op) {
		return (stmt, ctx) -> {
			List<RewriterStatement> operands = stmt.getOperands();
			String op1Str = operands.get(0).toString(ctx);
			if (operands.get(0) instanceof RewriterInstruction && operands.get(0).getOperands().size() > 1)
				op1Str = "(" + op1Str + ")";
			String op2Str = operands.get(1).toString(ctx);
			if (operands.get(1) instanceof RewriterInstruction && operands.get(1).getOperands().size() > 1)
				op2Str = "(" + op2Str + ")";
			return op1Str + op + op2Str;
		};
	}

	public static void mergeArgLists(RewriterStatement stmt, final RuleContext ctx) {

		stmt.forEachPreOrder(el -> {
			tryFlattenNestedArgList(ctx, el, el, -1);
			tryFlattenNestedOperatorPatterns(ctx, el);
			el.refreshReturnType(ctx);
			return true;
		}, true);

		stmt.prepareForHashing();
		stmt.recomputeHashCodes(ctx);
	}

	public static boolean tryFlattenNestedArgList(final RuleContext ctx, RewriterStatement stmt, RewriterStatement root, int insertAt) {
		if (!stmt.isArgumentList())
			return false;

		if (stmt == root) {
			boolean anyMatch = false;

			for (int i = 0; i < stmt.getOperands().size(); i++) {
				RewriterStatement op = stmt.getOperands().get(i);
				if (tryFlattenNestedArgList(ctx, op, root, i)) {
					stmt.getOperands().remove(i);
					anyMatch = true;
				}
			}

			return anyMatch;
		}

		String dt1 = root.getResultingDataType(ctx);
		String dt2 = stmt.getResultingDataType(ctx);

		String convertibleDataType = convertibleType(dt1.substring(0, dt1.length()-3), dt2.substring(0, dt2.length()-3));

		if (convertibleDataType == null)
			return false;

		root.getOperands().addAll(insertAt+1, stmt.getOperands());

		return true;
	}

	public static void tryFlattenNestedOperatorPatterns(final RuleContext ctx, RewriterStatement stmt) {
		if (!stmt.isInstruction())
			return;

		RewriterInstruction instr = (RewriterInstruction) stmt;

		if (instr.hasProperty("FusedOperator", ctx)) {
			for (int i = 0; i < instr.getOperands().get(0).getOperands().size(); i++)
				if (flattenNestedOperatorPatterns(ctx, instr.getOperands().get(0).getOperands().get(i), instr, i))
					i--;
		}
	}

	private static boolean flattenNestedOperatorPatterns(final RuleContext ctx, RewriterStatement stmt, RewriterInstruction rootInstr, int insertAt) {
		if (stmt.isInstruction() && ((RewriterInstruction)stmt).hasProperty("FusedOperator", ctx) && stmt.trueInstruction().equals(rootInstr.trueInstruction())) {
			RewriterStatement origArgList = rootInstr.getOperands().get(0);
			RewriterStatement subArgList = stmt.getOperands().get(0);

			origArgList.getOperands().set(insertAt, subArgList.getOperands().get(0));
			origArgList.getOperands().addAll(insertAt+1, subArgList.getOperands().subList(1, subArgList.getOperands().size()));

			return true;
		}

		return false;
	}

	public static RewriterStatement parse(String expr, final RuleContext ctx) {
		String[] split = expr.split("\n");
		return parse(split[split.length-1], ctx, Arrays.copyOfRange(split, 0, split.length-1));
	}

	public static RewriterRule parseRule(String expr, final RuleContext ctx) {
		// Remove empty lines
		expr = expr.replaceAll("\n\\s*\n", "\n");
		String[] split = expr.split("\n");
		Set<Integer> allowedMultiRefs = Collections.emptySet();
		boolean allowCombinations = false;
		boolean parsedExtendedHeader = false;

		if (split[0].startsWith("AllowedMultiRefs:")) {
			split[0] = split[0].substring(17);
			String[] sSplit = split[0].split(",");
			allowedMultiRefs = Arrays.stream(sSplit).map(s -> Integer.parseInt(s.substring(1))).collect(Collectors.toSet());

			if (!split[1].startsWith("AllowCombinations:"))
				throw new IllegalArgumentException();

			split[1] = split[1].substring(18);
			allowCombinations = Boolean.parseBoolean(split[1]);
			parsedExtendedHeader = true;
		}

		int condIdxStart = -1;
		for (int i = 2; i < split.length; i++) {
			if (split[i].startsWith("{")) {
				// Then we have a conditional rule
				condIdxStart = i;
				break;
			}
		}

		if (condIdxStart != -1) {
			// Then we have a conditional rule
			List<String> toExprs = Arrays.asList(split).subList(condIdxStart+1, split.length-1);
			return parseRule(split[condIdxStart-2], toExprs, allowedMultiRefs, allowCombinations, ctx, Arrays.copyOfRange(split, parsedExtendedHeader ? 2 : 0, condIdxStart-2));
		}

		return parseRule(split[split.length-3], split[split.length-1], allowedMultiRefs, allowCombinations, ctx, Arrays.copyOfRange(split, parsedExtendedHeader ? 2 : 0, split.length-3));
	}

	public static RewriterStatement parse(String expr, final RuleContext ctx, String... varDefinitions) {
		return parse(expr, ctx, new HashMap<>(), varDefinitions);
	}

	public static RewriterRule parseRule(String exprFrom, String exprTo, Set<Integer> allowedMultiRefs, boolean allowCombinations, final RuleContext ctx, String... varDefinitions) {
		return parseRule(exprFrom, exprTo, ctx, new HashMap<>(), allowedMultiRefs, allowCombinations, varDefinitions);
	}

	public static RewriterRule parseRule(String exprFrom, List<String> exprsTo, Set<Integer> allowedMultiRefs, boolean allowCombinations, final RuleContext ctx, String... varDefinitions) {
		return parseRule(exprFrom, exprsTo, ctx, new HashMap<>(), allowedMultiRefs, allowCombinations, true, varDefinitions);
	}

	public static RewriterStatement parse(String expr, final RuleContext ctx, Map<String, RewriterStatement> dataTypes, String... varDefinitions) {
		for (String def : varDefinitions)
			parseDataTypes(def, dataTypes, ctx);

		RewriterStatement parsed = parseExpression(expr, new HashMap<>(), dataTypes, ctx);
		if (ctx.metaPropagator == null)
			return parsed;
		else {
			RewriterStatement out = ctx.metaPropagator.apply(parsed);
			out.prepareForHashing();
			out.recomputeHashCodes(ctx);
			return out;
		}
	}

	public static RewriterRule parseRule(String exprFrom, String exprTo, final RuleContext ctx, Map<String, RewriterStatement> dataTypes, Set<Integer> allowedMultiRefs, boolean allowCombinations, String... varDefinitions) {
		for (String def : varDefinitions)
			parseDataTypes(def, dataTypes, ctx);

		HashMap<Integer, RewriterStatement> mmap = new HashMap<>();

		RewriterStatement parsedFrom = parseExpression(exprFrom, mmap, dataTypes, ctx);
		RewriterStatement parsedTo = parseExpression(exprTo, mmap, dataTypes, ctx);

		if (ctx.metaPropagator != null) {
			parsedFrom = ctx.metaPropagator.apply(parsedFrom);
			parsedTo = ctx.metaPropagator.apply(parsedTo);
		}

		return new RewriterRuleBuilder(ctx).completeRule(parsedFrom, parsedTo).withAllowedMultiRefs(allowedMultiRefs.stream().map(mmap::get).collect(Collectors.toSet()), allowCombinations).setUnidirectional(true).build();
	}

	public static RewriterRule parseRule(String exprFrom, List<String> exprsTo, final RuleContext ctx, Map<String, RewriterStatement> dataTypes, Set<Integer> allowedMultiRefs, boolean allowCombinations, boolean asConditional, String... varDefinitions) {
		if (!asConditional && exprsTo.size() > 1)
			throw new IllegalArgumentException();

		for (String def : varDefinitions)
			parseDataTypes(def, dataTypes, ctx);

		HashMap<Integer, RewriterStatement> mmap = new HashMap<>();

		RewriterStatement parsedFrom = parseExpression(exprFrom, mmap, dataTypes, ctx);
		if (ctx.metaPropagator != null) {
			parsedFrom = ctx.metaPropagator.apply(parsedFrom);
		}

		List<RewriterStatement> parsedTos = new ArrayList<>();
		for (String exprTo : exprsTo) {
			RewriterStatement parsedTo = parseExpression(exprTo, mmap, dataTypes, ctx);

			if (ctx.metaPropagator != null) {
				parsedTo = ctx.metaPropagator.apply(parsedTo);
				parsedTo.prepareForHashing();
				parsedTo.recomputeHashCodes(ctx);
			}

			parsedTos.add(parsedTo);
		}

		return new RewriterRuleBuilder(ctx)
				.completeConditionalRule(parsedFrom, parsedTos)
				.withAllowedMultiRefs(allowedMultiRefs.stream().map(mmap::get).collect(Collectors.toSet()), allowCombinations)
				.setUnidirectional(true).build();
	}

	/**
	 * Parses an expression
	 * @param expr the expression string
	 * @param refmap test
	 * @param dataTypes data type
	 * @param ctx context
	 * @return test
	 */
	public static RewriterStatement parseExpression(String expr, Map<Integer, RewriterStatement> refmap, Map<String, RewriterStatement> dataTypes, final RuleContext ctx) {
		RuleContext.currentContext = ctx;
		expr = expr.replaceAll("\\s+", "");
		MutableObject<String> mexpr = new MutableObject<>(expr);
		RewriterStatement stmt = doParseExpression(mexpr, refmap, dataTypes, ctx);
		stmt.prepareForHashing();
		stmt.consolidate(ctx);
		return stmt;
	}

	private static RewriterStatement doParseExpression(MutableObject<String> mexpr, Map<Integer, RewriterStatement> refmap, Map<String, RewriterStatement> dataTypes, final RuleContext ctx) {
		String expr = mexpr.getValue();
		if (expr.startsWith("$")) {
			expr = expr.substring(1);
			Pattern pattern = Pattern.compile("^\\d+");
			Matcher matcher = pattern.matcher(expr);

			if (matcher.find()) {
				String number = matcher.group();
				int n = Integer.parseInt(number);
				if (expr.charAt(matcher.end()) != ':') {
					// Then we inject the common subexpression
					String remainder = expr.substring(matcher.end());
					mexpr.setValue(remainder);
					RewriterStatement var = refmap.get(n);

					if (var == null)
						throw new IllegalArgumentException("Variable '$" + n + "' does not exist!");

					return var;
				}
				String remainder = expr.substring(matcher.end() + 1);
				mexpr.setValue(remainder);
				RewriterStatement stmt = parseRawExpression(mexpr, refmap, dataTypes, ctx);
				refmap.put(n, stmt);
				return stmt;
			} else {
				throw new IllegalArgumentException("Expected a number");
			}
		} else {
			return parseRawExpression(mexpr, refmap, dataTypes, ctx);
		}
	}

	public static boolean parseDataTypes(String expr, Map<String, RewriterStatement> dataTypes, final RuleContext ctx) {
		RuleContext.currentContext = ctx;
		Pattern pattern = Pattern.compile("([A-Za-z0-9]|_|\\.|\\*|\\?)([A-Za-z0-9]|_|\\.|\\*|-)*");
		Matcher matcher = pattern.matcher(expr);

		if (!matcher.find())
			return false;

		String dType = matcher.group();
		boolean intLiteral = dType.equals("LITERAL_INT");
		boolean boolLiteral = dType.equals("LITERAL_BOOL");
		boolean floatLiteral = dType.equals("LITERAL_FLOAT");

		if (intLiteral) {
			pattern = Pattern.compile("(-)?[0-9]+");
		} else if (boolLiteral) {
			pattern = Pattern.compile("(TRUE|FALSE)");
		} else if (floatLiteral) {
			pattern = Pattern.compile("((-)?([0-9]+(\\.[0-9]*)?(E(-)?[0-9]+)?|Infinity)|NaN)");
		}

		if (expr.charAt(matcher.end()) != ':')
			return false;

		expr = expr.substring(matcher.end() + 1);

		matcher = pattern.matcher(expr);

		while (matcher.find()) {
			String varName = matcher.group();

			RewriterDataType dt;

			if (intLiteral) {
				dt = new RewriterDataType().as(varName).ofType("INT").asLiteral(Long.parseLong(varName));
			} else if (boolLiteral) {
				dt = new RewriterDataType().as(varName).ofType("BOOL").asLiteral(Boolean.parseBoolean(varName));
			} else if (floatLiteral) {
				dt = new RewriterDataType().as(varName).ofType("FLOAT").asLiteral(Double.parseDouble(varName));
			} else {
				dt = new RewriterDataType().as(varName).ofType(dType);
			}

			dt.consolidate(ctx);
			dataTypes.put(varName, dt);

			if (expr.length() == matcher.end())
				return true;

			if (expr.charAt(matcher.end()) != ',')
				return false;

			expr = expr.substring(matcher.end()+1);
			matcher = pattern.matcher(expr);
		}

		return false;
	}

	private static RewriterStatement parseRawExpression(MutableObject<String> mexpr, Map<Integer, RewriterStatement> refmap, Map<String, RewriterStatement> dataTypes, final RuleContext ctx) {
		String expr = mexpr.getValue();

		Pattern pattern = Pattern.compile("^[^(),:]+");
		Matcher matcher = pattern.matcher(expr);

		if (matcher.find()) {
			String token = matcher.group();
			String remainder = expr.substring(matcher.end());

			if (remainder.isEmpty()) {
				mexpr.setValue(remainder);
				if (dataTypes.containsKey(token))
					return dataTypes.get(token);
				throw new IllegalArgumentException("DataType: '" + token + "' doesn't exist");
			}


			char nextChar = remainder.charAt(0);

			switch (nextChar) {
				case '(':
					// Then this is a function
					if (remainder.charAt(1) == ')') {
						RewriterInstruction mInstr = new RewriterInstruction().withInstruction(token).as(UUID.randomUUID().toString());
						handleSpecialInstructions(mInstr);
						mInstr.consolidate(ctx);
						mexpr.setValue(remainder.substring(2));
						return mInstr;
					} else {
						List<RewriterStatement> opList = new ArrayList<>();
						mexpr.setValue(remainder.substring(1));
						RewriterStatement cstmt = doParseExpression(mexpr, refmap, dataTypes, ctx);
						opList.add(cstmt);

						while (mexpr.getValue().charAt(0) == ',') {
							mexpr.setValue(mexpr.getValue().substring(1));
							cstmt = doParseExpression(mexpr, refmap, dataTypes, ctx);
							opList.add(cstmt);
						}

						if (mexpr.getValue().charAt(0) != ')')
							throw new IllegalArgumentException(mexpr.getValue());

						mexpr.setValue(mexpr.getValue().substring(1));
						RewriterInstruction instr = new RewriterInstruction().withInstruction(token).withOps(opList.toArray(RewriterStatement[]::new)).as(UUID.randomUUID().toString());
						handleSpecialInstructions(instr);
						instr.consolidate(ctx);
						return instr;
					}
				case ')':
				case ',':
					mexpr.setValue(remainder);
					if (dataTypes.containsKey(token))
						return dataTypes.get(token);
					throw new IllegalArgumentException("DataType: '" + token + "' doesn't exist");
				default:
					throw new NotImplementedException();
			}
		} else {
			throw new IllegalArgumentException(mexpr.getValue());
		}
	}

	private static void handleSpecialInstructions(RewriterInstruction instr) {
		if (instr.trueInstruction().equals("_m")) {
			UUID ownerId = UUID.randomUUID();
			instr.unsafePutMeta("ownerId", ownerId);

			if (instr.getOperands().get(0).isInstruction() && instr.getOperands().get(0).trueInstruction().equals("_idx")) {
				instr.getOperands().get(0).unsafePutMeta("ownerId", ownerId);
				instr.getOperands().get(0).unsafePutMeta("idxId", UUID.randomUUID());
			}

			if (instr.getOperands().get(1).isInstruction() && instr.getOperands().get(1).trueInstruction().equals("_idx")) {
				instr.getOperands().get(1).unsafePutMeta("ownerId", ownerId);
				instr.getOperands().get(1).unsafePutMeta("idxId", UUID.randomUUID());
			}
		} else if (instr.trueInstruction().equals("_idxExpr")) {
			UUID ownerId = UUID.randomUUID();
			instr.unsafePutMeta("ownerId", ownerId);

			if (instr.getOperands().get(0).isInstruction() && instr.getOperands().get(0).trueInstruction().equals("_idx")) {
				instr.getOperands().get(0).unsafePutMeta("ownerId", ownerId);
				instr.getOperands().get(0).unsafePutMeta("idxId", UUID.randomUUID());
			}
		}
	}

	public static void buildBinaryAlgebraInstructions(StringBuilder sb, String instr, List<String> instructions) {
		for (String arg1 : instructions) {
			for (String arg2 : instructions) {
				sb.append(instr + "(" + arg1 + "," + arg2 + ")::");

				if (arg1.equals("MATRIX") || arg2.equals("MATRIX"))
					sb.append("MATRIX\n");
				else if (arg1.equals("FLOAT") || arg2.equals("FLOAT"))
					sb.append("FLOAT\n");
				else
					sb.append("INT\n");
			}
		}
	}

	public static void buildTernaryPermutations(List<String> args, TriConsumer<String, String, String> func) {
		buildBinaryPermutations(args, (t1, t2) -> args.forEach(t3 -> func.accept(t1, t2, t3)));
	}

	public static void buildBinaryPermutations(List<String> args, BiConsumer<String, String> func) {
		buildBinaryPermutations(args, args, func);
	}

	public static void buildBinaryPermutations(List<String> args1, List<String> args2, BiConsumer<String, String> func) {
		for (String arg1 : args1)
			for (String arg2 : args2)
				func.accept(arg1, arg2);
	}

	public static String defaultTypeHierarchy(String t1, String t2) {
		boolean is1ArgList = t1.endsWith("...");
		boolean is2ArgList = t2.endsWith("...");

		if (is1ArgList)
			t1 = t1.substring(0, t1.length() - 3);

		if (is2ArgList)
			t2 = t2.substring(0, t2.length() - 3);

		if (t1.equals("BOOL") && t2.equals("BOOL"))
			return "BOOL";
		if (t1.equals("INT") && (t2.equals("INT") || t2.equals("BOOL")))
			return "INT";

		if (t2.equals("INT") && (t1.equals("INT") || t1.equals("BOOL")))
			return "INT";

		if (!t1.equals("MATRIX") && !t2.equals("MATRIX"))
			return "FLOAT";
		return "MATRIX";
	}

	public static String convertibleType(String t1, String t2) {
		if (t1.equals("MATRIX") && t2.equals("MATRIX"))
			return "MATRIX";

		if (t1.equals("MATRIX") || t2.equals("MATRIX"))
			return null; // Then it is not convertible

		if (!List.of("FLOAT", "INT", "BOOL").contains(t1) || !List.of("FLOAT", "INT", "BOOL").contains(t2))
			return null;

		if (t1.equals("FLOAT") || t2.equals("FLOAT"))
			return "FLOAT"; // This is the most "general" type

		if (t1.equals("INT") || t2.equals("INT"))
			return "INT";

		return "BOOL";
	}

	public static String convertImplicitly(String type, boolean allowTypeConversions) {
		if (!allowTypeConversions)
			return type;
		return convertImplicitly(type);
	}

	public static String convertImplicitly(String type) {
		if (type == null)
			return null;

		if (type.equals("INT") || type.equals("BOOL"))
			return "FLOAT";
		return type;
	}

	public static void putAsBinaryPrintable(String instr, List<String> types, HashMap<String, BiFunction<RewriterStatement, RuleContext, String>> printFunctions, BiFunction<RewriterStatement, RuleContext, String> function) {
		for (String type1 : types)
			for (String type2 : types)
				printFunctions.put(instr + "(" + type1 + "," + type2 + ")", function);
	}

	public static void putAsDefaultBinaryPrintable(List<String> instrs, List<String> types, HashMap<String, BiFunction<RewriterStatement, RuleContext, String>> funcs) {
		for (String instr : instrs)
			putAsBinaryPrintable(instr, types, funcs, binaryStringRepr(" " + instr + " "));
	}

	// Updates the references (including metadata UUIDs) for a copied _idxExpr(args(_idx(...),...),...)
	public static void copyIndexList(RewriterStatement idxExprRoot) {
		if (!idxExprRoot.isInstruction() || !idxExprRoot.trueInstruction().equals("_idxExpr"))
			throw new IllegalArgumentException();

		Map<UUID, RewriterStatement> replacements = new HashMap<>();
		UUID newOwnerId = UUID.randomUUID();
		idxExprRoot.unsafePutMeta("ownerId", newOwnerId);

		RewriterStatement newArgList = idxExprRoot.getChild(0).copyNode();
		idxExprRoot.getOperands().set(0, newArgList);

		List<RewriterStatement> operands = newArgList.getOperands();

		for (int i = 0; i < operands.size(); i++) {
			RewriterStatement idx = operands.get(i);
			RewriterStatement cpy = idx.copyNode();
			UUID newId = UUID.randomUUID();
			cpy.unsafePutMeta("idxId", newId);
			cpy.unsafePutMeta("ownerId", newOwnerId);
			replacements.put((UUID)idx.getMeta("idxId"), cpy);
			operands.set(i, cpy);
		}

		RewriterStatement out = RewriterUtils.replaceReferenceAware(idxExprRoot.getChild(1), stmt -> {
			UUID idxId = (UUID) stmt.getMeta("idxId");
			if (idxId != null) {
				RewriterStatement newStmt = replacements.get(idxId);
				if (newStmt != null)
					return newStmt;
			}

			return null;
		});

		if (out != null)
			idxExprRoot.getOperands().set(1, out);
	}

	public static RewriterStatement replaceReferenceAware(RewriterStatement root, Function<RewriterStatement, RewriterStatement> comparer) {
		return replaceReferenceAware(root, false, comparer, new HashMap<>());
	}

	// Replaces elements in a DAG. If a parent item has multiple references, the entire path is duplicated
	public static RewriterStatement replaceReferenceAware(RewriterStatement root, boolean duplicateReferences, Function<RewriterStatement, RewriterStatement> comparer, HashMap<RewriterStatement, RewriterStatement> visited) {
		if (visited.containsKey(root))
			return visited.get(root);

		RewriterStatement newOne = comparer.apply(root);

		if (newOne == root)
			newOne = null;

		root = newOne != null ? newOne : root;

		if (newOne == null)
			duplicateReferences |= root.refCtr > 1;

		if (root.getOperands() != null) {
			for (int i = 0; i < root.getOperands().size(); i++) {
				RewriterStatement newSub = replaceReferenceAware(root.getOperands().get(i), duplicateReferences, comparer, visited);

				if (newSub != null) {
					if (duplicateReferences && newOne == null) {
						root = root.copyNode();
						newOne = root;
					}

					root.getOperands().set(i, newSub);
				}
			}
		}

		return newOne;
	}

	// Deduplicates the DAG (removes duplicate references with new nodes except for leaf data-types)
	public static void unfoldExpressions(RewriterStatement root, RuleContext ctx) {
		for (int i = 0; i < root.getOperands().size(); i++) {
			RewriterStatement child = root.getChild(i);
			if (child.isInstruction() && child.refCtr > 1) {
				if (!child.trueInstruction().equals("_idx")
						&& !child.trueInstruction().equals("_m")
						&& !child.trueInstruction().equals("idxExpr")
						&& !child.trueInstruction().equals("rand")
						&& !child.trueInstruction().equals("_EClass")) {
					RewriterStatement cpy = child.copyNode();
					root.getOperands().set(i, cpy);
					child.refCtr--;
					cpy.getOperands().forEach(op -> op.refCtr++);
				}
			}

			unfoldExpressions(child, ctx);
		}
	}

	public static <T> boolean cartesianProduct(List<List<T>> list, T[] stack, Function<T[], Boolean> emitter) {
		if (list.size() == 0)
			return false;

		if (list.size() == 1) {
			list.get(0).forEach(t -> {
				stack[0] = t;
				emitter.apply(stack);
			});
			return true;
		}

		return _cartesianProduct(0, list, stack, emitter, new MutableBoolean(true));
	}

	private static <T> boolean _cartesianProduct(int index, List<List<T>> sets, T[] currentStack, Function<T[], Boolean> emitter, MutableBoolean doContinue) {
		if (index >= sets.size()) {
			if (!emitter.apply(currentStack))
				doContinue.setValue(false);
			return true;
		}

		int size = sets.get(index).size();
		boolean matchFound = false;

		for (int i = 0; i < size; i++) {
			currentStack[index] = sets.get(index).get(i);
			matchFound |= _cartesianProduct(index+1, sets, currentStack, emitter, doContinue);

			if (!doContinue.booleanValue())
				return matchFound;
		}

		return matchFound;
	}

	public static boolean isImplicitlyConvertible(String typeFrom, String typeTo) {
		if (typeFrom.equals(typeTo))
			return true;

		if (typeFrom.equals("INT") && typeTo.equals("FLOAT"))
			return true;

		return false;
	}

	public static boolean compareLiterals(RewriterDataType lit1, RewriterDataType lit2, boolean allowImplicitTypeConversions) {
		if (allowImplicitTypeConversions)
			return lit1.getLiteral().equals(literalAs(lit1.getType(), lit2));
		return lit1.getLiteral().equals(lit2.getLiteral());
	}

	public static Object literalAs(String type, RewriterDataType literal) {
		switch (type) {
			case "FLOAT":
				return literal.floatLiteral();
			case "INT":
				return literal.intLiteral(false);
			case "BOOL":
				return literal.boolLiteral();
			default:
				return null;
		}
	}

	public static RuleContext buildDefaultContext() {
		RuleContext ctx = RewriterContextSettings.getDefaultContext();
		ctx.metaPropagator = new MetaPropagator(ctx);
		return ctx;
	}

	private static RuleContext lastCtx;
	private static Function<RewriterStatement, RewriterStatement> lastUnfuse;
	public static RewriterStatement unfuseOperators(RewriterStatement stmt, final RuleContext ctx) {
		return unfuseOperators(ctx).apply(stmt);
	}
	public static Function<RewriterStatement, RewriterStatement> unfuseOperators(final RuleContext ctx) {
		if (lastCtx == ctx)
			return lastUnfuse;

		ArrayList<RewriterRule> unfuseRules = new ArrayList<>();
		RewriterRuleCollection.substituteFusedOps(unfuseRules, ctx);
		RewriterHeuristic heur = new RewriterHeuristic(new RewriterRuleSet(ctx, unfuseRules));
		lastCtx = ctx;
		lastUnfuse = heur::apply;
		return lastUnfuse;
	}

	public static Function<RewriterStatement, RewriterStatement> buildCanonicalFormConverter(final RuleContext ctx, boolean debug) {
		return buildCanonicalFormConverter(ctx, true, debug);
	}

	public static Function<RewriterStatement, RewriterStatement> buildCanonicalFormConverter(final RuleContext ctx, boolean allowInversionCanonicalization, boolean debug) {
		ArrayList<RewriterRule> algebraicCanonicalizationRules = new ArrayList<>();
		RewriterRuleCollection.substituteEquivalentStatements(algebraicCanonicalizationRules, ctx);
		RewriterRuleCollection.eliminateMultipleCasts(algebraicCanonicalizationRules, ctx);
		RewriterRuleCollection.canonicalizeBooleanStatements(algebraicCanonicalizationRules, ctx);
		RewriterRuleCollection.canonicalizeAlgebraicStatements(algebraicCanonicalizationRules, allowInversionCanonicalization, ctx);
		RewriterRuleCollection.eliminateMultipleCasts(algebraicCanonicalizationRules, ctx);
		RewriterRuleCollection.buildElementWiseAlgebraicCanonicalization(algebraicCanonicalizationRules, ctx);
		RewriterHeuristic algebraicCanonicalization = new RewriterHeuristic(new RewriterRuleSet(ctx, algebraicCanonicalizationRules));

		ArrayList<RewriterRule> expRules = new ArrayList<>();
		RewriterRuleCollection.expandStreamingExpressions(expRules, ctx);
		RewriterHeuristic streamExpansion = new RewriterHeuristic(new RewriterRuleSet(ctx, expRules));

		ArrayList<RewriterRule> expArbitraryMatricesRules = new ArrayList<>();
		RewriterRuleCollection.expandArbitraryMatrices(expArbitraryMatricesRules, ctx);
		RewriterHeuristic expandArbitraryMatrices = new RewriterHeuristic(new RewriterRuleSet(ctx, expArbitraryMatricesRules));

		ArrayList<RewriterRule> pd = new ArrayList<>();
		RewriterRuleCollection.pushdownStreamSelections(pd, ctx);
		RewriterRuleCollection.buildElementWiseAlgebraicCanonicalization(pd, ctx);
		RewriterRuleCollection.eliminateMultipleCasts(pd, ctx);
		RewriterRuleCollection.canonicalizeBooleanStatements(pd, ctx);
		RewriterRuleCollection.canonicalizeAlgebraicStatements(pd, allowInversionCanonicalization, ctx);
		RewriterHeuristic streamSelectPushdown = new RewriterHeuristic(new RewriterRuleSet(ctx, pd));

		ArrayList<RewriterRule> flatten = new ArrayList<>();
		RewriterRuleCollection.flattenOperations(flatten, ctx);
		RewriterHeuristic flattenOperations = new RewriterHeuristic(new RewriterRuleSet(ctx, flatten));

		RewriterHeuristics canonicalFormCreator = new RewriterHeuristics();
		canonicalFormCreator.add("ALGEBRAIC CANONICALIZATION", algebraicCanonicalization);
		canonicalFormCreator.add("EXPAND STREAMING EXPRESSIONS", streamExpansion);
		canonicalFormCreator.add("EXPAND ARBITRARY MATRICES", expandArbitraryMatrices);
		canonicalFormCreator.add("PUSHDOWN STREAM SELECTIONS", streamSelectPushdown);
		canonicalFormCreator.add("FOLD CONSTANTS", new RewriterHeuristic(t -> foldConstants(t, ctx)));
		//canonicalFormCreator.add("CANON ALGB", new RewriterHeuristic(new RewriterRuleSet(ctx, RewriterRuleCollection.buildElementWiseAlgebraicCanonicalization(new ArrayList<>(), ctx))));
		canonicalFormCreator.add("REPLACE NEGATIONS", new RewriterHeuristic(new RewriterRuleSet(ctx, RewriterRuleCollection.replaceNegation(new ArrayList<>(), ctx))));
		canonicalFormCreator.add("PUSHDOWN STREAM SELECTIONS", streamSelectPushdown);
		canonicalFormCreator.add("FLATTEN OPERATIONS", flattenOperations);

		ArrayList<RewriterRule> canonicalExpand = new ArrayList<>();
		RewriterRuleCollection.canonicalExpandAfterFlattening(canonicalExpand, ctx);
		RewriterHeuristic canonicalExpandOps = new RewriterHeuristic(new RewriterRuleSet(ctx, canonicalExpand));

		ArrayList<RewriterRule> flattenAlgebraicRewriteList = new ArrayList<>();
		RewriterRuleCollection.flattenedAlgebraRewrites(flattenAlgebraicRewriteList, ctx);
		RewriterHeuristic flattenedAlgebraicRewrites = new RewriterHeuristic(new RewriterRuleSet(ctx, flattenAlgebraicRewriteList));

		RewriterHeuristics afterFlattening = new RewriterHeuristics();
		afterFlattening.add("CANONICAL EXPAND", canonicalExpandOps);
		afterFlattening.add("FLATTENED ALGEBRA REWRITES", flattenedAlgebraicRewrites);

		return stmt -> {
			stmt = stmt.nestedCopy(true);
			stmt = canonicalFormCreator.apply(stmt, (t, r) -> {
				if (!debug)
					return true;

				if (r != null)
					System.out.println("Applying rule: " + r.getName());
				System.out.println(t.toParsableString(ctx));
				return true;
			}, debug);

			for (int i = 0; i < 2; i++) {
				RewriterUtils.mergeArgLists(stmt, ctx);
				stmt = RewriterUtils.pullOutConstants(stmt, ctx);
			}
			RewriterUtils.mergeArgLists(stmt, ctx);
			unfoldExpressions(stmt, ctx);
			stmt = RewriterUtils.pullOutConstants(stmt, ctx);
			cleanupUnecessaryIndexExpressions(stmt, ctx);
			stmt.prepareForHashing();
			stmt.recomputeHashCodes(ctx);

			stmt = afterFlattening.apply(stmt, (t, r) -> {
				if (!debug)
					return true;

				if (r != null)
					System.out.println("Applying rule: " + r.getName());
				System.out.println(t.toParsableString(ctx));
				return true;
			}, debug);

			stmt = foldConstants(stmt, ctx);

			for (int i = 0; i < 2; i++) {
				RewriterUtils.mergeArgLists(stmt, ctx);
				stmt = RewriterUtils.pullOutConstants(stmt, ctx);
			}
			RewriterUtils.mergeArgLists(stmt, ctx);

			stmt = stmt.getAssertions(ctx).cleanupEClasses(stmt);
			unfoldExpressions(stmt, ctx);
			stmt.prepareForHashing();

			if (debug)
				System.out.println("PRE1:   " + stmt.toParsableString(ctx, false));

			stmt.compress(); // To remove unnecessary metadata such as assertions that are not encoded in the graph
			TopologicalSort.sort(stmt, ctx);

			if (debug)
				System.out.println("FINAL1: " + stmt.toParsableString(ctx, false));

			return stmt;
		};
	}

	public static RewriterStatement pullOutConstants(RewriterStatement oldRoot, final RuleContext ctx) {
		RewriterStatement newRoot = pullOutConstantsRecursively(oldRoot, ctx, new HashMap<>());

		// Check if we have to move the assertions to new root
		if (newRoot != oldRoot)
			oldRoot.moveRootTo(newRoot);

		return newRoot;
	}

	private static RewriterStatement pullOutConstantsRecursively(RewriterStatement cur, final RuleContext ctx, Map<RewriterStatement, RewriterStatement> alreadyModified) {
		if (!cur.isInstruction())
			return cur;

		RewriterStatement modified = alreadyModified.get(cur);

		if (modified != null)
			return modified;

		alreadyModified.put(cur, cur);

		for (int i = 0; i < cur.getOperands().size(); i++)
			cur.getOperands().set(i, pullOutConstantsRecursively(cur.getChild(i), ctx, alreadyModified));

		cur.updateMetaObjects(el -> pullOutConstantsRecursively(el, ctx, alreadyModified));

		switch (cur.trueInstruction()) {
			case "sum":
				return tryPullOutSum(cur, ctx);
		}

		return cur;
	}

	private static RewriterStatement tryPullOutSum(RewriterStatement sum, final RuleContext ctx) {
		// TODO: What happens on multi-index? Then, some unnecessary indices will currently not be pulled out
		RewriterStatement idxExpr = sum.getChild(0);
		UUID ownerId = (UUID) idxExpr.getMeta("ownerId");
		RewriterStatement sumBody = idxExpr.getChild(1);

		Map<RewriterStatement, Boolean> checked = new HashMap<>();


		if (!checkSubgraphDependency(sumBody, ownerId, checked)) {
			// Then we have to remove the sum entirely
			List<RewriterStatement> indices = idxExpr.getChild(0).getOperands();
			List<RewriterStatement> components = new ArrayList<>();

			for (RewriterStatement idx : indices) {
				if (idx.isLiteral())
					continue;
				RewriterStatement idxFrom = idx.getChild(0);
				RewriterStatement idxTo = idx.getChild(1);
				RewriterStatement negation = new RewriterInstruction().as(UUID.randomUUID().toString()).withInstruction("-").withOps(/*RewriterStatement.ensureFloat(ctx, idxFrom)*/idxFrom).consolidate(ctx);
				RewriterStatement add = RewriterStatement.multiArgInstr(ctx, "+", /*RewriterStatement.ensureFloat(ctx, idxTo)*/idxTo, RewriterStatement.literal(ctx, 1.0D), negation);
				components.add(add);
			}

			RewriterStatement out = RewriterStatement.multiArgInstr(ctx, "*", sumBody);
			out.getChild(0).getOperands().addAll(components);
			return foldConstants(out, ctx);
		}

		if (isDirectlyDependent(sumBody, ownerId))
			return sum;

		if (sumBody.trueInstruction().equals("*")) {
			// We have to assume here, that this instruction is not referenced anywhere else in the graph
			List<RewriterStatement> argList = sumBody.getChild(0).getOperands();
			List<RewriterStatement> toRemove = new ArrayList<>(argList.size());

			for (RewriterStatement stmt : argList) {
				if (!checkSubgraphDependency(stmt, ownerId, checked))
					toRemove.add(stmt);
			}

			if (!toRemove.isEmpty()) {
				argList.removeAll(toRemove);

				if (argList.size() == 1) {
					idxExpr.getOperands().set(1, argList.get(0));
				}

				toRemove.add(sum);

				return RewriterStatement.multiArgInstr(ctx, "*", toRemove.toArray(RewriterStatement[]::new));
			}
		} else if (sumBody.trueInstruction().equals("+")) {
			// TODO: What about sum(+(A, *(a, B)))? We could pull out a

			// We have to assume here, that this instruction is not referenced anywhere else in the graph
			List<RewriterStatement> argList = sumBody.getChild(0).getOperands();
			List<RewriterStatement> toRemove = new ArrayList<>(argList.size());

			for (RewriterStatement stmt : argList) {
				if (!checkSubgraphDependency(stmt, ownerId, checked))
					toRemove.add(stmt);
			}

			if (!toRemove.isEmpty()) {
				argList.removeAll(toRemove);

				if (argList.size() == 1) {
					idxExpr.getOperands().set(1, argList.get(0));
				}

				RewriterStatement outerSum = RewriterStatement.multiArgInstr(ctx, "+", toRemove.toArray(RewriterStatement[]::new));
				List<RewriterStatement> mul = new ArrayList<>();

				for (RewriterStatement idx : idxExpr.getChild(0).getOperands()) {
					RewriterStatement neg = new RewriterInstruction().as(UUID.randomUUID().toString()).withInstruction("-").withOps(/*RewriterStatement.ensureFloat(ctx, idx.getChild(0))*/idx.getChild(0)).consolidate(ctx);
					RewriterStatement msum = RewriterStatement.multiArgInstr(ctx, "+", /*RewriterStatement.ensureFloat(ctx, idx.getChild(1))*/idx.getChild(1), neg, RewriterStatement.literal(ctx, 1.0));
					mul.add(msum);
				}

				mul.add(outerSum);
				RewriterStatement mulStmt = RewriterStatement.multiArgInstr(ctx, "*", mul.toArray(RewriterStatement[]::new));

				return RewriterStatement.multiArgInstr(ctx, "+", mulStmt, sum);
			}
		}

		return sum;
	}

	// Returns true if the subgraph is dependent on the corresponding owner
	private static boolean checkSubgraphDependency(RewriterStatement expr, UUID id, Map<RewriterStatement, Boolean> checked) {
		Boolean b = checked.get(expr);

		if (b != null)
			return b;

		if (expr.isInstruction() && expr.trueInstruction().equals("_idx")) {
			UUID mid = (UUID) expr.getMeta("ownerId");
			boolean isDependent = id.equals(mid);

			if (isDependent) {
				checked.put(expr, true);
				return true;
			}
		}

		for (RewriterStatement stmt : expr.getOperands()) {
			if (checkSubgraphDependency(stmt, id, checked)) {
				checked.put(expr, true);
				return true;
			}
		}

		checked.put(expr, false);
		return false;
	}

	private static boolean isDirectlyDependent(RewriterStatement child, UUID ownerId) {
		if (child.isInstruction() && child.trueInstruction().equals("_idx")) {
			UUID mid = (UUID) child.getMeta("_ownerId");
			return ownerId.equals(mid);
		}

		return false;
	}

	public static RewriterStatement foldConstants(RewriterStatement stmt, final RuleContext ctx) {
		Map<RewriterStatement, RewriterStatement> replaced = new HashMap<>();
		RewriterStatement ret = foldConstantsRecursively(stmt, ctx, replaced);
		ret.prepareForHashing();
		ret.recomputeHashCodes(ctx);
		return ret;
	}

	private static RewriterStatement foldConstantsRecursively(RewriterStatement cur, final RuleContext ctx, Map<RewriterStatement, RewriterStatement> alreadyFolded) {
		if (!cur.isInstruction())
			return cur;

		RewriterStatement folded = alreadyFolded.get(cur);

		if (folded != null)
			return folded;

		alreadyFolded.put(cur, cur);

		for (int i = 0; i < cur.getOperands().size(); i++)
			cur.getOperands().set(i, foldConstantsRecursively(cur.getChild(i), ctx, alreadyFolded));

		cur.updateMetaObjects(el -> foldConstantsRecursively(el, ctx, alreadyFolded));

		RewriterStatement ret = cur;

		switch (cur.trueInstruction()) {
			case "+":
			case "*":
			case "min":
			case "max":
				ret = foldNaryReducible(cur, ctx);
				break;
			case "_EClass":
				ret = foldEClass(cur, ctx);
				break;
			default:
				if (cur.getOperands().size() == 1)
					ret = foldUnary(cur, ctx);
				break;
		}

		ret.refreshReturnType(ctx);
		alreadyFolded.put(cur, ret);
		return ret;
	}

	private static RewriterStatement foldEClass(RewriterStatement stmt, final RuleContext ctx) {
		RewriterStatement lit = stmt.getLiteralStatement();
		if (lit != null)
			return lit;
		return stmt;
	}

	private static RewriterStatement foldNaryReducible(RewriterStatement stmt, final RuleContext ctx) {
		List<RewriterStatement> argList;
		if (stmt.getChild(0).isArgumentList())
			argList = stmt.getChild(0).getOperands();
		else
			argList = stmt.getOperands();

		if (argList.isEmpty())
			throw new IllegalArgumentException(stmt.toString(ctx));

		if (stmt.isInstruction() && (stmt.trueInstruction().equals("min") || stmt.trueInstruction().equals("max")) && argList.size() == 1 && !List.of("FLOAT", "INT", "BOOL").contains(argList.get(0).getResultingDataType(ctx)))
			return stmt;

		if (argList.size() < 2)
			return argList.get(0);

		int[] literals = IntStream.range(0, argList.size()).filter(i -> argList.get(i).isLiteral()).toArray();

		if (literals.length == 1) {
			Object literal = argList.get(literals[0]).getLiteral();
			if (literal instanceof Number) {
				RewriterStatement overwrite = ConstantFoldingUtils.overwritesLiteral((Number) literal, stmt.trueInstruction(), ctx);
				if (overwrite != null)
					return overwrite;
			}

			// Check if is neutral element
			if (ConstantFoldingUtils.isNeutralElement(argList.get(literals[0]).getLiteral(), stmt.trueInstruction())) {
				RewriterStatement neutral = argList.get(literals[0]);
				argList.remove(literals[0]);

				if (argList.size() == 1)
					return argList.get(0);
				else if (argList.isEmpty())
					return neutral;
			}
		}

		if (literals.length < 2)
			return stmt;

		String rType = stmt.getResultingDataType(ctx);

		BiFunction<Number, RewriterStatement, Number> foldingFunction = ConstantFoldingUtils.foldingBiFunction(stmt.trueInstruction(), rType);

		RewriterDataType foldedLiteral = new RewriterDataType();
		Number val = null;

		for (int literal : literals)
			val = foldingFunction.apply(val, argList.get(literal));


		RewriterStatement overwrite = ConstantFoldingUtils.overwritesLiteral(val, stmt.trueInstruction(), ctx);
		if (overwrite != null)
			return overwrite;

		foldedLiteral.as(val.toString()).ofType(rType).asLiteral(val).consolidate(ctx);

		argList.removeIf(RewriterStatement::isLiteral);

		if (argList.isEmpty() || !ConstantFoldingUtils.isNeutralElement(foldedLiteral.getLiteral(), stmt.trueInstruction()))
			argList.add(foldedLiteral);

		ConstantFoldingUtils.cancelOutNary(stmt.trueInstruction(), argList);

		if (argList.size() == 1)
			return argList.get(0);

		return stmt;
	}

	private static RewriterStatement foldUnary(RewriterStatement stmt, final RuleContext ctx) {
		RewriterStatement child = stmt.getChild(0);

		if (!child.isLiteral())
			return stmt;

		boolean isFloat = stmt.getResultingDataType(ctx).equals("FLOAT");

		switch (stmt.trueInstruction()) {
			case "inv":
				if (isFloat)
					return RewriterStatement.literal(ctx, 1.0 / child.floatLiteral());
				else
					return RewriterStatement.literal(ctx, 1L / child.intLiteral());
			case "-":
				if (isFloat)
					return RewriterStatement.literal(ctx, -child.floatLiteral());
				else
					return RewriterStatement.literal(ctx, -child.intLiteral());
		}

		// Not implemented yet
		return stmt;
	}

	public static RewriterStatement cleanupUnecessaryIndexExpressions(RewriterStatement stmt, final RuleContext ctx) {
		RewriterStatement mNew = cleanupIndexExprRecursively(stmt, ctx);

		if (mNew != null)
			stmt.moveRootTo(mNew);

		recursivePostCleanup(mNew != null ? mNew : stmt);

		return mNew;
	}

	private static RewriterStatement cleanupIndexExprRecursively(RewriterStatement cur, final RuleContext ctx) {
		for (int i = 0; i < cur.getOperands().size(); i++) {
			RewriterStatement mNew = cleanupIndexExprRecursively(cur.getChild(i), ctx);

			if (mNew != null)
				cur.getOperands().set(i, mNew);
		}

		return cleanupIndexExpr(cur);
	}

	private static void recursivePostCleanup(RewriterStatement cur) {
		for (RewriterStatement child : cur.getOperands())
			recursivePostCleanup(child);

		postCleanupIndexExpr(cur);
	}

	private static RewriterStatement cleanupIndexExpr(RewriterStatement cur) {
		if (!cur.isInstruction() || !cur.trueInstruction().equals("sum"))
			return null;

		RewriterStatement base = cur;
		cur = cur.getChild(0);

		if (!cur.isInstruction() || !cur.trueInstruction().equals("_idxExpr"))
			return null;

		if (!cur.getChild(1).isInstruction() || !cur.getChild(1).trueInstruction().equals("ifelse") || !cur.getChild(1,2).isLiteral() || cur.getChild(1,2).floatLiteral() != 0.0D)
			return null;

		RewriterStatement query = cur.getChild(1, 0);

		if (query.isInstruction() && query.trueInstruction().equals("==")) {
			RewriterStatement idx1 = query.getChild(0);
			RewriterStatement idx2 = query.getChild(1);

			if (idx1.isInstruction() && idx2.isInstruction() && idx1.trueInstruction().equals("_idx") && idx2.trueInstruction().equals("_idx")) {
				List<RewriterStatement> indices = cur.getChild(0).getOperands();
				RewriterStatement indexFromUpperLevel = null;
				if (idx1 == idx2) {
					cur.getOperands().set(1, cur.getChild(1, 1));
				} else if (indices.contains(idx1)) {
					boolean removed = indices.remove(idx2);
					indexFromUpperLevel = removed ? null : idx2;

					if (removed) {
						cur.getOperands().set(1, cur.getChild(1, 1));
						cur.getChild(1).forEachPreOrder(cur2 -> {
							for (int i = 0; i < cur2.getOperands().size(); i++) {
								if (cur2.getChild(i).equals(idx2))
									cur2.getOperands().set(i, idx1);
							}

							return true;
						}, true);
					}
				} else if (indices.contains(idx2)) {
					indexFromUpperLevel = idx1;
				}

				if (indexFromUpperLevel != null) {
					cur.getOperands().set(1, cur.getChild(1, 1));
					final RewriterStatement fIdxUpperLevel = indexFromUpperLevel;
					final RewriterStatement fIdxLowerLevel = idx1 == indexFromUpperLevel ? idx2 : idx1;
					cur.getChild(1).forEachPreOrder(cur2 -> {
						for (int i = 0; i < cur2.getOperands().size(); i++) {
							if (cur2.getChild(i).equals(fIdxLowerLevel))
								cur2.getOperands().set(i, fIdxUpperLevel);
						}

						return true;
					}, true);
					indices.remove(idx2);
				}

				if (indices.isEmpty()) {
					return cur.getChild(1);
				}
			}
		}

		return base;
	}

	// To unify ifelse (e.g. ifelse(a == b, a+b, a-b) => ifelse(a == b, a+a, a-b)
	private static void postCleanupIndexExpr(RewriterStatement cur) {
		if (!cur.isInstruction() || !cur.trueInstruction().equals("ifelse") || !cur.getChild(2).isLiteral() || cur.getChild(2).floatLiteral() != 0.0D)
			return;

		RewriterStatement query = cur.getChild(0);

		if (query.isInstruction() && query.trueInstruction().equals("==")) {
			RewriterStatement idx1 = query.getChild(0);
			RewriterStatement idx2 = query.getChild(1);

			if (idx1.isInstruction() && idx2.isInstruction() && idx1.trueInstruction().equals("_idx") && idx2.trueInstruction().equals("_idx")) {
				// Then we just choose the first index
				cur.getChild(1).forEachPreOrder(cur2 -> {
					for (int i = 0; i < cur2.getOperands().size(); i++) {
						if (cur2.getChild(i).equals(idx2))
							cur2.getOperands().set(i, idx1);
					}

					return true;
				}, true);
				cur.getChild(2).forEachPreOrder(cur2 -> {
					for (int i = 0; i < cur2.getOperands().size(); i++) {
						if (cur2.getChild(i).equals(idx2))
							cur2.getOperands().set(i, idx1);
					}

					return true;
				}, true);
			}
		}
	}

	public static void renameIllegalVarnames(final RuleContext ctx, RewriterStatement... stmts) {
		MutableInt matrixVarCtr = new MutableInt(0);
		MutableInt scalarVarCtr = new MutableInt(0);

		Set<String> varnames = new HashSet<>();
		for (RewriterStatement stmt : stmts) {
			stmt.forEachPreOrder(cur -> {
				if (cur.isInstruction())
					return true;

				varnames.add(cur.getId());
				return true;
			}, false);
		}

		for (RewriterStatement stmt : stmts) {
			stmt.forEachPreOrder(cur -> {
				if (cur.isInstruction() || cur.isLiteral())
					return true;

				boolean isMatrix = cur.getResultingDataType(ctx).equals("MATRIX");

				if (cur.getId().equals("?")) {
					cur.rename(getVarname(varnames, isMatrix ? matrixVarCtr : scalarVarCtr, isMatrix));
					return true;
				}

				if (cur.getId().contains("_")) {
					cur.rename(getVarname(varnames, isMatrix? matrixVarCtr : scalarVarCtr, isMatrix));
				}

				try {
					UUID.fromString(cur.getId());
					// If it could parse, then we should rename
					cur.rename(getVarname(varnames, isMatrix ? matrixVarCtr : scalarVarCtr, isMatrix));
					return true;
				} catch (Exception e) {
					// Then this is not a UUID
				}

				return true;
			}, false);
		}
	}

	private static String getVarname(Set<String> existingNames, MutableInt mInt, boolean matrix) {
		char origChar;

		if (matrix)
			origChar = 'A';
		else
			origChar = 'a';

		char ch = (char)(origChar + mInt.getAndIncrement());

		while (existingNames.contains(String.valueOf(ch)))
			ch = (char)(origChar + mInt.getAndIncrement());

		return String.valueOf(ch);
	}
}
