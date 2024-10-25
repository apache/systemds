package org.apache.sysds.hops.rewriter;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.mutable.MutableBoolean;
import org.apache.commons.lang3.mutable.MutableInt;
import org.apache.commons.lang3.mutable.MutableObject;
import org.apache.logging.log4j.util.TriConsumer;
import org.apache.spark.internal.config.R;
import org.apache.zookeeper.data.Id;
import scala.Tuple2;
import scala.Tuple3;
import scala.collection.parallel.ParIterableLike;
import scala.reflect.internal.Trees;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.Set;
import java.util.Stack;
import java.util.UUID;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class RewriterUtils {
	public static String typedToUntypedInstruction(String instr) {
		return instr.substring(0, instr.indexOf('('));
	}

	public static Function<RewriterStatement, Boolean> propertyExtractor(final List<String> desiredProperties, final RuleContext ctx) {
		return el -> {
			if (el instanceof RewriterInstruction) {
				Set<String> properties = ((RewriterInstruction) el).getProperties(ctx);
				String trueInstr = ((RewriterInstruction)el).trueTypedInstruction(ctx);
				//if (properties != null) {
					for (String desiredProperty : desiredProperties) {
						if (trueInstr.equals(desiredProperty) || (properties != null && properties.contains(desiredProperty))) {
							System.out.println("Found property: " + desiredProperty + " (for " + el + ")");
							String oldInstr = ((RewriterInstruction) el).changeConsolidatedInstruction(desiredProperty, ctx);
							if (el.getMeta("trueInstr") == null) {
								el.unsafePutMeta("trueInstr", oldInstr);
								el.unsafePutMeta("trueName", oldInstr);
							}
							break;
							//System.out.println("Property found: " + desiredProperty);
						}
					}
				//}
			}
			return true;
		};
	}

	public static BiFunction<RewriterStatement, RuleContext, String> binaryStringRepr(String op) {
		return (stmt, ctx) -> {
			List<RewriterStatement> operands = ((RewriterInstruction)stmt).getOperands();
			String op1Str = operands.get(0).toString(ctx);
			if (operands.get(0) instanceof RewriterInstruction && ((RewriterInstruction)operands.get(0)).getOperands().size() > 1)
				op1Str = "(" + op1Str + ")";
			String op2Str = operands.get(1).toString(ctx);
			if (operands.get(1) instanceof RewriterInstruction && ((RewriterInstruction)operands.get(1)).getOperands().size() > 1)
				op2Str = "(" + op2Str + ")";
			return op1Str + op + op2Str;
		};
	}

	public static BiFunction<RewriterStatement, RuleContext, String> wrappedBinaryStringRepr(String op) {
		return (stmt, ctx) -> {
			List<RewriterStatement> operands = ((RewriterInstruction)stmt).getOperands();
			return "(" + operands.get(0).toString(ctx) + ")" + op + "(" + operands.get(1).toString(ctx) + ")";
		};
	}

	public static RewriterStatement buildFusedPlan(RewriterStatement origStatement, final RuleContext ctx) {
		RewriterStatement cpy = origStatement.nestedCopy();
		MutableObject<RewriterStatement> mCpy = new MutableObject<>(cpy);

		Map<Tuple2<RewriterRule.IdentityRewriterStatement, RewriterRule.IdentityRewriterStatement>, List<RewriterStatement>> mmap = eraseAccessTypes(mCpy, ctx);
		cpy = mCpy.getValue();

		// Identify common element wise accesses (e.g. A[i, j] + B[i, j] for all i, j)
		//Map<Tuple2<RewriterRule.IdentityRewriterStatement, RewriterRule.IdentityRewriterStatement>, List<RewriterStatement>> mmap = new HashMap<>();

		/*for (Tuple3<RewriterRule.IdentityRewriterStatement, RewriterRule.IdentityRewriterStatement, RewriterRule.IdentityRewriterStatement> mTuple : mSet.keySet()) {
			List<RewriterStatement> accesses = mmap.compute(new Tuple2<>(mTuple._2(), mTuple._3()), (k, v) -> v == null ? new ArrayList<>() : v);
			accesses.add(mTuple._1().stmt);
		}

		List<RewriterStatement> fuseList = new ArrayList<>();

		MutableObject<RewriterStatement> mParent = new MutableObject<>(cpy);

		cpy.forEachPreOrder((current, parent, pIdx) -> {
			if (!current.isInstruction())
				return true;

			if (current.trueInstruction().equals("_m")) {
				if (parent != null)
					parent.getOperands().set(pIdx, current.getOperands().get(2));
				else
					mParent.setValue(current.getOperands().get(2));
			}

			return true;
		});

		mmap.forEach((k, v) -> {
			HashMap<String, RewriterStatement> args = new HashMap<>();
			args.put("idx1", k._1.stmt);
			args.put("idx2", k._2.stmt);
			args.put("valueFn", );
			RewriterStatement vFn = cpy.
			RewriterStatement newStmt = parse("_accessNary(idx1, idx2, valueFn)", ctx, );
			fuseList.add();
		});*/

		//

		if (mmap.size() == 1) {
			Map.Entry<Tuple2<RewriterRule.IdentityRewriterStatement, RewriterRule.IdentityRewriterStatement>, List<RewriterStatement>> entry = mmap.entrySet().iterator().next();
			HashMap<String, RewriterStatement> args = new HashMap<>();

			RewriterStatement mS = null;

			if (cpy.isInstruction()) {
				if (cpy.trueInstruction().equals("_m")) {
					args.put("stmt", cpy.getOperands().get(2));
					args.put("first", entry.getValue().get(0));

					mS = RewriterUtils.parse("_map(argList(first), stmt)", ctx, args);
					mS.getOperands().get(0).getOperands().addAll(entry.getValue().subList(1, entry.getValue().size()));
				} else if (cpy.trueInstruction().equals("sum")) {
					args.put("stmt", cpy.getOperands().get(0));
					args.put("first", entry.getValue().get(0));

					System.out.println(args.get("stmt"));
					mS = RewriterUtils.parse("_reduce(argList(first), +(_cur(), stmt))", ctx, args);
					mS.getOperands().get(0).getOperands().addAll(entry.getValue().subList(1, entry.getValue().size()));
				}
			}

			return mS;
		}

		return null;
	}

	public static Map<Tuple2<RewriterRule.IdentityRewriterStatement, RewriterRule.IdentityRewriterStatement>, List<RewriterStatement>> eraseAccessTypes(MutableObject<RewriterStatement> stmt, final RuleContext ctx) {
		//Map<Tuple3<RewriterRule.IdentityRewriterStatement, RewriterRule.IdentityRewriterStatement, RewriterRule.IdentityRewriterStatement>, RewriterStatement> out = new HashMap<>();

		Map<Tuple2<RewriterRule.IdentityRewriterStatement, RewriterRule.IdentityRewriterStatement>, List<RewriterStatement>> rewrites = new HashMap<>();

		HashMap<Integer, RewriterStatement> hooks = new HashMap<>();

		List<RewriterRule> rules = new ArrayList<>();

		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A")
				.parseGlobalVars("INT:i,j")
				.parseGlobalVars("FLOAT:v")
				.withParsedStatement("[](A, i, j)")
				.toParsedStatement("$1:_v(A)", hooks)
				.iff(match -> {
					List<RewriterStatement> ops = match.getMatchRoot().getOperands();
					return (ops.get(0).isInstruction() && ops.get(0).trueInstruction().equals("_idx"))
							|| (ops.get(1).isInstruction() && ops.get(1).trueInstruction().equals("_idx"));
				}, true)
				.apply(hooks.get(1).getId(), (t, m) -> {
					t.unsafePutMeta("data", m.getMatchRoot().getOperands().get(0));
					t.unsafePutMeta("idx1", m.getMatchRoot().getOperands().get(1));
					t.unsafePutMeta("idx2", m.getMatchRoot().getOperands().get(2));

					RewriterRule.IdentityRewriterStatement idx1 = new RewriterRule.IdentityRewriterStatement(m.getMatchRoot().getOperands().get(1));
					RewriterRule.IdentityRewriterStatement idx2 = new RewriterRule.IdentityRewriterStatement(m.getMatchRoot().getOperands().get(2));
					Tuple2<RewriterRule.IdentityRewriterStatement, RewriterRule.IdentityRewriterStatement> mT = new Tuple2<>(idx1, idx2);

					List<RewriterStatement> r = rewrites.get(mT);

					if (r == null) {
						r = new ArrayList<>();
						rewrites.put(mT, r);
					}

					r.add(t);
				}, true)
				.build());

		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A")
				.parseGlobalVars("INT...:i")
				.parseGlobalVars("FLOAT:v")
				.withParsedStatement("_idxExpr(i, v)")
				.toParsedStatement("$1:v", hooks)
				.iff(match -> {
					List<RewriterStatement> ops = match.getMatchRoot().getOperands().get(0).getOperands();
					return ops.stream().anyMatch(op -> op.isInstruction() && op.trueInstruction().equals("_idx"));
				}, true)
				.build());

		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A")
				.parseGlobalVars("INT...:i,j")
				.parseGlobalVars("FLOAT*:v")
				.withParsedStatement("_idxExpr(i, v)")
				.toParsedStatement("v", hooks)
				.iff(match -> {
					List<RewriterStatement> ops = match.getMatchRoot().getOperands().get(0).getOperands();
					return ops.stream().anyMatch(op -> op.isInstruction() && op.trueInstruction().equals("_idx"));
				}, true)
				.build());

		RewriterRuleSet rs = new RewriterRuleSet(ctx, rules);
		RewriterHeuristic heur = new RewriterHeuristic(rs, true);

		stmt.setValue(heur.apply(stmt.getValue()));

		return rewrites;

		/*stmt.getValue().forEachPostOrder((current, parent, pIdx) -> {
			if (!current.isInstruction())
				return;

			if (current.trueInstruction().equals("[]")) {
				boolean hasIndex = false;
				if (current.getOperands().get(1).isInstruction() && current.getOperands().get(1).trueInstruction().equals("_idx"))
					hasIndex = true;

				if (current.getOperands().get(2).isInstruction() && current.getOperands().get(2).trueInstruction().equals("_idx"))
					hasIndex = true;

				if (hasIndex) {
					current.getOperands().get(0).unsafePutMeta("idx1", current.getOperands().get(1));
					current.getOperands().get(0).unsafePutMeta("idx2", current.getOperands().get(2));
					out.put(new Tuple3<>(new RewriterRule.IdentityRewriterStatement(current.getOperands().get(0)),
							new RewriterRule.IdentityRewriterStatement(current.getOperands().get(1)),
							new RewriterRule.IdentityRewriterStatement(current.getOperands().get(2))),
							current.getOperands().get(0));

					if (parent != null)
						parent.getOperands().set(pIdx, current.getOperands().get(0));
					else
						stmt.setValue(current.getOperands().get(0));
				}
			} else if (current.trueInstruction().equals("idxExpr")) {
				if (parent != null)
					parent.getOperands().set(pIdx, current.getOperands().get(1));
				else
					stmt.setValue(current.getOperands().get(1));
			}
		});

		return out;*/
	}

	public static void mergeArgLists(RewriterStatement stmt, final RuleContext ctx) {

		stmt.forEachPreOrder(el -> {
			tryFlattenNestedArgList(ctx, el, el, -1);
			tryFlattenNestedOperatorPatterns(ctx, el);
			return true;
		});

		stmt.prepareForHashing();
		stmt.recomputeHashCodes(ctx);
	}

	private static boolean tryFlattenNestedArgList(final RuleContext ctx, RewriterStatement stmt, RewriterStatement root, int insertAt) {
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

	private static void tryFlattenNestedOperatorPatterns(final RuleContext ctx, RewriterStatement stmt) {
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

	public static RewriterStatement parse(String expr, final RuleContext ctx, String... varDefinitions) {
		return parse(expr, ctx, new HashMap<>(), varDefinitions);
	}

	public static RewriterStatement parse(String expr, final RuleContext ctx, HashMap<String, RewriterStatement> dataTypes, String... varDefinitions) {
		for (String def : varDefinitions)
			parseDataTypes(def, dataTypes, ctx);

		RewriterStatement parsed = parseExpression(expr, new HashMap<>(), dataTypes, ctx);
		return ctx.metaPropagator != null ? ctx.metaPropagator.apply(parsed) : parsed;
	}

	/**
	 * Parses an expression
	 * @param expr the expression string. Note that all whitespaces have to already be removed
	 * @param refmap test
	 * @param dataTypes data type
	 * @param ctx context
	 * @return test
	 */
	public static RewriterStatement parseExpression(String expr, HashMap<Integer, RewriterStatement> refmap, HashMap<String, RewriterStatement> dataTypes, final RuleContext ctx) {
		RuleContext.currentContext = ctx;
		expr = expr.replaceAll("\\s+", "");
		MutableObject<String> mexpr = new MutableObject<>(expr);
		RewriterStatement stmt = doParseExpression(mexpr, refmap, dataTypes, ctx);
		stmt.prepareForHashing();
		stmt.consolidate(ctx);
		return stmt;
	}

	private static RewriterStatement doParseExpression(MutableObject<String> mexpr, HashMap<Integer, RewriterStatement> refmap, HashMap<String, RewriterStatement> dataTypes, final RuleContext ctx) {
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
					//throw new IllegalArgumentException("Expected the token ':'");
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

	public static boolean parseDataTypes(String expr, HashMap<String, RewriterStatement> dataTypes, final RuleContext ctx) {
		RuleContext.currentContext = ctx;
		Pattern pattern = Pattern.compile("([A-Za-z0-9]|_|\\.|\\*)+");
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

	private static RewriterStatement parseRawExpression(MutableObject<String> mexpr, HashMap<Integer, RewriterStatement> refmap, HashMap<String, RewriterStatement> dataTypes, final RuleContext ctx) {
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

	public static HashMap<String, List<RewriterStatement>> createIndex(RewriterStatement stmt, final RuleContext ctx) {
		HashMap<String, List<RewriterStatement>> index = new HashMap<>();
		stmt.forEachPreOrderWithDuplicates(mstmt -> {
			if (mstmt instanceof RewriterInstruction) {
				RewriterInstruction instr = (RewriterInstruction)mstmt;
				index.compute(instr.trueTypedInstruction(ctx), (k, v) -> {
					if (v == null) {
						return List.of(mstmt);
					} else {
						if (v.stream().noneMatch(el -> el == instr))
							v.add(instr);
						return v;
					}
				});
			}
			return true;
		});
		return index;
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

	public static void buildBinaryBoolInstructions(StringBuilder sb, String instr, List<String> instructions) {
		for (String arg1 : instructions) {
			for (String arg2 : instructions) {
				sb.append(instr + "(" + arg1 + "," + arg2 + ")::BOOL\n");
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

	public static void putAsBinaryPrintable(String instr, List<String> types, HashMap<String, BiFunction<RewriterStatement, RuleContext, String>> printFunctions, BiFunction<RewriterStatement, RuleContext, String> function) {
		for (String type1 : types)
			for (String type2 : types)
				printFunctions.put(instr + "(" + type1 + "," + type2 + ")", function);
	}

	public static void putAsDefaultBinaryPrintable(List<String> instrs, List<String> types, HashMap<String, BiFunction<RewriterStatement, RuleContext, String>> funcs) {
		for (String instr : instrs)
			putAsBinaryPrintable(instr, types, funcs, binaryStringRepr(" " + instr + " "));
	}

	public static HashMap<String, Set<String>> mapToImplementedFunctions(final RuleContext ctx) {
		HashMap<String, Set<String>> out = new HashMap<>();
		Set<String> superTypes = new HashSet<>();

		for (Map.Entry<String, String> entry : ctx.instrTypes.entrySet()) {
			Set<String> props = ctx.instrProperties.get(entry.getKey());
			if (props != null && !props.isEmpty()) {
				for (String prop : props) {
					Set<String> impl = out.computeIfAbsent(prop, k -> new HashSet<>());
					impl.add(typedToUntypedInstruction(entry.getKey()));
					superTypes.add(typedToUntypedInstruction(prop));
				}
			}
		}

		for (Map.Entry<String, Set<String>> entry : out.entrySet())
			entry.getValue().removeAll(superTypes);

		return out;
	}

	public static RewriterStatement replaceReferenceAware(RewriterStatement root, Function<RewriterStatement, RewriterStatement> comparer) {
		return replaceReferenceAware(root, false, comparer, new HashMap<>());
	}

	public static RewriterStatement replaceReferenceAware(RewriterStatement root, boolean duplicateReferences, Function<RewriterStatement, RewriterStatement> comparer, HashMap<RewriterRule.IdentityRewriterStatement, RewriterStatement> visited) {
		RewriterRule.IdentityRewriterStatement is = new RewriterRule.IdentityRewriterStatement(root);
		if (visited.containsKey(is)) {
			return visited.get(is);
		}

		RewriterStatement oldRef = root;
		RewriterStatement newOne = comparer.apply(root);
		root = newOne != null ? newOne : root;

		if (newOne == null)
			duplicateReferences |= root.refCtr > 1;

		if (root.getOperands() != null) {
			for (int i = 0; i < root.getOperands().size(); i++) {
				RewriterStatement newSub = replaceReferenceAware(root.getOperands().get(i), duplicateReferences, comparer, visited);

				if (newSub != null) {
					if (duplicateReferences && newOne == null) {
						root = root.copyNode();
					}

					root.getOperands().set(i, newSub);
				}
			}
		}

		return newOne;
	}

	// Function to check if two lists match
	public static <T> boolean findMatchingOrderings(List<T> col1, List<T> col2, T[] stack, BiFunction<T, T, Boolean> matcher, Function<T[], Boolean> permutationEmitter, boolean symmetric) {
		if (col1.size() != col2.size())
			return false;  // Sizes must match

		if (stack.length < col2.size())
			throw new IllegalArgumentException("Mismatching stack sizes!");

		if (col1.size() == 1) {
			if (matcher.apply(col1.get(0), col2.get(0))) {
				stack[0] = col2.get(0);
				permutationEmitter.apply(stack);
				return true;
			}

			return false;
		}

		// We need to get true on the diagonal for it to be a valid permutation
		List<List<Integer>> possiblePermutations = new ArrayList<>(Collections.nCopies(col1.size(), null));

		boolean anyMatch;

		for (int i = 0; i < col1.size(); i++) {
			anyMatch = false;

			for (int j = 0; j < col2.size(); j++) {
				if (j > i && symmetric)
					break;

				if (matcher.apply(col1.get(i), col2.get(j))) {
					if (possiblePermutations.get(i) == null)
						possiblePermutations.set(i, new ArrayList<>());

					possiblePermutations.get(i).add(j);

					if (symmetric) {
						if (possiblePermutations.get(j) == null)
							possiblePermutations.set(j, new ArrayList<>());
						possiblePermutations.get(j).add(i);
					}

					anyMatch = true;
				}
			}

			if (!anyMatch) // Then there cannot be a matching permutation
				return false;
		}

		// Start recursive matching
		return cartesianProduct(possiblePermutations, new Integer[possiblePermutations.size()], arrangement -> {
			for (int i = 0; i < col2.size(); i++)
				stack[i] = col2.get(arrangement[i]);
			return permutationEmitter.apply(stack);
		});
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

	// TODO: This is broken --> remove
	public static void topologicalSort(RewriterStatement stmt, final RuleContext ctx, BiFunction<RewriterStatement, RewriterStatement, Boolean> arrangable) {
		MutableInt nameCtr = new MutableInt();
		stmt.forEachPostOrderWithDuplicates((el, parent, pIdx) -> {
			if (el.getOperands().isEmpty()) {
				el.unsafePutMeta("_tempName", nameCtr.intValue());
				nameCtr.increment();
			} else if (parent != null && arrangable.apply(el, parent)) {
				el.unsafePutMeta("_tempName", nameCtr.intValue());
				nameCtr.increment();
			}
		});

		//Map<RewriterRule.IdentityRewriterStatement, Map<RewriterRule.IdentityRewriterStatement, Integer>> votes = new HashMap<>();
		//Map<RewriterRule.IdentityRewriterStatement, Set<RewriterRule.IdentityRewriterStatement>> gotRatedBy = new HashMap<>();
		//List<Set<RewriterRule.IdentityRewriterStatement>> uncertainStatements = new ArrayList<>();

		// First pass (try to figure out everything)
		traversePostOrderWithDepthInfo(stmt, null, (el, depth, parent) -> {
			if (el.getOperands() == null)
				return;

			RewriterRule.IdentityRewriterStatement voter = new RewriterRule.IdentityRewriterStatement(el);
			createHierarchy(ctx, el, el.getOperands());

			/*if (votes.containsKey(voter))
				return;

			if (arrangable.apply(el, parent)) {
				List<Set<RewriterRule.IdentityRewriterStatement>> uStatements = createHierarchy(ctx, el, el.getOperands());
				if (uStatements.size() > 0) {
					uStatements.forEach(e -> System.out.println("Uncertain: " + e.stream().map(t -> t.stmt).collect(Collectors.toList())));
					uncertainStatements.addAll(uStatements);
				}
			} else {
				Map<RewriterRule.IdentityRewriterStatement, Integer> ratings = new HashMap<>();
				votes.put(voter, ratings);

				for (int i = 0; i < el.getOperands().size(); i++) {
					RewriterRule.IdentityRewriterStatement toRate = new RewriterRule.IdentityRewriterStatement(el.getOperands().get(i));

					if (votes.containsKey(toRate))
						continue;

					ratings.put(toRate, i);

					Set<RewriterRule.IdentityRewriterStatement> ratedBy = gotRatedBy.get(toRate);

					if (ratedBy == null) {
						ratedBy = new HashSet<>();
						gotRatedBy.put(toRate, ratedBy);
					}

					ratedBy.add(voter);
				}
			}*/
		}, 0);

		// TODO: Erase temp names

		/*while (!uncertainStatements.isEmpty()) {
			// Now, try to resolve the conflicts deterministically using element-wise comparison
			Map<Tuple2<RewriterRule.IdentityRewriterStatement, RewriterRule.IdentityRewriterStatement>, Integer> orderSet = new HashMap<>();

			for (Set<RewriterRule.IdentityRewriterStatement> requiredComparisons : uncertainStatements) {
				forEachDistinctBinaryCombination(new ArrayList<>(requiredComparisons), (s1, s2) -> {
					Optional<Boolean> myOpt = compareStatements(s1, s2, votes, gotRatedBy);
					if (myOpt.isPresent()) {
						orderSet.put(new Tuple2<>(s1, s2), myOpt);
						orderSet.put(new Tuple2<>(s2, s1), Optional.of(!myOpt.get()));
					} else {
						orderSet.put(new Tuple2<>(s1, s2), Optional.empty());
					}
				});
			}
		}*/

		// Trigger a recomputation of the hash codes
		stmt.prepareForHashing();
		stmt.recomputeHashCodes(ctx);
	}

	/*public static void setupOrderFacts(RewriterStatement root, final RuleContext ctx) {
		root.forEachPostOrder((el, parent, pIdx) -> {
			HashMap<RewriterRule.IdentityRewriterStatement, Integer> facts = new HashMap<>();
			List<Set<RewriterRule.IdentityRewriterStatement>> uncertainties = new ArrayList<>();
			List<RewriterStatement> subComparisons = new ArrayList<>();
			List<Object> semiLists = new ArrayList<>();
			el.unsafePutMeta("_facts", facts);
			//el.unsafePutMeta("_uncertainties", uncertainties);
			el.unsafePutMeta("_subComparisons", subComparisons);
			el.unsafePutMeta("_semiList", semiLists);

			el.getOperands().sort(Comparator.comparing(stmt -> toOrderString(ctx, stmt)));

			subComparisons.add(el);
			int range = 0;
			for (int i = 1; i < el.getOperands().size(); i++) {
				if (toOrderString(ctx, el.getOperands().get(i-1)).compareTo(toOrderString(ctx, el.getOperands().get(i))) != 0) {
					if (i - range > 1)
						// Then we have uncertain elements
						continue;

					range = i;
					// Then we can add i - 1 to the list of certainties
					subComparisons.addAll((List<RewriterStatement>)el.getOperands().get(i-1).getMeta("_subComparisons"));
				}
			}

			if (range == el.getOperands().size() - 1)
				subComparisons.addAll((List<RewriterStatement>)el.getOperands().get(range).getMeta("_subComparisons"));

			// Sort to the best of our knowledge
			el.getOperands().sort((el1, el2) -> compare(el1, el2, ctx));

			// Now we figure out the uncertain elements
			range = 0;
			List<RewriterRule.IdentityRewriterStatement> currentUncertainties = new ArrayList<>();
			for (int i = 1; i < el.getOperands().size(); i++) {
				if (compare(el.getOperands().get(i-1), el.getOperands().get(i), ctx) != 0) {
					if (i - range > 1) {
						// Then we still have uncertain elements
						if (i - range == 2)
							currentUncertainties.add(new RewriterRule.IdentityRewriterStatement(el.getOperands().get(i - 1)));

						currentUncertainties.add(new RewriterRule.IdentityRewriterStatement(el.getOperands().get(i)));
						facts.put(new RewriterRule.IdentityRewriterStatement(el.getOperands().get(i)), range);
						continue;
					}

					range = i;
					//facts.put(new RewriterRule.IdentityRewriterStatement(el.getOperands().get(i-1)), range);

					if (currentUncertainties.isEmpty()) {
						// Then we have a fact and we can add it to the facts list
						facts.put(new RewriterRule.IdentityRewriterStatement(el.getOperands().get(i-1)), i);
						semiLists.add(el.getOperands().get(i-1));
					} else {
						uncertainties.add(currentUncertainties);
						semiLists.add(new HashSet<>(currentUncertainties));
						currentUncertainties = new HashSet<>();
					}
				}
			}

			if (currentUncertainties.isEmpty()) {
				// Then we have a fact and we can add it to the facts list
				facts.put(new RewriterRule.IdentityRewriterStatement(el.getOperands().get(el.getOperands().size()-1)), range);
				semiLists.add(el.getOperands().get(el.getOperands().size()-1));
			} else {
				uncertainties.add(currentUncertainties);
				semiLists.add(new HashSet<>(currentUncertainties));
			}

			//facts.put(new RewriterRule.IdentityRewriterStatement(el.getOperands().get(el.getOperands().size()-1)), range);
		});
	}*/

	/*public static boolean factOrderingPass(RewriterStatement root) {
		MutableBoolean changeHappened = new MutableBoolean(true);

		while (changeHappened.booleanValue()) {
			root.forEachPostOrder((current, parent, pIdx) -> {
				// Here we try to bubble up the global order facts
			});
		}
	}*/

	// Returns null if there are not enough facts to determine the absolute order of elements
	/*private static Integer compare(RewriterStatement stmt1, RewriterStatement stmt2, RewriterFactTable factTable, final RuleContext ctx) {
		if (stmt1 == stmt2)
			return 0;

		int cmp = toOrderString(ctx, stmt1).compareTo(toOrderString(ctx, stmt2));

		if (cmp != 0)
			return cmp;

		if (stmt1.getOperands().size() == 0) {
			// Only then the fact can be in the fact table
			Integer cmpFromTable = factTable.compare(stmt1, stmt2);

			if (cmpFromTable != null)
				return cmpFromTable;
		}

		List<Object> compareHierarchy1 = (List<Object>)stmt1.getMeta("_semiList");
		List<Object> compareHierarchy2 = (List<Object>)stmt2.getMeta("_semiList");

		for (int i = 0; i < compareHierarchy1.size(); i++) {
			if (compareHierarchy1.get(i) instanceof RewriterStatement && compareHierarchy2.get(i) instanceof RewriterStatement) {
				// Then the order of elements is determined
				Integer comparison = compare((RewriterStatement) compareHierarchy1.get(i), (RewriterStatement) compareHierarchy2.get(i), factTable, ctx);

				if (comparison == null)
					return null; // Then the element cannot be compared yet

				if (comparison != 0)
					return comparison;
			}
		}

		return 0;
	}*/

	public static <T> void forEachDistinctBinaryCombination(List<T> l, BiConsumer<T, T> consumer) {
		for (int i = 0; i < l.size(); i++)
			for (int j = l.size() - 1; j > i; j--)
				consumer.accept(l.get(i), l.get(j));
	}

	private static void traversePostOrderWithDepthInfo(RewriterStatement stmt, RewriterStatement parent, TriConsumer<RewriterStatement, Integer, RewriterStatement> consumer, int currentDepth) {
		if (stmt.getOperands() != null)
			stmt.getOperands().forEach(el -> traversePostOrderWithDepthInfo(el, stmt, consumer, currentDepth + 1));

		consumer.accept(stmt, currentDepth, parent);
	}

	// Returns the range of uncertain elements [start, end)
	public static void createHierarchy(final RuleContext ctx, RewriterStatement voter, List<RewriterStatement> level) {
		if (level.isEmpty())
			return;

		//level.sort(Comparator.comparing(el -> toOrderString(ctx, el)));
		level.sort((el1, el2) -> compare(el1, el2, ctx));

		/*List<Set<RewriterRule.IdentityRewriterStatement>> ranges = new ArrayList<>();
		int currentRangeStart = 0;

		RewriterRule.IdentityRewriterStatement voterIds = new RewriterRule.IdentityRewriterStatement(voter);
		Map<RewriterRule.IdentityRewriterStatement, Integer> votes = new HashMap<>();

		{
			RewriterRule.IdentityRewriterStatement firstIds = new RewriterRule.IdentityRewriterStatement(level.get(0));

			Set<RewriterRule.IdentityRewriterStatement> voters = gotRatedBy.get(firstIds);

			if (voters == null) {
				voters = new HashSet<>();
				gotRatedBy.put(firstIds, voters);
			}

			voters.add(voterIds);

			allVotes.put(firstIds, votes);
			votes.put(firstIds, 0);
		}

		for (int i = 1; i < level.size(); i++) {
			System.out.println(toOrderString(ctx, level.get(i-1)) + " <=> " + toOrderString(ctx, level.get(i)));
			if (compare(level.get(i-1), level.get(i), ctx) == 0) {
				if (i - currentRangeStart > 1) {
					Set<RewriterRule.IdentityRewriterStatement> mSet = level.subList(currentRangeStart, i).stream().map(RewriterRule.IdentityRewriterStatement::new).collect(Collectors.toSet());

					if (mSet.size() > 1)
						ranges.add(mSet);

					System.out.println("E-Set: " + mSet.stream().map(id -> id.stmt.toParsableString(ctx, false)).collect(Collectors.toList()));

					currentRangeStart = i;
				}
			}

			RewriterRule.IdentityRewriterStatement ids = new RewriterRule.IdentityRewriterStatement(level.get(i));
			votes.put(ids, currentRangeStart);

			Set<RewriterRule.IdentityRewriterStatement> voters = gotRatedBy.get(ids);

			if (voters == null) {
				voters = new HashSet<>();
				gotRatedBy.put(ids, voters);
			}

			voters.add(voterIds);
		}

		if (level.size() - currentRangeStart > 1) {
			Set<RewriterRule.IdentityRewriterStatement> mSet = level
					.subList(currentRangeStart, level.size())
					.stream().map(RewriterRule.IdentityRewriterStatement::new)
					.collect(Collectors.toSet());

			if (mSet.size() > 1)
				ranges.add(mSet);

			System.out.println("E-Set: " + mSet.stream().map(id -> id.stmt.toParsableString(ctx, false)).collect(Collectors.toList()));
		}

		return ranges;*/
	}

	// Try to find new facts that enable comparisons between elements
	/*private static boolean decisionStep(RewriterRule.IdentityRewriterStatement unknown1, RewriterRule.IdentityRewriterStatement unknown2, Map<RewriterRule.IdentityRewriterStatement, Map<RewriterRule.IdentityRewriterStatement, Integer>> votes, Map<RewriterRule.IdentityRewriterStatement, Set<RewriterRule.IdentityRewriterStatement>> gotRatedBy, Map<Tuple2<RewriterRule.IdentityRewriterStatement, RewriterRule.IdentityRewriterStatement>, Integer> globalOrders, BiFunction<RewriterStatement, RewriterStatement, Boolean> arrangable, final RuleContext ctx) {
		Set<RewriterRule.IdentityRewriterStatement> commonRatings = gotRatedBy.getOrDefault(unknown1, Collections.emptySet());
		Set<RewriterRule.IdentityRewriterStatement> u2Ratings = gotRatedBy.getOrDefault(unknown2, Collections.emptySet());

		// Only keep parents that actually rated both of the children
		commonRatings.retainAll(u2Ratings);

		// Now, find the nodes that have actually decided this problem (if there is none, this problem is not decidable and we need to introduce facts)
		List<Tuple2<RewriterRule.IdentityRewriterStatement, Integer>> decisions = commonRatings.stream().filter(decider -> votes.get(decider).get(unknown1) != votes.get(decider).get(unknown2)).map(decider -> new Tuple2<>(decider, Integer.compare(votes.get(decider).get(unknown1), votes.get(decider).get(unknown2)))).collect(Collectors.toList());

		// Now, we try to find differences in uncertainties/orderings in the parent path
		// Otherwise take the first one that reaches the root
		int vote = 0;
		RewriterRule.IdentityRewriterStatement maxElement = null;
		int maxElementIndex = 0;
		for (int i = 0; i < decisions.size(); i++) {
			Tuple2<RewriterRule.IdentityRewriterStatement, Integer> currentDecision = decisions.get(i);

			if (currentDecision == null)
				continue;

			if (maxElement == null) {
				maxElement = currentDecision._1;
				vote = currentDecision._2;
				maxElementIndex = i;
				continue;
			}

			if (compare(maxElement.stmt, currentDecision._1.stmt, null, null, globalOrders, arrangable, ctx) > 0) {
				maxElementIndex = i;
			}
		}
	}*/

	public static int compare(RewriterStatement stmt1, RewriterStatement stmt2, /*RewriterStatement p1, RewriterStatement p2, Map<Tuple2<RewriterRule.IdentityRewriterStatement, RewriterRule.IdentityRewriterStatement>, Integer> globalOrders, BiFunction<RewriterStatement, RewriterStatement, Boolean> arrangable,*/ final RuleContext ctx) {
		/*boolean arrangable1 = arrangable.apply(stmt1, p1);
		boolean arrangable2 = arrangable.apply(stmt2, p2);

		if (arrangable1) {
			if (!arrangable2)
				return 1;
		} else {
			if (arrangable2)
				return -1;
		}

		RewriterRule.IdentityRewriterStatement id1 = new RewriterRule.IdentityRewriterStatement(stmt1);
		RewriterRule.IdentityRewriterStatement id2 = new RewriterRule.IdentityRewriterStatement(stmt2);

		if (!globalOrders.isEmpty()) {
			Integer result = globalOrders.get(new Tuple2<>(id1, id2));

			if (result == null)
				result = globalOrders.get(new Tuple2<>(id2, id1));

			if (result != null)
				return result;
		}*/

		int comp = toOrderString(ctx, stmt1).compareTo(toOrderString(ctx, stmt2));

		if (comp != 0 || stmt1.getOperands().isEmpty())
			return comp;

		for (int i = 0; i < stmt1.getOperands().size() && comp == 0; i++)
			comp = compare(stmt1.getOperands().get(i), stmt2.getOperands().get(i), ctx);

		if (comp == 0) {
			Integer mName1 = (Integer)stmt1.getMeta("_tempName");

			if (mName1 == null)
				return 0;

			return mName1.toString().compareTo(stmt2.getMeta("_tempName").toString());
		}

		return comp;
	}

	public static String toOrderString(final RuleContext ctx, RewriterStatement stmt) {
		return toOrderString(ctx, stmt, false);
	}

	public static String toOrderString(final RuleContext ctx, RewriterStatement stmt, boolean extendIfPossible) {
		if (stmt.isInstruction()) {
			Integer mName = (Integer)stmt.getMeta("_tempName");
			return stmt.getResultingDataType(ctx) + ":" + stmt.trueTypedInstruction(ctx) + "[" + stmt.refCtr + "](" + stmt.getOperands().size() + ")" + (mName == null ? "" : mName) + ";";
		} else {
			return stmt.getResultingDataType(ctx) + ":" + (stmt.isLiteral() ? "L:" + stmt.getLiteral() : "V") + "[" + stmt.refCtr + "](0)" + stmt.getMeta("_tempName") + ";";
		}
	}

	/*public static void subtreeCombinations(RewriterStatement stmt) {
		MutableInt numInts = new MutableInt(0);
		stmt.forEachPreOrder(s -> {
			numInts.add(s.getOperands().size());
			return true;
		});

		int n = numInts.getValue();
		long numCombinations = 2^n;
		long currentBitmask = 0;

		for (int i = 0; i < )
	}*/

	public static List<RewriterStatement> mergeSubtreeCombinations(RewriterStatement stmt, List<Integer> indices, List<List<RewriterStatement>> mList, final RuleContext ctx) {
		if (indices.isEmpty())
			return List.of(stmt);

		List<RewriterStatement> mergedTreeCombinations = new ArrayList<>();
		cartesianProduct(mList, new RewriterStatement[mList.size()], stack -> {
			RewriterStatement cpy = stmt.copyNode();
			for (int i = 0; i < stack.length; i++)
				cpy.getOperands().set(indices.get(i), stack[i]);
			cpy.consolidate(ctx);
			cpy.prepareForHashing();
			cpy.recomputeHashCodes(ctx);
			mergedTreeCombinations.add(cpy);
			return true;
		});

		return mergedTreeCombinations;
	}

	public static List<RewriterStatement> generateSubtrees(RewriterStatement stmt, final RuleContext ctx) {
		List<RewriterStatement> l = generateSubtrees(stmt, new HashMap<>(), ctx);

		if (ctx.metaPropagator != null)
			l.forEach(subtree -> ctx.metaPropagator.apply(subtree));

		return l.stream().map(subtree -> {
			if (ctx.metaPropagator != null)
				subtree = ctx.metaPropagator.apply(subtree);

			subtree.prepareForHashing();
			subtree.recomputeHashCodes(ctx);
			return subtree;
		}).collect(Collectors.toList());
	}

	private static List<RewriterStatement> generateSubtrees(RewriterStatement stmt, Map<RewriterRule.IdentityRewriterStatement, List<RewriterStatement>> visited, final RuleContext ctx) {
		if (stmt == null)
			return Collections.emptyList();

		RewriterRule.IdentityRewriterStatement is = new RewriterRule.IdentityRewriterStatement(stmt);
		List<RewriterStatement> alreadyVisited = visited.get(is);

		if (alreadyVisited != null)
			return alreadyVisited;

		if (stmt.getOperands().size() == 0)
			return List.of(stmt);

		// Scan if operand is not a DataType
		List<Integer> indices = new ArrayList<>();
		for (int i = 0; i < stmt.getOperands().size(); i++) {
			if (stmt.getOperands().get(i).isInstruction() || stmt.isLiteral())
				indices.add(i);
		}

		int n = indices.size();
		int totalSubsets = 1 << n;

		List<RewriterStatement> mList = new ArrayList<>();

		visited.put(is, mList);

		//if (totalSubsets == 0)
			//return List.of();

		List<List<RewriterStatement>> mOptions = indices.stream().map(i -> generateSubtrees(stmt.getOperands().get(i), visited, ctx)).collect(Collectors.toList());
		List<RewriterStatement> out = new ArrayList<>();

		for (int subsetMask = 0; subsetMask < totalSubsets; subsetMask++) {
			List<List<RewriterStatement>> mOptionCpy = new ArrayList<>(mOptions);

			for (int i = 0; i < n; i++) {
				// Check if the i-th child is included in the current subset
				if ((subsetMask & (1 << i)) == 0) {
					RewriterDataType mT = new RewriterDataType().as(UUID.randomUUID().toString()).ofType(stmt.getOperands().get(indices.get(i)).getResultingDataType(ctx));
					mT.consolidate(ctx);
					mOptionCpy.set(i, List.of(mT));
				}
			}

			out.addAll(mergeSubtreeCombinations(stmt, indices, mOptionCpy, ctx));
		}

		return out;
	}

	public static RuleContext buildDefaultContext() {
		RuleContext ctx = RewriterContextSettings.getDefaultContext(new Random());
		ctx.metaPropagator = new MetaPropagator(ctx);
		return ctx;
	}

	public static Function<RewriterStatement, RewriterStatement> buildCanonicalFormConverter(final RuleContext ctx, boolean debug) {
		ArrayList<RewriterRule> algebraicCanonicalizationRules = new ArrayList<>();
		RewriterRuleCollection.canonicalizeBooleanStatements(algebraicCanonicalizationRules, ctx);
		RewriterRuleCollection.canonicalizeAlgebraicStatements(algebraicCanonicalizationRules, ctx);
		RewriterHeuristic algebraicCanonicalization = new RewriterHeuristic(new RewriterRuleSet(ctx, algebraicCanonicalizationRules));

		ArrayList<RewriterRule> expRules = new ArrayList<>();
		RewriterRuleCollection.expandStreamingExpressions(expRules, ctx);
		RewriterHeuristic streamExpansion = new RewriterHeuristic(new RewriterRuleSet(ctx, expRules));

		ArrayList<RewriterRule> pd = new ArrayList<>();
		RewriterRuleCollection.pushdownStreamSelections(pd, ctx);
		RewriterRuleCollection.buildElementWiseAlgebraicCanonicalization(pd, ctx);
		RewriterHeuristic streamSelectPushdown = new RewriterHeuristic(new RewriterRuleSet(ctx, pd));

		ArrayList<RewriterRule> flatten = new ArrayList<>();
		RewriterRuleCollection.flattenOperations(flatten, ctx);
		RewriterHeuristic flattenOperations = new RewriterHeuristic(new RewriterRuleSet(ctx, flatten));

		RewriterHeuristics canonicalFormCreator = new RewriterHeuristics();
		canonicalFormCreator.add("ALGEBRAIC CANONICALIZATION", algebraicCanonicalization);
		canonicalFormCreator.add("EXPAND STREAMING EXPRESSIONS", streamExpansion);
		canonicalFormCreator.add("PUSHDOWN STREAM SELECTIONS", streamSelectPushdown);
		canonicalFormCreator.add("FLATTEN OPERATIONS", flattenOperations);

		return stmt -> {
			stmt = canonicalFormCreator.apply(stmt, (t, r) -> {
				if (!debug)
					return true;

				if (r != null)
					System.out.println("Applying rule: " + r.getName());
				System.out.println(t.toString(ctx));
				return true;
			}, debug);

			RewriterUtils.mergeArgLists(stmt, ctx);
			if (debug)
				System.out.println("PRE1: " + stmt.toParsableString(ctx, false));

			//RewriterUtils.topologicalSort(stmt, ctx, (el, parent) -> el.isArgumentList() && parent != null && Set.of("+", "-", "*", "_idxExpr").contains(parent.trueInstruction()));
			//stmt = stmt.getAssertions(ctx).buildEquivalences(stmt);
			System.out.println(stmt.getAssertions(ctx));
			TopologicalSort.sort(stmt, ctx);

			if (debug)
				System.out.println("FINAL1: " + stmt.toParsableString(ctx, false));

			return stmt;
		};
	}

	/*public static Function<RewriterStatement, RewriterStatement> buildFusedOperatorCreator(final RuleContext ctx, boolean debug) {
		ArrayList<RewriterRule> algebraicCanonicalizationRules = new ArrayList<>();
		RewriterRuleCollection.canonicalizeBooleanStatements(algebraicCanonicalizationRules, ctx);
		RewriterRuleCollection.canonicalizeAlgebraicStatements(algebraicCanonicalizationRules, ctx);
		RewriterHeuristic algebraicCanonicalization = new RewriterHeuristic(new RewriterRuleSet(ctx, algebraicCanonicalizationRules));

		ArrayList<RewriterRule> expRules = new ArrayList<>();
		RewriterRuleCollection.expandStreamingExpressions(expRules, ctx);
		RewriterHeuristic streamExpansion = new RewriterHeuristic(new RewriterRuleSet(ctx, expRules));

		ArrayList<RewriterRule> pd = new ArrayList<>();
		RewriterRuleCollection.pushdownStreamSelections(pd, ctx);
		RewriterHeuristic streamSelectPushdown = new RewriterHeuristic(new RewriterRuleSet(ctx, pd));

		ArrayList<RewriterRule> streamifyRules = new ArrayList<>();
		RewriterRuleCollection.streamifyExpressions(streamifyRules, ctx);
		RewriterHeuristic streamify = new RewriterHeuristic(new RewriterRuleSet(ctx, streamifyRules));

		ArrayList<RewriterRule> flatten = new ArrayList<>();
		RewriterRuleCollection.flattenOperations(flatten, ctx);
		RewriterHeuristic flattenOperations = new RewriterHeuristic(new RewriterRuleSet(ctx, flatten));

		RewriterHeuristics canonicalFormCreator = new RewriterHeuristics();
		canonicalFormCreator.add("ALGEBRAIC CANONICALIZATION", algebraicCanonicalization);
		canonicalFormCreator.add("EXPAND STREAMING EXPRESSIONS", streamExpansion);

		RewriterHeuristics pushDownAndStreamify = new RewriterHeuristics();
		pushDownAndStreamify.add("PUSHDOWN STREAM SELECTIONS", streamSelectPushdown);
		pushDownAndStreamify.add("STREAMIFY", streamify);
		canonicalFormCreator.addRepeated("PUSHDOWN AND STREAMIFY", pushDownAndStreamify);

		canonicalFormCreator.add("FLATTEN OPERATIONS", flattenOperations);

		return stmt -> {
			stmt = canonicalFormCreator.apply(stmt, (t, r) -> {
				if (!debug)
					return true;

				if (r != null)
					System.out.println("Applying rule: " + r.getName());
				System.out.println(t.toParsableString(ctx));
				return true;
			}, debug);

			RewriterUtils.mergeArgLists(stmt, ctx);
			if (debug)
				System.out.println("PRE1: " + stmt.toParsableString(ctx, false));

			TopologicalSort.sort(stmt, ctx);
			//RewriterUtils.topologicalSort(stmt, ctx, (el, parent) -> el.isArgumentList() && parent != null && Set.of("+", "-", "*", "_idxExpr").contains(parent.trueInstruction()));
			//TopologicalSort.setupOrderFacts(stmt, (el, parent) -> el.isArgumentList() && parent != null && Set.of("+", "-", "*", "_idxExpr").contains(parent.trueInstruction()), ctx);

			if (debug)
				System.out.println("FINAL1: " + stmt.toParsableString(ctx, false));

			return stmt;
		};
	}*/

	public static void doCSE(RewriterStatement stmt, final RuleContext ctx) {

	}
}
