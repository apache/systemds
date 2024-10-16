package org.apache.sysds.hops.rewriter;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.mutable.MutableBoolean;
import org.apache.commons.lang3.mutable.MutableInt;
import org.apache.commons.lang3.mutable.MutableObject;
import org.apache.logging.log4j.util.TriConsumer;
import org.apache.spark.internal.config.R;
import scala.Tuple2;
import scala.collection.parallel.ParIterableLike;
import scala.reflect.internal.Trees;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
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

	public static RewriterStatement parse(String expr, final RuleContext ctx, String... varDefinitions) {
		HashMap<String, RewriterStatement> dataTypes = new HashMap<>();

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
			pattern = Pattern.compile("(-)?[0-9]+(\\.[0-9]*)?");
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

	public static void topologicalSort(RewriterStatement stmt, final RuleContext ctx, BiFunction<RewriterStatement, RewriterStatement, Boolean> arrangable) {
		Map<RewriterRule.IdentityRewriterStatement, Map<RewriterRule.IdentityRewriterStatement, Integer>> votes = new HashMap<>();
		List<Set<RewriterRule.IdentityRewriterStatement>> uncertainStatements = new ArrayList<>();
		// First pass (try to figure out everything)
		traversePostOrderWithDepthInfo(stmt, null, (el, depth, parent) -> {
			if (el.getOperands() == null)
				return;

			if (arrangable.apply(el, parent)) {
				RewriterRule.IdentityRewriterStatement id = new RewriterRule.IdentityRewriterStatement(el);

				if (!votes.containsKey(id)) {
					//System.out.println("Sorting: " + el);
					List<Set<RewriterRule.IdentityRewriterStatement>> uStatements = createHierarchy(ctx, el.getOperands());
					if (uStatements.size() > 0) {
						uStatements.forEach(e -> System.out.println("Uncertain: " + e.stream().map(t -> t.stmt)));
						uncertainStatements.addAll(createHierarchy(ctx, el.getOperands()));
					}
				}
			}
		}, 0);

		// Trigger a recomputation of the hash codes
		stmt.prepareForHashing();
		stmt.recomputeHashCodes(ctx);
	}

	private static void traversePostOrderWithDepthInfo(RewriterStatement stmt, RewriterStatement parent, TriConsumer<RewriterStatement, Integer, RewriterStatement> consumer, int currentDepth) {
		if (stmt.getOperands() != null)
			stmt.getOperands().forEach(el -> traversePostOrderWithDepthInfo(el, stmt, consumer, currentDepth + 1));

		consumer.accept(stmt, currentDepth, parent);
	}

	// Returns the range of uncertain elements [start, end)
	public static List<Set<RewriterRule.IdentityRewriterStatement>> createHierarchy(final RuleContext ctx, List<RewriterStatement> level) {
		level.sort(Comparator.comparing(el -> toOrderString(ctx, el)));

		List<Set<RewriterRule.IdentityRewriterStatement>> ranges = new ArrayList<>();
		int currentRangeStart = 0;
		for (int i = 1; i < level.size(); i++) {
			//System.out.println(toOrderString(ctx, level.get(i)));
			if (toOrderString(ctx, level.get(i)).equals(toOrderString(ctx, level.get(i-1)))) {
				if (i - currentRangeStart > 1) {
					Set<RewriterRule.IdentityRewriterStatement> mSet = level.subList(currentRangeStart, i).stream().map(RewriterRule.IdentityRewriterStatement::new).collect(Collectors.toSet());

					if (mSet.size() > 1)
						ranges.add(mSet);
				}
				currentRangeStart = i;
			}
		}
		return ranges;
	}

	public static String toOrderString(final RuleContext ctx, RewriterStatement stmt) {
		if (stmt.isInstruction()) {
			return stmt.getResultingDataType(ctx) + ":" + stmt.trueTypedInstruction(ctx) + "[" + stmt.refCtr + "]";
		} else {
			return stmt.getResultingDataType(ctx) + ":" + (stmt.isLiteral() ? "L:" + stmt.getLiteral() : "V") + ";";
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

	public static List<RewriterStatement> generateSubtrees(RewriterStatement stmt, Map<RewriterRule.IdentityRewriterStatement, List<RewriterStatement>> visited, final RuleContext ctx) {
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
				System.out.println(t);
				return true;
			}, debug);

			RewriterUtils.mergeArgLists(stmt, ctx);
			if (debug)
				System.out.println("PRE1: " + stmt.toString(ctx));

			RewriterUtils.topologicalSort(stmt, ctx, (el, parent) -> el.isArgumentList() && parent != null && Set.of("+", "-", "*", "_idxExpr").contains(parent.trueInstruction()));

			if (debug)
				System.out.println("FINAL1: " + stmt.toString(ctx));

			return stmt;
		};
	}

	public static Function<RewriterStatement, RewriterStatement> buildFusedOperatorCreator(final RuleContext ctx, boolean debug) {
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
		canonicalFormCreator.add("PUSHDOWN STREAM SELECTIONS", streamSelectPushdown);
		canonicalFormCreator.add("STREAMIFY", streamify);
		canonicalFormCreator.add("FLATTEN OPERATIONS", flattenOperations);

		return stmt -> {
			stmt = canonicalFormCreator.apply(stmt, (t, r) -> {
				if (!debug)
					return true;

				if (r != null)
					System.out.println("Applying rule: " + r.getName());
				System.out.println(t);
				return true;
			}, debug);

			RewriterUtils.mergeArgLists(stmt, ctx);
			if (debug)
				System.out.println("PRE1: " + stmt.toString(ctx));

			RewriterUtils.topologicalSort(stmt, ctx, (el, parent) -> el.isArgumentList() && parent != null && Set.of("+", "-", "*", "_idxExpr").contains(parent.trueInstruction()));

			if (debug)
				System.out.println("FINAL1: " + stmt.toString(ctx));

			return stmt;
		};
	}
}
