package org.apache.sysds.hops.rewriter.codegen;


import org.apache.sysds.common.Types;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.hops.rewriter.assertions.RewriterAssertions;
import org.apache.sysds.hops.rewriter.estimators.RewriterCostEstimator;
import org.apache.sysds.hops.rewriter.RewriterDataType;
import org.apache.sysds.hops.rewriter.RewriterRule;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.codehaus.janino.SimpleCompiler;
import scala.Tuple2;

import java.util.AbstractCollection;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

public class RewriterCodeGen {
	public static boolean DEBUG = true;

	public static Function<Hop, Hop> compileRewrites(String className, List<Tuple2<String, RewriterRule>> rewrites, final RuleContext ctx, boolean ignoreErrors, boolean printErrors) throws Exception {
		String code = generateClass(className, rewrites, false, false, ctx, ignoreErrors, printErrors);
		System.out.println("Compiling code:\n" + code);
		SimpleCompiler compiler = new SimpleCompiler();
		compiler.cook(code);
		Class<?> mClass = compiler.getClassLoader().loadClass(className);
		Object instance = mClass.getDeclaredConstructor().newInstance();
		return (Function<Hop, Hop>) instance;
	}

	public static Function<Hop, Hop> compile(String javaCode, String className) {
		try {
			SimpleCompiler compiler = new SimpleCompiler();
			compiler.cook(javaCode);
			Class<?> mClass = compiler.getClassLoader().loadClass(className);
			Object instance = mClass.getDeclaredConstructor().newInstance();
			return (Function<Hop, Hop>) instance;
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}

	public static String generateClass(String className, List<Tuple2<String, RewriterRule>> rewrites, boolean optimize, boolean includePackageInfo, final RuleContext ctx, boolean ignoreErrors, boolean printErrors) {
		return generateClass(className, rewrites, optimize, includePackageInfo, ctx, ignoreErrors, printErrors, false);
	}

	public static String generateClass(String className, List<Tuple2<String, RewriterRule>> rewrites, boolean optimize, boolean includePackageInfo, final RuleContext ctx, boolean ignoreErrors, boolean printErrors, boolean maintainRewriteStats) {
		StringBuilder msb = new StringBuilder();

		if (includePackageInfo)
			msb.append("package org.apache.sysds.hops.rewriter;\n\n");

		msb.append("import java.util.ArrayList;\n");
		msb.append("import java.util.function.Function;\n");
		msb.append("\n");
		msb.append("import org.apache.sysds.utils.Statistics;\n");
		msb.append("import org.apache.sysds.hops.Hop;\n");
		msb.append("import org.apache.sysds.hops.LiteralOp;\n");
		msb.append("import org.apache.sysds.hops.UnaryOp;\n");
		msb.append("import org.apache.sysds.hops.BinaryOp;\n");
		msb.append("import org.apache.sysds.hops.ReorgOp;\n");
		msb.append("import org.apache.sysds.hops.AggUnaryOp;\n");
		msb.append("import org.apache.sysds.hops.AggBinaryOp;\n");
		msb.append("import org.apache.sysds.hops.DataGenOp;\n");
		msb.append("import org.apache.sysds.hops.TernaryOp;\n");
		msb.append("import org.apache.sysds.common.Types;\n");
		msb.append("import org.apache.sysds.hops.rewrite.HopRewriteUtils;\n");
		msb.append("import org.apache.sysds.hops.rewriter.dml.DMLExecutor;\n");
		msb.append("\n");
		msb.append("public class " + className + " implements Function {\n\n");

		StringBuilder implSb = new StringBuilder();
		Set<String> implemented = new HashSet<>();
		for (Tuple2<String, RewriterRule> appliedRewrites : rewrites) {
			String mRewriteFn;
			if (ignoreErrors) {
				try {
					mRewriteFn = generateRewriteFunction(appliedRewrites._2, appliedRewrites._1, 1, maintainRewriteStats, ctx);
				} catch (Exception e) {
					if (printErrors)
						e.printStackTrace();

					continue;
				}
			} else {
				mRewriteFn = generateRewriteFunction(appliedRewrites._2, appliedRewrites._1, 1, maintainRewriteStats, ctx);
			}

			implSb.append('\n');
			indent(1, implSb);
			implSb.append("// Implementation of the rule " + appliedRewrites._2 + "\n");
			implSb.append(mRewriteFn);
			implemented.add(appliedRewrites._1);
		}

		indent(1, msb);
		msb.append("@Override\n");
		indent(1, msb);
		msb.append("public Object apply( Object _hi ) {\n");
		indent(2, msb);
		msb.append("if ( _hi == null )\n");
		indent(3, msb);
		msb.append("return null;\n\n");
		indent(2, msb);
		msb.append("Hop hi = (Hop) _hi;\n\n");

		if (optimize) {
			List<Tuple2<String, RewriterRule>> implementedRewrites = rewrites.stream().filter(t -> implemented.contains(t._1)).collect(Collectors.toList());

			List<RewriterRule> rules = rewrites.stream().map(t -> t._2).collect(Collectors.toList());
			Map<RewriterRule, String> ruleNames = new HashMap<>();

			for (Tuple2<String, RewriterRule> t : implementedRewrites)
				ruleNames.put(t._2, t._1);

			List<CodeGenCondition> conditions = CodeGenCondition.buildCondition(rules, 5, ctx);
			CodeGenCondition.buildSelection(msb, conditions, 2, ruleNames, ctx);
		} else {
			for (Tuple2<String, RewriterRule> appliedRewrites : rewrites) {
				if (implemented.contains(appliedRewrites._1)) {
					indent(2, msb);
					msb.append("hi = " + appliedRewrites._1 + "((Hop) hi);\t\t// ");
					msb.append(appliedRewrites._2.toString());
					msb.append('\n');
				}
			}
		}

		indent(2, msb);
		msb.append("return hi;\n");

		indent(1, msb);
		msb.append("}\n");

		msb.append(implSb);

		msb.append('\n');
		buildTypeCastFunction(msb, 1);

		msb.append("}");
		return msb.toString();
	}

	private static String generateRewriteFunction(RewriterRule rule, String fName, int indentation, boolean maintainRewriteStats, final RuleContext ctx) {
		try {
			Tuple2<Set<RewriterStatement>, Boolean> t = RewriterCostEstimator.determineSingleReferenceRequirement(rule, ctx);
			Set<RewriterStatement> mSet = t._1;
			if (mSet instanceof AbstractCollection)
				mSet = new HashSet<>(mSet);
			mSet.add(rule.getStmt1());
			boolean allowCombinedMultiRefs = t._2;

			StringBuilder sb = new StringBuilder();

			// Append the function signature
			indent(indentation, sb);
			sb.append("private static Hop " + fName + "(Hop hi) {\n");

			if (!allowCombinedMultiRefs) {
				indent(indentation + 1, sb);
				sb.append("boolean _multiReference = false;\n");
			}

			// Build the function body
			buildMatchingSequence(rule.toString(), rule.getStmt1(), rule.getStmt2(), rule.getStmt1Cost(), rule.getStmt2Cost(), rule.getCombinedAssertions(), sb, ctx, indentation + 1, mSet, allowCombinedMultiRefs, maintainRewriteStats);
			indent(indentation, sb);

			sb.append("}\n");

			return sb.toString();
		} catch (Exception e) {
			e.addSuppressed(new Exception("Failed to generate rewrite rule: " + rule.toString() + "\nAssertions: " + rule.getCombinedAssertions()));
			throw e;
		}
	}

	private static void buildMatchingSequence(String name, RewriterStatement from, RewriterStatement to, RewriterStatement fromCost, RewriterStatement toCost, RewriterAssertions combinedAssertions, StringBuilder sb, final RuleContext ctx, int indentation, Set<RewriterStatement> allowedMultiRefs, boolean allowCombinations, boolean maintainRewriteStats) {
		Map<RewriterStatement, String> vars = new HashMap<>();
		vars.put(from, "hi");
		recursivelyBuildMatchingSequence(from, sb, "hi", ctx, indentation, vars, allowedMultiRefs, allowCombinations);

		if (fromCost != null && toCost != null) {
			//System.out.println("FromCost: " + fromCost.toParsableString(ctx));
			//System.out.println("ToCost: " + toCost.toParsableString(ctx));

			StringBuilder msb = new StringBuilder();
			StringBuilder msb2 = new StringBuilder();
			Set<Tuple2<String, String>> requirements = new HashSet<>();
			buildCostFnRecursively(fromCost, vars, ctx, msb, requirements);
			buildCostFnRecursively(toCost, vars, ctx, msb2, requirements);

			// First, we build the necessary checks (e.g. if we have nnz / dim information we need, otherwise this rewrite cannot be applied)
			if (!requirements.isEmpty()) {
				sb.append('\n');
				indent(indentation, sb);
				sb.append("if ( ");

				int ctr = 0;
				for (Tuple2<String, String> req : requirements) {
					if (ctr != 0)
						sb.append(" || ");

					sb.append(req._1);
					switch (req._2) {
						case "_nnz":
							sb.append(".getNnz() == -1");
							break;
						case "nrow":
							sb.append(".getDim1() == -1");
							break;
						case "ncol":
							sb.append(".getDim2() == -1");
							break;
						default:
							throw new IllegalArgumentException(req._2);
					}

					ctr++;
				}

				sb.append(" )\n");
				indent(indentation + 1, sb);
				sb.append("return hi;\n\n");
			}

			// Then we build the cost functions
			sb.append('\n');
			indent(indentation, sb);
			sb.append("double costFrom = ");
			sb.append(msb);
			sb.append(";\n");
			indent(indentation, sb);
			sb.append("double costTo = ");
			sb.append(msb2);
			sb.append(";\n\n");
			indent(indentation, sb);
			sb.append("if ( costFrom <= costTo )\n");
			indent(indentation + 1, sb);
			sb.append("return hi;\n\n");
		}


		sb.append('\n');
		indent(indentation, sb);
		sb.append("// Now, we start building the new Hop\n");

		if (DEBUG) {
			//indent(indentation, sb);
			//sb.append("System.out.println(\"Applying rewrite: " + name + "\");\n");
			indent(indentation, sb);
			sb.append("DMLExecutor.println(\"Applying rewrite: " + name + "\");\n");
		}

		if (maintainRewriteStats) {
			indent(indentation, sb);
			sb.append("Statistics.applyGeneratedRewrite(\"" + name + "\");\n");
		}

		Set<RewriterStatement> activeStatements = buildRewrite(to, sb, combinedAssertions, vars, ctx, indentation);

		String newRoot = vars.get(to);

		sb.append('\n');
		indent(indentation, sb);
		sb.append("Hop newRoot = " + newRoot + ";\n");
		indent(indentation, sb);
		sb.append("if ( " + newRoot + ".getValueType() != hi.getValueType() ) {\n");
		indent(indentation + 1, sb);
		sb.append("newRoot = castIfNecessary(newRoot, hi);\n");
		indent(indentation + 1, sb);
		sb.append("if ( newRoot == null )\n");
		indent(indentation + 2, sb);
		sb.append("return hi;\n");
		indent(indentation, sb);
		sb.append("}\n");


		sb.append('\n');
		indent(indentation, sb);
		sb.append("ArrayList<Hop> parents = new ArrayList<>(hi.getParent());\n\n");
		indent(indentation, sb);
		sb.append("for ( Hop p : parents )\n");
		indent(indentation + 1, sb);
		sb.append("HopRewriteUtils.replaceChildReference(p, hi, newRoot);\n\n");

		indent(indentation, sb);
		sb.append("// Remove old unreferenced Hops\n");
		removeUnreferencedHops(from, activeStatements, sb, vars, ctx, indentation);
		sb.append('\n');

		indent(indentation, sb);
		sb.append("return newRoot;\n");
	}

	private static void buildTypeCastFunction(StringBuilder sb, int indentation) {
		indent(indentation, sb);
		sb.append("private static Hop castIfNecessary(Hop newRoot, Hop oldRoot) {\n");
		indent(indentation + 1, sb);
		sb.append("Types.OpOp1 cast = null;\n");
		indent(indentation + 1, sb);
		sb.append("switch ( oldRoot.getValueType().toExternalString() ) {\n");
		indent(indentation + 2, sb);
		sb.append("case \"DOUBLE\":\n"); //Types.ValueType.FP64.toExternalString()
		indent(indentation + 3, sb);
		sb.append("cast = Types.OpOp1.CAST_AS_DOUBLE;\n");
		indent(indentation + 3, sb);
		sb.append("break;\n");
		indent(indentation + 2, sb);
		sb.append("case \"INT\":\n"); //Types.ValueType.FP64.toExternalString()
		indent(indentation + 3, sb);
		sb.append("cast = Types.OpOp1.CAST_AS_INT;\n");
		indent(indentation + 3, sb);
		sb.append("break;\n");
		indent(indentation + 2, sb);
		sb.append("case \"BOOLEAN\":\n"); //Types.ValueType.FP64.toExternalString()
		indent(indentation + 3, sb);
		sb.append("cast = Types.OpOp1.CAST_AS_BOOLEAN;\n");
		indent(indentation + 3, sb);
		sb.append("break;\n");
		indent(indentation + 2, sb);
		sb.append("default:\n");
		indent(indentation + 3, sb);
		sb.append("return null;\n");
		indent(indentation + 1, sb);
		sb.append("}\n\n");
		indent(indentation + 1, sb);
		//sb.append("return HopRewriteUtils.createUnary(newRoot" + ", cast);\n");
		sb.append("return new UnaryOp(\"tmp\", oldRoot.getDataType(), oldRoot.getValueType(), cast, newRoot);\n");
		indent(indentation, sb);
		sb.append("}\n");
	}

	private static void buildCostFnRecursively(RewriterStatement costFn, Map<RewriterStatement, String> vars, final RuleContext ctx, StringBuilder sb, Set<Tuple2<String, String>> requirements) {
		if (costFn.isLiteral()) {
			sb.append(costFn.floatLiteral());
			return;
		}

		if (!costFn.isInstruction())
			throw new IllegalArgumentException();

		List<RewriterStatement> operands;

		if (!costFn.getOperands().isEmpty() && costFn.getChild(0).isArgumentList())
			operands = costFn.getChild(0).getOperands();
		else
			operands = costFn.getOperands();

		String varName;

		// Then, the cost function is an instruction
		switch (costFn.trueInstruction()) {
			case "_nnz":
				varName = vars.get(costFn.getChild(0));

				if (varName == null)
					throw new IllegalArgumentException(costFn.toParsableString(ctx));

				requirements.add(new Tuple2<>(varName, "_nnz"));
				sb.append(varName);
				sb.append(".getNnz()");
				break;

			case "nrow":
				varName = vars.get(costFn.getChild(0));

				if (varName == null)
					throw new IllegalArgumentException();

				requirements.add(new Tuple2<>(varName, "nrow"));
				sb.append(varName);
				sb.append(".getDim1()");
				break;

			case "ncol":
				varName = vars.get(costFn.getChild(0));

				if (varName == null)
					throw new IllegalArgumentException();

				requirements.add(new Tuple2<>(varName, "ncol"));
				sb.append(varName);
				sb.append(".getDim2()");
				break;

			case "+":
			case "*":
				sb.append('(');

				for (int i = 0; i < operands.size(); i++) {
					if (i != 0) {
						sb.append(' ');
						sb.append(costFn.trueInstruction());
						sb.append(' ');
					}

					buildCostFnRecursively(operands.get(i), vars, ctx, sb, requirements);
				}

				sb.append(')');
				break;
			case "inv":
				sb.append("(1.0 / ");
				buildCostFnRecursively(operands.get(0), vars, ctx, sb, requirements);
				sb.append(')');
				break;
			case "min":
			case "max":
				sb.append("Math.");
				sb.append(costFn.trueInstruction());
				sb.append('(');
				for (int i = 0; i < operands.size(); i++) {
					if (i != 0)
						sb.append(", ");

					buildCostFnRecursively(operands.get(i), vars, ctx, sb, requirements);
				}
				sb.append(')');
				break;
			case "_EClass":
				// Here, we can just select a random representant
				// Ideally, we would choose one that has dimensions available, but for now, we just take the first
				buildCostFnRecursively(operands.get(0), vars, ctx, sb, requirements);
				break;
			default:
				throw new IllegalArgumentException(costFn.trueInstruction());
		}
	}

	// Returns the set of all active statements after the rewrite
	private static Set<RewriterStatement> buildRewrite(RewriterStatement newRoot, StringBuilder sb, RewriterAssertions assertions, Map<RewriterStatement, String> vars, final RuleContext ctx, int indentation) {
		Set<RewriterStatement> visited = new HashSet<>();
		recursivelyBuildNewHop(sb, newRoot, assertions, vars, ctx, indentation, 1, visited, newRoot.getResultingDataType(ctx).equals("FLOAT"));
		//indent(indentation, sb);
		//sb.append("hi = " + vars.get(newRoot) + ";\n");

		return visited;
	}

	private static void removeUnreferencedHops(RewriterStatement oldRoot, Set<RewriterStatement> activeStatements, StringBuilder sb, Map<RewriterStatement, String> vars, final RuleContext ctx, int indentation) {
		oldRoot.forEachPreOrder(cur -> {
			if (activeStatements.contains(cur))
				return true;

			indent(indentation, sb);
			sb.append("HopRewriteUtils.cleanupUnreferenced(" + vars.get(cur) + ");\n");
			return true;
		}, false);
	}

	private static int recursivelyBuildNewHop(StringBuilder sb, RewriterStatement cur, RewriterAssertions assertions, Map<RewriterStatement, String> vars, final RuleContext ctx, int indentation, int varCtr, Set<RewriterStatement> visited, boolean enforceRootDataType) {
		visited.add(cur);
		if (vars.containsKey(cur))
			return varCtr;

		for (RewriterStatement child : cur.getOperands())
			varCtr = recursivelyBuildNewHop(sb, child, assertions, vars, ctx, indentation, varCtr, visited, false);

		if (cur instanceof RewriterDataType) {
			if (cur.isLiteral()) {
				indent(indentation, sb);
				String name = "l" + (varCtr++);
				String literalStr = cur.getLiteral().toString();

				if (enforceRootDataType) {
					sb.append("LiteralOp " + name + ";");
					indent(indentation, sb);
					sb.append("switch (hi.getValueType()) {\n");
					indent(indentation+1, sb);
					sb.append("case Types.ValueType.FP64:\n");
					indent(indentation+2, sb);
					sb.append(name);
					sb.append(" = new LiteralOp( " + cur.floatLiteral() + " );\n");
					indent(indentation+2, sb);
					sb.append("break;\n");
					indent(indentation+1, sb);
					sb.append("case Types.ValueType.INT64:\n");
					indent(indentation+2, sb);
					sb.append(name);
					sb.append(" = new LiteralOp( " + cur.intLiteral(true) + " );\n");
					indent(indentation+2, sb);
					sb.append("break;\n");
					indent(indentation+1, sb);
					sb.append("case Types.ValueType.BOOLEAN:\n");
					indent(indentation+2, sb);
					sb.append(name);
					sb.append(" = new LiteralOp( " + cur.boolLiteral() + " );\n");
					indent(indentation+2, sb);
					sb.append("break;\n");
					indent(indentation, sb);
					sb.append("}\n");
				} else {
					sb.append("LiteralOp " + name + " = new LiteralOp( " + literalStr + " );\n");
				}
				vars.put(cur, name);
			}

			return varCtr;
		} else {
			String opClass = CodeGenUtils.getOpClass(cur, ctx);
			String constructor = CodeGenUtils.getHopConstructor(cur, assertions, vars, ctx, cur.getOperands().stream().map(vars::get).toArray(String[]::new));
			String name = "v"  + (varCtr++);
			indent(indentation, sb);
			sb.append(opClass + " " + name + " = " + constructor + ";\n");
			vars.put(cur, name);
		}

		return varCtr;
	}

	private static void recursivelyBuildMatchingSequence(RewriterStatement cur, StringBuilder sb, String curVar, final RuleContext ctx, int indentation, Map<RewriterStatement, String> map, Set<RewriterStatement> allowedMultiRefs, boolean allowCombinations) {
		if (cur.isLiteral()) {
			String[] types = CodeGenUtils.getReturnType(cur, ctx);
			indent(indentation, sb);
			sb.append("if ( !(" + curVar + " instanceof LiteralOp) )\n");
			indent(indentation + 1, sb);
			sb.append("return hi;\n\n");
			indent(indentation, sb);
			String lVar = "l_" + curVar;
			sb.append("LiteralOp " + lVar + " = (LiteralOp) " + curVar + ";\n\n");
			indent(indentation, sb);
			sb.append("if ( " + lVar + ".getDataType() != " + types[0]);
			sb.append("|| !" + lVar + ".getValueType().isNumeric()");

			/*for (int i = 1; i < types.length; i++) {
				if (i == 1) {
					sb.append(" || (" + lVar + ".getValueType() != " + types[1]);
					continue;
				}

				sb.append(" && " + lVar + ".getValueType() != " + types[i]);

				if (i == types.length - 1)
					sb.append(')');
			}*/

			sb.append(" )\n");

			indent(indentation + 1, sb);
			sb.append("return hi;\n\n");

			indent(indentation, sb);
			sb.append("if ( " + lVar + "." + CodeGenUtils.literalGetterFunction(cur, ctx) + " != " + cur.getLiteral() + " )\n");
			indent(indentation + 1, sb);
			sb.append("return hi;\n\n");

			return;
		}

		// Check if we have to ensure a single reference to this object
		// TODO: This check is not entirely correct
		if (cur.isInstruction() && !allowedMultiRefs.contains(cur)) {
			if (allowCombinations && !allowedMultiRefs.contains(cur)) {
				indent(indentation, sb);
				sb.append("if (");
				sb.append(curVar);
				sb.append(".getParent().size() > 1)\n");
				indent(indentation + 1, sb);
				sb.append("return hi;\n");
			} else if (!allowedMultiRefs.contains(cur)) {
				indent(indentation, sb);
				sb.append("if (");
				sb.append(curVar);
				sb.append(".getParent().size() > 1) {\n");
				indent(indentation + 1, sb);
				sb.append("if (_multiReference)\n");
				indent(indentation + 2, sb);
				sb.append("return hi;\n");
				indent(indentation + 1, sb);
				sb.append("else\n");
				indent(indentation + 2, sb);
				sb.append("_multiReference = true;\n");
			}
		}

		String specialOpCheck = CodeGenUtils.getSpecialOpCheck(cur, ctx, curVar);

		// E.g. A %*% B, which is an AggBinaryOp consisting of multiple OpCodes
		if (specialOpCheck != null) {
			indent(indentation, sb);
			sb.append("if ( !" + specialOpCheck + " )\n");
			indent(indentation + 1, sb);
			sb.append("return hi;\n\n");
		} else if (!cur.isDataOrigin()) {
			String opClass = CodeGenUtils.getOpClass(cur, ctx);

			// Generate initial class check
			indent(indentation, sb);
			sb.append("if ( !(" + curVar + " instanceof " + opClass + ") )\n");
			indent(indentation + 1, sb);
			sb.append("return hi;\n\n");

			// Cast the expression to the corresponding op-class
			String cCurVar = "c_" + curVar;
			indent(indentation, sb);
			sb.append(opClass + " " + cCurVar + " = (" + opClass + ") " + curVar + ";\n\n");

			String opCode = CodeGenUtils.getOpCode(cur, ctx);

			// Check if the instruction matches
			indent(indentation, sb);
			sb.append("if ( " + cCurVar + ".getOp() != " + opCode);
			sb.append(" || !" + cCurVar + ".getValueType().isNumeric()");
			//String[] types = CodeGenUtils.getReturnType(cur, ctx);
			//sb.append(" || " + cCurVar + ".getDataType() != " + types[0]);

			/*for (int i = 1; i < types.length; i++) {
				if (i == 1) {
					sb.append(" || (" + cCurVar + ".getValueType() != " + types[1]);
					continue;
				}

				sb.append(" && " + cCurVar + ".getValueType() != " + types[i]);

				if (i == types.length - 1)
					sb.append(')');
			}*/

			sb.append(" )\n");
			indent(indentation + 1, sb);
			sb.append("return hi;\n\n");

			String additionalCheck = CodeGenUtils.getAdditionalCheck(cur, ctx, cCurVar);

			if (additionalCheck != null) {
				indent(indentation, sb);
				sb.append("if ( !(" + additionalCheck + ") )\n");
				indent(indentation + 1, sb);
				sb.append("return hi;\n\n");
			}
		} else {
			indent(indentation, sb);
			String[] types = CodeGenUtils.getReturnType(cur, ctx);
			sb.append("if ( " + curVar + ".getDataType() != " + types[0]);
			sb.append(" || !" + curVar + ".getValueType().isNumeric()");

			/*for (int i = 1; i < types.length; i++) {
				if (i == 1) {
					sb.append(" || (" + curVar + ".getValueType() != " + types[1]);
					continue;
				}

				sb.append(" && " + curVar + ".getValueType() != " + types[i]);

				if (i == types.length - 1)
					sb.append(')');
			}*/

			if (cur.isRowVector()) {
				sb.append(" || " + curVar + ".getDim2() != 1L");
			} else if (cur.isColVector()) {
				sb.append(" || " + curVar + ".getDim1() != 1L");
			}

			sb.append(" )\n");
			indent(indentation + 1, sb);
			sb.append("return hi;\n\n");

			String additionalCheck = CodeGenUtils.getAdditionalCheck(cur, ctx, curVar);

			if (additionalCheck != null) {
				indent(indentation, sb);
				sb.append("if ( !(" + additionalCheck + ") )\n");
				indent(indentation + 1, sb);
				sb.append("return hi;\n\n");
			}
		}

		// Now, we match the children
		for (int i = 0; i < cur.getOperands().size(); i++) {
			RewriterStatement stmt = cur.getChild(i);

			String existingVar = map.get(stmt);

			if (existingVar != null) {
				String name = resolveOperand(cur, i, sb, curVar, ctx, indentation);
				sb.append('\n');
				// Just check if they are identical
				indent(indentation, sb);
				sb.append("if ( " + existingVar + " != " + name + " )\n");
				indent(indentation + 1, sb);
				sb.append("return hi;\n\n");
				continue;
			}

			//if (stmt.isLiteral() || stmt.isInstruction()) {
				// Build the variable definition
				String name = resolveOperand(cur, i, sb, curVar, ctx, indentation);
				map.put(stmt, name);
				sb.append('\n');
				recursivelyBuildMatchingSequence(stmt, sb, name, ctx, indentation, map, allowedMultiRefs, allowCombinations);
			/*} else {
				String name = resolveOperand(cur, i, sb, curVar, ctx, indentation);
				map.put(stmt, name);
				sb.append('\n');
			}*/
		}
	}

	private static String resolveOperand(RewriterStatement stmt, int idx, StringBuilder sb, String curVar, final RuleContext ctx, int indentation) {
		//RewriterStatement child = stmt.getChild(idx);
		String name = curVar + "_" + idx;
		indent(indentation, sb);
		sb.append("Hop " + name + " = " + curVar + ".getInput(" + idx + ");\n");
		return name;
	}

	public static void indent(int depth, StringBuilder sb) {
		for (int i = 0; i < depth; i++)
			sb.append('\t');
	}
}
