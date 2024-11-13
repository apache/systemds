package org.apache.sysds.hops.rewriter;


import org.apache.sysds.hops.Hop;
import org.codehaus.janino.SimpleCompiler;
import scala.Tuple2;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;

public class RewriterCodeGen {
	public static boolean DEBUG = false;

	public static Function<Hop, Hop> compileRewrites(String className, List<Tuple2<String, RewriterRule>> rewrites, final RuleContext ctx, boolean ignoreErrors, boolean printErrors) throws Exception {
		String code = generateClass(className, rewrites, ctx, ignoreErrors, printErrors);
		System.out.println(code);
		SimpleCompiler compiler = new SimpleCompiler();
		compiler.cook(code);
		Class<?> mClass = compiler.getClassLoader().loadClass(className);
		Object instance = mClass.getDeclaredConstructor().newInstance();
		return (Function<Hop, Hop>) instance;
	}

	public static String generateClass(String className, List<Tuple2<String, RewriterRule>> rewrites, final RuleContext ctx, boolean ignoreErrors, boolean printErrors) {
		StringBuilder msb = new StringBuilder();
		msb.append("import java.util.ArrayList;\n");
		msb.append("import java.util.function.Function;\n");
		msb.append("\n");
		msb.append("import org.apache.sysds.hops.Hop;\n");
		msb.append("import org.apache.sysds.hops.LiteralOp;\n");
		msb.append("import org.apache.sysds.hops.UnaryOp;\n");
		msb.append("import org.apache.sysds.hops.BinaryOp;\n");
		msb.append("import org.apache.sysds.hops.ReorgOp;\n");
		msb.append("import org.apache.sysds.hops.AggUnaryOp;\n");
		msb.append("import org.apache.sysds.hops.AggBinaryOp;\n");
		msb.append("import org.apache.sysds.common.Types;\n");
		msb.append("import org.apache.sysds.hops.rewrite.HopRewriteUtils;\n");
		msb.append("\n");
		msb.append("public class " + className + " implements Function {\n\n");

		StringBuilder implSb = new StringBuilder();
		Set<String> implemented = new HashSet<>();
		for (Tuple2<String, RewriterRule> appliedRewrites : rewrites) {
			String mRewriteFn;
			if (ignoreErrors) {
				try {
					mRewriteFn = generateRewriteFunction(appliedRewrites._2, appliedRewrites._1, 1, ctx);
				} catch (Exception e) {
					if (printErrors)
						e.printStackTrace();

					continue;
				}
			} else {
				mRewriteFn = generateRewriteFunction(appliedRewrites._2, appliedRewrites._1, 1, ctx);
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
		msb.append("public Object apply( Object hi ) {\n");
		indent(2, msb);
		msb.append("if ( hi == null )\n");
		indent(3, msb);
		msb.append("return null;\n\n");

		for (Tuple2<String, RewriterRule> appliedRewrites : rewrites) {
			if (implemented.contains(appliedRewrites._1)) {
				indent(2, msb);
				msb.append("hi = " + appliedRewrites._1 + "((Hop) hi);\n");
			}
		}

		indent(2, msb);
		msb.append("return hi;\n");

		indent(1, msb);
		msb.append("}\n");

		msb.append(implSb);

		msb.append("}");
		return msb.toString();
	}

	private static String generateRewriteFunction(RewriterRule rule, String fName, int indentation, final RuleContext ctx) {
		StringBuilder sb = new StringBuilder();

		// Append the function signature
		indent(indentation, sb);
		sb.append("private static Hop " + fName + "(Hop hi) {\n");

		// Build the function body
		buildMatchingSequence(rule.getStmt1(), rule.getStmt2(), sb, ctx, indentation + 1);
		indent(indentation, sb);

		sb.append("}\n");

		return sb.toString();
	}

	private static void buildMatchingSequence(RewriterStatement from, RewriterStatement to, StringBuilder sb, final RuleContext ctx, int indentation) {
		Map<RewriterStatement, String> vars = new HashMap<>();
		vars.put(from, "hi");
		recursivelyBuildMatchingSequence(from, sb, "hi", ctx, indentation, vars);
		sb.append("\n");
		indent(indentation, sb);
		sb.append("// Now, we start building the new Hop\n");

		if (DEBUG) {
			indent(indentation, sb);
			sb.append("System.out.println(\"HERE\");\n");
		}

		Set<RewriterStatement> activeStatements = buildRewrite(to, sb, vars, ctx, indentation);

		sb.append('\n');
		indent(indentation, sb);
		sb.append("ArrayList<Hop> parents = new ArrayList<>(hi.getParent());\n\n");
		indent(indentation, sb);
		sb.append("for ( Hop p : parents )\n");
		indent(indentation + 1, sb);
		sb.append("HopRewriteUtils.replaceChildReference(p, hi, " + vars.get(to) + ");\n\n");

		indent(indentation, sb);
		sb.append("// Remove old unreferenced Hops\n");
		removeUnreferencedHops(from, activeStatements, sb, vars, ctx, indentation);
		sb.append('\n');

		indent(indentation, sb);
		sb.append("return " + vars.get(to) + ";\n");
	}

	// Returns the set of all active statements after the rewrite
	private static Set<RewriterStatement> buildRewrite(RewriterStatement newRoot, StringBuilder sb, Map<RewriterStatement, String> vars, final RuleContext ctx, int indentation) {
		Set<RewriterStatement> visited = new HashSet<>();
		recursivelyBuildNewHop(sb, newRoot, vars, ctx, indentation, 1, visited);
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

	private static int recursivelyBuildNewHop(StringBuilder sb, RewriterStatement cur, Map<RewriterStatement, String> vars, final RuleContext ctx, int indentation, int varCtr, Set<RewriterStatement> visited) {
		visited.add(cur);
		if (vars.containsKey(cur))
			return varCtr;

		for (RewriterStatement child : cur.getOperands())
			varCtr = recursivelyBuildNewHop(sb, child, vars, ctx, indentation, varCtr, visited);

		if (cur instanceof RewriterDataType) {
			if (cur.isLiteral()) {
				indent(indentation, sb);
				String name = "l" + (varCtr++);
				sb.append("LiteralOp " + name + " = new LiteralOp( " + cur.getLiteral() + " );\n");
				vars.put(cur, name);
			}

			return varCtr;
		} else {
			String opClass = CodeGenUtils.getOpClass(cur, ctx);
			String constructor = CodeGenUtils.getHopConstructor(cur, ctx, cur.getOperands().stream().map(vars::get).toArray(String[]::new));
			String name = "v"  + (varCtr++);
			indent(indentation, sb);
			sb.append(opClass + " " + name + " = " + constructor + ";\n");
			vars.put(cur, name);
		}

		return varCtr;
	}

	private static void recursivelyBuildMatchingSequence(RewriterStatement cur, StringBuilder sb, String curVar, final RuleContext ctx, int indentation, Map<RewriterStatement, String> map) {
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

			for (int i = 1; i < types.length; i++) {
				if (i == 1) {
					sb.append(" || (" + lVar + ".getValueType() != " + types[1]);
					continue;
				}

				sb.append(" && " + lVar + ".getValueType() != " + types[i]);

				if (i == types.length - 1)
					sb.append(')');
			}

			sb.append(" )\n");

			indent(indentation + 1, sb);
			sb.append("return hi;\n\n");

			indent(indentation, sb);
			sb.append("if ( " + lVar + "." + CodeGenUtils.literalGetterFunction(cur, ctx) + " != " + cur.getLiteral() + " )\n");
			indent(indentation + 1, sb);
			sb.append("return hi;\n\n");

			return;
		}

		String specialOpCheck = CodeGenUtils.getSpecialOpCheck(cur, ctx, curVar);

		// E.g. A %*% B, which is an AggBinaryOp consisting of multiple OpCodes
		if (specialOpCheck != null) {
			indent(indentation, sb);
			sb.append("if ( !" + specialOpCheck + " )\n");
			indent(indentation + 1, sb);
			sb.append("return hi;\n\n");
		} else {
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
			String[] types = CodeGenUtils.getReturnType(cur, ctx);
			sb.append(" || " + cCurVar + ".getDataType() != " + types[0]);

			for (int i = 1; i < types.length; i++) {
				if (i == 1) {
					sb.append(" || (" + cCurVar + ".getValueType() != " + types[1]);
					continue;
				}

				sb.append(" && " + cCurVar + ".getValueType() != " + types[i]);

				if (i == types.length - 1)
					sb.append(')');
			}

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

			if (stmt.isLiteral() || stmt.isInstruction()) {
				// Build the variable definition
				String name = resolveOperand(cur, i, sb, curVar, ctx, indentation);
				map.put(stmt, name);
				sb.append('\n');
				recursivelyBuildMatchingSequence(stmt, sb, name, ctx, indentation, map);
			} else {
				String name = resolveOperand(cur, i, sb, curVar, ctx, indentation);
				map.put(stmt, name);
				sb.append('\n');
			}
		}
	}

	private static String resolveOperand(RewriterStatement stmt, int idx, StringBuilder sb, String curVar, final RuleContext ctx, int indentation) {
		//RewriterStatement child = stmt.getChild(idx);
		String name = curVar + "_" + idx;
		indent(indentation, sb);
		sb.append("Hop " + name + " = " + curVar + ".getInput(" + idx + ");\n");
		return name;
	}

	private static void indent(int depth, StringBuilder sb) {
		for (int i = 0; i < depth; i++)
			sb.append('\t');
	}
}
