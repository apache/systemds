package org.apache.sysds.hops.rewriter;

import org.apache.commons.lang3.NotImplementedException;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;

public class DMLCodeGenerator {
	public static final double EPS = 1e-10;
	public static Random rd = new Random(42);


	private static final HashSet<String> printAsBinary = new HashSet<>();
	private static final HashMap<String, BiFunction<RewriterStatement, StringBuilder, Boolean>> customEncoders = new HashMap<>();
	private static final RuleContext ctx = RewriterUtils.buildDefaultContext();

	static {
		printAsBinary.add("+");
		printAsBinary.add("-");
		printAsBinary.add("*");
		printAsBinary.add("/");
		printAsBinary.add("^");
		printAsBinary.add("==");
		printAsBinary.add("!=");
		printAsBinary.add(">");
		printAsBinary.add(">=");
		printAsBinary.add("<");
		printAsBinary.add("<=");

		customEncoders.put("[]", (stmt, sb) -> {
			if (stmt.getOperands().size() == 3) {
				sb.append('(');
				appendExpression(stmt.getChild(0), sb);
				sb.append(")[");
				appendExpression(stmt.getChild(1), sb);
				sb.append(", ");
				appendExpression(stmt.getChild(2), sb);
				sb.append(']');
				return true;
			} else if (stmt.getOperands().size() == 5) {
				sb.append('(');
				appendExpression(stmt.getChild(0), sb);
				sb.append(")[");
				appendExpression(stmt.getChild(1), sb);
				sb.append(" : ");
				appendExpression(stmt.getChild(2), sb);
				sb.append(", ");
				appendExpression(stmt.getChild(3), sb);
				sb.append(" : ");
				appendExpression(stmt.getChild(4), sb);
				sb.append(']');
				return true;
			}

			return false;
		});
	}

	public static Consumer<String> ruleValidationScript(String sessionId, Consumer<Boolean> validator) {
		return line -> {
			if (!line.startsWith(sessionId))
				return;

			if (line.endsWith("valid: TRUE")) {
				validator.accept(true);
			} else {
				DMLExecutor.println("An invalid rule was found!");
				validator.accept(false);
			}
		};
	}

	public static String generateRuleValidationDML(RewriterRule rule, String sessionId) {
		return generateRuleValidationDML(rule, EPS, sessionId);
	}

	public static String generateRuleValidationDML(RewriterRule rule, double eps, String sessionId) {
		RewriterStatement stmtFrom = rule.getStmt1();
		RewriterStatement stmtTo = rule.getStmt2();

		Set<RewriterStatement> vars = new HashSet<>();

		stmtFrom.forEachPostOrder((stmt, pred) -> {
			if (!stmt.isInstruction() && !stmt.isLiteral())
				vars.add(stmt);
		}, false);

		stmtTo.forEachPostOrder((stmt, pred) -> {
			if (!stmt.isInstruction() && !stmt.isLiteral())
				vars.add(stmt);
		}, false);

		StringBuilder sb = new StringBuilder();

		sb.append(generateDMLVariables(vars));
		/*for (RewriterStatement var : vars) {
			switch (var.getResultingDataType(ctx)) {
				case "MATRIX":
					sb.append(var.getId() + " = rand(rows=1000, cols=1000, min=0.0, max=1.0)\n");
					break;
				case "FLOAT":
					sb.append(var.getId() + " = as.scalar(rand())\n");
					break;
				case "INT":
					sb.append(var.getId() + " = as.integer(as.scalar(rand(min=0.0, max=10000.0)))\n");
					break;
				case "BOOL":
					sb.append(var.getId() + " = as.scalar(rand()) < 0.5\n");
					break;
				default:
					throw new NotImplementedException(var.getResultingDataType(ctx));
			}
		}*/

		sb.append('\n');
		sb.append("R1 = ");
		sb.append(generateDML(stmtFrom));
		sb.append('\n');
		sb.append("R2 = ");
		sb.append(generateDML(stmtTo));
		sb.append('\n');
		sb.append("print(\"");
		sb.append(sessionId);
		sb.append(" valid: \" + (");
		sb.append(generateEqualityCheck("R1", "R2", stmtFrom.getResultingDataType(ctx), eps));
		sb.append("))");

		return sb.toString();
	}

	public static Set<RewriterStatement> getVariables(RewriterStatement root) {
		Set<RewriterStatement> vars = new HashSet<>();
		root.forEachPostOrder((stmt, pred) -> {
			if (!stmt.isInstruction() && !stmt.isLiteral())
				vars.add(stmt);
		}, false);
		return vars;
	}

	public static String generateDMLVariables(RewriterStatement root) {
		return generateDMLVariables(getVariables(root));
	}

	public static String generateDMLVariables(Set<RewriterStatement> vars) {
		StringBuilder sb = new StringBuilder();

		for (RewriterStatement var : vars) {
			switch (var.getResultingDataType(ctx)) {
				case "MATRIX":
					sb.append(var.getId() + " = rand(rows=500, cols=500, min=(as.scalar(rand())+1.0), max=(as.scalar(rand())+2.0), seed=" + rd.nextInt(1000) + ")^as.scalar(rand())\n");
					break;
				case "FLOAT":
					sb.append(var.getId() + " = as.scalar(rand())\n");
					break;
				case "INT":
					sb.append(var.getId() + " = as.integer(as.scalar(rand(min=0.0, max=10000.0)))\n");
					break;
				case "BOOL":
					sb.append(var.getId() + " = as.scalar(rand()) < 0.5\n");
					break;
				default:
					throw new NotImplementedException(var.getResultingDataType(ctx));
			}
		}

		return sb.toString();
	}

	public static String generateEqualityCheck(String stmt1Var, String stmt2Var, String dataType, double eps) {
		switch (dataType) {
			case "MATRIX":
				return "sum(abs(" + stmt1Var + " - " + stmt2Var + ") < " + eps + ") == length(" + stmt1Var + ")";
			case "INT":
			case "BOOL":
				return stmt1Var + " == " + stmt2Var;
			case "FLOAT":
				return "abs(" + stmt1Var + " - " + stmt2Var + ") < " + eps;
		}

		throw new NotImplementedException();
	}

	public static String generateDMLDefs(RewriterStatement stmt) {
		Map<String, RewriterStatement> vars = new HashMap<>();

		stmt.forEachPostOrder((cur, pred) -> {
			if (!cur.isInstruction() && !cur.isLiteral())
				vars.put(cur.getId(), cur);
		}, false);

		return generateDMLDefs(vars);
	}

	public static String generateDMLDefs(Map<String, RewriterStatement> defs) {
		StringBuilder sb = new StringBuilder();

		defs.forEach((k, v) -> {
			sb.append(k);
			sb.append(" = ");
			sb.append(generateDML(v));
			sb.append('\n');
		});

		return sb.toString();
	}

	public static String generateDML(RewriterStatement root) {
		StringBuilder sb = new StringBuilder();
		appendExpression(root, sb);

		return sb.toString();
	}

	private static void appendExpression(RewriterStatement cur, StringBuilder sb) {
		if (cur.isInstruction()) {
			resolveExpression((RewriterInstruction) cur, sb);
		} else {
			if (cur.isLiteral())
				sb.append(cur.getLiteral());
			else
				sb.append(cur.getId());
		}
	}

	private static void resolveExpression(RewriterInstruction expr, StringBuilder sb) {
		String typedInstr = expr.trueTypedInstruction(ctx);
		String unTypedInstr = expr.trueInstruction();

		if (expr.getOperands().size() == 2 && (printAsBinary.contains(typedInstr) || printAsBinary.contains(unTypedInstr))) {
			sb.append('(');
			appendExpression(expr.getChild(0), sb);
			sb.append(") ");
			sb.append(unTypedInstr);
			sb.append(" (");
			appendExpression(expr.getChild(1), sb);
			sb.append(')');
			return;
		}

		BiFunction<RewriterStatement, StringBuilder, Boolean> customEncoder = customEncoders.get(typedInstr);

		if (customEncoder == null)
			customEncoder = customEncoders.get(unTypedInstr);

		if (customEncoder == null) {
			sb.append(unTypedInstr);
			sb.append('(');

			for (int i = 0; i < expr.getOperands().size(); i++) {
				if (i != 0)
					sb.append(", ");

				appendExpression(expr.getChild(i), sb);
			}

			sb.append(')');
		} else {
			customEncoder.apply(expr, sb);
		}
	}
}
