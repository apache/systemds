package org.apache.sysds.hops.rewriter.codegen;

import javassist.compiler.CodeGen;
import org.apache.sysds.hops.rewriter.RewriterRule;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RuleContext;
import scala.Tuple2;
import scala.Tuple3;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.stream.Collectors;

public class CodeGenCondition {
	public enum ConditionType {
		DATA_TYPE, VALUE_TYPE, UNIQUE_PARENTS, LITERAL, OP_CLASS, OP_CODE, NUM_INPUTS, ELSE
	}

	public enum ConditionDataType {
		SCALAR, MATRIX
	}

	private ConditionType conditionType;
	private Object conditionValue;
	private List<Object> rulesIf;
	private List<Integer> relativeChildPath;
	private RewriterStatement representant;

	private CodeGenCondition(ConditionType cType, Object cValue, List<Integer> relativeChildPath, RewriterStatement representant, final RuleContext ctx) {
		conditionType = cType;
		conditionValue = cValue;
		rulesIf = new ArrayList<>();
		this.relativeChildPath = relativeChildPath;
		this.representant = representant;

		if (conditionType != ConditionType.ELSE)
			buildConditionCheck(new StringBuilder(), ctx);
	}

	public static List<CodeGenCondition> buildCondition(List<RewriterRule> rules, int maxNumRules, final RuleContext ctx) {
		if (rules.isEmpty())
			return Collections.emptyList();
		List<Object> transformed = rules.stream().map(rule -> new Tuple2<RewriterRule, RewriterStatement>(rule, rule.getStmt1())).collect(Collectors.toList());
		List<Object> out = populateLayerRecursively(transformed, Collections.emptyList(), new LinkedList<>(), maxNumRules, ctx);
		List<CodeGenCondition> cond = out.stream().filter(o -> o instanceof CodeGenCondition).map(o -> ((CodeGenCondition)o)).collect(Collectors.toList());
		return cond.isEmpty() ? List.of(conditionalElse(transformed, Collections.emptyList(), ((Tuple2<RewriterRule, RewriterStatement>) transformed.get(0))._2, ctx)) : cond;
	}

	private static List<Object> populateLayerRecursively(List<Object> rules, List<Integer> relativeChildPath, Queue<Tuple2<List<Object>, List<Integer>>> queue, int maxNumRules, final RuleContext ctx) {
		if (rules.size() <= maxNumRules)
			return rules;

		List<Object> out = populateDataTypeLayer(rules, relativeChildPath, ctx);

		for (int i = 0; i < out.size(); i++) {
			CodeGenCondition c = (CodeGenCondition) out.get(i);
			c.rulesIf = populateOpClassLayer(c.rulesIf, relativeChildPath, ctx);

			for (int j = 0; j < c.rulesIf.size(); j++) {
				CodeGenCondition c2 = (CodeGenCondition) c.rulesIf.get(j);
				c2.rulesIf = populateOpCodeLayer(c2.rulesIf, relativeChildPath, ctx);

				for (int k = 0; k < c2.rulesIf.size(); k++) {
					CodeGenCondition c3 = (CodeGenCondition) c2.rulesIf.get(k);
					c3.rulesIf = populateInputSizeLayer(c3.rulesIf, relativeChildPath, ctx);
					//int maxChildSize = c3.rulesIf.stream().flatMap(o -> ((CodeGenCondition)o).rulesIf.stream()).mapToInt(o -> ((Tuple2<RewriterRule, RewriterStatement>) o)._2.getOperands().size()).max().getAsInt();

					for (int l = 0; l < c3.rulesIf.size(); l++) {
						CodeGenCondition c4 = (CodeGenCondition) c3.rulesIf.get(l);
						final int maxIndex = ((Tuple2<RewriterRule, RewriterStatement>) c4.rulesIf.get(0))._2.getOperands().size();
						Queue<Tuple2<List<Object>, List<Integer>>> mQueue = new LinkedList<>(queue);

						for (int idx = 0; idx < maxIndex; idx++) {
							final int mIdx = idx;
							final List<Integer> newRelativeChildPath = new ArrayList<>(relativeChildPath);
							newRelativeChildPath.add(mIdx);
							List<Object> mList = new ArrayList<>();
							mQueue.add(new Tuple2<>(mList, newRelativeChildPath));

							c4.rulesIf.forEach(o -> {
								Tuple2<RewriterRule, RewriterStatement> t = (Tuple2<RewriterRule, RewriterStatement>) o;
								mList.add(new Tuple2<RewriterRule, RewriterStatement>(t._1, t._2.getChild(mIdx)));
							});
						}

						if (!mQueue.isEmpty()) {
							Tuple2<List<Object>, List<Integer>> next = mQueue.poll();
							c4.rulesIf = populateLayerRecursively(next._1, next._2(), mQueue, maxNumRules, ctx);
						}
					}

					/*for (int childIndex = 0; childIndex < maxChildSize; childIndex++) {
						final List<Integer> newRelativeChildPath = new ArrayList<>(relativeChildPath);
						newRelativeChildPath.add(childIndex);*/

					//}
				}
			}
		}

		return out;
	}

	private static List<Object> populateDataTypeLayer(List<Object> rules, List<Integer> relativeChildPath, final RuleContext ctx) {
		List<Object> conds = new ArrayList<>();

		for (Object o : rules) {
			Tuple2<RewriterRule, RewriterStatement> t = (Tuple2<RewriterRule, RewriterStatement>) o;
			if (!conds.stream().anyMatch(cond -> ((CodeGenCondition) cond).insertIfMatches(t, ctx))) {
				CodeGenCondition cond = CodeGenCondition.conditionalDataType(t._2, relativeChildPath, t._2, ctx);
				cond.insertIfMatches(t, ctx);
				conds.add(cond);
			}
		}

		return conds;
	}

	private static List<Object> populateOpClassLayer(List<Object> l, List<Integer> relativeChildPath, final RuleContext ctx) {
		try {
			List<Object> conds = new ArrayList<>();
			List<Object> remaining = new ArrayList<>();

			for (Object o : l) {
				Tuple2<RewriterRule, RewriterStatement> t = (Tuple2<RewriterRule, RewriterStatement>) o;
				if (canGenerateOpClassCheck(t._2, ctx)) {
					if (!conds.stream().anyMatch(cond -> ((CodeGenCondition) cond).insertIfMatches(t, ctx))) {
						CodeGenCondition cond = CodeGenCondition.conditionalOpClass(t._2, relativeChildPath, t._2, ctx);
						cond.insertIfMatches(t, ctx);
						conds.add(cond);
					}
				} else {
					remaining.add(t);
				}
			}

			if (!remaining.isEmpty()) {
				conds.add(CodeGenCondition.conditionalElse(remaining, relativeChildPath, ((Tuple2<RewriterRule, RewriterStatement>) remaining.get(0))._2, ctx));
			}

			return conds;
		} catch (Exception e) {
			return Collections.emptyList();
		}
	}

	private static List<Object> populateOpCodeLayer(List<Object> l, List<Integer> relativeChildPath, final RuleContext ctx) {
		List<Object> conds = new ArrayList<>();
		List<Object> remaining = new ArrayList<>();

		for (Object o : l) {
			Tuple2<RewriterRule, RewriterStatement> t = (Tuple2<RewriterRule, RewriterStatement>) o;
			if (canGenerateOpCodeCheck(t._2, ctx)) {
				if (!conds.stream().anyMatch(cond -> ((CodeGenCondition) cond).insertIfMatches(t, ctx))) {
					CodeGenCondition cond = CodeGenCondition.conditionalOpCode(t._2, relativeChildPath, t._2, ctx);
					cond.insertIfMatches(t, ctx);
					conds.add(cond);
				}
			} else {
				remaining.add(t);
			}
		}

		if (!remaining.isEmpty()) {
			conds.add(CodeGenCondition.conditionalElse(remaining, relativeChildPath, ((Tuple2<RewriterRule, RewriterStatement>) remaining.get(0))._2, ctx));
		}

		return conds;
	}

	private static List<Object> populateInputSizeLayer(List<Object> l, List<Integer> relativeChildPath, final RuleContext ctx) {
		List<Object> conds = new ArrayList<>();
		List<Object> remaining = new ArrayList<>();

		for (Object o : l) {
			Tuple2<RewriterRule, RewriterStatement> t = (Tuple2<RewriterRule, RewriterStatement>) o;
			if (canGenerateInputSizeCheck(t._2, ctx)) {
				if (!conds.stream().anyMatch(cond -> ((CodeGenCondition) cond).insertIfMatches(t, ctx))) {
					CodeGenCondition cond = CodeGenCondition.conditionalInputSize(t._2.getOperands().size(), relativeChildPath, t._2, ctx);
					cond.insertIfMatches(t, ctx);
					conds.add(cond);
				}
			} else {
				remaining.add(t);
			}
		}

		if (!remaining.isEmpty()) {
			conds.add(CodeGenCondition.conditionalElse(remaining, relativeChildPath, ((Tuple2<RewriterRule, RewriterStatement>) remaining.get(0))._2, ctx));
		}

		return conds;
	}

	public String getVarName() {
		if (relativeChildPath.isEmpty())
			return "hi";
		return "hi_" + relativeChildPath.stream().map(Object::toString).collect(Collectors.joining("_"));
	}

	public void buildConditionCheck(StringBuilder sb, final RuleContext ctx) {
		switch (conditionType) {
			case DATA_TYPE:
				sb.append("hi");
				if (!relativeChildPath.isEmpty()) {
					sb.append("_");
					sb.append(relativeChildPath.stream().map(Object::toString).collect(Collectors.joining("_")));
				}
				sb.append(".getDataType() == ");
				sb.append(CodeGenUtils.getReturnType(getDataType() == ConditionDataType.MATRIX ? "MATRIX" : "FLOAT")[0]);
				break;
			case OP_CLASS:
				sb.append("hi");
				if (!relativeChildPath.isEmpty()) {
					sb.append("_");
					sb.append(relativeChildPath.stream().map(Object::toString).collect(Collectors.joining("_")));
				}
				sb.append(" instanceof " + CodeGenUtils.getOpClass(representant, ctx));
				break;
			case OP_CODE:
				String hopVar = "hi";
				if (!relativeChildPath.isEmpty()) {
					hopVar += "_";
					hopVar += relativeChildPath.stream().map(Object::toString).collect(Collectors.joining("_"));
				}
				String specialInstr = CodeGenUtils.getSpecialOpCheck(representant, ctx, hopVar);
				if (specialInstr != null) {
					sb.append(specialInstr);
				} else {
					sb.append(hopVar);
					sb.append(".getOp() == ");
					sb.append(CodeGenUtils.getOpCode(representant, ctx));
				}
				break;
			case NUM_INPUTS:
				sb.append("hi");
				if (!relativeChildPath.isEmpty()) {
					sb.append("_");
					sb.append(relativeChildPath.stream().map(Object::toString).collect(Collectors.joining("_")));
				}
				sb.append(".getInput().size() == ");
				sb.append(conditionValue.toString());
				break;
			default:
				throw new IllegalArgumentException(conditionType.name());
		}
	}

	public boolean insertIfMatches(Tuple2<RewriterRule, RewriterStatement> t, final RuleContext ctx) {
		if (matchesCondition(t._1, t._2, ctx)) {
			rulesIf.add(t);
			return true;
		}

		return false;
	}

	public boolean matchesCondition(RewriterRule rule, RewriterStatement stmt, final RuleContext ctx) {
		switch (conditionType) {
			case DATA_TYPE:
				return matchesDataTypeCondition(rule, stmt, ctx);
			case OP_CLASS:
				return matchesOpClassCondition(rule, stmt, ctx);
			case OP_CODE:
				return matchesOpCodeCondition(rule, stmt, ctx);
			case NUM_INPUTS:
				return matchesNumInputs(rule, stmt, ctx);
		}
		return false;
	}

	public ConditionDataType getDataType() {
		return (ConditionDataType) conditionValue;
	}

	private boolean matchesNumInputs(RewriterRule rule, RewriterStatement stmt, final RuleContext ctx) {
		return ((int)conditionValue) == stmt.getOperands().size();
	}

	private boolean matchesDataTypeCondition(RewriterRule rule, RewriterStatement stmt, final RuleContext ctx) {
		ConditionDataType cdt = getDataType();
		String dType = stmt.getResultingDataType(ctx);

		if (dType.equals("MATRIX"))
			return cdt.equals(ConditionDataType.MATRIX);
		else
			return cdt.equals(ConditionDataType.SCALAR);
	}

	private boolean matchesOpClassCondition(RewriterRule rule, RewriterStatement stmt, final RuleContext ctx) {
		String opClass = (String) conditionValue;
		String actualClass = CodeGenUtils.getOpClass(stmt, ctx);

		return opClass.equals(actualClass);
	}

	private boolean matchesOpCodeCondition(RewriterRule rule, RewriterStatement stmt, final RuleContext ctx) {
		String opType = (String) conditionValue;
		String actualOpType = CodeGenUtils.getOpCode(stmt, ctx);

		return actualOpType.equals(opType);
	}


	public static CodeGenCondition conditionalDataType(RewriterStatement stmt, List<Integer> i, RewriterStatement representant, final RuleContext ctx) {
		ConditionDataType cdt = stmt.getResultingDataType(ctx).equals("MATRIX") ? ConditionDataType.MATRIX : ConditionDataType.SCALAR;
		return new CodeGenCondition(ConditionType.DATA_TYPE, cdt, i, representant, ctx);
	}

	public static CodeGenCondition conditionalOpClass(RewriterStatement op, List<Integer> i, RewriterStatement representant, final RuleContext ctx) {
		String opClass = CodeGenUtils.getOpClass(op, ctx);
		return new CodeGenCondition(ConditionType.OP_CLASS, opClass, i, representant, ctx);
	}

	public static boolean canGenerateOpClassCheck(RewriterStatement op, final RuleContext ctx) {
		return !op.isDataOrigin();
	}

	public static CodeGenCondition conditionalOpCode(RewriterStatement op, List<Integer> i, RewriterStatement representant, final RuleContext ctx) {
		String opCode = CodeGenUtils.getOpCode(op, ctx);
		return new CodeGenCondition(ConditionType.OP_CODE, opCode, i, representant, ctx);
	}

	public static boolean canGenerateOpCodeCheck(RewriterStatement op, final RuleContext ctx) {
		return !op.isDataOrigin();
	}

	public static CodeGenCondition conditionalInputSize(int inputSize, List<Integer> i, RewriterStatement representant, final RuleContext ctx) {
		return new CodeGenCondition(ConditionType.NUM_INPUTS, inputSize, i, representant, ctx);
	}

	public static boolean canGenerateInputSizeCheck(RewriterStatement op, final RuleContext ctx) {
		return !op.isDataOrigin();
	}

	public static CodeGenCondition conditionalElse(List<Object> l, List<Integer> relativeChildPath, RewriterStatement representant, final RuleContext ctx) {
		CodeGenCondition cond = new CodeGenCondition(ConditionType.ELSE, null, relativeChildPath, representant, ctx);
		cond.rulesIf = l;
		return cond;
	}

	public static String getSelectionString(List<CodeGenCondition> conds, int indentation, Map<RewriterRule, String> ruleFunctionMappings, final RuleContext ctx) {
		StringBuilder sb = new StringBuilder();
		buildSelection(sb, conds, indentation, ruleFunctionMappings, ctx);
		return sb.toString();
	}

	public static void buildSelection(StringBuilder sb, List<CodeGenCondition> conds, int indentation, Map<RewriterRule, String> ruleFunctionMappings, final RuleContext ctx) {
		if (conds.isEmpty())
			return;

		CodeGenCondition firstCond = conds.get(0);

		if (firstCond.conditionType == ConditionType.ELSE) {
			List<CodeGenCondition> nestedCondition = firstCond.rulesIf.stream().filter(o -> o instanceof CodeGenCondition).map(o -> (CodeGenCondition)o).collect(Collectors.toList());
			buildSelection(sb, nestedCondition, indentation, ruleFunctionMappings, ctx);
			if (nestedCondition.isEmpty()) {
				List<Tuple2<RewriterRule, RewriterStatement>> cur = firstCond.rulesIf.stream().map(o -> (Tuple2<RewriterRule, RewriterStatement>)o).collect(Collectors.toList());

				for (Tuple2<RewriterRule, RewriterStatement> t : cur) {
					String fMapping = ruleFunctionMappings.get(t._1);
					if (fMapping != null) {
						RewriterCodeGen.indent(indentation, sb);
						sb.append("hi = ");
						sb.append(fMapping);
						sb.append("(hi);");
						sb.append("\n");
					}
				}
			}
			return;
		}

		RewriterCodeGen.indent(indentation, sb);
		sb.append("if ( ");
		firstCond.buildConditionCheck(sb, ctx);
		sb.append(" ) {\n");
		List<CodeGenCondition> nestedCondition = firstCond.rulesIf.stream().filter(o -> o instanceof CodeGenCondition).map(o -> (CodeGenCondition)o).collect(Collectors.toList());
		buildSelection(sb, nestedCondition, indentation + 1, ruleFunctionMappings, ctx);

		if (nestedCondition.isEmpty()) {
			List<Tuple2<RewriterRule, RewriterStatement>> cur = firstCond.rulesIf.stream().map(o -> (Tuple2<RewriterRule, RewriterStatement>)o).collect(Collectors.toList());

			for (Tuple2<RewriterRule, RewriterStatement> t : cur) {
				String fMapping = ruleFunctionMappings.get(t._1);
				if (fMapping != null) {
					RewriterCodeGen.indent(indentation, sb);
					sb.append("hi = ");
					sb.append(fMapping);
					sb.append("(hi);");
					sb.append("\n");
				}
			}
		}

		RewriterCodeGen.indent(indentation, sb);
		sb.append("}");

		for (CodeGenCondition cond : conds.subList(1, conds.size())) {
			if (cond.conditionType == ConditionType.ELSE) {
				sb.append(" else {\n");
			} else {
				sb.append(" else if ( ");
				cond.buildConditionCheck(sb, ctx);
				sb.append(" ) {\n");
			}

			List<CodeGenCondition> mNestedCondition = cond.rulesIf.stream().filter(o -> o instanceof CodeGenCondition).map(o -> (CodeGenCondition)o).collect(Collectors.toList());
			buildSelection(sb, mNestedCondition, indentation + 1, ruleFunctionMappings, ctx);

			if (mNestedCondition.isEmpty()) {
				List<Tuple2<RewriterRule, RewriterStatement>> cur = cond.rulesIf.stream().map(o -> (Tuple2<RewriterRule, RewriterStatement>)o).collect(Collectors.toList());

				for (Tuple2<RewriterRule, RewriterStatement> t : cur) {
					String fMapping = ruleFunctionMappings.get(t._1);
					if (fMapping != null) {
						RewriterCodeGen.indent(indentation, sb);
						sb.append("hi = ");
						sb.append(fMapping);
						sb.append("(hi);");
						sb.append("\n");
					}
				}
			}

			RewriterCodeGen.indent(indentation, sb);
			sb.append("}");
		}

		sb.append("\n");
	}
}
