package org.apache.sysds.hops.rewriter.codegen;

import org.apache.sysds.hops.rewriter.RewriterRule;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RuleContext;
import scala.Tuple2;
import scala.Tuple3;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
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

	private CodeGenCondition(ConditionType cType, Object cValue) {
		conditionType = cType;
		conditionValue = cValue;
		rulesIf = new ArrayList<>();
	}

	public static List<CodeGenCondition> buildCondition(List<RewriterRule> rules, final RuleContext ctx) {
		List<Object> transformed = rules.stream().map(rule -> new Tuple2<RewriterRule, RewriterStatement>(rule, rule.getStmt1())).collect(Collectors.toList());
		List<Object> out = populateLayerRecursively(transformed, Collections.emptyList(), new LinkedList<>(), ctx);
		return out.stream().map(o -> ((CodeGenCondition)o)).collect(Collectors.toList());
	}

	private static List<Object> populateLayerRecursively(List<Object> rules, List<Integer> relativeChildPath, Queue<Tuple2<List<Object>, List<Integer>>> queue, final RuleContext ctx) {
		System.out.println("Current: " + ((Tuple2<RewriterRule, RewriterStatement>) rules.get(0))._2);
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
							c4.rulesIf = populateLayerRecursively(next._1, next._2(), mQueue, ctx);
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
				CodeGenCondition cond = CodeGenCondition.conditionalDataType(t._2, ctx);
				cond.insertIfMatches(t, ctx);
				conds.add(cond);
			}
		}

		return conds;
	}

	private static List<Object> populateOpClassLayer(List<Object> l, List<Integer> relativeChildPath, final RuleContext ctx) {
		List<Object> conds = new ArrayList<>();
		List<Object> remaining = new ArrayList<>();

		for (Object o : l) {
			Tuple2<RewriterRule, RewriterStatement> t = (Tuple2<RewriterRule, RewriterStatement>) o;
			if (canGenerateOpClassCheck(t._2, ctx)) {
				if (!conds.stream().anyMatch(cond -> ((CodeGenCondition) cond).insertIfMatches(t, ctx))) {
					CodeGenCondition cond = CodeGenCondition.conditionalOpClass(t._2, ctx);
					cond.insertIfMatches(t, ctx);
					conds.add(cond);
				}
			} else {
				remaining.add(t);
			}
		}

		if (!remaining.isEmpty()) {
			conds.add(CodeGenCondition.conditionalElse(remaining, ctx));
		}

		return conds;
	}

	private static List<Object> populateOpCodeLayer(List<Object> l, List<Integer> relativeChildPath, final RuleContext ctx) {
		List<Object> conds = new ArrayList<>();
		List<Object> remaining = new ArrayList<>();

		for (Object o : l) {
			Tuple2<RewriterRule, RewriterStatement> t = (Tuple2<RewriterRule, RewriterStatement>) o;
			if (canGenerateOpCodeCheck(t._2, ctx)) {
				if (!conds.stream().anyMatch(cond -> ((CodeGenCondition) cond).insertIfMatches(t, ctx))) {
					CodeGenCondition cond = CodeGenCondition.conditionalOpCode(t._2, ctx);
					cond.insertIfMatches(t, ctx);
					conds.add(cond);
				}
			} else {
				remaining.add(t);
			}
		}

		if (!remaining.isEmpty()) {
			conds.add(CodeGenCondition.conditionalElse(remaining, ctx));
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
					CodeGenCondition cond = CodeGenCondition.conditionalInputSize(t._2.getOperands().size(), ctx);
					cond.insertIfMatches(t, ctx);
					conds.add(cond);
				}
			} else {
				remaining.add(t);
			}
		}

		if (!remaining.isEmpty()) {
			conds.add(CodeGenCondition.conditionalElse(remaining, ctx));
		}

		return conds;
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


	public static CodeGenCondition conditionalDataType(RewriterStatement stmt, final RuleContext ctx) {
		ConditionDataType cdt = stmt.getResultingDataType(ctx).equals("MATRIX") ? ConditionDataType.MATRIX : ConditionDataType.SCALAR;
		return new CodeGenCondition(ConditionType.DATA_TYPE, cdt);
	}

	public static CodeGenCondition conditionalOpClass(RewriterStatement op, final RuleContext ctx) {
		String opClass = CodeGenUtils.getOpClass(op, ctx);
		return new CodeGenCondition(ConditionType.OP_CLASS, opClass);
	}

	public static boolean canGenerateOpClassCheck(RewriterStatement op, final RuleContext ctx) {
		return !op.isDataOrigin();
	}

	public static CodeGenCondition conditionalOpCode(RewriterStatement op, final RuleContext ctx) {
		String opCode = CodeGenUtils.getOpCode(op, ctx);
		return new CodeGenCondition(ConditionType.OP_CODE, opCode);
	}

	public static boolean canGenerateOpCodeCheck(RewriterStatement op, final RuleContext ctx) {
		return !op.isDataOrigin();
	}

	public static CodeGenCondition conditionalInputSize(int inputSize, final RuleContext ctx) {
		return new CodeGenCondition(ConditionType.NUM_INPUTS, inputSize);
	}

	public static boolean canGenerateInputSizeCheck(RewriterStatement op, final RuleContext ctx) {
		return !op.isDataOrigin();
	}

	public static CodeGenCondition conditionalElse(List<Object> l, final RuleContext ctx) {
		return new CodeGenCondition(ConditionType.ELSE, l);
	}
}
