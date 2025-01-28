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

package org.apache.sysds.hops.rewriter.codegen;

import org.apache.sysds.hops.rewriter.RewriterDataType;
import org.apache.sysds.hops.rewriter.rule.RewriterRule;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.hops.rewriter.utils.CodeGenUtils;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
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
	private List<RewriterStatement> applyAnyway;
	private List<Integer> relativeChildPath;
	private RewriterStatement representant;

	private CodeGenCondition(ConditionType cType, Object cValue, List<Integer> relativeChildPath, RewriterStatement representant, final RuleContext ctx) {
		conditionType = cType;
		conditionValue = cValue;
		rulesIf = new ArrayList<>();
		applyAnyway = new ArrayList<>();
		this.relativeChildPath = relativeChildPath;
		this.representant = representant;

		if (conditionType != ConditionType.ELSE)
			buildConditionCheck(new StringBuilder(), ctx);
	}

	public static List<CodeGenCondition> buildCondition(List<RewriterRule> rules, int maxNumRules, final RuleContext ctx) {
		return buildCondition(rules, 3, maxNumRules, ctx);
	}

	public static List<CodeGenCondition> buildCondition(List<RewriterRule> rules, int maxDepth, int maxNumRules, final RuleContext ctx) {
		if (rules.isEmpty())
			return Collections.emptyList();
		List<Object> transformed = rules.stream().map(rule -> new Tuple2<RewriterRule, RewriterStatement>(rule, rule.getStmt1())).collect(Collectors.toList());
		List<Object> out = populateLayerRecursively(transformed, Collections.emptyList(), new LinkedList<>(), maxDepth, maxNumRules, ctx);
		List<CodeGenCondition> cond = out.stream().filter(o -> o instanceof CodeGenCondition).map(o -> ((CodeGenCondition)o)).collect(Collectors.toList());
		return cond.isEmpty() ? List.of(conditionalElse(transformed, Collections.emptyList(), ((Tuple2<RewriterRule, RewriterStatement>) transformed.get(0))._2, ctx)) : cond;
	}

	private static List<Object> populateLayerRecursively(List<Object> rules, List<Integer> relativeChildPath, Queue<Tuple2<List<Object>, List<Integer>>> queue, int maxDepth, int maxNumRules, final RuleContext ctx) {
		if (rules.size() <= maxNumRules || maxDepth == 0)
			return rules;

		List<Object> out = populateDataTypeLayer(rules, relativeChildPath, ctx);

		for (int i = 0; i < out.size(); i++) {
			CodeGenCondition c = (CodeGenCondition) out.get(i);

			if (c.rulesIf.size() <= maxNumRules)
				continue;

			c.rulesIf = populateOpClassLayer(c.rulesIf, relativeChildPath, ctx);

			for (int j = 0; j < c.rulesIf.size(); j++) {
				CodeGenCondition c2 = (CodeGenCondition) c.rulesIf.get(j);

				if (c2.rulesIf.size() <= maxNumRules)
					continue;

				c2.rulesIf = populateOpCodeLayer(c2.rulesIf, relativeChildPath, ctx);

				for (int k = 0; k < c2.rulesIf.size(); k++) {
					CodeGenCondition c3 = (CodeGenCondition) c2.rulesIf.get(k);

					if (c3.rulesIf.size() <= maxNumRules)
						continue;

					c3.rulesIf = populateInputSizeLayer(c3.rulesIf, relativeChildPath, ctx);

					for (int l = 0; l < c3.rulesIf.size(); l++) {
						CodeGenCondition c4 = (CodeGenCondition) c3.rulesIf.get(l);

						if (((Tuple2<RewriterRule, RewriterStatement>) c4.rulesIf.get(0))._2 == null)
							continue;

						final int maxIndex = ((Tuple2<RewriterRule, RewriterStatement>) c4.rulesIf.get(0))._2.getOperands().size();
						Set<RewriterRule> activeRules = c4.rulesIf.stream().map(o -> ((Tuple2<RewriterRule, RewriterStatement>) o)._1).collect(Collectors.toSet());
						Queue<Tuple2<List<Object>, List<Integer>>> mQueue = new LinkedList<>();

						for (Tuple2<List<Object>, List<Integer>> t : queue) {
							List<Object> mObj = new ArrayList<>();
							for (Object o : t._1) {
								if (activeRules.contains(((Tuple2<RewriterRule, RewriterStatement>) o)._1))
									mObj.add(o);
							}

							if (!mObj.isEmpty())
								mQueue.add(new Tuple2<>(mObj, t._2));
						}

						for (int idx = 0; idx < maxIndex; idx++) {
							final int mIdx = idx;
							final List<Integer> newRelativeChildPath = new ArrayList<>(relativeChildPath);
							newRelativeChildPath.add(mIdx);
							List<Object> mList = new ArrayList<>();
							mQueue.add(new Tuple2<>(mList, newRelativeChildPath));

							c4.rulesIf.forEach(o -> {
								Tuple2<RewriterRule, RewriterStatement> t = (Tuple2<RewriterRule, RewriterStatement>) o;
								mList.add(new Tuple2<RewriterRule, RewriterStatement>(t._1, (t._2 == null ? null : (t._2.getOperands().isEmpty() ? null : t._2.getChild(mIdx)))));
							});
						}

						if (!mQueue.isEmpty()) {
							Tuple2<List<Object>, List<Integer>> next = mQueue.poll();
							c4.rulesIf = populateLayerRecursively(next._1, next._2(), mQueue, maxDepth-1, maxNumRules, ctx);
						}
					}
				}
			}
		}

		return out;
	}

	private static boolean validateSizeMaintenance(List<Object> rules, List<Object> generatedConditions) {
		int origSize = rules.size();
		int newSize = generatedConditions.stream().mapToInt(o -> ((CodeGenCondition)o).rulesIf.size()).sum();
		return origSize <= newSize;
	}

	private static List<Object> populateDataTypeLayer(List<Object> rules, List<Integer> relativeChildPath, final RuleContext ctx) {
		List<Object> conds = new ArrayList<>();
		List<Tuple2<RewriterRule, RewriterStatement>> defer = new ArrayList<>();

		//System.out.println("=====");

		for (Object o : rules) {
			Tuple2<RewriterRule, RewriterStatement> t = (Tuple2<RewriterRule, RewriterStatement>) o;

			if (t._2 == null) {
				defer.add(t);
				continue;
			}

			if (!conds.stream().anyMatch(cond -> ((CodeGenCondition) cond).insertIfMatches(t, ctx))) {
				CodeGenCondition cond = CodeGenCondition.conditionalDataType(t._2, relativeChildPath, t._2, ctx);
				cond.insertIfMatches(t, ctx);
				conds.add(cond);
				StringBuilder sb = new StringBuilder();
				cond.buildConditionCheck(sb, ctx);
			} else {
				CodeGenCondition condse = (CodeGenCondition) conds.stream().filter(cond -> ((CodeGenCondition) cond).matchesCondition(t._1, t._2, ctx)).findFirst().get();
				StringBuilder sb = new StringBuilder();
				condse.buildConditionCheck(sb, ctx);
			}
		}

		if (!defer.isEmpty()) {
			conds.add(CodeGenCondition.conditionalElse(new ArrayList<>(defer), relativeChildPath, null, ctx));
		}

		for (Tuple2<RewriterRule, RewriterStatement> deferred : defer) {
			for (Object obj : conds)
				((CodeGenCondition) obj).rulesIf.add(deferred);
		}

		if (!validateSizeMaintenance(rules, conds))
			throw new IllegalArgumentException();

		return conds;
	}

	private static List<Object> populateOpClassLayer(List<Object> l, List<Integer> relativeChildPath, final RuleContext ctx) {
		List<Object> conds = new ArrayList<>();
		List<Object> remaining = new ArrayList<>();
		List<Tuple2<RewriterRule, RewriterStatement>> defer = new ArrayList<>();

		for (Object o : l) {
			try {
				Tuple2<RewriterRule, RewriterStatement> t = (Tuple2<RewriterRule, RewriterStatement>) o;

				if (t._2 == null || (t._2 instanceof RewriterDataType && !t._2.isLiteral())) {
					defer.add(t);
					continue;
				}

				if (canGenerateOpClassCheck(t._2, ctx)) {
					if (!conds.stream().anyMatch(cond -> ((CodeGenCondition) cond).insertIfMatches(t, ctx))) {
						CodeGenCondition cond = CodeGenCondition.conditionalOpClass(t._2, relativeChildPath, t._2, ctx);
						cond.insertIfMatches(t, ctx);
						conds.add(cond);
					}
				} else {
					remaining.add(t);
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		remaining.addAll(defer);

		for (Tuple2<RewriterRule, RewriterStatement> deferred : defer) {
			for (Object obj : conds)
				((CodeGenCondition) obj).rulesIf.add(deferred);
		}

		if (!remaining.isEmpty()) {
			conds.add(CodeGenCondition.conditionalElse(remaining, relativeChildPath, ((Tuple2<RewriterRule, RewriterStatement>) remaining.get(0))._2, ctx));
		}

		return conds;
	}

	private static List<Object> populateOpCodeLayer(List<Object> l, List<Integer> relativeChildPath, final RuleContext ctx) {
		List<Object> conds = new ArrayList<>();
		List<Object> remaining = new ArrayList<>();
		List<Tuple2<RewriterRule, RewriterStatement>> defer = new ArrayList<>();

		for (Object o : l) {
			Tuple2<RewriterRule, RewriterStatement> t = (Tuple2<RewriterRule, RewriterStatement>) o;

			if (t._2 == null || (t._2 instanceof RewriterDataType && !t._2.isLiteral())) {
				defer.add(t);
				continue;
			}

			if (canGenerateOpCodeCheck(t._2, ctx)) {
				if (!conds.stream().anyMatch(cond -> ((CodeGenCondition) cond).insertIfMatches(t, ctx))) {
					CodeGenCondition cond = CodeGenCondition.conditionalOpCode(t._2, relativeChildPath, t._2, ctx);
					cond.insertIfMatches(t, ctx);
					conds.add(cond);
				}
			} else if (t._2 instanceof RewriterDataType && !t._2.isLiteral()) {
				// Then we must add it to all conditions
				for (Object obj : conds)
					((CodeGenCondition) obj).rulesIf.add(t);
			} else {
				remaining.add(t);
			}
		}

		remaining.addAll(defer);

		for (Tuple2<RewriterRule, RewriterStatement> deferred : defer) {
			for (Object obj : conds)
				((CodeGenCondition) obj).rulesIf.add(deferred);
		}

		if (!remaining.isEmpty()) {
			conds.add(CodeGenCondition.conditionalElse(remaining, relativeChildPath, ((Tuple2<RewriterRule, RewriterStatement>) remaining.get(0))._2, ctx));
		}

		if (!validateSizeMaintenance(l, conds))
			throw new IllegalArgumentException();

		return conds;
	}

	private static List<Object> populateInputSizeLayer(List<Object> l, List<Integer> relativeChildPath, final RuleContext ctx) {
		List<Object> conds = new ArrayList<>();
		List<Object> remaining = new ArrayList<>();
		List<Tuple2<RewriterRule, RewriterStatement>> defer = new ArrayList<>();

		for (Object o : l) {
			Tuple2<RewriterRule, RewriterStatement> t = (Tuple2<RewriterRule, RewriterStatement>) o;

			if (t._2 == null || (t._2 instanceof RewriterDataType && !t._2.isLiteral())) {
				defer.add(t);
				continue;
			}

			if (canGenerateInputSizeCheck(t._2, ctx)) {
				if (!conds.stream().anyMatch(cond -> ((CodeGenCondition) cond).insertIfMatches(t, ctx))) {
					CodeGenCondition cond = CodeGenCondition.conditionalInputSize(t._2.getOperands().size(), relativeChildPath, t._2, ctx);
					cond.insertIfMatches(t, ctx);
					conds.add(cond);
				}
			} else if (t._2 instanceof RewriterDataType && !t._2.isLiteral()) {
				// Then we must add it to all conditions
				for (Object obj : conds)
					((CodeGenCondition) obj).rulesIf.add(t);
			} else {
				remaining.add(t);
			}
		}

		remaining.addAll(defer);

		for (Tuple2<RewriterRule, RewriterStatement> deferred : defer) {
			for (Object obj : conds)
				((CodeGenCondition) obj).rulesIf.add(deferred);
		}

		if (!remaining.isEmpty()) {
			conds.add(CodeGenCondition.conditionalElse(remaining, relativeChildPath, ((Tuple2<RewriterRule, RewriterStatement>) remaining.get(0))._2, ctx));
		}

		if (!validateSizeMaintenance(l, conds))
			throw new IllegalArgumentException();

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
					// Some type casting
					sb.append("(( ");
					sb.append(CodeGenUtils.getOpClass(representant, ctx));
					sb.append(" ) ");
					sb.append(hopVar);
					sb.append(" )");
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
						sb.append("(hi); // ");
						sb.append(t._1.toString());
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

		if (firstCond.conditionType == ConditionType.NUM_INPUTS) {
			int numInputs = (int)firstCond.conditionValue;

			for (int i = 0; i < numInputs; i++) {
				RewriterCodeGen.indent(indentation + 1, sb);
				sb.append("Hop ");
				sb.append(firstCond.getVarName());
				sb.append("_");
				sb.append(i);
				sb.append(" = ");
				sb.append(firstCond.getVarName());
				sb.append(".getInput(");
				sb.append(i);
				sb.append(");\n");
			}
		}

		List<CodeGenCondition> nestedCondition = firstCond.rulesIf.stream().filter(o -> o instanceof CodeGenCondition).map(o -> (CodeGenCondition)o).collect(Collectors.toList());
		buildSelection(sb, nestedCondition, indentation + 1, ruleFunctionMappings, ctx);

		if (nestedCondition.isEmpty()) {
			List<Tuple2<RewriterRule, RewriterStatement>> cur = firstCond.rulesIf.stream().map(o -> (Tuple2<RewriterRule, RewriterStatement>)o).collect(Collectors.toList());

			if (cur.isEmpty())
				throw new IllegalArgumentException(firstCond.rulesIf.toString());

			for (Tuple2<RewriterRule, RewriterStatement> t : cur) {
				String fMapping = ruleFunctionMappings.get(t._1);
				if (fMapping != null) {
					RewriterCodeGen.indent(indentation + 1, sb);
					sb.append("hi = ");
					sb.append(fMapping);
					sb.append("(hi); // ");
					sb.append(t._1.toString());
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

			if (cond.conditionType == ConditionType.NUM_INPUTS) {
				int numInputs = (int)cond.conditionValue;

				for (int i = 0; i < numInputs; i++) {
					RewriterCodeGen.indent(indentation + 1, sb);
					sb.append("Hop ");
					sb.append(cond.getVarName());
					sb.append("_");
					sb.append(i);
					sb.append(" = ");
					sb.append(cond.getVarName());
					sb.append(".getInput(");
					sb.append(i);
					sb.append(");");
				}
			}

			List<CodeGenCondition> mNestedCondition = cond.rulesIf.stream().filter(o -> o instanceof CodeGenCondition).map(o -> (CodeGenCondition)o).collect(Collectors.toList());
			buildSelection(sb, mNestedCondition, indentation + 1, ruleFunctionMappings, ctx);

			if (mNestedCondition.isEmpty()) {
				List<Tuple2<RewriterRule, RewriterStatement>> cur = cond.rulesIf.stream().map(o -> (Tuple2<RewriterRule, RewriterStatement>)o).collect(Collectors.toList());

				if (cur.isEmpty())
					throw new IllegalArgumentException(cond.rulesIf.toString());

				for (Tuple2<RewriterRule, RewriterStatement> t : cur) {
					String fMapping = ruleFunctionMappings.get(t._1);
					if (fMapping != null) {
						RewriterCodeGen.indent(indentation + 1, sb);
						sb.append("hi = ");
						sb.append(fMapping);
						sb.append("(hi); // ");
						sb.append(t._1.toString());
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
