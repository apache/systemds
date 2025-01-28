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

package org.apache.sysds.hops.rewriter.rule;

import org.apache.sysds.hops.rewriter.RewriterDataType;
import org.apache.sysds.hops.rewriter.RewriterInstruction;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Function;

public class RewriterRuleBuilder {
	private final RuleContext ctx;
	private String ruleName = "?";
	private ArrayList<RewriterInstruction> instrSeq = new ArrayList<>();
	private ArrayList<RewriterInstruction> mappingSeq = new ArrayList<>();
	private HashMap<String, RewriterStatement> globalIds = new HashMap<>();
	private HashMap<String, RewriterStatement> instrSeqIds = new HashMap<>();
	private HashMap<String, RewriterStatement> mappingSeqIds = new HashMap<>();
	private HashMap<RewriterStatement, RewriterRule.LinkObject> linksStmt1ToStmt2 = new HashMap<>();
	private ArrayList<Tuple2<RewriterStatement, BiConsumer<RewriterStatement, RewriterStatement.MatchingSubexpression>>> applyStmt1ToStmt2 = new ArrayList<>();
	private HashMap<RewriterStatement, RewriterRule.LinkObject> linksStmt2ToStmt1 = new HashMap<>();
	private ArrayList<Tuple2<RewriterStatement, BiConsumer<RewriterStatement, RewriterStatement.MatchingSubexpression>>> applyStmt2ToStmt1 = new ArrayList<>();
	private RewriterStatement fromRoot = null;
	private RewriterStatement toRoot = null;
	private List<RewriterStatement> multiRuleRoots = null;
	private Function<RewriterStatement.MatchingSubexpression, Boolean> iff1to2 = null;
	private Function<RewriterStatement.MatchingSubexpression, Boolean> iff2to1 = null;
	private boolean isUnidirectional = false;
	private boolean buildSingleDAG = false;

	private RewriterStatement currentStatement = null;
	private boolean mappingState = false;

	private boolean canBeModified = true;

	private Set<RewriterStatement> allowedMultiReferences = Collections.emptySet();
	private boolean allowCombinations = false;

	public RewriterRuleBuilder(final RuleContext ctx) {
		this.ctx = ctx;
	}

	public RewriterRuleBuilder(final RuleContext ctx, String ruleName) {
		this.ctx = ctx;
		this.ruleName = ruleName;
	}

	public RewriterRuleBuilder iff(Function<RewriterStatement.MatchingSubexpression, Boolean> iff, boolean forward) {
		if (buildSingleDAG)
			throw new IllegalArgumentException();

		if (forward)
			iff1to2 = iff;
		else
			iff2to1 = iff;

		return this;
	}

	public RewriterRuleBuilder parseGlobalVars(String globalVarDefinition) {
		if (!canBeModified)
			throw new IllegalArgumentException();
		RewriterUtils.parseDataTypes(globalVarDefinition, globalIds, ctx);
		return this;
	}

	public RewriterRuleBuilder intLiteral(String id, int value) {
		return intLiteral(id, value, "global");
	}

	public RewriterRuleBuilder intLiteral(String id, int value, String scope) {
		switch (scope) {
			case "global":
				globalIds.put(id, new RewriterDataType().as(id).ofType("INT").asLiteral(value));
				break;
			case "from":
				instrSeqIds.put(id, new RewriterDataType().as(id).ofType("INT").asLiteral(value));
				break;
			case "to":
				mappingSeqIds.put(id, new RewriterDataType().as(id).ofType("INT").asLiteral(value));
				break;
		}

		return this;
	}

	public RewriterRuleBuilder parseGlobalStatementAsVariable(String varName, String expr) {
		return parseGlobalStatementAsVariable(varName, expr, new HashMap<>());
	}

	public RewriterRuleBuilder parseGlobalStatementAsVariable(String varName, String expr, HashMap<Integer, RewriterStatement> refMap) {
		if (!canBeModified)
			throw new IllegalArgumentException();

		RewriterStatement parsed = RewriterUtils.parseExpression(expr, refMap, globalIds, ctx);
		parsed.consolidate(ctx);
		globalIds.put(varName, parsed);
		return this;
	}

	public RewriterRuleBuilder withParsedStatement(String stmt) {
		return withParsedStatement(stmt, new HashMap<>());
	}

	public RewriterRuleBuilder withParsedStatement(String stmt, HashMap<Integer, RewriterStatement> refMap) {
		if (!canBeModified)
			throw new IllegalArgumentException();
		fromRoot = RewriterUtils.parseExpression(stmt, refMap, globalIds, ctx);
		fromRoot.forEachPreOrderWithDuplicates(el -> {
			instrSeqIds.put(el.getId(), el);
			return true;
		});
		return this;
	}

	public RewriterRuleBuilder toParsedStatement(String stmt) {
		return toParsedStatement(stmt, new HashMap<>());
	}

	public RewriterRuleBuilder toParsedStatement(String stmt, HashMap<Integer, RewriterStatement> refMap) {
		if (!canBeModified)
			throw new IllegalArgumentException();
		mappingState = true;
		toRoot = RewriterUtils.parseExpression(stmt, refMap, globalIds, ctx);
		toRoot.forEachPreOrderWithDuplicates(el -> {
			mappingSeqIds.put(el.getId(), el);
			return true;
		});
		return this;
	}

	public RewriterRuleBuilder prepare() {
		if (!canBeModified)
			return this;
		if (buildSingleDAG) {
			getCurrentInstruction().consolidate(ctx);
			fromRoot.prepareForHashing();
			fromRoot.recomputeHashCodes(ctx);
			canBeModified = false;
		} else {
			if (getCurrentInstruction() != null)
				getCurrentInstruction().consolidate(ctx);
			fromRoot.prepareForHashing();
			if (toRoot != null)
				toRoot.prepareForHashing();
			else
				multiRuleRoots.forEach(RewriterStatement::prepareForHashing);
			fromRoot.recomputeHashCodes(ctx);
			if (toRoot != null)
				toRoot.recomputeHashCodes(ctx);
			else
				multiRuleRoots.forEach(rt -> rt.recomputeHashCodes(ctx));
			canBeModified = false;
		}

		return this;
	}

	public RewriterRule build() {
		if (buildSingleDAG)
			throw new IllegalArgumentException("Cannot build a rule if DAG was specified");
		if (!mappingState)
			throw new IllegalArgumentException("No mapping expression");
		if (fromRoot == null)
			throw new IllegalArgumentException("From-root statement cannot be null");
		if (toRoot == null && multiRuleRoots == null)
			throw new IllegalArgumentException("To-root statement cannot be null");
		if (getCurrentInstruction() != null)
			getCurrentInstruction().consolidate(ctx);
		prepare();
		RewriterRule rule = new RewriterRule(ctx, ruleName, fromRoot, toRoot, isUnidirectional, linksStmt1ToStmt2, linksStmt2ToStmt1, iff1to2, iff2to1, applyStmt1ToStmt2, applyStmt2ToStmt1);
		rule.setAllowedMultiReferences(allowedMultiReferences, allowCombinations);
		if (multiRuleRoots != null)
			rule.setConditional(multiRuleRoots);
		return rule;
	}

	public RewriterStatement buildDAG() {
		if (!buildSingleDAG)
			throw new IllegalArgumentException("Cannot build a DAG if rule was specified");
		prepare();
		return fromRoot;
	}

	public RewriterRuleBuilder asDAGBuilder() {
		buildSingleDAG = true;
		return this;
	}

	public RewriterRuleBuilder setUnidirectional(boolean unidirectional) {
		this.isUnidirectional = unidirectional;
		return this;
	}

	public RewriterStatement getCurrentInstruction() {
		if (mappingState)
			if (mappingSeq.size() > 0)
				return mappingSeq.get(mappingSeq.size()-1);
			else if (toRoot != null)
				return toRoot;
			else if (multiRuleRoots != null)
				return multiRuleRoots.get(0); // Just as a dummy
			else
				throw new IllegalArgumentException("There is no current instruction in the mapping sequence");
		else
			if (instrSeq.size() > 0)
				return instrSeq.get(instrSeq.size()-1);
			else if (fromRoot != null)
				return fromRoot;
			else
				throw new IllegalArgumentException("There is no current instruction in the instruction sequence");
	}

	public RewriterDataType getCurrentOperand() {
		if (currentStatement instanceof RewriterDataType)
			return (RewriterDataType)currentStatement;
		else
			throw new IllegalArgumentException("The current operand is not a data type");
	}

	public RewriterRuleBuilder withDataType(String id, String type) {
		withDataType(id, type, null);
		return this;
	}

	public RewriterRuleBuilder withDataType(String id, String type, Object literal) {
		if (!canBeModified)
			throw new IllegalArgumentException("The DAG is final and cannot be modified");
		if (!instrSeq.isEmpty())
			throw new IllegalArgumentException("To define a single data type, the instruction sequence must be empty");
		fromRoot = new RewriterDataType().ofType(type).asLiteral(literal).as(id);
		storeVar(fromRoot);
		return this;
	}

	public RewriterRuleBuilder withInstruction(String instr) {
		if (!canBeModified)
			throw new IllegalArgumentException("The DAG is final and cannot be modified");
		if (mappingState)
			throw new IllegalArgumentException("Cannot add an instruction when a mapping instruction was already defined");
		if (instrSeq.size() > 0)
			getCurrentInstruction().consolidate(ctx);
		instrSeq.add(new RewriterInstruction().withInstruction(instr));
		return this;
	}

	public RewriterRuleBuilder completeRule(RewriterStatement from, RewriterStatement to) {
		if (!canBeModified)
			throw new IllegalArgumentException("The DAG is final and cannot be modified");
		if (mappingState)
			throw new IllegalArgumentException("Cannot add an instruction when a mapping instruction was already defined");
		this.fromRoot = from;
		this.toRoot = to;
		this.mappingState = true;
		return this;
	}

	public RewriterRuleBuilder completeConditionalRule(RewriterStatement from, List<RewriterStatement> to) {
		if (!canBeModified)
			throw new IllegalArgumentException("The DAG is final and cannot be modified");
		if (mappingState)
			throw new IllegalArgumentException("Cannot add an instruction when a mapping instruction was already defined");
		this.fromRoot = from;
		this.multiRuleRoots = to;
		this.mappingState = true;
		return this;
	}

	public RewriterRuleBuilder withAllowedMultiRefs(Set<RewriterStatement> allowedMultiRefs, boolean allowCombinations) {
		if (!canBeModified)
			throw new IllegalArgumentException("The DAG is final and cannot be modified");
		if (!mappingState)
			throw new IllegalArgumentException("Cannot add an instruction when a mapping instruction was already defined");

		this.allowedMultiReferences = allowedMultiRefs;
		this.allowCombinations = allowCombinations;
		return this;
	}

	public RewriterRuleBuilder withOps(RewriterDataType... operands) {
		if (!canBeModified)
			throw new IllegalArgumentException("The DAG is final and cannot be modified");
		((RewriterInstruction)getCurrentInstruction()).withOps(operands);
		currentStatement = null;
		return this;
	}

	public RewriterRuleBuilder addOp(String id) {
		if (!canBeModified)
			throw new IllegalArgumentException("The DAG is final and cannot be modified");
		RewriterDataType dt = new RewriterDataType().as(id);
		storeVar(dt);
		((RewriterInstruction)getCurrentInstruction()).addOp(dt);
		if (currentStatement != null)
			currentStatement.consolidate(ctx);
		currentStatement = dt;
		return this;
	}

	public RewriterRuleBuilder addDynamicOpListInstr(String id, String type, boolean fromInstr, String... ops) {
		if (!canBeModified)
			throw new IllegalArgumentException("The DAG is final and cannot be modified");

		if (fromInstr)
			withInstruction("argList");
		else
			toInstruction("argList");

		if (ops.length == 0 && type.endsWith("...")) {
			// Add one placeholder operand to implicitly determine the data type
			addOp(UUID.randomUUID().toString()).ofType(type.substring(0, type.length()-3));
		} else {
			for (String op : ops)
				addExistingOp(op);
		}

		as(id);
		return this;
	}

	public RewriterRuleBuilder asLiteral(Object literal) {
		if (!canBeModified)
			throw new IllegalArgumentException("The DAG is final and cannot be modified");
		getCurrentOperand().asLiteral(literal);
		return this;
	}

	public RewriterRuleBuilder as(String id) {
		if (!canBeModified)
			throw new IllegalArgumentException("The DAG is final and cannot be modified");
		getCurrentInstruction().as(id);
		currentVars().put(id, getCurrentInstruction());
		storeVar(getCurrentInstruction());
		return this;
	}

	public RewriterRuleBuilder asRootInstruction() {
		if (!canBeModified)
			throw new IllegalArgumentException("The DAG is final and cannot be modified");
		if (mappingState) {
			if (toRoot != null)
				throw new IllegalArgumentException("Cannot have more than one root instruction");
			toRoot = getCurrentInstruction().as("result");
			mappingSeqIds.put("result", toRoot);
		} else {
			if (fromRoot != null)
				throw new IllegalArgumentException("Cannot have more than one root instruction");
			fromRoot = getCurrentInstruction().as("result");
			instrSeqIds.put("result", fromRoot);
		}
		return this;
	}

	public RewriterRuleBuilder addExistingOp(String id) {
		if (!canBeModified)
			throw new IllegalArgumentException("The DAG is final and cannot be modified");
		RewriterStatement operand = findVar(id);

		if (operand == null)
			throw new IllegalArgumentException("Operand with id '" + id + "' does not exist");

		if (currentStatement != null)
			currentStatement.consolidate(ctx);

		currentStatement = operand;
		((RewriterInstruction)getCurrentInstruction()).addOp(operand);

		return this;
	}

	public RewriterRuleBuilder ofType(String type) {
		if (!canBeModified)
			throw new IllegalArgumentException("The DAG is final and cannot be modified");
		getCurrentOperand().ofType(type);
		return this;
	}

	public RewriterRuleBuilder instrMeta(String key, Object value) {
		if (!canBeModified)
			throw new IllegalArgumentException("The DAG is final and cannot be modified");
		getCurrentInstruction().putMeta(key, value);
		return this;
	}

	public RewriterRuleBuilder operandMeta(String key, Object value) {
		if (!canBeModified)
			throw new IllegalArgumentException("The DAG is final and cannot be modified");
		getCurrentOperand().putMeta(key, value);
		return this;
	}

	public RewriterRuleBuilder toInstruction(String instr) {
		if (!canBeModified)
			throw new IllegalArgumentException("The DAG is final and cannot be modified");
		if (buildSingleDAG)
			throw new IllegalArgumentException("Cannot create a mapping instruction when building a single DAG");
		getCurrentInstruction().consolidate(ctx);
		mappingSeq.add(new RewriterInstruction().withInstruction(instr));
		mappingState = true;
		return this;
	}

	public RewriterRuleBuilder linkUnidirectional(String idFrom, String idTo, Consumer<RewriterRule.ExplicitLink> transferFunction, boolean forward) {
		return linkManyUnidirectional(idFrom, List.of(idTo), transferFunction, forward);
	}

	public RewriterRuleBuilder linkManyUnidirectional(String idFrom, List<String> idsTo, Consumer<RewriterRule.ExplicitLink> transferFunction, boolean forward) {
		prepare();
		RewriterStatement stmt1 = forward ? instrSeqIds.get(idFrom) : mappingSeqIds.get(idFrom);
		if (stmt1 == null)
			stmt1 = globalIds.get(idFrom);
		if (stmt1 == null)
			throw new IllegalArgumentException("Could not find instruction id: " + idFrom);
		if (!stmt1.isConsolidated())
			stmt1.consolidate(ctx);

		List<RewriterStatement> stmts2 = new ArrayList<>();

		for (String idTo : idsTo) {
			RewriterStatement stmt2 = forward ? mappingSeqIds.get(idTo) : instrSeqIds.get(idTo);
			if (stmt2 == null)
				stmt2 = globalIds.get(idTo);
			if (stmt2 == null)
				throw new IllegalArgumentException("Could not find instruction id: " + idTo);
			if (!stmt2.isConsolidated())
				stmt2.consolidate(ctx);

			stmts2.add(stmt2);
		}

		HashMap<RewriterStatement, RewriterRule.LinkObject> links = forward ? linksStmt1ToStmt2 : linksStmt2ToStmt1;

		RewriterRule.LinkObject lnk = new RewriterRule.LinkObject(stmts2, transferFunction);

		if (links.containsKey(stmt1) || links.containsValue(lnk))
			throw new IllegalArgumentException("Key or value already exists in explicit link map.");

		links.put(stmt1, lnk);
		return this;
	}

	public RewriterRuleBuilder link(String id, String id2, Consumer<RewriterRule.ExplicitLink> transferFunction) {
		linkUnidirectional(id, id2, transferFunction, true);
		linkUnidirectional(id2, id, transferFunction, false);
		return this;
	}

	public RewriterRuleBuilder apply(String id, Consumer<RewriterStatement> applicationFunction, boolean forward) {
		return apply(id, (stmt, match) -> applicationFunction.accept(stmt), forward);
	}

	public RewriterRuleBuilder apply(String id, BiConsumer<RewriterStatement, RewriterStatement.MatchingSubexpression> applicationFunction, boolean forward) {
		prepare();
		RewriterStatement stmt1 = forward ?  mappingSeqIds.get(id) : instrSeqIds.get(id);
		if (stmt1 == null)
			stmt1 = globalIds.get(id);
		if (stmt1 == null)
			throw new IllegalArgumentException("Could not find instruction id: " + id);
		if (!stmt1.isConsolidated())
			stmt1.consolidate(ctx);

		if (forward)
			applyStmt1ToStmt2.add(new Tuple2<>(stmt1, applicationFunction));
		else
			applyStmt2ToStmt1.add(new Tuple2<>(stmt1, applicationFunction));

		return this;
	}

	public RewriterRuleBuilder toDataType(String id, String type) {
		toDataType(id, type, null);
		return this;
	}

	public RewriterRuleBuilder toDataType(String id, String type, Object literal) {
		if (!mappingSeq.isEmpty())
			throw new IllegalArgumentException("To define a single data type, the mapping sequence must be empty");
		toRoot = new RewriterDataType().ofType(type).asLiteral(literal).as(id);
		storeVar(toRoot);
		return this;
	}

	private HashMap<String, RewriterStatement> currentVars() {
		return mappingState ? mappingSeqIds : instrSeqIds;
	}

	private RewriterStatement findVar(String id) {
		RewriterStatement stmt = null;

		if (mappingState) {
			stmt = mappingSeqIds.get(id);
			if (stmt != null)
				return stmt;
		} else {
			stmt = instrSeqIds.get(id);
			if (stmt != null)
				return stmt;
		}
		return globalIds.get(id);
	}

	private void storeVar(RewriterStatement var) {
		if (var.getId() == null)
			throw new IllegalArgumentException("The id of a statement cannot be null!");

		if (mappingState) {
			mappingSeqIds.put(var.getId(), var);
		} else {
			if (var instanceof RewriterDataType)
				globalIds.put(var.getId(), var);
			else
				instrSeqIds.put(var.getId(), var);
		}
	}
}
