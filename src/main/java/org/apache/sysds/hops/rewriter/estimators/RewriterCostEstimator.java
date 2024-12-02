package org.apache.sysds.hops.rewriter.estimators;

import org.apache.commons.lang3.mutable.MutableLong;
import org.apache.commons.lang3.mutable.MutableObject;
import org.apache.sysds.hops.rewriter.RewriterInstruction;
import org.apache.sysds.hops.rewriter.RewriterRule;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.hops.rewriter.assertions.RewriterAssertionUtils;
import org.apache.sysds.hops.rewriter.assertions.RewriterAssertions;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.function.Function;
import java.util.stream.Collectors;

public class RewriterCostEstimator {
	private static final long INSTRUCTION_OVERHEAD = 10;
	private static final long MALLOC_COST = 10000;
	public static final Function<RewriterStatement, Long> DEFAULT_COST_FN = el -> 2000L;

	public static Tuple2<Set<RewriterStatement>, Boolean> determineSingleReferenceRequirement(RewriterRule rule, final RuleContext ctx) {
		MutableObject<RewriterAssertions> assertionRef = new MutableObject<>();
		long fullCost = RewriterCostEstimator.estimateCost(rule.getStmt1(), ctx, assertionRef);
		long maxCost = RewriterCostEstimator.estimateCost(rule.getStmt2(), ctx);
		return RewriterCostEstimator.determineSingleReferenceRequirement(rule.getStmt2(), RewriterCostEstimator.DEFAULT_COST_FN, assertionRef.getValue(), fullCost, maxCost, ctx);
	}

	// Returns all (upmost) sub-DAGs that can have multiple references and true as a second arg if all statements can have multiple references at once
	public static Tuple2<Set<RewriterStatement>, Boolean> determineSingleReferenceRequirement(RewriterStatement root, Function<RewriterStatement, Long> costFn, RewriterAssertions assertions, long fullCost, long maxCost, final RuleContext ctx) {
		if (fullCost >= maxCost)
			return new Tuple2<>(Collections.emptySet(), true);

		List<Tuple2<RewriterStatement, Long>> subDAGCosts = new ArrayList<>();

		root.forEachPreOrder((cur, pred) -> {
			if (pred.isRoot() || !cur.isInstruction())
				return true;

			long cost = estimateCost(cur, costFn, ctx, new MutableObject<>(assertions));

			if (fullCost + cost <= maxCost) {
				subDAGCosts.add(new Tuple2<>(cur, cost));
				return false;
			}

			return true;
		}, true);

		boolean canCombine = true;
		long curCost = fullCost;

		for (Tuple2<RewriterStatement, Long> t : subDAGCosts) {
			curCost += t._2;

			if (curCost > maxCost) {
				canCombine = false;
				break;
			}
		}

		return new Tuple2<>(subDAGCosts.stream().map(t -> t._1).collect(Collectors.toSet()), canCombine);
	}

	public static long estimateCost(RewriterStatement stmt, final RuleContext ctx) {
		return estimateCost(stmt, DEFAULT_COST_FN, ctx);
	}

	public static long estimateCost(RewriterStatement stmt, final RuleContext ctx, MutableObject<RewriterAssertions> assertionRef) {
		return estimateCost(stmt, DEFAULT_COST_FN, ctx, assertionRef);
	}

	public static long estimateCost(RewriterStatement stmt, Function<RewriterStatement, Long> propertyGenerator, final RuleContext ctx) {
		return estimateCost(stmt, propertyGenerator, ctx, null);
	}

	public static long estimateCost(RewriterStatement stmt, Function<RewriterStatement, Long> propertyGenerator, final RuleContext ctx, MutableObject<RewriterAssertions> assertionRef) {
		RewriterAssertions assertions = assertionRef != null && assertionRef.getValue() != null ? assertionRef.getValue() : new RewriterAssertions(ctx);

		if (assertionRef != null)
			assertionRef.setValue(assertions);

		RewriterStatement costFn = propagateCostFunction(stmt, ctx, assertions);
		costFn = assertions.update(costFn);
		// TODO: Something makes this necessary
		costFn = RewriterUtils.foldConstants(costFn, ctx);

		//System.out.println(costFn.toParsableString(ctx));
		//System.out.println(RewriterUtils.foldConstants(costFn, ctx).toParsableString(ctx));

		Map<RewriterStatement, RewriterStatement> map = new HashMap<>();

		//System.out.println("Cost1: " + costFn.toParsableString(ctx));

		costFn.forEachPreOrder((cur, pred) -> {
			for (int i = 0; i < cur.getOperands().size(); i++) {
				RewriterStatement op = cur.getChild(i);

				RewriterStatement mNew = map.get(op);
				if (mNew != null) {
					cur.getOperands().set(i, mNew);
					continue;
				}

				if (op.isEClass()) {
					mNew = RewriterStatement.literal(ctx, propertyGenerator.apply(op));
					map.put(op, mNew);
					cur.getOperands().set(i, mNew);
				} else if (op.isInstruction()) {
					if (op.trueInstruction().equals("ncol") || op.trueInstruction().equals("nrow")) {
						mNew = RewriterStatement.literal(ctx, propertyGenerator.apply(op));
						map.put(op, mNew);
						cur.getOperands().set(i, mNew);
					}
				}
			}

			return true;
		}, false);

		//System.out.println("Cost2: " + costFn.toParsableString(ctx));

		costFn = RewriterUtils.foldConstants(costFn, ctx);

		if (!costFn.isLiteral()) {
			throw new IllegalArgumentException("Cost function must be a literal: " + costFn.toParsableString(ctx) + "\nCorresponding statement:\n" + stmt.toParsableString(ctx));
		}

		return (long)costFn.getLiteral();
	}

	private static RewriterStatement propagateCostFunction(RewriterStatement stmt, final RuleContext ctx, RewriterAssertions assertions) {
		List<RewriterStatement> includedCosts = new ArrayList<>();
		MutableLong instructionOverhead = new MutableLong(0);

		stmt.forEachPostOrder((cur, pred) -> {
			if (!(cur instanceof RewriterInstruction))
				return;

			computeCostOf((RewriterInstruction) cur, ctx, includedCosts, assertions, instructionOverhead);
			instructionOverhead.add(INSTRUCTION_OVERHEAD);
		}, false);

		includedCosts.add(RewriterStatement.literal(ctx, instructionOverhead.longValue()));

		RewriterStatement argList = RewriterStatement.argList(ctx, includedCosts);
		RewriterStatement add = new RewriterInstruction().as(UUID.randomUUID().toString()).withInstruction("+").withOps(argList).consolidate(ctx);
		add.unsafePutMeta("_assertions", assertions);

		//System.out.println("Cost0: " + add.toParsableString(ctx));
		//System.out.println("Assertions: " + assertions);

		// TODO: Validate that this is not needed
		//add = RewriterUtils.buildCanonicalFormConverter(ctx, false).apply(add);
		return add;
	}

	private static RewriterStatement computeCostOf(RewriterInstruction instr, final RuleContext ctx, List<RewriterStatement> uniqueCosts, RewriterAssertions assertions, MutableLong instructionOverhead) {
		if (instr.getResultingDataType(ctx).equals("MATRIX"))
			return computeMatrixOpCost(instr, ctx, uniqueCosts, assertions, instructionOverhead);
		else
			return computeScalarOpCost(instr, ctx, uniqueCosts, assertions, instructionOverhead);
	}

	private static RewriterStatement computeMatrixOpCost(RewriterInstruction instr, final RuleContext ctx, List<RewriterStatement> uniqueCosts, RewriterAssertions assertions, MutableLong overhead) {
		RewriterAssertionUtils.buildImplicitAssertion(instr, assertions, ctx);

		RewriterStatement cost = null;
		Map<String, RewriterStatement> map = new HashMap<>();

		switch (instr.trueInstruction()) {
			case "%*%":
				map.put("nrowA", instr.getChild(0).getNRow());
				map.put("ncolA", instr.getChild(0).getNCol());
				map.put("nrowB", instr.getChild(1).getNRow());
				map.put("ncolB", instr.getChild(1).getNCol());
				map.put("mulCost", atomicOpCostStmt("*", ctx));
				map.put("sumCost", atomicOpCostStmt("+", ctx));
				// Rough estimation
				cost = RewriterUtils.parse("*(argList(nrowA, ncolA, ncolB, +(argList(mulCost, sumCost))))", ctx, map);
				//assertions.addEqualityAssertion(map.get("ncolA"), map.get("nrowB"));
				overhead.add(MALLOC_COST);
				break;
			case "t":
			case "rev":
			case "rowSums":
			case "colSums":
				map.put("nrowA", instr.getChild(0).getNRow());
				map.put("ncolA", instr.getChild(0).getNCol());
				// Rough estimation
				cost = RewriterUtils.parse("*(argList(nrowA, ncolA))", ctx, map);
				overhead.add(MALLOC_COST);
				break;
			case "diag":
				map.put("nrowA", instr.getChild(0).getNRow());
				map.put("ncolA", instr.getChild(0).getNCol());
				cost = map.get("nrowA");
				//assertions.addEqualityAssertion(map.get("nrowA"), map.get("ncolA"));
				overhead.add(MALLOC_COST);
				break;
			case "cast.MATRIX":
				cost = RewriterStatement.literal(ctx, 5L);
				break;
			case "[]":
				cost = RewriterStatement.literal(ctx, 0L);
				break; // I assume that nothing is materialized
			case "RBind":
				map.put("nrowA", instr.getChild(0).getNRow());
				map.put("ncolA", instr.getChild(0).getNCol());
				map.put("nrowB", instr.getChild(1).getNRow());
				map.put("ncolB", instr.getChild(1).getNCol());
				cost = RewriterUtils.parse("+(argList(*(argList(nrowA, ncolA)), *(argList(nrowB, ncolB))))", ctx, map);
				//assertions.addEqualityAssertion(instr.getChild(0).getNCol(), instr.getChild(1).getNCol());
				overhead.add(MALLOC_COST);
				break;
			case "CBind":
				map.put("nrowA", instr.getChild(0).getNRow());
				map.put("ncolA", instr.getChild(0).getNCol());
				map.put("nrowB", instr.getChild(1).getNRow());
				map.put("ncolB", instr.getChild(1).getNCol());
				cost = RewriterUtils.parse("+(argList(*(argList(nrowA, ncolA)), *(argList(nrowB, ncolB))))", ctx, map);
				//assertions.addEqualityAssertion(instr.getChild(0).getNRow(), instr.getChild(1).getNRow());
				overhead.add(MALLOC_COST);
				break;
			case "rand":
				map.put("nrowA", instr.getNRow());
				map.put("ncolA", instr.getNCol());
				cost = RewriterUtils.parse("*(argList(nrowA, ncolA))", ctx, map);
				overhead.add(MALLOC_COST);
				break;
			case "1-*":
				RewriterStatement subtractionCost = atomicOpCostStmt("-", ctx);
				RewriterStatement mulCost = atomicOpCostStmt("*", ctx);
				RewriterStatement sum = RewriterStatement.multiArgInstr(ctx, "+", subtractionCost, mulCost);
				cost = RewriterStatement.multiArgInstr(ctx, "*", sum, instr.getNCol(), instr.getNRow());
				//assertions.addEqualityAssertion(instr.getChild(0).getNCol(), instr.getChild(1).getNCol());
				//assertions.addEqualityAssertion(instr.getChild(0).getNRow(), instr.getChild(1).getNRow());
				overhead.add(MALLOC_COST);
				break;
			case "+*":
				RewriterStatement additionCost = atomicOpCostStmt("+", ctx);
				mulCost = atomicOpCostStmt("*", ctx);
				sum = RewriterStatement.multiArgInstr(ctx, "+", additionCost, mulCost);
				cost = RewriterStatement.multiArgInstr(ctx, "*", sum, instr.getNCol(), instr.getNRow());
				//assertions.addEqualityAssertion(instr.getChild(0).getNCol(), instr.getChild(2).getNCol());
				//assertions.addEqualityAssertion(instr.getChild(0).getNRow(), instr.getChild(2).getNRow());
				overhead.add(MALLOC_COST + 50); // To make it worse than 1-*
				break;
			case "-*":
				subtractionCost = atomicOpCostStmt("-", ctx);
				mulCost = atomicOpCostStmt("*", ctx);
				sum = RewriterStatement.multiArgInstr(ctx, "+", subtractionCost, mulCost);
				cost = RewriterStatement.multiArgInstr(ctx, "*", sum, instr.getNCol(), instr.getNRow());
				//assertions.addEqualityAssertion(instr.getChild(0).getNCol(), instr.getChild(2).getNCol());
				//assertions.addEqualityAssertion(instr.getChild(0).getNRow(), instr.getChild(2).getNRow());
				overhead.add(MALLOC_COST + 50); // To make it worse than 1-*
				break;
			case "*2":
				cost = RewriterStatement.multiArgInstr(ctx, "*", atomicOpCostStmt("*2", ctx), instr.getChild(0).getNRow(), instr.getChild(0).getNCol());
				overhead.add(MALLOC_COST);
				break;
			case "sq":
				cost = RewriterStatement.multiArgInstr(ctx, "*", atomicOpCostStmt("sq", ctx), instr.getChild(0).getNRow(), instr.getChild(0).getNCol());
				overhead.add(MALLOC_COST);
				break;
			case "log_nz": {
				// Must be a matrix
				RewriterStatement logCost = atomicOpCostStmt("log", ctx);
				RewriterStatement twoLogCost = RewriterStatement.multiArgInstr(ctx, "*", RewriterStatement.literal(ctx, 2L), logCost);
				RewriterStatement neqCost = atomicOpCostStmt("!=", ctx);
				sum = RewriterStatement.multiArgInstr(ctx, "+", neqCost, instr.getOperands().size() == 2 ? twoLogCost : logCost);
				cost = RewriterStatement.multiArgInstr(ctx, "*", sum, instr.getNCol(), instr.getNRow());
				overhead.add(MALLOC_COST);
				break;
			}
			case "log":
				if (instr.getChild(0).getResultingDataType(ctx).equals("MATRIX")) {
					RewriterStatement logCost = atomicOpCostStmt("log", ctx);
					RewriterStatement twoLogCost = RewriterStatement.multiArgInstr(ctx, "*", RewriterStatement.literal(ctx, 2L), logCost);
					cost = RewriterStatement.multiArgInstr(ctx, "*", instr.getOperands().size() == 2 ? twoLogCost : logCost, instr.getNCol(), instr.getNRow());
					overhead.add(MALLOC_COST);
				} else {
					RewriterStatement logCost = atomicOpCostStmt("log", ctx);
					RewriterStatement twoLogCost = RewriterStatement.multiArgInstr(ctx, "*", RewriterStatement.literal(ctx, 2L), logCost);
					cost = instr.getOperands().size() == 2 ? twoLogCost : logCost;
				}
				break;
			case "const":
			case "rowVec":
			case "colVec":
				cost = RewriterStatement.literal(ctx, 0L);
				break;
		}

		if (cost == null) {
			if (instr.hasProperty("ElementWiseInstruction", ctx)) {
				RewriterStatement firstMatrix = null;
				RewriterStatement secondMatrix = null;
				if (instr.getChild(0).getResultingDataType(ctx).equals("MATRIX")) {
					firstMatrix = instr.getChild(0);
				}

				if (instr.getChild(1).getResultingDataType(ctx).equals("MATRIX")) {
					if (firstMatrix == null)
						firstMatrix = instr.getChild(1);
					else
						secondMatrix = instr.getChild(1);
				}

				RewriterStatement opCost = atomicOpCostStmt(instr.trueInstruction(), ctx);
				cost = new RewriterInstruction().as(UUID.randomUUID().toString())
						.withInstruction("*")
						.withOps(RewriterStatement.argList(ctx, opCost, instr.getNCol(), instr.getNRow()));

				if (secondMatrix != null) {
					//assertions.addEqualityAssertion(firstMatrix.getNCol(), secondMatrix.getNCol());
					//assertions.addEqualityAssertion(firstMatrix.getNRow(), secondMatrix.getNRow());
				}

				overhead.add(MALLOC_COST);
			} else if (instr.hasProperty("UnaryElementWiseOperator", ctx)) {
				RewriterStatement opCost = atomicOpCostStmt(instr.trueInstruction(), ctx);
				cost = new RewriterInstruction().as(UUID.randomUUID().toString())
						.withInstruction("*")
						.withOps(RewriterStatement.argList(ctx, opCost, instr.getNCol(), instr.getNRow()));
				overhead.add(MALLOC_COST);
			} else {
				throw new IllegalArgumentException("Unknown instruction: " + instr.trueTypedInstruction(ctx));
			}
		}

		uniqueCosts.add(cost);
		return cost;
	}

	private static RewriterStatement computeScalarOpCost(RewriterInstruction instr, final RuleContext ctx, List<RewriterStatement> uniqueCosts, RewriterAssertions assertions, MutableLong overhead) {
		RewriterAssertionUtils.buildImplicitAssertion(instr, assertions, ctx);
		Map<String, RewriterStatement> map = new HashMap<>();
		switch (instr.trueTypedInstruction(ctx)) {
			case "sum(MATRIX)":
			case "min(MATRIX)":
			case "max(MATRIX)":
				map.put("nrowA", instr.getChild(0).getNRow());
				map.put("ncolA", instr.getChild(0).getNCol());
				uniqueCosts.add(RewriterUtils.parse("*(argList(nrowA, ncolA))", ctx, map));
				return uniqueCosts.get(uniqueCosts.size()-1);
			case "trace(MATRIX)":
				map.put("nrowA", instr.getChild(0).getNRow());
				map.put("ncolA", instr.getChild(0).getNCol());
				uniqueCosts.add(map.get("nrowA"));
				//assertions.addEqualityAssertion(map.get("nrowA"), map.get("ncolA"));
				return uniqueCosts.get(uniqueCosts.size()-1);
			case "[](MATRIX,INT,INT)":
				return RewriterStatement.literal(ctx, 0L);
			case "cast.FLOAT(MATRIX)":
				map.put("nrowA", instr.getChild(0).getNRow());
				map.put("ncolA", instr.getChild(0).getNCol());
				uniqueCosts.add(map.get("nrowA"));
				//assertions.addEqualityAssertion(map.get("nrowA"), map.get("ncolA"));
				//assertions.addEqualityAssertion(map.get("nrowA"), RewriterStatement.literal(ctx, 1L));
				return uniqueCosts.get(uniqueCosts.size()-1);
			case "const(MATRIX,FLOAT)":
			case "_nnz":
				return RewriterStatement.literal(ctx, 0L);
		}

		long opCost = atomicOpCost(instr.trueInstruction());
		uniqueCosts.add(RewriterUtils.parse(Long.toString(opCost), ctx, "LITERAL_INT:" + opCost));
		return uniqueCosts.get(uniqueCosts.size()-1);
	}

	private static RewriterStatement atomicOpCostStmt(String op, final RuleContext ctx) {
		long opCost = atomicOpCost(op);
		return RewriterUtils.parse(Long.toString(opCost), ctx, "LITERAL_INT:" + opCost);
	}

	private static long atomicOpCost(String op) {
		switch (op) {
			case "+":
			case "-":
				return 1;
			case "*":
				return 2;
			case "*2":
				return 1; // To make *2 cheaper than *
			case "/":
			case "inv":
				return 3;
			case "length":
			case "nrow":
			case "ncol":
			case "_nnz":
				return 0; // These just fetch metadata
			case "sqrt":
				return 10;
			case "sq":
				return 5;
			case "exp":
			case "log":
			case "^":
				return 20;
			case "!":
			case "|":
			case "&":
			case ">":
			case ">=":
			case "<":
			case "<=":
			case "==":
			case "!=":
				return 1;
			case "round":
				return 2;
			case "abs":
				return 2;
			case "cast.FLOAT":
				return 1;
		}

		throw new IllegalArgumentException("Unknown instruction: " + op);
	}
}
