package org.apache.sysds.hops.rewriter;

import org.apache.commons.lang3.mutable.MutableLong;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;

public class RewriterCostEstimator {
	private static final long INSTRUCTION_OVERHEAD = 10;
	private static final long MALLOC_COST = 10000;
	private static final Function<RewriterStatement, Long> DEFAULT_COST_FN = el -> 2000L;

	public static long estimateCost(RewriterStatement stmt, final RuleContext ctx) {
		return estimateCost(stmt, DEFAULT_COST_FN, ctx);
	}

	public static long estimateCost(RewriterStatement stmt, Function<RewriterStatement, Long> propertyGenerator, final RuleContext ctx) {
		RewriterAssertions assertions = new RewriterAssertions(ctx);
		RewriterStatement costFn = propagateCostFunction(stmt, ctx, assertions);

		Map<RewriterStatement, RewriterStatement> map = new HashMap<>();

		//System.out.println("Cost1: " + costFn.toParsableString(ctx));

		costFn.forEachPostOrder((cur, parent, pIdx) -> {
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
		});

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

		stmt.forEachPostOrder((cur, parent, pIdx) -> {
			if (!(cur instanceof RewriterInstruction))
				return;

			computeCostOf((RewriterInstruction) cur, ctx, includedCosts, assertions, instructionOverhead);
			instructionOverhead.add(INSTRUCTION_OVERHEAD);
		});

		includedCosts.add(RewriterStatement.literal(ctx, instructionOverhead.longValue()));

		RewriterStatement argList = RewriterStatement.argList(ctx, includedCosts);
		RewriterStatement add = new RewriterInstruction().as(UUID.randomUUID().toString()).withInstruction("+").withOps(argList).consolidate(ctx);
		add.unsafePutMeta("_assertions", assertions);

		//System.out.println("Cost0: " + add.toParsableString(ctx));
		//System.out.println("Assertions: " + assertions);

		add = RewriterUtils.buildCanonicalFormConverter(ctx, false).apply(add);
		return add;
	}

	private static void computeCostOf(RewriterInstruction instr, final RuleContext ctx, List<RewriterStatement> uniqueCosts, RewriterAssertions assertions, MutableLong instructionOverhead) {
		if (instr.getResultingDataType(ctx).equals("MATRIX"))
			computeMatrixOpCost(instr, ctx, uniqueCosts, assertions, instructionOverhead);
		else
			computeScalarOpCost(instr, ctx, uniqueCosts, assertions, instructionOverhead);
	}

	private static void computeMatrixOpCost(RewriterInstruction instr, final RuleContext ctx, List<RewriterStatement> uniqueCosts, RewriterAssertions assertions, MutableLong overhead) {
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
				assertions.addEqualityAssertion(map.get("ncolA"), map.get("nrowB"));
				overhead.add(MALLOC_COST);
				break;
			case "t":
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
				assertions.addEqualityAssertion(map.get("nrowA"), map.get("ncolA"));
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
				assertions.addEqualityAssertion(instr.getChild(0).getNCol(), instr.getChild(1).getNCol());
				overhead.add(MALLOC_COST);
				break;
			case "CBind":
				map.put("nrowA", instr.getChild(0).getNRow());
				map.put("ncolA", instr.getChild(0).getNCol());
				map.put("nrowB", instr.getChild(1).getNRow());
				map.put("ncolB", instr.getChild(1).getNCol());
				cost = RewriterUtils.parse("+(argList(*(argList(nrowA, ncolA)), *(argList(nrowB, ncolB))))", ctx, map);
				assertions.addEqualityAssertion(instr.getChild(0).getNRow(), instr.getChild(1).getNRow());
				overhead.add(MALLOC_COST);
				break;
			case "rand":
				map.put("nrowA", instr.getNRow());
				map.put("ncolA", instr.getNCol());
				cost = RewriterUtils.parse("*(argList(nrowA, ncolA))", ctx, map);
				overhead.add(MALLOC_COST);
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
					assertions.addEqualityAssertion(firstMatrix.getNCol(), secondMatrix.getNCol());
					assertions.addEqualityAssertion(firstMatrix.getNRow(), secondMatrix.getNRow());
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
	}

	private static void computeScalarOpCost(RewriterInstruction instr, final RuleContext ctx, List<RewriterStatement> uniqueCosts, RewriterAssertions assertions, MutableLong overhead) {
		Map<String, RewriterStatement> map = new HashMap<>();
		switch (instr.trueTypedInstruction(ctx)) {
			case "sum(MATRIX)":
			case "min(MATRIX)":
			case "max(MATRIX)":
				map.put("nrowA", instr.getChild(0).getNRow());
				map.put("ncolA", instr.getChild(0).getNCol());
				uniqueCosts.add(RewriterUtils.parse("*(argList(nrowA, ncolA))", ctx, map));
				return;
			case "trace(MATRIX)":
				map.put("nrowA", instr.getChild(0).getNRow());
				map.put("ncolA", instr.getChild(0).getNCol());
				uniqueCosts.add(map.get("nrowA"));
				assertions.addEqualityAssertion(map.get("nrowA"), map.get("ncolA"));
				return;
			case "[](MATRIX,INT,INT)":
				return;
			case "cast.FLOAT(MATRIX)":
				map.put("nrowA", instr.getChild(0).getNRow());
				map.put("ncolA", instr.getChild(0).getNCol());
				uniqueCosts.add(map.get("nrowA"));
				assertions.addEqualityAssertion(map.get("nrowA"), map.get("ncolA"));
				assertions.addEqualityAssertion(map.get("nrowA"), RewriterStatement.literal(ctx, 1L));
				return;
		}

		long opCost = atomicOpCost(instr.trueInstruction());
		uniqueCosts.add(RewriterUtils.parse(Long.toString(opCost), ctx, "LITERAL_INT:" + opCost));
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
			case "/":
			case "inv":
				return 3;
			case "length":
			case "nrow":
			case "ncol":
				return 0; // These just fetch metadata
			case "sqrt":
				return 10;
			case "exp":
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
