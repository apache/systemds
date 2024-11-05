package org.apache.sysds.hops.rewriter;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;

public class RewriterCostEstimator {
	public static long estimateCost(RewriterStatement stmt, Function<RewriterStatement, Long> propertyGenerator, final RuleContext ctx) {
		RewriterAssertions assertions = new RewriterAssertions(ctx);
		RewriterStatement costFn = propagateCostFunction(stmt, ctx, assertions);

		// Now, assign
		System.out.println(costFn);

		Map<RewriterStatement, RewriterStatement> map = new HashMap<>();

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

		costFn = RewriterUtils.foldConstants(costFn, ctx);
		return (long)costFn.getLiteral();
	}

	private static RewriterStatement propagateCostFunction(RewriterStatement stmt, final RuleContext ctx, RewriterAssertions assertions) {
		List<RewriterStatement> includedCosts = new ArrayList<>();

		stmt.forEachPostOrder((cur, parent, pIdx) -> {
			if (!(cur instanceof RewriterInstruction))
				return;

			computeCostOf((RewriterInstruction) cur, ctx, includedCosts, assertions);
		});

		RewriterStatement argList = RewriterStatement.argList(ctx, includedCosts);
		RewriterStatement add = new RewriterInstruction().as(UUID.randomUUID().toString()).withInstruction("+").withOps(argList).consolidate(ctx);
		add.unsafePutMeta("_assertions", assertions);

		add = RewriterUtils.buildCanonicalFormConverter(ctx, false).apply(add);
		return add;
	}

	private static void computeCostOf(RewriterInstruction instr, final RuleContext ctx, List<RewriterStatement> uniqueCosts, RewriterAssertions assertions) {
		if (instr.getResultingDataType(ctx).equals("MATRIX"))
			computeMatrixOpCost(instr, ctx, uniqueCosts, assertions);
		else
			computeScalarOpCost(instr, ctx, uniqueCosts, assertions);
	}

	private static void computeMatrixOpCost(RewriterInstruction instr, final RuleContext ctx, List<RewriterStatement> uniqueCosts, RewriterAssertions assertions) {
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
				break;
			case "t":
				map.put("nrowA", instr.getChild(0).getNRow());
				map.put("ncolA", instr.getChild(0).getNCol());
				// Rough estimation
				cost = RewriterUtils.parse("*(argList(nrowA, ncolA))", ctx, map);
		}

		if (cost == null) {
			if (instr.hasProperty("ElementWiseInstruction", ctx)) {
				RewriterStatement opCost = atomicOpCostStmt(instr.trueInstruction(), ctx);
				cost = new RewriterInstruction().as(UUID.randomUUID().toString())
						.withInstruction("*")
						.withOps(RewriterStatement.argList(ctx, opCost, instr.getNCol(), instr.getNRow()));
				assertions.addEqualityAssertion(instr.getChild(0).getNCol(), instr.getChild(1).getNCol());
				assertions.addEqualityAssertion(instr.getChild(0).getNRow(), instr.getChild(1).getNRow());
			} else {
				throw new IllegalArgumentException();
			}
		}

		uniqueCosts.add(cost);
	}

	private static void computeScalarOpCost(RewriterInstruction instr, final RuleContext ctx, List<RewriterStatement> uniqueCosts, RewriterAssertions assertions) {
		RewriterStatement cost = null;
		long opCost = atomicOpCost(instr.trueInstruction());


		uniqueCosts.add(RewriterUtils.parse(Long.toString(opCost), ctx, "LITERAL_INT:" + opCost));
	}

	private static RewriterStatement atomicOpCostStmt(String op, final RuleContext ctx) {
		long opCost = atomicOpCost(op);
		return RewriterUtils.parse(Long.toString(opCost), ctx, "LITERAL_INT:" + opCost);
	}

	private static RewriterStatement literalInt(long value) {
		return new RewriterDataType().as(Long.toString(value)).ofType("INT").asLiteral(value);
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
		}

		throw new IllegalArgumentException();
	}
}
