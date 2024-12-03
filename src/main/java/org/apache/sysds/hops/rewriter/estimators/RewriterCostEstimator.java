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
import org.apache.sysds.hops.rewriter.utils.StatementUtils;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;

public class RewriterCostEstimator {
	private static final long INSTRUCTION_OVERHEAD = 10;
	private static final long MALLOC_COST = 10000;
	public static final Function<RewriterStatement, Long> DEFAULT_COST_FN = el -> 2000L;
	public static final BiFunction<RewriterStatement, Tuple2<Long, Long>, Long> DEFAULT_NNZ_FN = (el, tpl) -> tpl._1 * tpl._2;

	// Computes the cost of an expression using different matrix dimensions and sparsities
	public static void compareCosts(RewriterStatement stmt1, RewriterStatement stmt2, RewriterAssertions jointAssertions, final RuleContext ctx) {
		Map<RewriterStatement, RewriterStatement> estimates1 = RewriterSparsityEstimator.estimateAllNNZ(stmt1, ctx);
		Map<RewriterStatement, RewriterStatement> estimates2 = RewriterSparsityEstimator.estimateAllNNZ(stmt2, ctx);

		MutableObject<RewriterAssertions> assertionRef = new MutableObject<>(jointAssertions);
		RewriterStatement costFn1 = getRawCostFunction(stmt1, ctx, assertionRef);
		RewriterStatement costFn2 = getRawCostFunction(stmt2, ctx, assertionRef);

		costFn1 = RewriterSparsityEstimator.rollupSparsities(costFn1, estimates1, ctx);
		costFn2 = RewriterSparsityEstimator.rollupSparsities(costFn2, estimates2, ctx);

		long[] dimVals = new long[] {1, 5000};
		double[] sparsities = new double[] {1.0D, 0.2D, 0.001D};


		costFn1.unsafePutMeta("_assertions", jointAssertions);
		Map<RewriterStatement, RewriterStatement> createdObjects = new HashMap<>();
		RewriterStatement costFn1Cpy = costFn1.nestedCopy(true, createdObjects);
		RewriterStatement costFn2Cpy = costFn2.nestedCopy(false, createdObjects);
		costFn2Cpy.unsafePutMeta("_assertions", costFn1Cpy.getAssertions(ctx));

		Set<RewriterStatement> dimsToPopulate = new HashSet<>();
		Set<RewriterStatement> nnzsToPopulate = new HashSet<>();

		long cost1 = computeCostFunction(costFn1Cpy, el -> {
				dimsToPopulate.add(el);
				return 2000L;
			}, (nnz, tpl) -> {
				nnzsToPopulate.add(nnz.getChild(0));
				return tpl._1 * tpl._2;
			}, costFn1Cpy.getAssertions(ctx), ctx);
		long cost2 = computeCostFunction(costFn2Cpy, el -> {
			dimsToPopulate.add(el);
				return 2000L;
			}, (nnz, tpl) -> {
				nnzsToPopulate.add(nnz.getChild(0));
				return tpl._1 * tpl._2;
			}, costFn2Cpy.getAssertions(ctx), ctx);

		int nDimsToPopulate = dimsToPopulate.size();
		int nNNZsToPopulate = nnzsToPopulate.size();

		System.out.println("nDimsToPopulate: " + nDimsToPopulate);
		System.out.println(dimsToPopulate);
		System.out.println("nNNZsToPopulate: " + nNNZsToPopulate);
		System.out.println(nnzsToPopulate);
	}

	public static Tuple2<Set<RewriterStatement>, Boolean> determineSingleReferenceRequirement(RewriterRule rule, final RuleContext ctx) {
		MutableObject<RewriterAssertions> assertionRef = new MutableObject<>();
		long fullCost = RewriterCostEstimator.estimateCost(rule.getStmt1(), ctx, assertionRef);
		long maxCost = RewriterCostEstimator.estimateCost(rule.getStmt2(), ctx);
		return RewriterCostEstimator.determineSingleReferenceRequirement(rule.getStmt2(), RewriterCostEstimator.DEFAULT_COST_FN, RewriterCostEstimator.DEFAULT_NNZ_FN, assertionRef.getValue(), fullCost, maxCost, ctx);
	}

	public static Tuple2<Set<RewriterStatement>, Boolean> determineSingleReferenceRequirement(RewriterStatement root, Function<RewriterStatement, Long> costFn, RewriterAssertions assertions, long fullCost, long maxCost, final RuleContext ctx) {
		return determineSingleReferenceRequirement(root, costFn, RewriterCostEstimator.DEFAULT_NNZ_FN, assertions, fullCost, maxCost, ctx);
	}

	// Returns all (upmost) sub-DAGs that can have multiple references and true as a second arg if all statements can have multiple references at once
	public static Tuple2<Set<RewriterStatement>, Boolean> determineSingleReferenceRequirement(RewriterStatement root, Function<RewriterStatement, Long> costFn, BiFunction<RewriterStatement, Tuple2<Long, Long>, Long> nnzFn, RewriterAssertions assertions, long fullCost, long maxCost, final RuleContext ctx) {
		if (fullCost >= maxCost)
			return new Tuple2<>(Collections.emptySet(), true);

		List<Tuple2<RewriterStatement, Long>> subDAGCosts = new ArrayList<>();

		root.forEachPreOrder((cur, pred) -> {
			if (pred.isRoot() || !cur.isInstruction())
				return true;

			long cost = estimateCost(cur, costFn, nnzFn, ctx, new MutableObject<>(assertions));

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
		return estimateCost(stmt, DEFAULT_COST_FN, DEFAULT_NNZ_FN, ctx, assertionRef);
	}

	public static long estimateCost(RewriterStatement stmt, Function<RewriterStatement, Long> propertyGenerator, final RuleContext ctx) {
		return estimateCost(stmt, propertyGenerator, DEFAULT_NNZ_FN, ctx, null);
	}

	public static long estimateCost(RewriterStatement stmt, Function<RewriterStatement, Long> propertyGenerator, BiFunction<RewriterStatement, Tuple2<Long, Long>, Long> nnzGenerator, final RuleContext ctx, MutableObject<RewriterAssertions> assertionRef) {
		if (assertionRef == null)
			assertionRef = new MutableObject<>();

		RewriterStatement costFn = getRawCostFunction(stmt, ctx, assertionRef);
		return computeCostFunction(costFn, propertyGenerator, nnzGenerator, assertionRef.getValue(), ctx);
	}

	public static RewriterStatement getRawCostFunction(RewriterStatement stmt, final RuleContext ctx, MutableObject<RewriterAssertions> assertionRef) {
		RewriterAssertions assertions = assertionRef != null && assertionRef.getValue() != null ? assertionRef.getValue() : new RewriterAssertions(ctx);

		if (assertionRef != null)
			assertionRef.setValue(assertions);

		RewriterStatement costFn = propagateCostFunction(stmt, ctx, assertions);
		costFn = assertions.update(costFn);
		// TODO: Something makes this necessary
		costFn = RewriterUtils.foldConstants(costFn, ctx);

		return costFn;
	}

	public static long computeCostFunction(RewriterStatement costFn, Function<RewriterStatement, Long> propertyGenerator, BiFunction<RewriterStatement, Tuple2<Long, Long>, Long> nnzGenerator, RewriterAssertions assertions, final RuleContext ctx) {
		Map<RewriterStatement, RewriterStatement> map = new HashMap<>();

		costFn.forEachPreOrder((cur, pred) -> {
			for (int i = 0; i < cur.getOperands().size(); i++) {
				RewriterStatement op = cur.getChild(i);

				RewriterStatement mNew = map.get(op);
				if (mNew != null) {
					cur.getOperands().set(i, mNew);
					continue;
				}

				if (op.isEClass()) {
					RewriterAssertions.RewriterAssertion assertion = assertions.getAssertionObj(op);
					Optional<RewriterStatement> literal = assertion != null ? assertion.getLiteral() : Optional.empty();

					mNew = literal.orElseGet(() -> RewriterStatement.literal(ctx, propertyGenerator.apply(op)));

					map.put(op, mNew);
					cur.getOperands().set(i, mNew);
				} else if (op.isInstruction()) {
					if (op.trueInstruction().equals("ncol") || op.trueInstruction().equals("nrow")) {
						RewriterStatement eClassStmt = assertions.getAssertionStatement(op, null);
						mNew = RewriterStatement.literal(ctx, propertyGenerator.apply(eClassStmt));
						map.put(eClassStmt, mNew);
						cur.getOperands().set(i, mNew);
					}
				}
			}

			return true;
		}, false);

		costFn.forEachPreOrder((cur, pred) -> {
			for (int i = 0; i < cur.getOperands().size(); i++) {
				RewriterStatement op = cur.getChild(i);

				RewriterStatement mNew = map.get(op);
				if (mNew != null) {
					cur.getOperands().set(i, mNew);
					continue;
				}

				if (op.isInstruction() && op.trueInstruction().equals("_nnz")) {
					RewriterStatement ncolLiteral = map.get(op.getChild(0).getNCol());

					if (ncolLiteral == null) {
						RewriterAssertions.RewriterAssertion assertion = assertions.getAssertionObj(op.getChild(0).getNCol());

						if (assertion != null) {
							RewriterStatement assStmt = assertion.getEClassStmt(ctx, assertions);
							ncolLiteral = map.get(assStmt);

							if (ncolLiteral == null) {
								ncolLiteral = RewriterStatement.literal(ctx, propertyGenerator.apply(assStmt));
								map.put(assStmt, ncolLiteral);
							}
						} else {
							ncolLiteral = RewriterStatement.literal(ctx, propertyGenerator.apply(op.getChild(0).getNCol()));
							map.put(op.getChild(0).getNCol(), ncolLiteral);
						}
					}

					RewriterStatement nrowLiteral = map.get(op.getChild(0).getNRow());

					if (nrowLiteral == null) {
						RewriterAssertions.RewriterAssertion assertion = assertions.getAssertionObj(op.getChild(0).getNRow());

						if (assertion != null) {
							RewriterStatement assStmt = assertion.getEClassStmt(ctx, assertions);
							nrowLiteral = map.get(assStmt);

							if (nrowLiteral == null) {
								nrowLiteral = RewriterStatement.literal(ctx, propertyGenerator.apply(assStmt));
								map.put(assStmt, nrowLiteral);
							}
						} else {
							nrowLiteral = RewriterStatement.literal(ctx, propertyGenerator.apply(op.getChild(0).getNRow()));
							map.put(op.getChild(0).getNRow(), nrowLiteral);
						}
					}

					mNew = RewriterStatement.literal(ctx, nnzGenerator.apply(op, new Tuple2<>(nrowLiteral.intLiteral(), ncolLiteral.intLiteral())));
					map.put(op, mNew);
					cur.getOperands().set(i, mNew);
				}
			}

			return true;
		}, false);

		//System.out.println("Cost2: " + costFn.toParsableString(ctx));

		costFn = RewriterUtils.foldConstants(costFn, ctx);

		if (!costFn.isLiteral()) {
			throw new IllegalArgumentException("Cost function must be a literal: " + costFn.toParsableString(ctx));
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
				map.put("A", instr.getChild(0));
				map.put("B", instr.getChild(1));
				map.put("nrowA", instr.getChild(0).getNRow());
				map.put("ncolA", instr.getChild(0).getNCol());
				map.put("nrowB", instr.getChild(1).getNRow());
				map.put("ncolB", instr.getChild(1).getNCol());
				map.put("mulCost", atomicOpCostStmt("*", ctx));
				map.put("sumCost", atomicOpCostStmt("+", ctx));
				// Rough estimation
				cost = RewriterUtils.parse("*(argList(min(_nnz(A), _nnz(B)), ncolA, +(argList(mulCost, sumCost))))", ctx, map);
				//assertions.addEqualityAssertion(map.get("ncolA"), map.get("nrowB"));
				overhead.add(MALLOC_COST);
				break;
			case "t":
			case "rev":
				map.put("A", instr.getChild(0));
				cost = RewriterUtils.parse("_nnz(A)", ctx, map);
				overhead.add(MALLOC_COST);
				break;
			case "rowSums":
			case "colSums":
				map.put("A", instr.getChild(0));
				RewriterStatement aoc = atomicOpCostStmt("+", ctx);
				map.put("opcost", aoc);
				// Rough estimation
				cost = RewriterUtils.parse("*(argList(_nnz(A), opcost))", ctx, map);
				overhead.add(MALLOC_COST);
				break;
			case "diag":
				map.put("nrowA", instr.getChild(0).getNRow());
				map.put("A", instr.getChild(0));
				cost = RewriterUtils.parse("min(_nnz(A), nrowA)", ctx, map);//map.get("nrowA");
				//assertions.addEqualityAssertion(map.get("nrowA"), map.get("ncolA"));
				overhead.add(MALLOC_COST);
				break;
			case "cast.MATRIX":
				cost = RewriterStatement.literal(ctx, 20L);
				break;
			case "[]":
				cost = RewriterStatement.literal(ctx, 0L);
				break; // I assume that nothing is materialized
			case "RBind":
			case "CBind":
				map.put("A", instr.getChild(0));
				map.put("B", instr.getChild(1));
				cost = RewriterUtils.parse("+(argList(_nnz(A), _nnz(B)))", ctx, map);
				//assertions.addEqualityAssertion(instr.getChild(0).getNCol(), instr.getChild(1).getNCol());
				overhead.add(MALLOC_COST);
				break;
			case "rand":
				map.put("A", instr);
				cost = RewriterUtils.parse("_nnz(A)", ctx, map);
				overhead.add(MALLOC_COST);
				break;
			case "1-*":
				RewriterStatement subtractionCost = atomicOpCostStmt("-", ctx);
				RewriterStatement mulCost = atomicOpCostStmt("*", ctx);
				RewriterStatement sparsityAwareMul = RewriterStatement.multiArgInstr(ctx, "*", mulCost, StatementUtils.min(ctx, RewriterStatement.nnz(instr.getChild(0), ctx), RewriterStatement.nnz(instr.getChild(1), ctx)));
				RewriterStatement oneMinus = RewriterStatement.multiArgInstr(ctx, "*", subtractionCost, instr.getNCol(), instr.getNRow());
				//RewriterStatement sum = RewriterStatement.multiArgInstr(ctx, "+", subtractionCost, mulCost);
				cost = RewriterStatement.multiArgInstr(ctx, "+", oneMinus, sparsityAwareMul);
				//assertions.addEqualityAssertion(instr.getChild(0).getNCol(), instr.getChild(1).getNCol());
				//assertions.addEqualityAssertion(instr.getChild(0).getNRow(), instr.getChild(1).getNRow());
				overhead.add(MALLOC_COST);
				break;
			case "+*":
				RewriterStatement additionCost = atomicOpCostStmt("+", ctx);
				mulCost = atomicOpCostStmt("*", ctx);
				RewriterStatement sum = RewriterStatement.multiArgInstr(ctx, "+", additionCost, mulCost);
				cost = RewriterStatement.multiArgInstr(ctx, "*", sum, StatementUtils.min(ctx, RewriterStatement.nnz(instr.getChild(0), ctx), RewriterStatement.nnz(instr.getChild(2), ctx)));
				//assertions.addEqualityAssertion(instr.getChild(0).getNCol(), instr.getChild(2).getNCol());
				//assertions.addEqualityAssertion(instr.getChild(0).getNRow(), instr.getChild(2).getNRow());
				overhead.add(MALLOC_COST + 50); // To make it worse than 1-*
				break;
			case "-*":
				subtractionCost = atomicOpCostStmt("-", ctx);
				mulCost = atomicOpCostStmt("*", ctx);
				sum = RewriterStatement.multiArgInstr(ctx, "+", subtractionCost, mulCost);
				cost = RewriterStatement.multiArgInstr(ctx, "*", sum, StatementUtils.min(ctx, RewriterStatement.nnz(instr.getChild(0), ctx), RewriterStatement.nnz(instr.getChild(2), ctx)));
				//assertions.addEqualityAssertion(instr.getChild(0).getNCol(), instr.getChild(2).getNCol());
				//assertions.addEqualityAssertion(instr.getChild(0).getNRow(), instr.getChild(2).getNRow());
				overhead.add(MALLOC_COST + 50); // To make it worse than 1-*
				break;
			case "*2":
				cost = RewriterStatement.multiArgInstr(ctx, "*", atomicOpCostStmt("*2", ctx), RewriterStatement.nnz(instr.getChild(0), ctx));
				overhead.add(MALLOC_COST);
				break;
			case "sq":
				cost = RewriterStatement.multiArgInstr(ctx, "*", atomicOpCostStmt("sq", ctx), RewriterStatement.nnz(instr.getChild(0), ctx));
				overhead.add(MALLOC_COST);
				break;
			case "log_nz": {
				// Must be a matrix
				RewriterStatement logCost = atomicOpCostStmt("log", ctx);
				RewriterStatement twoLogCost = RewriterStatement.multiArgInstr(ctx, "*", RewriterStatement.literal(ctx, 2L), logCost);
				RewriterStatement neqCost = atomicOpCostStmt("!=", ctx);
				sum = RewriterStatement.multiArgInstr(ctx, "+", neqCost, instr.getOperands().size() == 2 ? twoLogCost : logCost);
				cost = RewriterStatement.multiArgInstr(ctx, "*", sum, RewriterStatement.nnz(instr.getChild(0), ctx));
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

				if (firstMatrix != null) {
					switch (instr.trueInstruction()) {
						case "*":
							cost = new RewriterInstruction().as(UUID.randomUUID().toString())
									.withInstruction("*")
									.withOps(RewriterStatement.argList(ctx, opCost, secondMatrix != null ? StatementUtils.min(ctx, RewriterStatement.nnz(firstMatrix, ctx), RewriterStatement.nnz(secondMatrix, ctx)) : RewriterStatement.nnz(firstMatrix, ctx)));
							break;
						case "/":
							if (instr.getChild(0).getResultingDataType(ctx).equals("MATRIX"))
								cost = new RewriterInstruction().as(UUID.randomUUID().toString())
										.withInstruction("*")
										.withOps(RewriterStatement.argList(ctx, opCost, RewriterStatement.nnz(instr.getChild(0), ctx)));
							else
								cost = new RewriterInstruction().as(UUID.randomUUID().toString())
										.withInstruction("*")
										.withOps(RewriterStatement.argList(ctx, opCost, StatementUtils.length(ctx, firstMatrix)));

							break;
						case "+":
						case "-":
							cost = new RewriterInstruction().as(UUID.randomUUID().toString())
									.withInstruction("*")
									.withOps(RewriterStatement.argList(ctx, opCost, secondMatrix != null ? StatementUtils.add(ctx, RewriterStatement.nnz(firstMatrix, ctx), RewriterStatement.nnz(secondMatrix, ctx)) : RewriterStatement.nnz(firstMatrix, ctx)));
							break;
						default:
							cost = RewriterStatement.multiArgInstr(ctx, "*", opCost, instr.getNRow(), instr.getNCol());
							break;
					}

					overhead.add(MALLOC_COST);
				} else {
					cost = opCost;
				}
			} else if (instr.hasProperty("UnaryElementWiseOperator", ctx)) {
				RewriterStatement opCost = atomicOpCostStmt(instr.trueInstruction(), ctx);
				cost = new RewriterInstruction().as(UUID.randomUUID().toString())
						.withInstruction("*")
						.withOps(RewriterStatement.argList(ctx, opCost, RewriterStatement.nnz(instr.getChild(0), ctx)));
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
				map.put("A", instr.getChild(0));
				uniqueCosts.add(RewriterUtils.parse("_nnz(A)", ctx, map));
				return uniqueCosts.get(uniqueCosts.size()-1);
			case "trace(MATRIX)":
				uniqueCosts.add(StatementUtils.min(ctx, RewriterStatement.nnz(instr.getChild(0), ctx), instr.getChild(0).getNRow()));
				//assertions.addEqualityAssertion(map.get("nrowA"), map.get("ncolA"));
				return uniqueCosts.get(uniqueCosts.size()-1);
			case "[](MATRIX,INT,INT)":
				return RewriterStatement.literal(ctx, 0L);
			case "cast.FLOAT(MATRIX)":
				//uniqueCosts.add(map.get("nrowA"));
				//assertions.addEqualityAssertion(map.get("nrowA"), map.get("ncolA"));
				//assertions.addEqualityAssertion(map.get("nrowA"), RewriterStatement.literal(ctx, 1L));
				return RewriterStatement.literal(ctx, INSTRUCTION_OVERHEAD);
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
