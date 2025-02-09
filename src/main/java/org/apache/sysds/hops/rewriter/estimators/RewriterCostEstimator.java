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

package org.apache.sysds.hops.rewriter.estimators;

import org.apache.commons.lang3.mutable.MutableInt;
import org.apache.commons.lang3.mutable.MutableLong;
import org.apache.commons.lang3.mutable.MutableObject;
import org.apache.sysds.hops.rewriter.RewriterFramework;
import org.apache.sysds.hops.rewriter.RewriterInstruction;
import org.apache.sysds.hops.rewriter.rule.RewriterRule;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.hops.rewriter.assertions.RewriterAssertionUtils;
import org.apache.sysds.hops.rewriter.assertions.RewriterAssertions;
import org.apache.sysds.hops.rewriter.utils.StatementUtils;
import scala.Tuple2;
import scala.Tuple3;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
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

	// This is an important check as many intermediate matrices do not contain any sparsity information
	// Thus, we want to use cost functions without sparsity information if possible
	public static boolean doesHaveAnImpactOnOptimalExpression(List<Tuple3<List<Number>, Long, Long>> list, boolean sparsity, boolean sort, int costThreshhold) {
		if (sort)
			sort(list);

		int diff = 0;
		Tuple3<List<Number>, Long, Long> last = null;

		for (Tuple3<List<Number>, Long, Long> t : list) {
			if (Math.abs(t._2() - t._3()) < costThreshhold)
				continue;

			if (last == null || (sparsity && !hasSameDims(last._1(), t._1()))) {
				last = t;
				diff = Long.signum(t._2() - t._3());
				continue;
			}

			int mDiff = Long.signum(t._2() - t._3());

			if (diff != mDiff && Math.abs(t._2() - t._3() - last._2() + last._3()) > costThreshhold)
				return true;
		}

		return false;
	}

	private static boolean hasSameDims(List<Number> l1, List<Number> l2) {
		int maxN = Math.min(l1.size(), l2.size());

		for (int i = 0; i < maxN; i++) {
			Number el1 = l1.get(i);
			Number el2 = l2.get(i);

			if (el1 instanceof Long && el1.longValue() != el2.longValue())
				return false;
		}

		return true;
	}

	private static void sort(List<Tuple3<List<Number>, Long, Long>> list) {
		list.sort((t1, t2) -> {
			int size = Math.min(t1._1().size(), t2._1().size());
			for (int i = 0; i < size; i++) {
				int cmp = Double.compare(t1._1().get(i).doubleValue(), t2._1().get(i).doubleValue());
				if (cmp != 0)
					return cmp; // Return non-zero comparison result if elements differ
			}

			return Integer.compare(t1._1().size(), t2._1().size());
		});
	}

	public static Set<Tuple2<Integer, Integer>> findOptima(List<Tuple2<List<Number>, List<Long>>> data) {
		Set<Tuple2<Integer, Integer>> outSet = new HashSet<>();
		data.stream().forEach(t -> {
			int minIdx = -1;
			long minValue = Long.MAX_VALUE;
			for (int i = 0; i < t._2.size(); i++) {
				if (t._2.get(i) < minValue) {
					minValue = t._2.get(i);
					minIdx = i;
				}
			}

			for (int i = 0; i < t._2.size(); i++) {
				if (t._2.get(i) > minValue)
					outSet.add(new Tuple2<>(i, minIdx));
			}
		});

		return outSet;
	}

	public static List<Tuple2<List<Number>, List<Long>>> compareCosts(List<RewriterStatement> statements, RewriterAssertions jointAssertions, final RuleContext ctx, boolean sample, int sampleSize) {
		List<Map<RewriterStatement, RewriterStatement>> estimates = statements.stream().map(stmt -> RewriterSparsityEstimator.estimateAllNNZ(stmt, ctx)).collect(Collectors.toList());

		MutableObject<RewriterAssertions> assertionRef = new MutableObject<>(jointAssertions);
		List<RewriterStatement> costFns = statements.stream().map(stmt -> getRawCostFunction(stmt, ctx, assertionRef, false)).collect(Collectors.toList());

		for (int i = 0; i < estimates.size(); i++) {
			costFns.set(i, RewriterSparsityEstimator.rollupSparsities(costFns.get(i), estimates.get(i), ctx));
		}

		long[] dimVals = new long[] {10, 5000};
		double[] sparsities = new double[] {1.0D, 0.000001D};

		Map<RewriterStatement, RewriterStatement> createdObjects = new HashMap<>();
		List<RewriterStatement> costFnCpys = costFns.stream().map(fn -> fn.nestedCopy(false, createdObjects)).collect(Collectors.toList());
		RewriterAssertions jointAssertionsCpy = RewriterAssertions.copy(jointAssertions, createdObjects, false);

		Set<RewriterStatement> dimsToPopulate = new HashSet<>();
		Set<RewriterStatement> nnzsToPopulate = new HashSet<>();

		List<Long> costs = costFnCpys.stream().map(costFnCpy -> {
			try {
				return computeCostFunction(costFnCpy, el -> {
					dimsToPopulate.add(el);
					return 2000L;
				}, (nnz, tpl) -> {
					nnzsToPopulate.add(nnz.getChild(0));
					return tpl._1 * tpl._2;
				}, jointAssertionsCpy, ctx);
			} catch (Exception e) {
				//e.printStackTrace();
				if (RewriterFramework.DEBUG)
					System.err.println("Error while estimating the cost: " + e.getMessage());
				return null;
			}
		}).collect(Collectors.toList());

		int nDimsToPopulate = dimsToPopulate.size();
		int nNNZsToPopulate = nnzsToPopulate.size();

		List<Number> firstList = new ArrayList<>();
		for (int i = 0; i < nDimsToPopulate; i++)
			firstList.add(2000L);
		for (int i = 0; i < nNNZsToPopulate; i++)
			firstList.add(1.0D);

		List<Tuple2<List<Number>, List<Long>>> out = new ArrayList<>();
		out.add(new Tuple2<>(firstList, costs));

		if (sampleSize < 2)
			return out;

		List<List<Number>> nums = new ArrayList<>();
		List<Number> dimList = Arrays.stream(dimVals).mapToObj(dim -> ((Number)dim)).collect(Collectors.toList());
		List<Number> sparsityList = Arrays.stream(sparsities).mapToObj(s -> ((Number)s)).collect(Collectors.toList());

		int numCombinations = 1;

		for (int i = 0; i < nDimsToPopulate; i++) {
			nums.add(dimList);
			numCombinations *= dimList.size();
		}

		for (int i = 0; i < nNNZsToPopulate; i++) {
			nums.add(sparsityList);
			numCombinations *= sparsityList.size();
		}

		Set<Integer> samples = new HashSet<>();

		if (sample) {
			if (sampleSize < numCombinations) {
				Random rd = new Random();

				while (samples.size() < sampleSize)
					samples.add(rd.nextInt(numCombinations));
			} else {
				sample = false;
			}
		}

		final boolean doSample = sample;

		MutableInt ctr = new MutableInt();

		if (nums.size() > 16) {
			System.err.println("Could not properly sample: " + statements);
			return out;
		}

		RewriterUtils.cartesianProduct(nums, new Number[nums.size()], stack -> {
			if (doSample && !samples.contains(ctr.getAndIncrement()))
				return true;

			int sparsityStart = 0;

			for (Number num : stack) {
				if (num instanceof Double)
					break;

				sparsityStart++;
			}

			final int fSparsityStart = sparsityStart;

			Map<RewriterStatement, Long> replace = new HashMap<>();

			MutableInt dimCtr = new MutableInt();
			MutableInt sCtr = new MutableInt();

			Map<RewriterStatement, RewriterStatement> mCreatedObjects = new HashMap<>();
			List<RewriterStatement> mCostFnCpys = costFns.stream().map(cpy -> cpy.nestedCopy(false, mCreatedObjects)).collect(Collectors.toList());
			RewriterAssertions mAssertionsCpy = RewriterAssertions.copy(jointAssertions, mCreatedObjects, false);

			List<Long> mCosts = mCostFnCpys.stream().map(mCpy -> {
				try {
					return computeCostFunction(mCpy, el -> {
						Long literal = replace.get(el);

						if (literal == null) {
							literal = (Long) stack[dimCtr.getAndIncrement()];
							//System.out.println("populated size with: " + literal);
							replace.put(el, literal);
						}

						return literal;
					}, (nnz, tpl) -> {
						Long literal = replace.get(nnz.getChild(0));

						if (literal == null) {
							double sparsity = (double) stack[fSparsityStart + sCtr.getAndIncrement()];
							literal = (long) Math.ceil(sparsity * tpl._1 * tpl._2);
							replace.put(nnz.getChild(0), literal);
						}

						return literal;
					}, mAssertionsCpy, ctx);
				} catch (Exception e) {
					if (RewriterFramework.DEBUG)
						e.printStackTrace();
					return null;
				}
			}).collect(Collectors.toList());

			out.add(new Tuple2<>(new ArrayList<>(Arrays.asList(stack)), mCosts));

			return true;
		});

		return out;
	}

	// Computes the cost of an expression using different matrix dimensions and sparsities
	public static List<Tuple3<List<Number>, Long, Long>> compareCosts(RewriterStatement stmt1, RewriterStatement stmt2, RewriterAssertions jointAssertions, final RuleContext ctx, boolean sample, int sampleSize, boolean returnOnDifference) {
		Map<RewriterStatement, RewriterStatement> estimates1 = RewriterSparsityEstimator.estimateAllNNZ(stmt1, ctx);
		Map<RewriterStatement, RewriterStatement> estimates2 = RewriterSparsityEstimator.estimateAllNNZ(stmt2, ctx);

		MutableObject<RewriterAssertions> assertionRef = new MutableObject<>(jointAssertions);
		RewriterStatement costFn1 = getRawCostFunction(stmt1, ctx, assertionRef, false);
		RewriterStatement costFn2 = getRawCostFunction(stmt2, ctx, assertionRef, false);

		costFn1 = RewriterSparsityEstimator.rollupSparsities(costFn1, estimates1, ctx);
		costFn2 = RewriterSparsityEstimator.rollupSparsities(costFn2, estimates2, ctx);

		final RewriterStatement fCostFn1 = costFn1;
		final RewriterStatement fCostFn2 = costFn2;

		long[] dimVals = new long[] {10, 5000};
		double[] sparsities = new double[] {1.0D, 0.05D};

		Map<RewriterStatement, RewriterStatement> createdObjects = new HashMap<>();
		RewriterStatement costFn1Cpy = costFn1.nestedCopy(true, createdObjects);
		RewriterStatement costFn2Cpy = costFn2.nestedCopy(false, createdObjects);
		RewriterAssertions jointAssertionsCpy = RewriterAssertions.copy(jointAssertions, createdObjects, false);

		Set<RewriterStatement> dimsToPopulate = new HashSet<>();
		Set<RewriterStatement> nnzsToPopulate = new HashSet<>();

		long cost1 = computeCostFunction(costFn1Cpy, el -> {
				dimsToPopulate.add(el);
				return 2000L;
			}, (nnz, tpl) -> {
				nnzsToPopulate.add(nnz.getChild(0));
				return tpl._1 * tpl._2;
			}, jointAssertionsCpy, ctx);
		long cost2 = computeCostFunction(costFn2Cpy, el -> {
			dimsToPopulate.add(el);
				return 2000L;
			}, (nnz, tpl) -> {
				nnzsToPopulate.add(nnz.getChild(0));
				return tpl._1 * tpl._2;
			}, jointAssertionsCpy, ctx);

		int nDimsToPopulate = dimsToPopulate.size();
		int nNNZsToPopulate = nnzsToPopulate.size();

		List<Number> firstList = new ArrayList<>();
		for (int i = 0; i < nDimsToPopulate; i++)
			firstList.add(2000L);
		for (int i = 0; i < nNNZsToPopulate; i++)
			firstList.add(1.0D);

		List<Tuple3<List<Number>, Long, Long>> out = new ArrayList<>();
		out.add(new Tuple3<>(firstList, cost1, cost2));

		if (returnOnDifference && cost1 != cost2)
			return out;

		List<List<Number>> nums = new ArrayList<>();
		List<Number> dimList = Arrays.stream(dimVals).mapToObj(dim -> ((Number)dim)).collect(Collectors.toList());
		List<Number> sparsityList = Arrays.stream(sparsities).mapToObj(s -> ((Number)s)).collect(Collectors.toList());

		int numCombinations = 1;

		for (int i = 0; i < nDimsToPopulate; i++) {
			nums.add(dimList);
			numCombinations *= dimList.size();
		}

		for (int i = 0; i < nNNZsToPopulate; i++) {
			nums.add(sparsityList);
			numCombinations *= sparsityList.size();
		}

		Set<Integer> samples = new HashSet<>();

		if (sample) {
			if (sampleSize < numCombinations) {
				Random rd = new Random();

				while (samples.size() < sampleSize)
					samples.add(rd.nextInt(numCombinations));
			} else {
				sample = false;
			}
		}

		final boolean doSample = sample;

		MutableInt ctr = new MutableInt();

		RewriterUtils.cartesianProduct(nums, new Number[nums.size()], stack -> {
			if (doSample && !samples.contains(ctr.getAndIncrement()))
				return true;

			int sparsityStart = 0;

			for (Number num : stack) {
				if (num instanceof Double)
					break;

				sparsityStart++;
			}

			final int fSparsityStart = sparsityStart;

			Map<RewriterStatement, Long> replace = new HashMap<>();

			MutableInt dimCtr = new MutableInt();
			MutableInt sCtr = new MutableInt();

			Map<RewriterStatement, RewriterStatement> mCreatedObjects = new HashMap<>();
			RewriterStatement mCpy1 = fCostFn1.nestedCopy(false, mCreatedObjects);
			RewriterStatement mCpy2 = fCostFn2.nestedCopy(false, mCreatedObjects);
			RewriterAssertions mAssertionsCpy = RewriterAssertions.copy(jointAssertions, mCreatedObjects, false);

			long mCost1 = computeCostFunction(mCpy1, el -> {
				Long literal = replace.get(el);

				if (literal == null) {
					literal = (Long) stack[dimCtr.getAndIncrement()];
					replace.put(el, literal);
				}

				return literal;
			}, (nnz, tpl) -> {
				Long literal = replace.get(nnz.getChild(0));

				if (literal == null) {
					double sparsity = (double) stack[fSparsityStart + sCtr.getAndIncrement()];
					literal = (long)Math.ceil(sparsity * tpl._1 * tpl._2);
					replace.put(nnz.getChild(0), literal);
				}

				return literal;
			}, mAssertionsCpy, ctx);
			long mCost2 = computeCostFunction(mCpy2, el -> {
				Long literal = replace.get(el);

				if (literal == null) {
					literal = (Long) stack[dimCtr.getAndIncrement()];
					replace.put(el, literal);
				}

				return literal;
			}, (nnz, tpl) -> {
				Long literal = replace.get(nnz.getChild(0));

				if (literal == null) {
					double sparsity = (double) stack[fSparsityStart + sCtr.getAndIncrement()];
					literal = (long)Math.ceil(sparsity * tpl._1 * tpl._2);
					replace.put(nnz.getChild(0), literal);
				}

				return literal;
			}, mAssertionsCpy, ctx);

			out.add(new Tuple3<>(new ArrayList<>(Arrays.asList(stack)), mCost1, mCost2));

			return !returnOnDifference || mCost1 == mCost2;
		});

		return out;
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

		RewriterStatement costFn = getRawCostFunction(stmt, ctx, assertionRef, false);
		return computeCostFunction(costFn, propertyGenerator, nnzGenerator, assertionRef.getValue(), ctx);
	}

	public static RewriterStatement getRawCostFunction(RewriterStatement stmt, final RuleContext ctx, MutableObject<RewriterAssertions> assertionRef, boolean treatAsDense) {
		RewriterAssertions assertions = assertionRef != null && assertionRef.getValue() != null ? assertionRef.getValue() : new RewriterAssertions(ctx);

		if (assertionRef != null)
			assertionRef.setValue(assertions);

		RewriterStatement costFn = propagateCostFunction(stmt, ctx, assertions, treatAsDense);
		Map<RewriterStatement, RewriterStatement> estimations = RewriterSparsityEstimator.estimateAllNNZ(costFn, ctx);
		RewriterSparsityEstimator.rollupSparsities(costFn, estimations, ctx);
		costFn = assertions.update(costFn);
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

					mNew = RewriterStatement.literal(ctx, nnzGenerator.apply(op, new Tuple2<>(nrowLiteral.intLiteral(false), ncolLiteral.intLiteral(false))));
					map.put(op, mNew);
					cur.getOperands().set(i, mNew);
				}
			}

			return true;
		}, false);

		costFn.forEachPreOrder(cur -> {
			if (cur.isInstruction())
				cur.refreshReturnType(ctx);

			return true;
		}, false);

		costFn = RewriterUtils.foldConstants(costFn, ctx);

		if (!costFn.isLiteral()) {
			throw new IllegalArgumentException("Cost function must be a literal: " + costFn.toParsableString(ctx));
		}

		if (costFn.getLiteral() instanceof Double)
			return (long)((double)costFn.getLiteral());

		return (long)costFn.getLiteral();
	}

	private static RewriterStatement propagateCostFunction(RewriterStatement stmt, final RuleContext ctx, RewriterAssertions assertions, boolean treatAsDense) {
		List<RewriterStatement> includedCosts = new ArrayList<>();
		MutableLong instructionOverhead = new MutableLong(0);

		stmt.forEachPostOrder((cur, pred) -> {
			if (!(cur instanceof RewriterInstruction))
				return;

			computeCostOf((RewriterInstruction) cur, ctx, includedCosts, assertions, instructionOverhead, treatAsDense, stmt);
			instructionOverhead.add(INSTRUCTION_OVERHEAD);
		}, false);

		includedCosts.add(RewriterStatement.literal(ctx, instructionOverhead.longValue()));

		RewriterStatement argList = RewriterStatement.argList(ctx, includedCosts);
		RewriterStatement add = new RewriterInstruction().as(UUID.randomUUID().toString()).withInstruction("+").withOps(argList).consolidate(ctx);
		add.unsafePutMeta("_assertions", assertions);
		return add;
	}

	private static RewriterStatement computeCostOf(RewriterInstruction instr, final RuleContext ctx, List<RewriterStatement> uniqueCosts, RewriterAssertions assertions, MutableLong instructionOverhead, boolean treatAsDense, RewriterStatement exprRoot) {
		if (instr.getResultingDataType(ctx).equals("MATRIX"))
			return computeMatrixOpCost(instr, ctx, uniqueCosts, assertions, instructionOverhead, treatAsDense, exprRoot);
		else
			return computeScalarOpCost(instr, ctx, uniqueCosts, assertions, instructionOverhead, treatAsDense, exprRoot);
	}

	private static RewriterStatement computeMatrixOpCost(RewriterInstruction instr, final RuleContext ctx, List<RewriterStatement> uniqueCosts, RewriterAssertions assertions, MutableLong overhead, boolean treatAsDense, RewriterStatement exprRoot) {
		RewriterAssertionUtils.buildImplicitAssertion(instr, assertions, exprRoot, ctx);

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
				map.put("nnzA", RewriterStatement.nnz(instr.getChild(0), ctx, treatAsDense));
				map.put("nnzB", RewriterStatement.nnz(instr.getChild(1), ctx, treatAsDense));
				// Rough estimation
				cost = RewriterUtils.parse("*(argList(min(nnzA, nnzB), ncolA, +(argList(mulCost, sumCost))))", ctx, map);
				overhead.add(MALLOC_COST);
				break;
			case "t":
			case "rev":
				cost = RewriterStatement.nnz(instr.getChild(0), ctx, treatAsDense);//RewriterUtils.parse("_nnz(A)", ctx, map);
				overhead.add(MALLOC_COST);
				break;
			case "rowSums":
			case "colSums":
				map.put("A", instr.getChild(0));
				map.put("nnzA", RewriterStatement.nnz(instr.getChild(0), ctx, treatAsDense));
				RewriterStatement aoc = atomicOpCostStmt("+", ctx);
				map.put("opcost", aoc);
				// Rough estimation
				cost = RewriterUtils.parse("*(argList(nnzA, opcost))", ctx, map);
				overhead.add(MALLOC_COST);
				break;
			case "diag":
				map.put("nrowA", instr.getChild(0).getNRow());
				map.put("nnzA", RewriterStatement.nnz(instr.getChild(0), ctx, treatAsDense));
				map.put("A", instr.getChild(0));
				cost = RewriterUtils.parse("min(nnzA, nrowA)", ctx, map);
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
				map.put("nnzA", RewriterStatement.nnz(instr.getChild(0), ctx, treatAsDense));
				map.put("nnzB", RewriterStatement.nnz(instr.getChild(1), ctx, treatAsDense));
				cost = RewriterUtils.parse("+(argList(nnzA, nnzB))", ctx, map);
				overhead.add(MALLOC_COST);
				break;
			case "rand":
				cost = RewriterStatement.nnz(instr, ctx, treatAsDense);
				overhead.add(MALLOC_COST);
				break;
			case "1-*":
				RewriterStatement subtractionCost = atomicOpCostStmt("-", ctx);
				RewriterStatement mulCost = atomicOpCostStmt("*", ctx);
				RewriterStatement sparsityAwareMul = RewriterStatement.multiArgInstr(ctx, "*", mulCost, StatementUtils.min(ctx, RewriterStatement.nnz(instr.getChild(0), ctx, treatAsDense), RewriterStatement.nnz(instr.getChild(1), ctx, treatAsDense)));
				RewriterStatement oneMinus = RewriterStatement.multiArgInstr(ctx, "*", subtractionCost, instr.getNCol(), instr.getNRow());
				cost = RewriterStatement.multiArgInstr(ctx, "+", oneMinus, sparsityAwareMul);
				overhead.add(MALLOC_COST);
				break;
			case "+*":
				RewriterStatement additionCost = atomicOpCostStmt("+", ctx);
				mulCost = atomicOpCostStmt("*", ctx);
				RewriterStatement sum = RewriterStatement.multiArgInstr(ctx, "+", additionCost, mulCost);
				cost = RewriterStatement.multiArgInstr(ctx, "*", sum, StatementUtils.min(ctx, RewriterStatement.nnz(instr.getChild(0), ctx, treatAsDense), RewriterStatement.nnz(instr.getChild(2), ctx, treatAsDense)));
				overhead.add(MALLOC_COST + 50); // To make it worse than 1-*
				break;
			case "-*":
				subtractionCost = atomicOpCostStmt("-", ctx);
				mulCost = atomicOpCostStmt("*", ctx);
				sum = RewriterStatement.multiArgInstr(ctx, "+", subtractionCost, mulCost);
				cost = RewriterStatement.multiArgInstr(ctx, "*", sum, StatementUtils.min(ctx, RewriterStatement.nnz(instr.getChild(0), ctx, treatAsDense), RewriterStatement.nnz(instr.getChild(2), ctx, treatAsDense)));
				overhead.add(MALLOC_COST + 50); // To make it worse than 1-*
				break;
			case "*2":
				cost = RewriterStatement.multiArgInstr(ctx, "*", atomicOpCostStmt("*2", ctx), RewriterStatement.nnz(instr.getChild(0), ctx, treatAsDense));
				overhead.add(MALLOC_COST);
				break;
			case "sq":
				cost = RewriterStatement.multiArgInstr(ctx, "*", atomicOpCostStmt("sq", ctx), RewriterStatement.nnz(instr.getChild(0), ctx, treatAsDense));
				overhead.add(MALLOC_COST);
				break;
			case "sqrt":
				cost = RewriterStatement.multiArgInstr(ctx, "*", atomicOpCostStmt("sqrt", ctx), RewriterStatement.nnz(instr.getChild(0), ctx, treatAsDense));
				overhead.add(MALLOC_COST);
				break;
			case "exp":
				cost = RewriterStatement.multiArgInstr(ctx, "*", atomicOpCostStmt("exp", ctx), RewriterStatement.nnz(instr.getChild(0), ctx, treatAsDense));
				overhead.add(MALLOC_COST);
				break;
			case "log_nz": {
				// Must be a matrix
				RewriterStatement logCost = atomicOpCostStmt("log", ctx);
				RewriterStatement twoLogCost = RewriterStatement.multiArgInstr(ctx, "*", RewriterStatement.literal(ctx, 2L), logCost);
				RewriterStatement neqCost = atomicOpCostStmt("!=", ctx);
				sum = RewriterStatement.multiArgInstr(ctx, "+", neqCost, instr.getOperands().size() == 2 ? twoLogCost : logCost);
				cost = RewriterStatement.multiArgInstr(ctx, "*", sum, RewriterStatement.nnz(instr.getChild(0), ctx, treatAsDense));
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
				overhead.add(MALLOC_COST);
			case "rowVec":
			case "colVec":
			case "cellMat":
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
									.withOps(RewriterStatement.argList(ctx, opCost, secondMatrix != null ? StatementUtils.min(ctx, RewriterStatement.nnz(firstMatrix, ctx, treatAsDense), RewriterStatement.nnz(secondMatrix, ctx, treatAsDense)) : RewriterStatement.nnz(firstMatrix, ctx, treatAsDense)));
							break;
						case "/":
							if (instr.getChild(0).getResultingDataType(ctx).equals("MATRIX"))
								cost = new RewriterInstruction().as(UUID.randomUUID().toString())
										.withInstruction("*")
										.withOps(RewriterStatement.argList(ctx, opCost, RewriterStatement.nnz(instr.getChild(0), ctx, treatAsDense)));
							else
								cost = new RewriterInstruction().as(UUID.randomUUID().toString())
										.withInstruction("*")
										.withOps(RewriterStatement.argList(ctx, opCost, StatementUtils.length(ctx, firstMatrix)));

							break;
						case "+":
						case "-":
							cost = new RewriterInstruction().as(UUID.randomUUID().toString())
									.withInstruction("*")
									.withOps(RewriterStatement.argList(ctx, opCost, secondMatrix != null ? StatementUtils.add(ctx, RewriterStatement.nnz(firstMatrix, ctx, treatAsDense), RewriterStatement.nnz(secondMatrix, ctx, treatAsDense)) : RewriterStatement.nnz(firstMatrix, ctx, treatAsDense)));
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
						.withOps(RewriterStatement.argList(ctx, opCost, RewriterStatement.nnz(instr.getChild(0), ctx, treatAsDense)));
				overhead.add(MALLOC_COST);
			} else {
				throw new IllegalArgumentException("Unknown instruction: " + instr.trueTypedInstruction(ctx));
			}
		}

		uniqueCosts.add(cost);
		return cost;
	}

	private static RewriterStatement computeScalarOpCost(RewriterInstruction instr, final RuleContext ctx, List<RewriterStatement> uniqueCosts, RewriterAssertions assertions, MutableLong overhead, boolean treatAsDense, RewriterStatement exprRoot) {
		RewriterAssertionUtils.buildImplicitAssertion(instr, assertions, exprRoot, ctx);
		Map<String, RewriterStatement> map = new HashMap<>();
		switch (instr.trueTypedInstruction(ctx)) {
			case "sum(MATRIX)":
			case "min(MATRIX)":
			case "max(MATRIX)":
				map.put("nnzA", RewriterStatement.nnz(instr.getChild(0), ctx, treatAsDense));
				uniqueCosts.add(RewriterUtils.parse("nnzA", ctx, map));
				return uniqueCosts.get(uniqueCosts.size()-1);
			case "sumSq(MATRIX)":
				map.put("nnzA", RewriterStatement.nnz(instr.getChild(0), ctx, treatAsDense));
				uniqueCosts.add(RewriterStatement.multiArgInstr(ctx, "*", RewriterStatement.nnz(instr.getChild(0), ctx, treatAsDense), RewriterStatement.literal(ctx, 2L)));
				return uniqueCosts.get(uniqueCosts.size()-1);
			case "trace(MATRIX)":
				uniqueCosts.add(StatementUtils.min(ctx, RewriterStatement.nnz(instr.getChild(0), ctx, treatAsDense), instr.getChild(0).getNRow()));
				return uniqueCosts.get(uniqueCosts.size()-1);
			case "[](MATRIX,INT,INT)":
				return RewriterStatement.literal(ctx, 0L);
			case "cast.FLOAT(MATRIX)":
				return RewriterStatement.literal(ctx, INSTRUCTION_OVERHEAD);
			case "const(MATRIX,FLOAT)":
			case "_nnz(MATRIX)":
				return RewriterStatement.literal(ctx, 0L);
		}

		double opCost = atomicOpCost(instr.trueInstruction());
		uniqueCosts.add(RewriterUtils.parse(Double.toString(opCost), ctx, "LITERAL_FLOAT:" + opCost));
		return uniqueCosts.get(uniqueCosts.size()-1);
	}

	private static RewriterStatement atomicOpCostStmt(String op, final RuleContext ctx) {
		double opCost = atomicOpCost(op);
		return RewriterUtils.parse(Double.toString(opCost), ctx, "LITERAL_FLOAT:" + opCost);
	}

	private static double atomicOpCost(String op) {
		switch (op) {
			case "+":
			case "-":
				return 1;
			case "*":
				return 2;
			case "*2":
				return 1; // To make *2 cheaper than A+A
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
				return 1.8; // To make it cheaper than *(A,A)
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
			case "literal.FLOAT":
			case "literal.INT":
			case "literal.BOOL":
				return 0;
		}

		throw new IllegalArgumentException("Unknown instruction: " + op);
	}
}
