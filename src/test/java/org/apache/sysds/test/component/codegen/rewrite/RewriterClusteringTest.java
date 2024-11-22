package org.apache.sysds.test.component.codegen.rewrite;

import org.apache.commons.lang3.mutable.MutableInt;
import org.apache.sysds.hops.rewriter.RewriteAutomaticallyGenerated;
import org.apache.sysds.hops.rewriter.RewriterAlphabetEncoder;
import org.apache.sysds.hops.rewriter.RewriterCostEstimator;
import org.apache.sysds.hops.rewriter.RewriterDatabase;
import org.apache.sysds.hops.rewriter.RewriterEquivalenceDatabase;
import org.apache.sysds.hops.rewriter.RewriterHeuristic;
import org.apache.sysds.hops.rewriter.RewriterRule;
import org.apache.sysds.hops.rewriter.RewriterRuleCollection;
import org.apache.sysds.hops.rewriter.RewriterRuleCreator;
import org.apache.sysds.hops.rewriter.RewriterRuleSet;
import org.apache.sysds.hops.rewriter.RewriterRuntimeUtils;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.hops.rewriter.TopologicalSort;
import org.apache.sysds.performance.TimingUtils;
import org.junit.BeforeClass;
import org.junit.Test;
import scala.Tuple2;
import scala.Tuple3;
import scala.Tuple5;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class RewriterClusteringTest {
	private static RuleContext ctx;
	private static Function<RewriterStatement, RewriterStatement> converter;
	private static RewriterDatabase db;
	private static Function<RewriterStatement, RewriterStatement> flattenAndMerge;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
		converter = RewriterUtils.buildCanonicalFormConverter(ctx, false);
		db = new RewriterDatabase();

		try(BufferedReader reader = new BufferedReader(new FileReader(RewriterRuntimeUtils.dbFile))) {
			db.deserialize(reader, ctx);
		} catch (IOException e) {
			e.printStackTrace();
		}

		ArrayList<RewriterRule> flatten = new ArrayList<>();
		RewriterRuleCollection.flattenOperations(flatten, ctx);
		RewriterHeuristic flattenOperations = new RewriterHeuristic(new RewriterRuleSet(ctx, flatten));

		flattenAndMerge = el -> {
			el = flattenOperations.apply(el, null, false);
			RewriterUtils.mergeArgLists(el, ctx);
			return el;
		};
	}

	@Test
	public void testExpressionClustering() {
		boolean useData = false;
		boolean useRandomized = true;

		long startTime = System.currentTimeMillis();
		AtomicLong generatedExpressions = new AtomicLong(0);
		AtomicLong evaluatedExpressions = new AtomicLong(0);
		AtomicLong failures = new AtomicLong(0);
		AtomicLong totalCanonicalizationMillis = new AtomicLong(0);

		RewriterDatabase exactExprDB = new RewriterDatabase();
		RewriterEquivalenceDatabase canonicalExprDB = new RewriterEquivalenceDatabase();

		List<RewriterEquivalenceDatabase.DBEntry> foundEquivalences = Collections.synchronizedList(new ArrayList<>());

		int size = db.size();
		MutableInt ctr = new MutableInt(0);

		if (useData) {
			db.parForEach(expr -> {
				if (ctr.incrementAndGet() % 10 == 0)
					System.out.println("Done: " + ctr.intValue() + " / " + size);
				//if (ctr.intValue() > 100)
				//	return; // Skip
				// First, build all possible subtrees
				//System.out.println("Eval:\n" + expr.toParsableString(ctx, true));
				List<RewriterStatement> subExprs = RewriterUtils.generateSubtrees(expr, ctx, 300);
				if (subExprs.size() > 100)
					System.out.println("Critical number of subtrees: " + subExprs.size());
				if (subExprs.size() > 500) {
					System.out.println("Skipping subtrees...");
					subExprs = List.of(expr);
				}
				//List<RewriterStatement> subExprs = List.of(expr);
				long evaluationCtr = 0;
				long mCanonicalizationMillis = 0;

				for (RewriterStatement subExpr : subExprs) {
					try {
						if (!exactExprDB.insertEntry(ctx, subExpr))
							continue;

						//System.out.println("Evaluating expression: \n" + subExpr.toParsableString(ctx, true));

						evaluationCtr++;

						//System.out.println("Eval: " + subExpr.toParsableString(ctx, true));

						// Duplicate the statement as we do not want to canonicalize the original statement
						long startMillis = System.currentTimeMillis();
						RewriterStatement canonicalForm = converter.apply(subExpr.nestedCopy(true));
						mCanonicalizationMillis += System.currentTimeMillis() - startMillis;

						computeCost(subExpr, ctx);

						// Insert the canonical form or retrieve the existing entry
						RewriterEquivalenceDatabase.DBEntry entry = canonicalExprDB.insert(ctx, canonicalForm, subExpr);

						if (entry.equivalences.size() == 2) {
							foundEquivalences.add(entry);
						}

						/*if (existingEntry == null) {
							List<RewriterStatement> equivalentExpressions = new ArrayList<>();
							equivalentExpressions.add(subExpr);
							canonicalForm.unsafePutMeta("equivalentExpressions", equivalentExpressions);
						} else {
							List<RewriterStatement> equivalentExpressions = (List<RewriterStatement>) existingEntry.getMeta("equivalentExpressions");
							equivalentExpressions.add(subExpr);

							if (equivalentExpressions.size() == 2)
								foundEquivalences.add(existingEntry);

							//System.out.println("Found equivalent statement!");
						}*/
					} catch (Exception e) {
						e.printStackTrace();
						failures.incrementAndGet();
					}
				}

				generatedExpressions.addAndGet(subExprs.size());
				evaluatedExpressions.addAndGet(evaluationCtr);
				totalCanonicalizationMillis.addAndGet(mCanonicalizationMillis);
			});
		}

		if (useRandomized) {
			long MAX_MILLIS = 100000000; // Should be bound by number of ops
			int BATCH_SIZE = 200;
			int maxN = RewriterAlphabetEncoder.getMaxSearchNumberForNumOps(3);
			long startMillis = System.currentTimeMillis();

			for (int batch = 0; batch < 100 && System.currentTimeMillis() - startMillis < MAX_MILLIS && batch * BATCH_SIZE < maxN; batch++) {
				List<Integer> indices = IntStream.range(batch * BATCH_SIZE, Math.min((batch + 1) * BATCH_SIZE - 1, maxN)).boxed().collect(Collectors.toList());
				Collections.shuffle(indices);
				MutableInt ctr2 = new MutableInt(0);
				int maxSize = indices.size();
				final int mBATCH = batch;
				indices.parallelStream().forEach(idx -> {
					if (ctr2.incrementAndGet() % 10 == 0)
						System.out.println("Done: " + (mBATCH * BATCH_SIZE + ctr2.intValue()) + " / " + (mBATCH * BATCH_SIZE + maxSize));

					List<RewriterAlphabetEncoder.Operand> ops = RewriterAlphabetEncoder.decodeOrderedStatements(idx);
					List<RewriterStatement> stmts = RewriterAlphabetEncoder.buildAllPossibleDAGs(ops, ctx, true);

					for (RewriterStatement dag : stmts) {
						List<RewriterStatement> expanded = new ArrayList<>();
						expanded.addAll(RewriterAlphabetEncoder.buildAssertionVariations(dag, ctx, true));
						expanded.addAll(RewriterAlphabetEncoder.buildVariations(dag, ctx));
						for (RewriterStatement stmt : expanded) {
							try {
								RewriterStatement canonicalForm = converter.apply(stmt);
								computeCost(stmt, ctx);

								List<RewriterStatement> equivalentExpressions = new ArrayList<>();
								equivalentExpressions.add(stmt);

								// TODO: Better handling
								if (!canonicalForm.isLiteral())
									canonicalForm.unsafePutMeta("equivalentExpressions", equivalentExpressions);

								stmt.getCost(ctx); // Fetch cost already
								RewriterEquivalenceDatabase.DBEntry entry = canonicalExprDB.insert(ctx, canonicalForm, stmt);

								if (entry.equivalences.size() == 2)
									foundEquivalences.add(entry);

								// Insert the canonical form or retrieve the existing entry
							/*RewriterStatement existingEntry = canonicalExprDB.insertOrReturn(ctx, canonicalForm);

							if (existingEntry != null) {
								equivalentExpressions = (List<RewriterStatement>) existingEntry.getMeta("equivalentExpressions");
								// TODO: Better handling
								if (equivalentExpressions != null) {
									equivalentExpressions.add(stmt);

									if (equivalentExpressions.size() == 2)
										foundEquivalences.add(existingEntry);
								}

								//System.out.println("Found equivalent statement!");
							}*/
							} catch (Exception e) {
								e.printStackTrace();
							}
						}
					}
				});
			}
		}

		printEquivalences(/*foundEquivalences*/ Collections.emptyList(), System.currentTimeMillis() - startTime, generatedExpressions.longValue(), evaluatedExpressions.longValue(), totalCanonicalizationMillis.longValue(), failures.longValue(), true);

		System.out.println("===== SUGGESTED REWRITES =====");
		List<Tuple5<Double, Long, Long, RewriterStatement, RewriterStatement>> rewrites = findSuggestedRewrites(foundEquivalences);

		// Here, we create any rule
		List<Tuple3<RewriterRule, Long, Long>> allRules = new ArrayList<>();
		int mCtr = 0;
		for (Tuple5<Double, Long, Long, RewriterStatement, RewriterStatement> rewrite : rewrites) {
			if (++mCtr % 100 == 0)
				System.out.println("Creating rule: " + mCtr + " / " + rewrites.size());

			RewriterStatement canonicalFormFrom = converter.apply(rewrite._4());
			RewriterStatement canonicalFormTo = converter.apply(rewrite._5());
			try {
				RewriterRule rule = RewriterRuleCreator.createRule(rewrite._4(), rewrite._5(), canonicalFormFrom, canonicalFormTo, ctx);

				allRules.add(new Tuple3<>(rule, rewrite._2(), rewrite._3()));
				//ruleCreator.registerRule(rule, rewrite._2(), rewrite._3());
			} catch (Exception e) {
				System.err.println("An error occurred while trying to create a rule:");
				System.err.println(rewrite._4().toParsableString(ctx, true));
				System.err.println(rewrite._5().toParsableString(ctx, true));
				e.printStackTrace();
			}
		}



		RewriterRuleCreator ruleCreator = new RewriterRuleCreator(ctx);

		for (Tuple3<RewriterRule, Long, Long> t : allRules) {
			// First, without validating correctness
			// This might throw out some fallback options if a rule turns out to be incorrect but we there is a huge performance benefit
			ruleCreator.registerRule(t._1(), t._2(), t._3(), false);
		}

		allRules = null;

		RewriterRuleSet rawRuleSet = ruleCreator.getRuleSet();

		try (FileWriter writer = new FileWriter(RewriteAutomaticallyGenerated.RAW_FILE_PATH)) {
			writer.write(rawRuleSet.serialize(ctx));
		} catch (IOException ex) {
			ex.printStackTrace();
		}

		ruleCreator.throwOutInvalidRules();

		/*RewriterRuleCreator ruleCreator = new RewriterRuleCreator(ctx);

		for (Tuple3<RewriterRule, Long, Long> t : allRules)
			ruleCreator.registerRule(t._1(), t._2(), t._3());*/

		ruleCreator.forEachRule(rule -> {
			System.out.println(rule);
			//System.out.println("Score: " + rewrite._1());
			System.out.println("Cost1: " + rule.getStmt1().getCost(ctx));
			System.out.println("Cost2: " + rule.getStmt2().getCost(ctx));
		});

		String serialized = ruleCreator.getRuleSet().serialize(ctx);
		System.out.println(serialized);

		try (FileWriter writer = new FileWriter(RewriteAutomaticallyGenerated.FILE_PATH)) {
			writer.write(serialized);
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}

	private void computeCost(RewriterStatement subExpr, final RuleContext ctx) {
		if (subExpr.isLiteral())
			return;

		if (subExpr.getMeta("_cost") == null) {
			long cost = -1;
			try {
				cost = RewriterCostEstimator.estimateCost(subExpr, el -> 2000L, ctx);
			} catch (Exception e) {
			}
			subExpr.unsafePutMeta("_cost", cost);
		}
	}

	// This function should be called regularly if the number of equivalent expressions get too big
	/*private void updateOptimum(RewriterStatement dbEntry) {
		long optimalCost = -1;
		RewriterStatement currentOptimum = (RewriterStatement) dbEntry.getMeta("_optimum");
		List<RewriterStatement> equivalences = (List<RewriterStatement>) dbEntry.getMeta("equivalentExpressions");

		if (currentOptimum == null) {
			for (int i = 0; i < equivalences.size(); i++) {
				currentOptimum = equivalences.get(i);
				// TODO: Failures will be recomputed as _cost is still null
				if (currentOptimum.getMeta("_cost") == null) {
					try {
						optimalCost = RewriterCostEstimator.estimateCost(currentOptimum, el -> 2000L, ctx);
						currentOptimum.unsafePutMeta("_cost", optimalCost);
					} catch (Exception e) {
						currentOptimum.unsafePutMeta("_cost", -1L);
					}
				} else {
					optimalCost = (Long) currentOptimum.getMeta("_cost");
				}

				if (optimalCost != -1)
					break;
			}
		}

		if (optimalCost == -1)
			return;

		for (RewriterStatement eq : equivalences) {
			if (eq != currentOptimum) {
				Object obj = eq.getMeta("_cost");
				long cost;
				if (obj == null) {
					try {
						cost = RewriterCostEstimator.estimateCost(eq, el -> 2000L, ctx);
						eq.unsafePutMeta("_cost", cost);
					} catch (Exception e) {
						cost = -1;
						eq.unsafePutMeta("_cost", -1L);
					}
				} else {
					cost = (Long) obj;
				}

				if (cost != -1 && cost < optimalCost) {
					currentOptimum = eq;
					optimalCost = cost;
				}
			}
		}

		dbEntry.unsafePutMeta("_optimum", currentOptimum);
	}*/

	private void printEquivalences(List<RewriterStatement> equivalentStatements, long cpuTime, long generatedExpressions, long evaluatedExpressions, long canonicalizationMillis, long failures, boolean preFilter) {
		System.out.println("===== ALL EQUIVALENCES =====");
		if (preFilter)
			System.out.println("Pre-filtering is active! Note that this hides some (probably less impactful) equivalences");

		for (RewriterStatement eStmt : equivalentStatements) {
			List<RewriterStatement> equivalences = (List<RewriterStatement>)eStmt.getMeta("equivalentExpressions");
			if (preFilter && !checkRelevance(equivalences))
				continue; // Then this equivalence is not that relevant as it is just a shuffling of operands

			System.out.println();
			System.out.println();
			System.out.println("===================================");
			System.out.println("Canonical form: " + eStmt.toParsableString(ctx) + "\n");
			equivalences.forEach(stmt -> System.out.println(stmt.toParsableString(ctx, true) + "\nHC: " + stmt.hashCode()  + "\n"));

			if (equivalences.size() == 0)
				System.out.println("All statements were actually equivalent!");
			//System.out.println(equivalences.get(0).match(new RewriterStatement.MatcherContext(ctx, equivalences.get(0))));
		}

		System.out.println();
		System.out.println("Total rewriter CPU time: " + cpuTime + "ms");
		System.out.println("Total generated expressions: " + generatedExpressions);
		System.out.println("Total evaluated unique expressions: " + evaluatedExpressions);
		System.out.println("Avg canonicalization time: " + Math.round(((double)canonicalizationMillis)/evaluatedExpressions) + "ms");
		System.out.println("Total failures: " + failures);
	}

	private boolean checkRelevance(List<RewriterStatement> stmts) {
		boolean match = true;

		for (int i = 0; i < stmts.size(); i++) {
			for (int j = stmts.size() - 1; j > i; j--) {
				RewriterStatement stmt1 = stmts.get(i).nestedCopy(true);
				RewriterStatement stmt2 = stmts.get(j).nestedCopy(true);

				stmt1 = flattenAndMerge.apply(stmt1);
				stmt2 = flattenAndMerge.apply(stmt2);

				TopologicalSort.sort(stmt1, ctx);
				TopologicalSort.sort(stmt2, ctx);

				if (!stmt1.match(RewriterStatement.MatcherContext.exactMatchWithDifferentLiteralValues(ctx, stmt2, stmt1))) {
					// TODO: Minimal difference can still prune valid rewrites (e.g. sum(A %*% B) -> sum(A * t(B)))
					RewriterStatement.MatcherContext mCtx = RewriterStatement.MatcherContext.findMinimalDifference(ctx, stmts.get(j), stmts.get(i));
					stmts.get(i).match(mCtx);
					Tuple2<RewriterStatement, RewriterStatement> minimalDifference = mCtx.getFirstMismatch();

					if (minimalDifference._1 == stmts.get(i))
						match = false;
					else {
						// Otherwise we need to work ourselves backwards to the root if both canonical forms don't match now
						RewriterStatement minStmt1 = minimalDifference._1.nestedCopy(true);
						RewriterStatement minStmt2 = minimalDifference._2.nestedCopy(true);
						minStmt1 = converter.apply(minStmt1);
						minStmt2 = converter.apply(minStmt2);

						if (minStmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, minStmt2, minStmt1))) {
							// Then the minimal difference does not imply equivalence
							// For now, just keep every result then
							match = false;
						}
					}
				}
			}
		}

		return !match;
	}

	private List<Tuple5<Double, Long, Long, RewriterStatement, RewriterStatement>> findSuggestedRewrites(List<RewriterEquivalenceDatabase.DBEntry> equivalences) {
		List<Tuple5<Double, Long, Long, RewriterStatement, RewriterStatement>> suggestedRewrites = new ArrayList<>();

		for (RewriterEquivalenceDatabase.DBEntry entry : equivalences) {
			List<RewriterStatement> mEq = entry.equivalences;
			RewriterStatement optimalStatement = null;
			long minCost = -1;

			for (RewriterStatement eq : mEq) {
				try {
					long cost = eq.getCost(ctx);

					if (cost == -1)
						continue;

					if (optimalStatement == null) {
						minCost = cost;
						optimalStatement = eq;
						continue;
					}

					if (cost < minCost) {
						optimalStatement = eq;
						minCost = cost;
					}
				} catch (Exception e) {
					// TODO: Enable
					//e.printStackTrace();
				}
			}

			if (optimalStatement != null) {
				for (RewriterStatement eq : mEq) {
					if (eq == optimalStatement)
						continue;

					long cost = eq.getCost();

					if (cost != -1) {
						double score = (((double)cost) / minCost - 1) * 1000; // Relative cost reduction
						score *= cost - minCost; // Absolute cost reduction
						if (score > 1e-10)
							suggestedRewrites.add(new Tuple5<>(score, cost, minCost, eq, optimalStatement));
					}
				}
			}
		}

		suggestedRewrites.sort(Comparator.comparing(Tuple5::_1));
		Collections.reverse(suggestedRewrites);
		return suggestedRewrites;
	}
}
