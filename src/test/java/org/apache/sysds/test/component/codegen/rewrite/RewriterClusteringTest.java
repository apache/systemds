package org.apache.sysds.test.component.codegen.rewrite;

import org.apache.commons.collections.list.SynchronizedList;
import org.apache.commons.lang3.mutable.MutableInt;
import org.apache.sysds.hops.rewriter.RewriteAutomaticallyGenerated;
import org.apache.sysds.hops.rewriter.RewriterAlphabetEncoder;
import org.apache.sysds.hops.rewriter.assertions.RewriterAssertionUtils;
import org.apache.sysds.hops.rewriter.assertions.RewriterAssertions;
import org.apache.sysds.hops.rewriter.estimators.RewriterCostEstimator;
import org.apache.sysds.hops.rewriter.RewriterDatabase;
import org.apache.sysds.hops.rewriter.RewriterEquivalenceDatabase;
import org.apache.sysds.hops.rewriter.RewriterHeuristic;
import org.apache.sysds.hops.rewriter.RewriterRule;
import org.apache.sysds.hops.rewriter.RewriterRuleCollection;
import org.apache.sysds.hops.rewriter.RewriterRuleCreator;
import org.apache.sysds.hops.rewriter.RewriterRuleSet;
import org.apache.sysds.hops.rewriter.RewriterRuntimeUtils;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.hops.rewriter.TopologicalSort;
import scala.Tuple2;
import scala.Tuple3;
import scala.Tuple4;
import scala.Tuple5;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class RewriterClusteringTest {
	private static RuleContext ctx;
	private static Function<RewriterStatement, RewriterStatement> converter;
	private static RewriterDatabase db;
	private static Function<RewriterStatement, RewriterStatement> flattenAndMerge;

	public static void main(String[] args) {
		setup();
		testExpressionClustering();
	}

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

	public static void testExpressionClustering() {
		boolean useData = true;
		boolean useSystematic = true;
		boolean pruneNovelExpressions = false; // To drop all "irrelevant" statements (those that don't appear in the data set)
		boolean useRandomLarge = false;
		int systematicSearchDepth = 3;
		int BATCH_SIZE = 1000;
		int pruneDataSubexrBiggerThan = 1000;
		int maxCostSamples = 50;

		AtomicLong generatedExpressions = new AtomicLong(0);
		AtomicLong evaluatedExpressions = new AtomicLong(0);
		AtomicLong failures = new AtomicLong(0);
		AtomicLong totalCanonicalizationMillis = new AtomicLong(0);

		RewriterDatabase exactExprDB = new RewriterDatabase();
		RewriterEquivalenceDatabase canonicalExprDB = new RewriterEquivalenceDatabase();

		List<RewriterEquivalenceDatabase.DBEntry> foundEquivalences = Collections.synchronizedList(new ArrayList<>());

		MutableInt ctr = new MutableInt(0);

		Object lock = new Object();

		if (useData) {
			int size = db.size();
			db.parForEach(expr -> {
				if (ctr.getValue() > 20000)
					return;
				if (ctr.incrementAndGet() % 10 == 0)
					System.out.println("Done: " + ctr.intValue() + " / " + size);
				//if (ctr.intValue() > 100)
				//	return; // Skip
				// First, build all possible subtrees
				//System.out.println("Eval:\n" + expr.toParsableString(ctx, true));
				List<RewriterStatement> subExprs = RewriterUtils.generateSubtrees(expr, ctx, pruneDataSubexrBiggerThan);
				if (subExprs.size() > pruneDataSubexrBiggerThan)
					System.out.println("Critical number of subtrees: " + subExprs.size());
				if (subExprs.size() > 2 * pruneDataSubexrBiggerThan) {
					System.out.println("Skipping subtrees...");
					subExprs = List.of(expr);
				}
				//List<RewriterStatement> subExprs = List.of(expr);
				long evaluationCtr = 0;
				long mCanonicalizationMillis = 0;

				for (RewriterStatement subExpr : subExprs) {
					try {
						String mstmt = subExpr.toParsableString(ctx, true);
						//System.out.println(mstmt);
						subExpr = RewriterUtils.parse(mstmt, ctx);

						if (!exactExprDB.insertEntry(ctx, subExpr))
							continue;

						//System.out.println("Evaluating expression: \n" + subExpr.toParsableString(ctx, true));

						evaluationCtr++;

						//System.out.println("Eval: " + subExpr.toParsableString(ctx, true));

						// Duplicate the statement as we do not want to canonicalize the original statement
						long startMillis = System.currentTimeMillis();
						RewriterStatement canonicalForm = converter.apply(subExpr);
						mCanonicalizationMillis += System.currentTimeMillis() - startMillis;

						synchronized (lock) {
							RewriterEquivalenceDatabase.DBEntry entry = canonicalExprDB.insert(ctx, canonicalForm, subExpr);

							// Now, we use common variables
							if (entry.equivalences.size() > 1) {
								RewriterStatement commonForm = RewriterRuleCreator.createCommonForm(subExpr, entry.equivalences.get(0), canonicalForm, entry.canonicalForm, ctx)._1;
								entry.equivalences.set(entry.equivalences.size()-1, commonForm);
							}

							if (entry.equivalences.size() == 2)
								foundEquivalences.add(entry);
						}

						//computeCost(subExpr, ctx);
						//subExpr.getCost(ctx);

						// Insert the canonical form or retrieve the existing entry
						/*RewriterEquivalenceDatabase.DBEntry entry = canonicalExprDB.insert(ctx, canonicalForm, subExpr);

						if (entry.equivalences.size() == 2) {
							foundEquivalences.add(entry);
						}*/

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
						try {
							System.err.println("Error from expression: " + subExpr.toParsableString(ctx));
						} catch (Exception e2) {
						}
						e.printStackTrace();
						failures.incrementAndGet();
					}
				}

				generatedExpressions.addAndGet(subExprs.size());
				evaluatedExpressions.addAndGet(evaluationCtr);
				totalCanonicalizationMillis.addAndGet(mCanonicalizationMillis);
			});
		}

		db = null;

		if (useSystematic) {
			long MAX_MILLIS = 12000000; // Should be bound by number of ops
			//int BATCH_SIZE = 400;
			int maxN = RewriterAlphabetEncoder.getMaxSearchNumberForNumOps(systematicSearchDepth);
			System.out.println("MaxN: " + maxN);
			long startMillis = System.currentTimeMillis();

			for (int batch = 0; batch < 10000 && System.currentTimeMillis() - startMillis < MAX_MILLIS && batch * BATCH_SIZE < maxN; batch++) {
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
					long actualCtr = 0;

					for (RewriterStatement dag : stmts) {
						List<RewriterStatement> expanded = new ArrayList<>();
						expanded.add(dag);
						//expanded.addAll(RewriterAlphabetEncoder.buildAssertionVariations(dag, ctx, false));
						expanded.addAll(RewriterAlphabetEncoder.buildVariations(dag, ctx));
						actualCtr += expanded.size();
						for (RewriterStatement stmt : expanded) {
							try {
								String mstmt = stmt.toParsableString(ctx, true);
								stmt = RewriterUtils.parse(mstmt, ctx);
								ctx.metaPropagator.apply(stmt);
								RewriterStatement canonicalForm = converter.apply(stmt);

								synchronized (lock) {
									if (pruneNovelExpressions && !canonicalExprDB.containsEntry(canonicalForm))
										return;

									RewriterEquivalenceDatabase.DBEntry entry = canonicalExprDB.insert(ctx, canonicalForm, stmt);

									// Now, we use common variables
									if (entry.equivalences.size() > 1) {
										RewriterStatement commonForm = RewriterRuleCreator.createCommonForm(stmt, entry.equivalences.get(0), canonicalForm, entry.canonicalForm, ctx)._1;
										entry.equivalences.set(entry.equivalences.size()-1, commonForm);
										/*System.out.println("HERE: " + entry.equivalences.get(0));
										System.out.println("BEFORE: " + stmt);
										System.out.println("HERE2: " + commonForm);*/
									}

									if (entry.equivalences.size() == 2)
										foundEquivalences.add(entry);
								}
							} catch (Exception e) {
								System.err.println("Faulty expression: " + stmt.toParsableString(ctx));
								e.printStackTrace();
							}
						}
					}

					//System.out.println(ops + " >> " + actualCtr);
				});
			}

			if (useRandomLarge) {
				// Now we will just do random sampling for a few rounds
				Random rd = new Random(42);
				int nMaxN = RewriterAlphabetEncoder.getMaxSearchNumberForNumOps(4);
				for (int batch = 0; batch < 200 && System.currentTimeMillis() - startMillis < MAX_MILLIS && batch * BATCH_SIZE < maxN; batch++) {
					List<Integer> indices = IntStream.range(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE - 1).boxed().map(v -> maxN + rd.nextInt(nMaxN)).collect(Collectors.toList());
					//Collections.shuffle(indices);
					MutableInt ctr2 = new MutableInt(0);
					int maxSize = indices.size();
					final int mBATCH = batch;
					indices.parallelStream().forEach(idx -> {
						if (ctr2.incrementAndGet() % 10 == 0)
							System.out.println("Done: " + (mBATCH * BATCH_SIZE + ctr2.intValue()) + " / " + (mBATCH * BATCH_SIZE + maxSize));

						List<RewriterAlphabetEncoder.Operand> ops = RewriterAlphabetEncoder.decodeOrderedStatements(idx);
						List<RewriterStatement> stmts = RewriterAlphabetEncoder.buildAllPossibleDAGs(ops, ctx, true);
						long actualCtr = 0;

						for (RewriterStatement dag : stmts) {
							List<RewriterStatement> expanded = new ArrayList<>();
							expanded.add(dag);
							//expanded.addAll(RewriterAlphabetEncoder.buildAssertionVariations(dag, ctx, true));
							expanded.addAll(RewriterAlphabetEncoder.buildVariations(dag, ctx));
							actualCtr += expanded.size();
							for (RewriterStatement stmt : expanded) {
								try {
									String mstmt = stmt.toParsableString(ctx, true);
									stmt = RewriterUtils.parse(mstmt, ctx);
									ctx.metaPropagator.apply(stmt);
									RewriterStatement canonicalForm = converter.apply(stmt);

									//canonicalForm.compress();
									//stmt.compress();
									synchronized (lock) {
										RewriterEquivalenceDatabase.DBEntry entry = canonicalExprDB.insert(ctx, canonicalForm, stmt);

										// Now, we use common variables
										if (entry.equivalences.size() > 1) {
											RewriterStatement commonForm = RewriterRuleCreator.createCommonForm(stmt, entry.equivalences.get(0), canonicalForm, entry.canonicalForm, ctx)._1;
											entry.equivalences.set(entry.equivalences.size()-1, commonForm);
										}

										if (entry.equivalences.size() == 2)
											foundEquivalences.add(entry);
									}
								} catch (Exception e) {
									System.err.println("Faulty expression: " + stmt.toParsableString(ctx));
									e.printStackTrace();
								}
							}
						}

						//System.out.println(ops + " >> " + actualCtr);
					});
				}
			}
		}

		//printEquivalences(/*foundEquivalences*/ Collections.emptyList(), System.currentTimeMillis() - startTime, generatedExpressions.longValue(), evaluatedExpressions.longValue(), totalCanonicalizationMillis.longValue(), failures.longValue(), true);

		System.out.println("===== SUGGESTED REWRITES =====");
		//List<Tuple5<Double, Long, Long, RewriterStatement, RewriterStatement>> rewrites = findSuggestedRewrites(foundEquivalences);
		List<Tuple4<RewriterStatement, List<RewriterStatement>, Long, Boolean>> rewrites = findSuggestedRewrites(foundEquivalences, maxCostSamples);
		foundEquivalences.clear();
		exactExprDB.clear();
		canonicalExprDB.clear();

		// Here, we create any rule
		//List<Tuple4<RewriterRule, Long, Long, Integer>> allRules = new ArrayList<>();
		List<Tuple4<RewriterRule, Long, Integer, Boolean>> allRules = new ArrayList<>();
		int mCtr = 0;
		//for (Tuple5<Double, Long, Long, RewriterStatement, RewriterStatement> rewrite : rewrites) {
		for (Tuple4<RewriterStatement, List<RewriterStatement>, Long, Boolean> rewrite : rewrites) {
			if (++mCtr % 100 == 0)
				System.out.println("Creating rule: " + mCtr + " / " + rewrites.size());

			try {
				RewriterRule rule;
				if (rewrite._4())
					rule = RewriterRuleCreator.createRuleFromCommonStatements(rewrite._1(), rewrite._2().get(0), ctx);
				else
					rule = RewriterRuleCreator.createConditionalRuleFromCommonStatements(rewrite._1(), rewrite._2(), ctx);

				allRules.add(new Tuple4<>(rule, rewrite._3(), rule.getStmt1().countInstructions(), rewrite._4()));
			} catch (Exception e) {
				System.err.println("An error occurred while trying to create a rule:");
				System.err.println(rewrite._1().toParsableString(ctx, true));
				for (RewriterStatement stmt : rewrite._2())
					System.err.println(stmt.toParsableString(ctx, true));
				e.printStackTrace();
			}
		}

		System.out.println("Rule creation complete!");

		allRules.sort(Comparator.comparing(Tuple4::_3));

		System.out.println("Rules sorted!");

		RewriterRuleCreator ruleCreator = new RewriterRuleCreator(ctx);
		List<RewriterRule> conditionalRules = new ArrayList<>();

		mCtr = 0;

		//for (Tuple4<RewriterRule, Long, Long, Integer> t : allRules) {
		for (Tuple4<RewriterRule, Long, Integer, Boolean> t : allRules) {
			if (++mCtr % 100 == 0)
				System.out.println("Registering rule: " + mCtr + " / " + allRules.size());

			try {
				// First, without validating correctness
				// This might throw out some fallback options if a rule turns out to be incorrect but we there is a huge performance benefit
				if (!t._1().isConditionalMultiRule()) {
					ruleCreator.registerRule(t._1(), converter, ctx);
				} else {
					conditionalRules.add(t._1());
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		System.out.println("Writing raw to files...");

		allRules = null;

		RewriterRuleSet rawRuleSet = ruleCreator.getRuleSet();

		try (FileWriter writer = new FileWriter(RewriteAutomaticallyGenerated.RAW_FILE_PATH)) {
			writer.write(rawRuleSet.serialize(ctx));
		} catch (IOException ex) {
			ex.printStackTrace();
		}

		System.out.println("Throwing out incorrect rules...");

		ruleCreator.throwOutInvalidRules(true, false);

		try (FileWriter writer = new FileWriter(RewriteAutomaticallyGenerated.VALIDATED_FILE_PATH)) {
			writer.write(rawRuleSet.serialize(ctx));
		} catch (IOException ex) {
			ex.printStackTrace();
		}

		System.out.println("Throwing out non-applicable rules...");
		ruleCreator.throwOutInvalidRules(false, true);

		System.out.println("Writing results...");

		try (FileWriter writer = new FileWriter(RewriteAutomaticallyGenerated.FILE_PATH)) {
			String serialized = ruleCreator.getRuleSet().serialize(ctx);
			System.out.println(serialized);
			writer.write(serialized);
		} catch (IOException ex) {
			ex.printStackTrace();
		}

		try (FileWriter writer = new FileWriter(RewriteAutomaticallyGenerated.FILE_PATH_CONDITIONAL)) {
			String serialized = new RewriterRuleSet(ctx, conditionalRules).serialize(ctx);
			System.out.println(serialized);
			writer.write(serialized);
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}

	private static boolean checkRelevance(List<RewriterStatement> stmts) {
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

	/**
	 * This function computes rewrite suggestions based on cost-estimates. To enable random sampling, sample_size should be bigger than 1.
	 * Note that random sampling might generate incorrect suggestions due to inaccurate cost-estimates (especially for fused ops)
	 * @param equivalences
	 * @param sample_size how many sparsity and dimension values should be sampled; a sample size of 1 uses a fixed cost esimtate with ncols=nrows=2000 and fully dense matrices
	 * @return
	 */
	private static /*List<Tuple5<Double, Long, Long, RewriterStatement, RewriterStatement>>*/List<Tuple4<RewriterStatement, List<RewriterStatement>, Long, Boolean>> findSuggestedRewrites(List<RewriterEquivalenceDatabase.DBEntry> equivalences, int sample_size) {
		//List<Tuple5<Double, Long, Long, RewriterStatement, RewriterStatement>> suggestedRewrites = SynchronizedList.decorate(new ArrayList<>());
		List<Tuple4<RewriterStatement, List<RewriterStatement>, Long, Boolean>> suggestions = SynchronizedList.decorate(new ArrayList<>());

		AtomicLong idCtr = new AtomicLong();
		equivalences.parallelStream().forEach(entry -> {
			try {
				List<RewriterStatement> mEq = entry.equivalences;
				RewriterAssertions assertions = RewriterAssertionUtils.buildImplicitAssertions(mEq.get(0), ctx);

				for (int i = 1; i < mEq.size(); i++)
					RewriterAssertionUtils.buildImplicitAssertions(mEq.get(i), assertions, ctx);

				List<Tuple2<List<Number>, List<Long>>> costs = RewriterCostEstimator.compareCosts(mEq, assertions, ctx, true, sample_size);
				Set<Tuple2<Integer, Integer>> rewriteProposals = RewriterCostEstimator.findOptima(costs);
				long mId = idCtr.incrementAndGet();

				if (!rewriteProposals.isEmpty()) {
					int targetIdx = rewriteProposals.stream().findFirst().get()._2;
					boolean hasOneTarget = rewriteProposals.stream().allMatch(t -> t._2 == targetIdx);

					// Group by origin expression
					Map<Integer, List<Tuple2<Integer, Integer>>> grouped = rewriteProposals.stream().collect(Collectors.groupingBy(Tuple2::_1));

					//System.out.println("Grouped: " + grouped.values());
					int found = 0;
					List<Integer> idx = null;

					for (List<Tuple2<Integer, Integer>> proposalsFromSameOrigin : grouped.values()) {
						if (mEq.get(proposalsFromSameOrigin.get(0)._1).toParsableString(ctx).matches("%\\*%\\(\\*\\((\\w+,\\w+)\\),t\\(\\w+\\)\\)")) {
							found++;
							if (idx == null)
								idx = new ArrayList<>();
							idx.add(proposalsFromSameOrigin.get(0)._1);
						}
						suggestions.add(new Tuple4<>(mEq.get(proposalsFromSameOrigin.get(0)._1), proposalsFromSameOrigin.stream().map(t -> mEq.get(t._2)).collect(Collectors.toList()), mId, hasOneTarget));
					}

					/*for (Tuple2<Integer, Integer> proposal : rewriteProposals) {
						if (!proposal._1.equals(proposal._2))
							suggestions.add(new Tuple4<>(mEq.get(proposal._1), mEq.get(proposal._2), mId, hasOneTarget));
					}*/
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		});

		return suggestions;
		//return Collections.emptyList();

		/*for (RewriterEquivalenceDatabase.DBEntry entry : equivalences) {
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
					System.out.println("Could not compute cost for: " + eq.toParsableString(ctx));
					e.printStackTrace();
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
						if (score > 1e-10) {
							//ctx.metaPropagator.apply(eq); // To (partially) decompress the statement
							//ctx.metaPropagator.apply(optimalStatement);
							suggestedRewrites.add(new Tuple5<>(score, cost, minCost, eq, optimalStatement));
						}
					}
				}
			}
		}

		suggestedRewrites.sort(Comparator.comparing(Tuple5::_1));
		Collections.reverse(suggestedRewrites);
		return suggestedRewrites;*/
	}
}
