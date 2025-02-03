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

package org.apache.sysds.hops.rewriter;

import org.apache.commons.collections.list.SynchronizedList;
import org.apache.commons.lang3.mutable.MutableInt;
import org.apache.sysds.hops.rewriter.assertions.RewriterAssertionUtils;
import org.apache.sysds.hops.rewriter.assertions.RewriterAssertions;
import org.apache.sysds.hops.rewriter.estimators.RewriterCostEstimator;
import org.apache.sysds.hops.rewriter.generated.RewriteAutomaticallyGenerated;
import org.apache.sysds.hops.rewriter.rule.RewriterRule;
import org.apache.sysds.hops.rewriter.rule.RewriterRuleCreator;
import org.apache.sysds.hops.rewriter.rule.RewriterRuleSet;
import org.apache.sysds.hops.rewriter.utils.RewriterSearchUtils;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;
import scala.Tuple2;
import scala.Tuple4;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
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

public class RewriterFramework {

	// To test the framework
	public static void main(String[] args) {
		String dbPath = "./src/test/resources/rewriterframework/expressions.db";
		RewriterFramework rwf = new RewriterFramework(dbPath);
		rwf.init(true,true);
		rwf.dataDrivenSearch(1000);
		rwf.systematicSearch(3);
		//rwf.randomSearch(4, 4, 5000);
		rwf.createRules(true);
		rwf.removeInvalidRules();
		// Note that unconditional rules are not 'static' rules.
		// It is a set of equivalences that have a single optimal expression
		System.out.println(rwf.getUnconditionalRuleSet());
		//rwf.removeInapplicableRules();
		//System.out.println(rwf.getUnconditionalRuleSet().toJavaCode("GeneratedRewriteClass", true));

		/*RewriterRuleSet rs = loadRuleSet(rPath);
		saveJavaCode(sPath, rs, "GeneratedRewriteClass", true);*/
	}


	private RuleContext ctx;
	private Function<RewriterStatement, RewriterStatement> converter;
	private RewriterDatabase db;
	private String dbFile;

	private int BATCH_SIZE = 1000;
	private int MAX_COST_SAMPLES = 50;

	private RewriterEquivalenceDatabase equivalenceDB;
	private List<RewriterEquivalenceDatabase.DBEntry> foundEquivalences;
	private boolean pruneNovelExpressions = false;

	private RewriterRuleCreator unconditionalRuleCreator;
	private RewriterRuleSet conditionalRuleSet;

	public RewriterFramework(String dbFile) {
		this.dbFile = dbFile;
	}

	private void setupDataDrivenSearch() {
		if (db != null && db.size() > 0)
			return; // Then a database has already been loaded

		try(BufferedReader reader = new BufferedReader(new FileReader(dbFile))) {
			db.deserialize(reader, ctx);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Initializes the rewriter framework
	 * @param allowInversionCanonicalization if the conversion from a/c =&gt; a*(c^-1) should be applied (during canonicalization)
	 * @param pruneNovelExpressions if only equivalence groups should be stored, where at least one expression was in the data-set
	 */
	public void init(boolean allowInversionCanonicalization, boolean pruneNovelExpressions) {
		ctx = RewriterUtils.buildDefaultContext();
		converter = RewriterUtils.buildCanonicalFormConverter(ctx, allowInversionCanonicalization, false);
		db = new RewriterDatabase();
		equivalenceDB = new RewriterEquivalenceDatabase();
		foundEquivalences = new ArrayList<>();
		this.pruneNovelExpressions = pruneNovelExpressions;
	}

	public RuleContext getContext() {
		return ctx;
	}

	/**
	 * Performs a data-driven search where existing expressions and their subexpressions are considered
	 * @param exprPruningThreshold the maximum number of generated subexpressions (to avoid exploding numbers of subgraphs for big graphs)
	 */
	public void dataDrivenSearch(int exprPruningThreshold) {
		setupDataDrivenSearch(); // Load the expression DB

		int size = db.size();
		RewriterDatabase exactExprDB = new RewriterDatabase();

		MutableInt ctr = new MutableInt(0);
		MutableInt failures = new MutableInt(0);
		MutableInt generatedExpressions = new MutableInt(0);
		MutableInt evaluatedExpressions = new MutableInt(0);
		MutableInt totalCanonicalizationMillis = new MutableInt(0);
		db.parForEach(expr -> {
			if (ctr.incrementAndGet() % 10 == 0)
				System.out.println("Done: " + ctr.intValue() + " / " + size);

			List<RewriterStatement> subExprs = RewriterSearchUtils.generateSubtrees(expr, ctx, exprPruningThreshold);
			if (subExprs.size() > exprPruningThreshold)
				System.out.println("Critical number of subtrees: " + subExprs.size());
			if (subExprs.size() > 2 * exprPruningThreshold) {
				System.out.println("Skipping subtrees...");
				subExprs = List.of(expr);
			}
			long evaluationCtr = 0;
			long mCanonicalizationMillis = 0;

			for (RewriterStatement subExpr : subExprs) {
				try {
					if (!exactExprDB.insertEntry(ctx, subExpr))
						continue;

					evaluationCtr++;

					// Duplicate the statement as we do not want to canonicalize the original statement
					long startMillis = System.currentTimeMillis();
					RewriterStatement canonicalForm = converter.apply(subExpr);
					mCanonicalizationMillis += System.currentTimeMillis() - startMillis;

					synchronized (this) {
						RewriterEquivalenceDatabase.DBEntry entry = equivalenceDB.insert(ctx, canonicalForm, subExpr);

						// Now, we use common variables
						if (entry.equivalences.size() > 1) {
							RewriterStatement commonForm = RewriterRuleCreator.createCommonForm(subExpr, entry.equivalences.get(0), canonicalForm, entry.canonicalForm, ctx)._1;
							entry.equivalences.set(entry.equivalences.size()-1, commonForm);
						}

						if (entry.equivalences.size() == 2)
							foundEquivalences.add(entry);
					}
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

	/**
	 * Performs a systematic search
	 * @param maxDepth the maximum number of (virtual) operands
	 */
	public void systematicSearch(int maxDepth) {
		systematicSearch(0, RewriterSearchUtils.getMaxSearchNumberForNumOps(maxDepth), true, false);
	}

	/**
	 * Performs a systematic search
	 * @param maxDepth the maximum number of (virtual) operands
	 * @param includeDuplicateReferences if the search space should be extended to contain a shared variable (e.g. +(A,B) =&gt; [+(A,B), +(A,A)])
	 */
	public void systematicSearch(int maxDepth, boolean includeDuplicateReferences) {
		systematicSearch(0, RewriterSearchUtils.getMaxSearchNumberForNumOps(maxDepth), includeDuplicateReferences, false);
	}

	/**
	 * Performs a systematic search
	 * @param fromIdx the start index
	 * @param toIdx the end index
	 * @param includeDuplicateReferences if the search space should be extended to contain a shared variable (e.g. +(A,B) =&gt; [+(A,B), +(A,A)])
	 * @param includeRowColVectors if row-vectors and col-vectors should be included in the search (note that the data-driven approach does not support this)
	 */
	public void systematicSearch(int fromIdx, int toIdx, boolean includeDuplicateReferences, boolean includeRowColVectors) {
		int diff = toIdx - fromIdx;
		int maxN = toIdx;

		for (int batch = 0; batch < 10000 && batch * BATCH_SIZE < diff; batch++) {
			List<Integer> indices = IntStream.range(fromIdx + batch * BATCH_SIZE, fromIdx + Math.min((batch + 1) * BATCH_SIZE - 1, maxN)).boxed().collect(Collectors.toList());
			Collections.shuffle(indices);
			MutableInt ctr2 = new MutableInt(0);
			int maxSize = indices.size();
			final int mBATCH = batch;
			indices.parallelStream().forEach(idx -> {
				if (ctr2.incrementAndGet() % 10 == 0)
					System.out.println("Done: " + (mBATCH * BATCH_SIZE + ctr2.intValue()) + " / " + (mBATCH * BATCH_SIZE + maxSize));


				List<RewriterSearchUtils.Operand> ops = RewriterSearchUtils.decodeOrderedStatements(idx);
				List<RewriterStatement> stmts = RewriterSearchUtils.buildAllPossibleDAGs(ops, ctx, true);

				for (RewriterStatement dag : stmts) {
					List<RewriterStatement> expanded = new ArrayList<>();
					expanded.add(dag);
					if (includeDuplicateReferences)
						expanded.addAll(RewriterSearchUtils.buildVariations(dag, ctx));
					if (includeRowColVectors)
						expanded.addAll(RewriterSearchUtils.buildAssertionVariations(dag, ctx));

					insertEquivalences(expanded);
				}
			});
		}
	}

	public void randomSearch(int minExprSize, int maxExprSize, int numSamples) {
		randomSearchFromIndex(RewriterSearchUtils.getMaxSearchNumberForNumOps(minExprSize-1)+1, RewriterSearchUtils.getMaxSearchNumberForNumOps(maxExprSize), numSamples, true, false);
	}

	/**
	 * Performs a random search. Samples numSamples expression groups (groups of expressions encoded by a single integer)
	 * @param fromIdx the start index
	 * @param toIdx the end index
	 * @param numSamples the number of sampmles
	 * @param includeDuplicateReferences if expressions such as +(A,A) should be included in the search
	 * @param includeRowColVectors if row-col vectors should be included in the search
	 */
	public void randomSearchFromIndex(int fromIdx, int toIdx, int numSamples, boolean includeDuplicateReferences, boolean includeRowColVectors) {
		// Now we will just do random sampling for a few rounds
		Random rd = new Random(42);
		for (int batch = 0; batch < 200 && batch * BATCH_SIZE < numSamples; batch++) {
			List<Integer> indices = IntStream.range(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE - 1).boxed().map(v -> fromIdx + rd.nextInt(toIdx-fromIdx)).collect(Collectors.toList());
			MutableInt ctr2 = new MutableInt(0);
			int maxSize = indices.size();
			final int mBATCH = batch;
			indices.parallelStream().forEach(idx -> {
				if (ctr2.incrementAndGet() % 10 == 0)
					System.out.println("Done: " + (mBATCH * BATCH_SIZE + ctr2.intValue()) + " / " + (mBATCH * BATCH_SIZE + maxSize));

				List<RewriterSearchUtils.Operand> ops = RewriterSearchUtils.decodeOrderedStatements(idx);
				List<RewriterStatement> stmts = RewriterSearchUtils.buildAllPossibleDAGs(ops, ctx, true);

				for (RewriterStatement dag : stmts) {
					List<RewriterStatement> expanded = new ArrayList<>();
					expanded.add(dag);
					if (includeDuplicateReferences)
						expanded.addAll(RewriterSearchUtils.buildVariations(dag, ctx));
					if (includeRowColVectors)
						expanded.addAll(RewriterSearchUtils.buildAssertionVariations(dag, ctx));

					insertEquivalences(expanded);
				}
			});
		}
	}

	private void insertEquivalences(List<RewriterStatement> stmts) {
		for (RewriterStatement stmt : stmts) {
			try {
				RewriterStatement canonicalForm = converter.apply(stmt);

				synchronized (this) {
					if (pruneNovelExpressions && !equivalenceDB.containsEntry(canonicalForm))
						return;

					RewriterEquivalenceDatabase.DBEntry entry = equivalenceDB.insert(ctx, canonicalForm, stmt);

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

	/**
	 * Create rules from all observed equivalences
	 * @param freeDBMemory if all the stored equivalences that are not needed for rule creation should be dropped immediately
	 */
	public void createRules(boolean freeDBMemory) {
		System.out.println("===== SUGGESTED REWRITES =====");
		List<Tuple4<RewriterStatement, List<RewriterStatement>, Long, Boolean>> rewrites = findSuggestedRewrites(foundEquivalences, MAX_COST_SAMPLES);

		if (freeDBMemory) {
			db.clear();
			foundEquivalences.clear();
			equivalenceDB.clear();
		}

		// Here, we create any rule
		List<Tuple4<RewriterRule, Long, Integer, Boolean>> allRules = new ArrayList<>();
		int mCtr = 0;
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

		unconditionalRuleCreator = new RewriterRuleCreator(ctx);
		List<RewriterRule> conditionalRules = new ArrayList<>();

		mCtr = 0;

		for (Tuple4<RewriterRule, Long, Integer, Boolean> t : allRules) {
			if (++mCtr % 100 == 0)
				System.out.println("Registering rule: " + mCtr + " / " + allRules.size());

			try {
				// First, without validating correctness
				// This might throw out some fallback options if a rule turns out to be incorrect but we there is a huge performance benefit
				if (!t._1().isConditionalMultiRule()) {
					unconditionalRuleCreator.registerRule(t._1(), converter, ctx);
				} else {
					conditionalRules.add(t._1());
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		conditionalRuleSet = new RewriterRuleSet(ctx, conditionalRules);
	}

	/**
	 * This function removes rules where the output of the origin expression does not match
	 * the output of the target expression.
	 */
	public void removeInvalidRules() {
		unconditionalRuleCreator.throwOutInvalidRules(true, false);
	}

	/**
	 * This function removes rules where the origin expression is modified by the HOP-DAG rewriter.
	 * We aim to remove rules that are already implemented by intercepting the HOP-DAG after rewriting.
	 * We disable operator fusion and sum-product rewrites during execution.
	 * However, we throw away any rule that does not match our expected DAG structure, which may affect
	 * valid rules that are not correctly extracted during runtime.
	 */
	public void removeInapplicableRules() {
		unconditionalRuleCreator.throwOutInvalidRules(false, true);
	}

	/**
	 *
	 * @return the unconditional rule set (includes rules where there is exactly one possible optimum per equality set)
	 */
	public RewriterRuleSet getUnconditionalRuleSet() {
		return unconditionalRuleCreator.getRuleSet();
	}

	/**
	 *
	 * @return the conditional rule set (rules where the optimal expression may change, e.g., (A*B)+(A*C) &lt;=&gt; A*(B+C))
	 */
	public RewriterRuleSet getConditionalRuleSet() {
		return conditionalRuleSet;
	}

	public static boolean saveRuleSet(String filePath, RewriterRuleSet ruleSet) {
		try (FileWriter writer = new FileWriter(filePath)) {
			writer.write(ruleSet.serialize());
		} catch (IOException ex) {
			ex.printStackTrace();
			return false;
		}

		return true;
	}

	public static RewriterRuleSet loadRuleSet(String filePath) {
		try {
			List<String> lines = Files.readAllLines(Paths.get(filePath));
			return RewriterRuleSet.deserialize(lines, RewriterUtils.buildDefaultContext());
		} catch (IOException ex) {
			ex.printStackTrace();
			return null;
		}
	}

	public static boolean saveJavaCode(String filePath, RewriterRuleSet ruleSet, String className, boolean optimize) {
		try (FileWriter writer = new FileWriter(filePath)) {
			writer.write(ruleSet.toJavaCode(className, optimize));
		} catch (IOException ex) {
			ex.printStackTrace();
			return false;
		}

		return true;
	}

	/**
	 * This function computes rewrite suggestions based on cost-estimates. To enable random sampling, sample_size should be bigger than 1.
	 * Note that random sampling might generate incorrect suggestions due to inaccurate cost-estimates (especially for fused ops)
	 * @param equivalences
	 * @param sample_size how many sparsity and dimension values should be sampled; a sample size of 1 uses a fixed cost esimtate with ncols=nrows=2000 and fully dense matrices
	 * @return
	 */
	private List<Tuple4<RewriterStatement, List<RewriterStatement>, Long, Boolean>> findSuggestedRewrites(List<RewriterEquivalenceDatabase.DBEntry> equivalences, int sample_size) {
		List<Tuple4<RewriterStatement, List<RewriterStatement>, Long, Boolean>> suggestions = SynchronizedList.decorate(new ArrayList<>());

		AtomicLong idCtr = new AtomicLong();
		equivalences.parallelStream().forEach(entry -> {
			try {
				List<RewriterStatement> mEq = entry.equivalences;
				RewriterAssertions assertions = RewriterAssertionUtils.buildImplicitAssertions(mEq.get(0), ctx);

				for (int i = 1; i < mEq.size(); i++)
					RewriterAssertionUtils.buildImplicitAssertions(mEq.get(i), assertions, ctx);

				List<Tuple2<List<Number>, List<Long>>> costs = RewriterCostEstimator.compareCosts(mEq, assertions, ctx, true, 0);

				Set<Tuple2<Integer, Integer>> rewriteProposals = RewriterCostEstimator.findOptima(costs);
				long mId = idCtr.incrementAndGet();

				if (!rewriteProposals.isEmpty()) {
					int targetIdx = rewriteProposals.stream().findFirst().get()._2;
					boolean hasOneTarget = rewriteProposals.stream().allMatch(t -> t._2 == targetIdx);

					// Group by origin expression
					Map<Integer, List<Tuple2<Integer, Integer>>> grouped = rewriteProposals.stream().collect(Collectors.groupingBy(Tuple2::_1));

					for (List<Tuple2<Integer, Integer>> proposalsFromSameOrigin : grouped.values()) {
						suggestions.add(new Tuple4<>(mEq.get(proposalsFromSameOrigin.get(0)._1), proposalsFromSameOrigin.stream().map(t -> mEq.get(t._2)).collect(Collectors.toList()), mId, hasOneTarget));
					}
				}
			} catch (Exception e) {
				//e.printStackTrace();
			}
		});

		return suggestions;
	}
}
