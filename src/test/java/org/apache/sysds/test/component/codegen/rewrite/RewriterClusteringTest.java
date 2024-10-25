package org.apache.sysds.test.component.codegen.rewrite;

import org.apache.commons.lang3.mutable.MutableLong;
import org.apache.sysds.hops.rewriter.RewriterDatabase;
import org.apache.sysds.hops.rewriter.RewriterHeuristic;
import org.apache.sysds.hops.rewriter.RewriterRule;
import org.apache.sysds.hops.rewriter.RewriterRuleCollection;
import org.apache.sysds.hops.rewriter.RewriterRuleSet;
import org.apache.sysds.hops.rewriter.RewriterRuntimeUtils;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.hops.rewriter.TopologicalSort;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

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
		long startTime = System.currentTimeMillis();
		MutableLong generatedExpressions = new MutableLong(0);
		MutableLong evaluatedExpressions = new MutableLong(0);
		MutableLong failures = new MutableLong(0);

		RewriterDatabase exactExprDB = new RewriterDatabase();
		RewriterDatabase canonicalExprDB = new RewriterDatabase();
		List<RewriterStatement> foundEquivalences = new ArrayList<>();

		db.forEach(expr -> {
			// First, build all possible subtrees
			List<RewriterStatement> subExprs = RewriterUtils.generateSubtrees(expr, ctx);
			//List<RewriterStatement> subExprs = List.of(expr);
			long evaluationCtr = 0;

			for (RewriterStatement subExpr : subExprs) {
				try {
					if (!exactExprDB.insertEntry(ctx, subExpr))
						continue;

					evaluationCtr++;

					//System.out.println("Eval: " + subExpr.toParsableString(ctx, true));

					// Duplicate the statement as we do not want to canonicalize the original statement
					RewriterStatement canonicalForm = converter.apply(subExpr.nestedCopy());

					// Insert the canonical form or retrieve the existing entry
					RewriterStatement existingEntry = canonicalExprDB.insertOrReturn(ctx, canonicalForm);

					if (existingEntry == null) {
						List<RewriterStatement> equivalentExpressions = new ArrayList<>();
						equivalentExpressions.add(subExpr);
						canonicalForm.unsafePutMeta("equivalentExpressions", equivalentExpressions);
					} else {
						List<RewriterStatement> equivalentExpressions = (List<RewriterStatement>) existingEntry.getMeta("equivalentExpressions");
						equivalentExpressions.add(subExpr);

						if (equivalentExpressions.size() == 2)
							foundEquivalences.add(existingEntry);

						//System.out.println("Found equivalent statement!");
					}
				} catch (Exception e) {
					e.printStackTrace();
					failures.increment();
				}
			}

			generatedExpressions.add(subExprs.size());
			evaluatedExpressions.add(evaluationCtr);
		});

		printEquivalences(foundEquivalences, System.currentTimeMillis() - startTime, generatedExpressions.longValue(), evaluatedExpressions.longValue(), failures.longValue(), true);
	}

	private void printEquivalences(List<RewriterStatement> equivalentStatements, long cpuTime, long generatedExpressions, long evaluatedExpressions, long failures, boolean preFilter) {
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
		System.out.println("Total failures: " + failures);
	}

	private boolean checkRelevance(List<RewriterStatement> stmts) {
		boolean match = true;

		for (int i = 0; i < stmts.size(); i++) {
			for (int j = stmts.size() - 1; j > i; j--) {
				RewriterStatement stmt1 = stmts.get(i).nestedCopy();
				RewriterStatement stmt2 = stmts.get(j).nestedCopy();

				stmt1 = flattenAndMerge.apply(stmt1);
				stmt2 = flattenAndMerge.apply(stmt2);

				TopologicalSort.sort(stmt1, ctx);
				TopologicalSort.sort(stmt2, ctx);

				match &= stmt1.match(RewriterStatement.MatcherContext.exactMatchWithDifferentLiteralValues(ctx, stmt2));

				if (match && stmt2.toString(ctx).contains("t(t(")) {
					System.out.println("MATCH: " + stmt1.toParsableString(ctx) + " [" + stmt1.hashCode() + "]; " + stmt2.toParsableString(ctx) + "[" + stmt2.hashCode() + "]");
					stmt1.match(RewriterStatement.MatcherContext.exactMatchWithDifferentLiteralValues(ctx, stmt2).debug(true));
				}

				/*if (match)
					System.out.println("Equals: " + stmt1 + "; " + stmt2);
				else
					System.out.println("NEquals: " + stmt1 + "; " + stmt2);*/
			}
		}

		return !match;
	}
}
