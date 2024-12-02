package org.apache.sysds.test.component.codegen.rewrite.functions;

import org.apache.commons.lang3.mutable.MutableObject;
import org.apache.sysds.hops.rewriter.assertions.RewriterAssertions;
import org.apache.sysds.hops.rewriter.estimators.RewriterCostEstimator;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.junit.BeforeClass;
import org.junit.Test;
import scala.Tuple2;

import java.util.Set;
import java.util.function.Function;

public class CostEstimates {
	private static RuleContext ctx;
	private static Function<RewriterStatement, RewriterStatement> canonicalConverter;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
		canonicalConverter = RewriterUtils.buildCanonicalFormConverter(ctx, true);
	}

	@Test
	public void test1() {
		RewriterStatement stmt = RewriterUtils.parse("%*%(+(A,B), C)", ctx, "MATRIX:A,B,C");
		MutableObject<RewriterAssertions> assertionRef = new MutableObject<>();
		long cost1 = RewriterCostEstimator.estimateCost(stmt, el -> 2000L, ctx, assertionRef);
		System.out.println(cost1);
		long cost2 = RewriterCostEstimator.estimateCost(stmt.getChild(0), el -> 2000L, ctx, assertionRef);
		System.out.println(cost2);
		assert cost2 < cost1;
	}

	@Test
	public void test2() {
		RewriterStatement stmt = RewriterUtils.parse("*(+(1, 1), 2)", ctx, "LITERAL_INT:1,2");
		System.out.println(canonicalConverter.apply(stmt));
	}

	@Test
	public void test3() {
		RewriterStatement stmt = RewriterUtils.parse("_EClass(argList(1, ncol(X)))", ctx, "LITERAL_INT:1", "MATRIX:X");
		System.out.println(canonicalConverter.apply(stmt));
	}

	@Test
	public void test4() {
		RewriterStatement stmt1 = RewriterUtils.parse("t(%*%(+(A,B), C))", ctx, "MATRIX:A,B,C");
		RewriterStatement stmt2 = RewriterUtils.parse("%*%(t(C), t(+(A,B)))", ctx, "MATRIX:A,B,C");
		long cost1 = RewriterCostEstimator.estimateCost(stmt1, el -> 2000L, ctx);
		long cost2 = RewriterCostEstimator.estimateCost(stmt2, el -> 2000L, ctx);
		System.out.println("Cost1: " + cost1);
		System.out.println("Cost2: " + cost2);
		System.out.println("Ratio: " + ((double)cost1)/cost2);
		assert cost1 < cost2;
	}

	@Test
	public void test5() {
		RewriterStatement stmt1 = RewriterUtils.parse("t(/(*(A, B), C))", ctx, "MATRIX:A,B,C");
		RewriterStatement stmt2 = RewriterUtils.parse("/(*(t(A), t(B)), t(C))", ctx, "MATRIX:A,B,C");
		long cost1 = RewriterCostEstimator.estimateCost(stmt1, el -> 2000L, ctx);
		long cost2 = RewriterCostEstimator.estimateCost(stmt2, el -> 2000L, ctx);
		System.out.println("Cost1: " + cost1);
		System.out.println("Cost2: " + cost2);
		System.out.println("Ratio: " + ((double)cost1)/cost2);
		assert cost1 < cost2;

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void test6() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(+(A, B))", ctx, "MATRIX:A,B,C");
		RewriterStatement stmt2 = RewriterUtils.parse("+(sum(A), sum(B))", ctx, "MATRIX:A,B,C");
		long cost1 = RewriterCostEstimator.estimateCost(stmt1, el -> 2000L, ctx);
		long cost2 = RewriterCostEstimator.estimateCost(stmt2, el -> 2000L, ctx);
		System.out.println("Cost1: " + cost1);
		System.out.println("Cost2: " + cost2);
		System.out.println("Ratio: " + ((double)cost2)/cost1);
		assert cost2 < cost1;

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);
		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void test7() {
		RewriterStatement stmt1 = RewriterUtils.parse("cast.MATRIX(sum(A))", ctx, "MATRIX:A,B,C");
		RewriterStatement stmt2 = RewriterUtils.parse("rowSums(colSums(A))", ctx, "MATRIX:A,B,C");
		long cost1 = RewriterCostEstimator.estimateCost(stmt1, el -> 2000L, ctx);
		long cost2 = RewriterCostEstimator.estimateCost(stmt2, el -> 2000L, ctx);
		System.out.println("Cost1: " + cost1);
		System.out.println("Cost2: " + cost2);
		System.out.println("Ratio: " + ((double)cost1)/cost2);
		assert cost1 < cost2;

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void test8() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(*(diag(A), diag(B)))", ctx, "MATRIX:A,B,C");
		RewriterStatement stmt2 = RewriterUtils.parse("trace(*(A, B))", ctx, "MATRIX:A,B,C");

		long cost1 = RewriterCostEstimator.estimateCost(stmt1, el -> 2000L, ctx);
		long cost2 = RewriterCostEstimator.estimateCost(stmt2, el -> 2000L, ctx);
		System.out.println("Cost1: " + cost1);
		System.out.println("Cost2: " + cost2);
		System.out.println("Ratio: " + ((double)cost1)/cost2);
		assert cost1 < cost2;

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void test9() {
		String stmt1Str = "MATRIX:WM\n" +
				"FLOAT:m2X,c19b086e-34d2-46dd-9651-7b6d1d16e459\n" +
				"LITERAL_FLOAT:1.0\n" +
				"sqrt(*(m2X,/(sum(WM),-(c19b086e-34d2-46dd-9651-7b6d1d16e459,1.0))))";
		String stmt2Str = "MATRIX:1167aa9b-102a-4bae-9801-8b18d210f954\n" +
				"FLOAT:m2,41d7e6fb-d4a7-45cf-89cb-cea8ecf3430a\n" +
				"LITERAL_FLOAT:1.0\n" +
				"sqrt(/(*(m2,sum(1167aa9b-102a-4bae-9801-8b18d210f954)),-(41d7e6fb-d4a7-45cf-89cb-cea8ecf3430a,1.0)))";

		RewriterStatement stmt1 = RewriterUtils.parse(stmt1Str, ctx);
		RewriterStatement stmt2 = RewriterUtils.parse(stmt2Str, ctx);

		long cost1 = RewriterCostEstimator.estimateCost(stmt1, el -> 2000L, ctx);
		long cost2 = RewriterCostEstimator.estimateCost(stmt2, el -> 2000L, ctx);
		System.out.println("Cost1: " + cost1);
		System.out.println("Cost2: " + cost2);
		System.out.println("Ratio: " + ((double)cost1)/cost2);
		assert cost1 == cost2;

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void test10() {
		String stmt1Str = "INT:num_records\n" +
				"LITERAL_INT:3\n" +
				"*(num_records,3)";
		String stmt2Str = "LITERAL_INT:3\n" +
				"INT:run_index\n" +
				"*(3,run_index)";

		RewriterStatement stmt1 = RewriterUtils.parse(stmt1Str, ctx);
		RewriterStatement stmt2 = RewriterUtils.parse(stmt2Str, ctx);

		long cost1 = RewriterCostEstimator.estimateCost(stmt1, el -> 2000L, ctx);
		long cost2 = RewriterCostEstimator.estimateCost(stmt2, el -> 2000L, ctx);
		System.out.println("Cost1: " + cost1);
		System.out.println("Cost2: " + cost2);
		System.out.println("Ratio: " + ((double)cost1)/cost2);
		assert cost1 == cost2;

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void test11() {
		String stmtStr1 = "MATRIX:A,p_CG,z\n" +
				"FLOAT:trust_delta_sq\n" +
				"*(cast.FLOAT(A),cast.FLOAT(%*%(p_CG,z)))";
		String stmtStr2 = "MATRIX:A,p_CG,z\n" +
				"FLOAT:trust_delta_sq\n" +
				"*(cast.FLOAT(%*%(p_CG,z)),cast.FLOAT(A))";

		RewriterStatement stmt1 = RewriterUtils.parse(stmtStr1, ctx);
		RewriterStatement stmt2 = RewriterUtils.parse(stmtStr2, ctx);

		long cost1 = RewriterCostEstimator.estimateCost(stmt1, el -> 2000L, ctx);
		long cost2 = RewriterCostEstimator.estimateCost(stmt2, el -> 2000L, ctx);
		System.out.println("Cost1: " + cost1);
		System.out.println("Cost2: " + cost2);
		System.out.println("Ratio: " + ((double)cost1)/cost2);

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void test12() {
		String stmtStr1 = "MATRIX:A,B\n" +
				"LITERAL_INT:1\n" +
				"+([](A, 1, nrow(A), 1, 1),B)";
		String stmtStr2 = "MATRIX:A,B\n" +
				"LITERAL_INT:1\n" +
				"+([](A, 1, nrow(A), 1, ncol(A)), B)";

		RewriterStatement stmt1 = RewriterUtils.parse(stmtStr1, ctx);
		RewriterStatement stmt2 = RewriterUtils.parse(stmtStr2, ctx);

		long cost1 = RewriterCostEstimator.estimateCost(stmt1, el -> 2000L, ctx);
		long cost2 = RewriterCostEstimator.estimateCost(stmt2, el -> 2000L, ctx);
		System.out.println("Cost1: " + cost1);
		System.out.println("Cost2: " + cost2);
		System.out.println("Ratio: " + ((double)cost1)/cost2);

		assert cost1 < cost2;
	}

	@Test
	public void test13() {
		String stmtStr1 = "MATRIX:A,B\n" +
				"LITERAL_INT:1\n" +
				"[](rowSums(A), 1, nrow(A), 1, 1)";
		String stmtStr2 = "MATRIX:A,B\n" +
				"LITERAL_INT:1\n" +
				"rowSums(A)";

		RewriterStatement stmt1 = RewriterUtils.parse(stmtStr1, ctx);
		RewriterStatement stmt2 = RewriterUtils.parse(stmtStr2, ctx);

		long cost1 = RewriterCostEstimator.estimateCost(stmt1, el -> 2000L, ctx);
		long cost2 = RewriterCostEstimator.estimateCost(stmt2, el -> 2000L, ctx);
		System.out.println("Cost1: " + cost1);
		System.out.println("Cost2: " + cost2);
		System.out.println("Ratio: " + ((double)cost1)/cost2);

		assert cost2 < cost1;

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void test14() {
		RewriterStatement stmt1 = RewriterUtils.parse("t(t(A))", ctx, "MATRIX:A");
		MutableObject<RewriterAssertions> assertionRef = new MutableObject<>();
		long maxCost = RewriterCostEstimator.estimateCost(stmt1, ctx, assertionRef);
		Tuple2<Set<RewriterStatement>, Boolean> allowedCombinations = RewriterCostEstimator.determineSingleReferenceRequirement(stmt1, RewriterCostEstimator.DEFAULT_COST_FN, assertionRef.getValue(), 0, maxCost, ctx);
		System.out.println(allowedCombinations._1);
		System.out.println("AllowCombinations: " + allowedCombinations._2);
		assert allowedCombinations._1.size() == 1;
	}

	@Test
	public void test15() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(rowSums(A))", ctx, "MATRIX:A");
		RewriterStatement stmt2 = RewriterUtils.parse("sum(A)", ctx, "MATRIX:A");
		MutableObject<RewriterAssertions> assertionRef = new MutableObject<>();
		long maxCost = RewriterCostEstimator.estimateCost(stmt1, ctx, assertionRef);
		long fullCost = RewriterCostEstimator.estimateCost(stmt2, ctx, assertionRef);
		Tuple2<Set<RewriterStatement>, Boolean> allowedCombinations = RewriterCostEstimator.determineSingleReferenceRequirement(stmt1, RewriterCostEstimator.DEFAULT_COST_FN, assertionRef.getValue(), fullCost, maxCost, ctx);
		System.out.println(allowedCombinations._1);
		System.out.println("AllowCombinations: " + allowedCombinations._2);
		assert allowedCombinations._1.isEmpty();
	}

	@Test
	public void test16() {
		RewriterStatement stmt1 = RewriterUtils.parse("+(colSums(A),[](B,1,1,1,ncol(B)))", ctx, "MATRIX:A,B", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("+(colSums(A),colSums([](B,1,1,1,ncol(B))))", ctx, "MATRIX:A,B", "LITERAL_INT:1");
		long cost1 = RewriterCostEstimator.estimateCost(stmt1, ctx);
		long cost2 = RewriterCostEstimator.estimateCost(stmt2, ctx);
		assert cost1 < cost2;
	}

	@Test
	public void test17() {
		RewriterStatement stmt1 = RewriterUtils.parse("%*%(colVec(A),B)", ctx, "MATRIX:A,B", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("%*%(colSums(colVec(A)),B)", ctx, "MATRIX:A,B", "LITERAL_INT:1");
		long cost1 = RewriterCostEstimator.estimateCost(stmt1, ctx);
		long cost2 = RewriterCostEstimator.estimateCost(stmt2, ctx);
		System.out.println("Cost1: " + cost1);
		System.out.println("Cost2: " + cost2);
		assert cost1 < cost2;
	}
}
