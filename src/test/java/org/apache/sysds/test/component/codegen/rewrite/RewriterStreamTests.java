package org.apache.sysds.test.component.codegen.rewrite;

import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.function.Function;

public class RewriterStreamTests {

	private static RuleContext ctx;
	private static Function<RewriterStatement, RewriterStatement> converter;
	private static Function<RewriterStatement, RewriterStatement> canonicalConverter;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
		converter = RewriterUtils.buildFusedOperatorCreator(ctx, true);
		canonicalConverter = RewriterUtils.buildCanonicalFormConverter(ctx, true);
	}

	@Test
	public void testAdditionFloat1() {
		RewriterStatement stmt = RewriterUtils.parse("+(+(a, b), 1)", ctx, "MATRIX:A,B,C", "FLOAT:a,b", "LITERAL_INT:0,1");
		stmt = converter.apply(stmt);
		System.out.println(stmt.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, RewriterUtils.parse("+(argList(a, b, 1))", ctx, "FLOAT:a,b", "LITERAL_INT:0,1")));
	}

	@Test
	public void testAdditionFloat2() {
		RewriterStatement stmt = RewriterUtils.parse("+(1, +(a, b))", ctx, "FLOAT:a,b", "LITERAL_INT:0,1");
		stmt = converter.apply(stmt);
		System.out.println(stmt.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, RewriterUtils.parse("+(argList(a, b, 1))", ctx, "FLOAT:a,b", "LITERAL_INT:0,1")));
	}

	@Test
	public void testAdditionMatrix1() {
		RewriterStatement stmt = RewriterUtils.parse("+(+(A, B), 1)", ctx, "MATRIX:A,B", "LITERAL_INT:0,1");
		stmt = converter.apply(stmt);
		System.out.println(stmt.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, RewriterUtils.parse("_m($1:_idx(1, nrow(A)), $2:_idx(1, ncol(A)), +(argList([](B, $1, $2), [](A, $1, $2), 1)))", ctx, "MATRIX:A,B", "LITERAL_INT:0,1")));
	}

	@Test
	public void testAdditionMatrix2() {
		RewriterStatement stmt = RewriterUtils.parse("+(1, +(A, B))", ctx, "MATRIX:A,B", "LITERAL_INT:0,1");
		stmt = converter.apply(stmt);
		System.out.println(stmt.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, RewriterUtils.parse("_m($1:_idx(1, nrow(A)), $2:_idx(1, ncol(A)), +(argList([](A, $1, $2), [](B, $1, $2), 1)))", ctx, "MATRIX:A,B", "LITERAL_INT:0,1")));
	}

	@Test
	public void testSubtractionFloat1() {
		RewriterStatement stmt = RewriterUtils.parse("+(-(a, b), 1)", ctx, "MATRIX:A,B,C", "FLOAT:a,b", "LITERAL_INT:0,1");
		stmt = converter.apply(stmt);
		System.out.println(stmt.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, RewriterUtils.parse("+(argList(-(b), a, 1))", ctx, "FLOAT:a,b", "LITERAL_INT:0,1")));
	}

	@Test
	public void testSubtractionFloat2() {
		RewriterStatement stmt = RewriterUtils.parse("+(1, -(a, -(b, c)))", ctx, "MATRIX:A,B,C", "FLOAT:a,b,c", "LITERAL_INT:0,1");
		stmt = converter.apply(stmt);
		System.out.println(stmt.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, RewriterUtils.parse("+(argList(-(b), a, c, 1))", ctx, "FLOAT:a,b, c", "LITERAL_INT:0,1")));
	}

	@Test
	public void testFusedPlanMatrixGeneration() {
		RewriterStatement stmt = RewriterUtils.parse("+(1, +(A, B))", ctx, "MATRIX:A,B", "LITERAL_INT:0,1");
		stmt = converter.apply(stmt);
		RewriterStatement fused = RewriterUtils.buildFusedPlan(stmt, ctx);
		System.out.println("Orig: " + stmt.toParsableString(ctx, true));
		System.out.println("Fused: " + (fused == null ? null : fused.toParsableString(ctx, true)));
	}

	@Test
	public void testFusedPlanAggregationGeneration() {
		RewriterStatement stmt = RewriterUtils.parse("sum(*(/(A, B), B))", ctx, "MATRIX:A,B", "LITERAL_INT:0,1");
		stmt = converter.apply(stmt);
		RewriterStatement fused = RewriterUtils.buildFusedPlan(stmt, ctx);
		System.out.println("Orig: " + stmt.toParsableString(ctx, true));
		System.out.println("Fused: " + (fused == null ? null : fused.toParsableString(ctx, true)));
	}

	@Test
	public void testFusedPlanAdvancedAggregationGeneration() {
		RewriterStatement stmt = RewriterUtils.parse("sum(*(t(A), B))", ctx, "MATRIX:A,B", "LITERAL_INT:0,1");
		stmt = converter.apply(stmt);
		RewriterStatement fused = RewriterUtils.buildFusedPlan(stmt, ctx);
		System.out.println("Orig: " + stmt.toParsableString(ctx, true));
		System.out.println("Fused: " + (fused == null ? null : fused.toParsableString(ctx, true)));
	}

	@Test
	public void testReorgEquivalence() {
		RewriterStatement stmt1 = RewriterUtils.parse("t(t(A))", ctx, "MATRIX:A");
		RewriterStatement stmt2 = RewriterUtils.parse("A", ctx, "MATRIX:A");
		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2));
	}

	@Test
	public void testTraceEquivalence1() {
		RewriterStatement stmt = RewriterUtils.parse("trace(%*%(A, B))", ctx, "MATRIX:A,B");
		RewriterStatement stmt2 = RewriterUtils.parse("sum(*(A, t(B)))", ctx, "MATRIX:A,B");
		stmt = canonicalConverter.apply(stmt);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2));
	}

	@Test
	public void testTraceEquivalence2() {
		RewriterStatement stmt = RewriterUtils.parse("trace(%*%(A, B))", ctx, "MATRIX:A,B");
		RewriterStatement stmt2 = RewriterUtils.parse("sum(*(A, t(B)))", ctx, "MATRIX:A,B");
		stmt = canonicalConverter.apply(stmt);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2));
	}

	@Test
	public void testTraceEquivalence3() {
		RewriterStatement stmt = RewriterUtils.parse("trace(*(A, B))", ctx, "MATRIX:A,B");
		RewriterStatement stmt2 = RewriterUtils.parse("sum(*(diag(A), diag(B)))", ctx, "MATRIX:A,B");
		stmt = canonicalConverter.apply(stmt);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2));
	}

	@Test
	public void testAggEquivalence() {
		RewriterStatement stmt = RewriterUtils.parse("sum(%*%(A, B))", ctx, "MATRIX:A,B");
		RewriterStatement stmt2 = RewriterUtils.parse("sum(*(colSums(A), t(rowSums(B))))", ctx, "MATRIX:A,B");
		stmt = canonicalConverter.apply(stmt);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2));
	}

	@Test
	public void testSumInequality() {
		RewriterStatement stmt = RewriterUtils.parse("sum(+(B, sum(*(a, A))))", ctx, "MATRIX:A,B", "FLOAT:a");
		RewriterStatement stmt2 = RewriterUtils.parse("sum(+(B, *(a, sum(A))))", ctx, "MATRIX:A,B", "FLOAT:a");
		stmt = canonicalConverter.apply(stmt);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert !stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2));
	}

	@Test
	public void testSumEquality() {
		RewriterStatement stmt = RewriterUtils.parse("sum(+(B, sum(*(a, A))))", ctx, "MATRIX:A,B", "FLOAT:a");
		RewriterStatement stmt2 = RewriterUtils.parse("sum(+(B, *(*(*(nrow(A), ncol(A)), a), sum(A))))", ctx, "MATRIX:A,B", "FLOAT:a");
		stmt = canonicalConverter.apply(stmt);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2));
	}

	@Test
	public void testArgListSelectionPushdown() {
		RewriterStatement stmt = RewriterUtils.parse("[](+(A, 1), 1, 1)", ctx, "MATRIX:A", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("+([](A, 1, 1), 1)", ctx, "MATRIX:A", "LITERAL_INT:1");
		stmt = canonicalConverter.apply(stmt);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2));
	}

	@Test
	public void testDistributiveLaw1() {
		RewriterStatement stmt = RewriterUtils.parse("*(+(a, b), c)", ctx, "FLOAT:a,b,c");
		RewriterStatement stmt2 = RewriterUtils.parse("+(*(a, c), *(b, c))", ctx, "FLOAT:a,b,c");
		stmt = canonicalConverter.apply(stmt);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2));
	}

	@Test
	public void testDistributiveLaw2() {
		RewriterStatement stmt = RewriterUtils.parse("*(a, +(b, c))", ctx, "FLOAT:a,b,c");
		RewriterStatement stmt2 = RewriterUtils.parse("+(*(a, b), *(a, c))", ctx, "FLOAT:a,b,c");
		stmt = canonicalConverter.apply(stmt);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2));
	}

	@Test
	public void testEClassProperties() {
		RewriterStatement stmt = RewriterUtils.parse("*(+(A, B), nrow(A))", ctx, "MATRIX:A,B");
		RewriterStatement stmt2 = RewriterUtils.parse("*(+(A, B), nrow(B))", ctx, "MATRIX:A,B");
		stmt = canonicalConverter.apply(stmt);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2));
	}

	@Test
	public void testRealExamples1() {
		RewriterStatement stmt1 = RewriterUtils.parse("t(%*%(t(U),V))", ctx, "MATRIX:U,V");
		RewriterStatement stmt2 = RewriterUtils.parse("%*%(t(V), U)", ctx, "MATRIX:U,V");
		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2));
	}

	@Test
	public void test() {
		RewriterStatement stmt = RewriterUtils.parse("t(A)", ctx, "MATRIX:A,B");
		RewriterStatement stmt2 = RewriterUtils.parse("A", ctx, "FLOAT:A,B");
		stmt = canonicalConverter.apply(stmt);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert !stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2));
	}

	@Test
	public void test2() {
		RewriterStatement stmt = RewriterUtils.parse("+(0.0,*(2,%*%(t(X),T)))", ctx, "MATRIX:T,X", "FLOAT:0.0", "INT:2");
		stmt = canonicalConverter.apply(stmt);

		System.out.println(stmt.toParsableString(ctx));
	}
}
