package org.apache.sysds.test.component.codegen.rewrite;

import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.function.Function;

public class RewriterNormalFormTests {
	private static RuleContext ctx;
	//private static Function<RewriterStatement, RewriterStatement> converter;
	private static Function<RewriterStatement, RewriterStatement> canonicalConverter;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
		//converter = RewriterUtils.buildFusedOperatorCreator(ctx, true);
		canonicalConverter = RewriterUtils.buildCanonicalFormConverter(ctx, true);
	}

	//e.g., matrix(1,nrow(X),ncol(X))/X -> 1/X
	@Test
	public void testUnnecessaryVectorize() {
		RewriterStatement stmt1 = RewriterUtils.parse("/(const(A, 1.0), A)", ctx, "MATRIX:A", "LITERAL_FLOAT:1.0");
		RewriterStatement stmt2 = RewriterUtils.parse("/(1.0, A)", ctx, "MATRIX:A", "LITERAL_FLOAT:1.0");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert RewriterStatement.MatcherContext.exactMatch(ctx, stmt1, stmt2).match();
	}

	@Test
	public void testRemoveUnnecessaryBinaryOperation() {
		RewriterStatement stmt1 = RewriterUtils.parse("*(1.0, A)", ctx, "MATRIX:A", "LITERAL_FLOAT:1.0");
		RewriterStatement stmt2 = RewriterUtils.parse("A", ctx, "MATRIX:A", "LITERAL_FLOAT:1.0");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert RewriterStatement.MatcherContext.exactMatch(ctx, stmt1, stmt2).match();
	}

	@Test
	public void testFuseDatagenAndBinaryOperation() {
		RewriterStatement stmt1 = RewriterUtils.parse("*(rand(nrow(A), ncol(A), -1.0, 1.0), a)", ctx, "MATRIX:A", "FLOAT:a", "LITERAL_FLOAT:1.0,-1.0");
		RewriterStatement stmt2 = RewriterUtils.parse("rand(nrow(A), ncol(A), -(a), a)", ctx, "MATRIX:A", "FLOAT:a");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert RewriterStatement.MatcherContext.exactMatch(ctx, stmt1, stmt2).match();
	}

	@Test
	public void testFuseDatagenAndMinusOperation() {
		RewriterStatement stmt1 = RewriterUtils.parse("-(rand(nrow(A), ncol(A), -2.0, 1.0))", ctx, "MATRIX:A", "LITERAL_FLOAT:1.0,-2.0");
		RewriterStatement stmt2 = RewriterUtils.parse("rand(nrow(A), ncol(A), -1.0, 2.0)", ctx, "MATRIX:A", "LITERAL_FLOAT:-1.0,2.0");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert RewriterStatement.MatcherContext.exactMatch(ctx, stmt1, stmt2).match();
	}

	@Test
	public void testCanonicalizeMatrixMultScalarAdd() {
		RewriterStatement stmt1 = RewriterUtils.parse("+(eps, %*%(A, t(B)))", ctx, "MATRIX:A,B", "FLOAT:eps");
		RewriterStatement stmt2 = RewriterUtils.parse("+(%*%(A, t(B)), eps)", ctx, "MATRIX:A,B", "FLOAT:eps");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert RewriterStatement.MatcherContext.exactMatch(ctx, stmt1, stmt2).match();
	}

	@Test
	public void testCanonicalizeMatrixMultScalarAdd2() {
		RewriterStatement stmt1 = RewriterUtils.parse("-(%*%(A, t(B)), eps)", ctx, "MATRIX:A,B", "FLOAT:eps");
		RewriterStatement stmt2 = RewriterUtils.parse("+(%*%(A, t(B)), -(eps))", ctx, "MATRIX:A,B", "FLOAT:eps");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert RewriterStatement.MatcherContext.exactMatch(ctx, stmt1, stmt2).match();
	}

	@Test
	public void testSimplifyMultiBinaryToBinaryOperation() {
		RewriterStatement stmt1 = RewriterUtils.parse("-(1.0, *(A,B))", ctx, "MATRIX:A,B", "LITERAL_FLOAT:1.0");
		RewriterStatement stmt2 = RewriterUtils.parse("1-*(A, B)", ctx, "MATRIX:A,B", "LITERAL_FLOAT:1.0");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert RewriterStatement.MatcherContext.exactMatch(ctx, stmt1, stmt2).match();
	}

	@Test
	public void testSimplifyDistributiveBinaryOperation() {
		RewriterStatement stmt1 = RewriterUtils.parse("-(A, *(B,A))", ctx, "MATRIX:A,B,C", "LITERAL_FLOAT:1.0");
		RewriterStatement stmt2 = RewriterUtils.parse("*(-(1.0,B), A)", ctx, "MATRIX:A,B,C", "LITERAL_FLOAT:1.0");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert RewriterStatement.MatcherContext.exactMatch(ctx, stmt1, stmt2).match();
	}

	@Test
	public void testSimplifyBushyBinaryOperation() {
		RewriterStatement stmt1 = RewriterUtils.parse("*(A,*(B, %*%(C, rowVec(D))))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0");
		RewriterStatement stmt2 = RewriterUtils.parse("*(*(A,B), %*%(C, rowVec(D)))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert RewriterStatement.MatcherContext.exactMatch(ctx, stmt1, stmt2).match();
	}

	@Test
	public void testSimplifyUnaryAggReorgOperation() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(t(A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0");
		RewriterStatement stmt2 = RewriterUtils.parse("sum(A)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert RewriterStatement.MatcherContext.exactMatch(ctx, stmt1, stmt2).match();
	}

	@Test
	public void testRemoveUnnecessaryAggregates() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(rowSums(A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0");
		RewriterStatement stmt2 = RewriterUtils.parse("sum(A)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert RewriterStatement.MatcherContext.exactMatch(ctx, stmt1, stmt2).match();
	}

	@Test
	public void testSimplifyBinaryMatrixScalarOperation() {
		RewriterStatement stmt1 = RewriterUtils.parse("as.scalar(*(A,a))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0");
		RewriterStatement stmt2 = RewriterUtils.parse("*(as.scalar(A),a)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert RewriterStatement.MatcherContext.exactMatch(ctx, stmt1, stmt2).match();
	}

	@Test
	public void testPushdownUnaryAggTransposeOperation() {
		RewriterStatement stmt1 = RewriterUtils.parse("colSums(t(A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0");
		RewriterStatement stmt2 = RewriterUtils.parse("t(rowSums(A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert RewriterStatement.MatcherContext.exactMatch(ctx, stmt1, stmt2).match();
	}

	@Test
	public void testPushdownCSETransposeScalarOperation() {
		// Introduce a dummy instruction * as I don't support the assignment operator
		RewriterStatement stmt1 = RewriterUtils.parse("*(t(A), t(^(A,2.0)))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0");
		RewriterStatement stmt2 = RewriterUtils.parse("*(t(A), ^(t(A),2.0))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert RewriterStatement.MatcherContext.exactMatch(ctx, stmt1, stmt2).match();
	}

	@Test
	public void testPushdownSumBinaryMult() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(*(a,A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0");
		RewriterStatement stmt2 = RewriterUtils.parse("*(a, sum(A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert RewriterStatement.MatcherContext.exactMatch(ctx, stmt1, stmt2).match();
	}

	@Test
	public void testSimplifyTraceMatrixMult() {
		RewriterStatement stmt1 = RewriterUtils.parse("trace(%*%(A,B))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0");
		RewriterStatement stmt2 = RewriterUtils.parse("sum(*(A, t(B)))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert RewriterStatement.MatcherContext.exactMatch(ctx, stmt1, stmt2).match();
	}

	@Test
	public void testSimplifySlicedMatrixMult() {
		RewriterStatement stmt1 = RewriterUtils.parse("[](%*%(A,B), 1, 1)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("%*%(colVec(A), rowVec(B))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0", "LITERAL_INT:1");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert RewriterStatement.MatcherContext.exactMatch(ctx, stmt1, stmt2).match();
	}

	@Test
	public void testRemoveUnnecessaryReorgOperation() {
		RewriterStatement stmt1 = RewriterUtils.parse("t(t(A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("A", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0", "LITERAL_INT:1");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testRemoveUnnecessaryReorgOperation2() {
		RewriterStatement stmt1 = RewriterUtils.parse("rev(rev(A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("A", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0", "LITERAL_INT:1");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testSimplifyTransposeAggBinBinaryChains() {
		RewriterStatement stmt1 = RewriterUtils.parse("t(+(%*%(t(A),t(B)), C))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("+(%*%(B,A), t(C))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0", "LITERAL_INT:1");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testRemoveUnnecessaryMinus() {
		RewriterStatement stmt1 = RewriterUtils.parse("-(-(A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("A", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0", "LITERAL_INT:1");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testFuseLogNzUnaryOperation() {
		RewriterStatement stmt1 = RewriterUtils.parse("*(!=(A,0.0), log(A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("log_nz(A)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testAdditionMatrix1() {
		RewriterStatement stmt1 = RewriterUtils.parse("*(A, %*%(rowVec(B), const([](B, 1, 1, 1, 1), 1.0)))", ctx, "MATRIX:A,B", "LITERAL_FLOAT:1.0", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("*(A, rowVec(B))", ctx, "MATRIX:A,B");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert RewriterStatement.MatcherContext.exactMatch(ctx, stmt1, stmt2).match();
	}

	private boolean match(RewriterStatement stmt1, RewriterStatement stmt2) {
		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		return RewriterStatement.MatcherContext.exactMatch(ctx, stmt1, stmt2).match();
	}
}
