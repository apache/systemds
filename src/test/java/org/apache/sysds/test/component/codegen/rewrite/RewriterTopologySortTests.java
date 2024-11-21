package org.apache.sysds.test.component.codegen.rewrite;

import org.apache.sysds.hops.rewriter.RewriterRuntimeUtils;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.hops.rewriter.TopologicalSort;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.function.Function;

public class RewriterTopologySortTests {

	private static RuleContext ctx;
	private static Function<RewriterStatement, RewriterStatement> converter;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
		converter = RewriterUtils.buildCanonicalFormConverter(ctx, true);
	}

	@Test
	public void testSimpleEquivalence1() {
		RewriterStatement stmt = RewriterUtils.parse("+(*(a, b), *(a, c))", ctx, "FLOAT:a,b,c");
		RewriterStatement stmt2 = RewriterUtils.parse("+(*(b, a), *(c, a))", ctx, "FLOAT:a,b,c");
		stmt = converter.apply(stmt);
		stmt2 = converter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt));
	}

	@Test
	public void testSimpleEquivalence2() {
		// Here, a and b are indistinguishable
		// Thus, the topological sort has to decide a random but consistent order
		RewriterStatement stmt = RewriterUtils.parse("+(*(a, b), *(b, a))", ctx, "FLOAT:a,b");
		RewriterStatement stmt2 = RewriterUtils.parse("+(*(b, a), *(b, a))", ctx, "FLOAT:a,b");
		stmt = converter.apply(stmt);
		stmt2 = converter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt));
	}

	@Test
	public void testSimpleEquivalence3() {
		RewriterStatement stmt = RewriterUtils.parse("+(-(*(a, b)), *(b, a))", ctx, "FLOAT:a,b");
		RewriterStatement stmt2 = RewriterUtils.parse("+(*(b, a), -(*(b, a)))", ctx, "FLOAT:a,b");
		stmt = converter.apply(stmt);
		stmt2 = converter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt));
	}

	@Test
	public void testSimpleEquivalence4() {
		RewriterStatement stmt = RewriterUtils.parse("+(*(-(a), b), *(b, a))", ctx, "FLOAT:a,b");
		RewriterStatement stmt2 = RewriterUtils.parse("+(*(a, -(b)), *(b, a))", ctx, "FLOAT:a,b");
		stmt = converter.apply(stmt);
		stmt2 = converter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt));
	}

	@Test
	public void testSimpleEquivalence5() {
		RewriterStatement stmt = RewriterUtils.parse("+(1, 2)", ctx, "LITERAL_INT:1,2");
		RewriterStatement stmt2 = RewriterUtils.parse("+(2, 1)", ctx, "LITERAL_INT:1,2");
		stmt = converter.apply(stmt);
		stmt2 = converter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt));
	}

	@Test
	public void testSimpleEquivalence6() {
		RewriterStatement stmt = RewriterUtils.parse("+(*(a, b), *(*(a, b), c))", ctx, "FLOAT:a,b,c");
		RewriterStatement stmt2 = RewriterUtils.parse("+(*(*(a, b), c), *(a, b))", ctx, "FLOAT:a,b,c");
		stmt = converter.apply(stmt);
		stmt2 = converter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt));
	}

	@Test
	public void testSimpleEquivalence7() {
		RewriterStatement stmt = RewriterUtils.parse("+(*(a, b), *(/(a, b), /(b, a)))", ctx, "FLOAT:a,b,c");
		RewriterStatement stmt2 = RewriterUtils.parse("+(*(/(a, b), /(b, a)), *(a, b))", ctx, "FLOAT:a,b,c");
		stmt = converter.apply(stmt);
		stmt2 = converter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt));
	}

	@Test
	public void testSimpleEquivalence8() {
		RewriterStatement stmt = RewriterUtils.parse("+(*(a, b), f(b, a))", ctx, "FLOAT:a,b");
		RewriterStatement stmt2 = RewriterUtils.parse("+(*(b, a), f(b, a))", ctx, "FLOAT:a,b");
		stmt = converter.apply(stmt);
		stmt2 = converter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt));
	}

	@Test
	public void testSimpleEquivalence9() {
		RewriterStatement stmt = RewriterUtils.parse("+(*(-(a), b), *(a, a))", ctx, "FLOAT:a,b");
		RewriterStatement stmt2 = RewriterUtils.parse("+(*(a, -(b)), *(a, a))", ctx, "FLOAT:a,b");
		stmt = converter.apply(stmt);
		stmt2 = converter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt));
	}

	@Test
	public void test4() {
		RewriterStatement stmt = RewriterUtils.parse("sum(*(A, A))", ctx, "MATRIX:A");
		stmt = converter.apply(stmt);

		System.out.println(stmt.toParsableString(ctx, true));
	}

	@Test
	public void test5() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(_idxExpr($1:_idx(1,_EClass(argList(ncol(A),nrow(A)))),*(argList([](B,$1,$1),[](A,$1,$1)))))", ctx, "MATRIX:A,B", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("sum(_idxExpr($1:_idx(1,_EClass(argList(nrow(B),nrow(A)))),*(argList([](B,$1,$1),[](A,$1,$1)))))", ctx, "MATRIX:A,B", "LITERAL_INT:1");
		TopologicalSort.sort(stmt1, ctx);
		TopologicalSort.sort(stmt2, ctx);
		System.out.println(stmt1.toParsableString(ctx));
		System.out.println(stmt2.toParsableString(ctx));
		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testComplex1() {
		RewriterStatement stmt1 = RewriterUtils.parse("_m($1:_idx(1,ncol(V)),$2:_idx(1,ncol(U)),sum(_idxExpr($3:_idx(1,_EClass(argList(nrow(V),nrow(U)))),*(argList([](V,$3,$1),[](U,$3,$2))))))", ctx, "MATRIX:U,V", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("_m($1:_idx(1,ncol(V)),$2:_idx(1,ncol(U)),sum(_idxExpr($3:_idx(1,_EClass(argList(nrow(U),nrow(V)))),*(argList([](U,$3,$2),[](V,$3,$1))))))", ctx, "MATRIX:U,V", "LITERAL_INT:1");

		TopologicalSort.sort(stmt1, ctx);
		TopologicalSort.sort(stmt2, ctx);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testComplex2() {
		RewriterStatement stmt1 = RewriterUtils.parse("_m($1:_idx(1,ncol(V)),$2:_idx(1,ncol(U)),sum(_idxExpr($3:_idx(1,_EClass(argList(nrow(V),nrow(U)))),1.0)))", ctx, "MATRIX:U,V", "LITERAL_INT:1", "LITERAL_FLOAT:1.0");
		RewriterStatement stmt2 = RewriterUtils.parse("_m($1:_idx(1,ncol(V)),$2:_idx(1,ncol(U)),sum(_idxExpr($3:_idx(1,_EClass(argList(nrow(U),nrow(V)))),1.0)))", ctx, "MATRIX:U,V", "LITERAL_INT:1", "LITERAL_FLOAT:1.0");

		TopologicalSort.sort(stmt1, ctx);
		TopologicalSort.sort(stmt2, ctx);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testComplex3() {
		RewriterStatement stmt1 = RewriterUtils.parse("_m(ncol(V),ncol(U),as.float(_EClass(argList(nrow(V),nrow(U))))))", ctx, "MATRIX:U,V", "LITERAL_INT:1", "LITERAL_FLOAT:1.0");
		RewriterStatement stmt2 = RewriterUtils.parse("_m(ncol(V),ncol(U),as.float(_EClass(argList(nrow(U),nrow(V))))))", ctx, "MATRIX:U,V", "LITERAL_INT:1", "LITERAL_FLOAT:1.0");

		TopologicalSort.sort(stmt1, ctx);
		TopologicalSort.sort(stmt2, ctx);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

}
