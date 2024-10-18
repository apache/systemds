package org.apache.sysds.test.component.codegen.rewrite;

import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
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
		assert stmt.match(new RewriterStatement.MatcherContext(ctx, stmt2));
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
		assert stmt.match(new RewriterStatement.MatcherContext(ctx, stmt2));
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
		assert stmt.match(new RewriterStatement.MatcherContext(ctx, stmt2));
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
		assert stmt.match(new RewriterStatement.MatcherContext(ctx, stmt2));
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
		assert stmt.match(new RewriterStatement.MatcherContext(ctx, stmt2));
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
		assert stmt.match(new RewriterStatement.MatcherContext(ctx, stmt2));
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
		assert stmt.match(new RewriterStatement.MatcherContext(ctx, stmt2));
	}

}
