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

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
		converter = RewriterUtils.buildFusedOperatorCreator(ctx, true);
	}

	@Test
	public void testAdditionFloat1() {
		RewriterStatement stmt = RewriterUtils.parse("+(+(a, b), 1)", ctx, "MATRIX:A,B,C", "FLOAT:a,b", "LITERAL_INT:0,1");
		stmt = converter.apply(stmt);
		System.out.println(stmt);
		assert stmt.match(new RewriterStatement.MatcherContext(ctx, RewriterUtils.parse("+(argList(a, b, 1))", ctx, "FLOAT:a,b", "LITERAL_INT:0,1")));
	}

	@Test
	public void testAdditionFloat2() {
		RewriterStatement stmt = RewriterUtils.parse("+(1, +(a, b))", ctx, "FLOAT:a,b", "LITERAL_INT:0,1");
		stmt = converter.apply(stmt);
		System.out.println(stmt);
		assert stmt.match(new RewriterStatement.MatcherContext(ctx, RewriterUtils.parse("+(argList(a, b, 1))", ctx, "FLOAT:a,b", "LITERAL_INT:0,1")));
	}

	@Test
	public void testAdditionMatrix1() {
		RewriterStatement stmt = RewriterUtils.parse("+(+(A, B), 1)", ctx, "MATRIX:A,B", "LITERAL_INT:0,1");
		stmt = converter.apply(stmt);
		System.out.println(stmt);
		assert stmt.match(new RewriterStatement.MatcherContext(ctx, RewriterUtils.parse("_m($1:_idx(1, nrow(A)), $2:_idx(1, ncol(A)), +(argList([](A, $1, $2), [](B, $1, $2), 1)))", ctx, "MATRIX:A,B", "LITERAL_INT:0,1")));
	}

	@Test
	public void testAdditionMatrix2() {
		RewriterStatement stmt = RewriterUtils.parse("+(1, +(A, B))", ctx, "MATRIX:A,B", "LITERAL_INT:0,1");
		stmt = converter.apply(stmt);
		System.out.println(stmt);
		assert stmt.match(new RewriterStatement.MatcherContext(ctx, RewriterUtils.parse("_m($1:_idx(1, nrow(A)), $2:_idx(1, ncol(A)), +(argList([](A, $1, $2), [](B, $1, $2), 1)))", ctx, "MATRIX:A,B", "LITERAL_INT:0,1")));
	}

	@Test
	public void testSubtractionFloat1() {
		RewriterStatement stmt = RewriterUtils.parse("+(-(a, b), 1)", ctx, "MATRIX:A,B,C", "FLOAT:a,b", "LITERAL_INT:0,1");
		stmt = converter.apply(stmt);
		System.out.println(stmt);
		assert stmt.match(new RewriterStatement.MatcherContext(ctx, RewriterUtils.parse("+(argList(-(b), a, 1))", ctx, "FLOAT:a,b", "LITERAL_INT:0,1")));
	}

	@Test
	public void testSubtractionFloat2() {
		RewriterStatement stmt = RewriterUtils.parse("+(1, -(a, -(b, c)))", ctx, "MATRIX:A,B,C", "FLOAT:a,b,c", "LITERAL_INT:0,1");
		stmt = converter.apply(stmt);
		System.out.println(stmt);
		assert stmt.match(new RewriterStatement.MatcherContext(ctx, RewriterUtils.parse("+(argList(-(b), a, c, 1))", ctx, "FLOAT:a,b, c", "LITERAL_INT:0,1")));
	}
}
