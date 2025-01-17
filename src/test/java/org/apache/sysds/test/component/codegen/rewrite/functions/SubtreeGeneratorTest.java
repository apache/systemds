package org.apache.sysds.test.component.codegen.rewrite.functions;

import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.List;
import java.util.function.Function;

public class SubtreeGeneratorTest {

	private static RuleContext ctx;
	private static Function<RewriterStatement, RewriterStatement> canonicalConverter;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
		canonicalConverter = RewriterUtils.buildCanonicalFormConverter(ctx, true);
	}

	@Test
	public void test1() {
		RewriterStatement stmt = RewriterUtils.parse("+(1, a)", ctx, "LITERAL_INT:1", "FLOAT:a");
		List<RewriterStatement> subtrees = RewriterUtils.generateSubtrees(stmt, ctx, 100);

		for (RewriterStatement sub : subtrees) {
			System.out.println("==========");
			System.out.println(sub.toParsableString(ctx, true));
		}

		assert subtrees.size() == 2;
	}

	@Test
	public void test2() {
		RewriterStatement stmt = RewriterUtils.parse("+(+(1, b), a)", ctx, "LITERAL_INT:1", "FLOAT:a,b");
		List<RewriterStatement> subtrees = RewriterUtils.generateSubtrees(stmt, ctx, 100);

		for (RewriterStatement sub : subtrees) {
			System.out.println("==========");
			System.out.println(sub.toParsableString(ctx, true));
		}

		assert subtrees.size() == 3;
	}

	@Test
	public void test3() {
		RewriterStatement stmt = RewriterUtils.parse("-(+(1.0,A),B)", ctx, "LITERAL_FLOAT:1.0", "MATRIX:A,B");
		List<RewriterStatement> subtrees = RewriterUtils.generateSubtrees(stmt, ctx, 100);

		for (RewriterStatement sub : subtrees) {
			System.out.println("==========");
			System.out.println(sub.toParsableString(ctx, true));
		}

		assert subtrees.size() == 3;
	}
}
