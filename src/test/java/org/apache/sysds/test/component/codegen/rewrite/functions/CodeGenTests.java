package org.apache.sysds.test.component.codegen.rewrite.functions;

import org.apache.sysds.common.Types;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.rewriter.RewriterCodeGen;
import org.apache.sysds.hops.rewriter.RewriterRule;
import org.apache.sysds.hops.rewriter.RewriterRuleBuilder;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.junit.BeforeClass;
import org.junit.Test;
import scala.Tuple2;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

public class CodeGenTests {

	private static RuleContext ctx;
	private static Function<RewriterStatement, RewriterStatement> canonicalConverter;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
		canonicalConverter = RewriterUtils.buildCanonicalFormConverter(ctx, false);
	}

	@Test
	public void test1() {
		RewriterStatement stmt1 = RewriterUtils.parse("+(1, 1)", ctx, "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("2", ctx, "LITERAL_INT:2");

		RewriterRule rule = new RewriterRuleBuilder(ctx, "testRule")
				.setUnidirectional(true)
				.completeRule(stmt1, stmt2)
				.build();

		System.out.println(RewriterCodeGen.generateClass("MRuleTest", List.of(new Tuple2<>("testRule", rule)), ctx));

		try {
			Function<Hop, Hop> f = RewriterCodeGen.compileRewrites("MRuleTest", List.of(new Tuple2<>("testRule", rule)), ctx);
			Hop l = new LiteralOp(1);
			Hop add = new BinaryOp("test", Types.DataType.SCALAR, Types.ValueType.INT64, Types.OpOp2.PLUS, l, l);
			System.out.println(f.apply(add));
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
