package org.apache.sysds.test.component.codegen.rewrite.functions;

import org.apache.sysds.common.Types;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.DataGenOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.ReorgOp;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.hops.rewriter.RewriterCodeGen;
import org.apache.sysds.hops.rewriter.RewriterRule;
import org.apache.sysds.hops.rewriter.RewriterRuleBuilder;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.parser.DataIdentifier;
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
			Hop result = f.apply(add);

			assert result instanceof LiteralOp && ((LiteralOp) result).getLongValue() == 2;
		} catch (Exception e) {
			e.printStackTrace();
			assert false;
		}
	}

	@Test
	public void test2() {
		HashMap<String, RewriterStatement> vars = new HashMap<>();
		vars.put("A", RewriterUtils.parse("A", ctx, "MATRIX:A"));
		vars.put("B", RewriterUtils.parse("B", ctx, "MATRIX:B"));
		RewriterStatement stmt1 = RewriterUtils.parse("+(t(A), t(B))", ctx, vars);
		RewriterStatement stmt2 = RewriterUtils.parse("t(+(A, B))", ctx, vars);

		RewriterRule rule = new RewriterRuleBuilder(ctx, "testRule")
				.setUnidirectional(true)
				.completeRule(stmt1, stmt2)
				.build();

		System.out.println(RewriterCodeGen.generateClass("MRuleTest", List.of(new Tuple2<>("testRule", rule)), ctx));

		try {
			Function<Hop, Hop> f = RewriterCodeGen.compileRewrites("MRuleTest", List.of(new Tuple2<>("testRule", rule)), ctx);
			HashMap<String, Hop> inputParams = new HashMap<>();
			inputParams.put(DataExpression.RAND_ROWS, new LiteralOp(100));
			inputParams.put(DataExpression.RAND_COLS, new LiteralOp(100));
			inputParams.put(DataExpression.RAND_MIN, new LiteralOp(0.0));
			inputParams.put(DataExpression.RAND_MAX, new LiteralOp(1.0));
			Hop A = new DataGenOp(Types.OpOpDG.RAND, new DataIdentifier("A", Types.DataType.MATRIX, Types.ValueType.FP64), inputParams);
			Hop B = new DataGenOp(Types.OpOpDG.RAND, new DataIdentifier("B", Types.DataType.MATRIX, Types.ValueType.FP64), inputParams);
			Hop tA = new ReorgOp("t", Types.DataType.MATRIX, Types.ValueType.FP64, Types.ReOrgOp.TRANS, A);
			Hop tB = new ReorgOp("t", Types.DataType.MATRIX, Types.ValueType.FP64, Types.ReOrgOp.TRANS, B);
			Hop add = new BinaryOp("test", Types.DataType.MATRIX, Types.ValueType.FP64, Types.OpOp2.PLUS, tA, tB);
			Hop result = f.apply(add);

			assert result instanceof ReorgOp && result.getInput().size() == 1 && result.getInput(0) instanceof BinaryOp;
		} catch (Exception e) {
			e.printStackTrace();
			assert false;
		}
	}

	@Test
	public void test3() {
		HashMap<String, RewriterStatement> vars = new HashMap<>();
		vars.put("A", RewriterUtils.parse("A", ctx, "MATRIX:A"));
		vars.put("B", RewriterUtils.parse("B", ctx, "MATRIX:B"));
		RewriterStatement stmt1 = RewriterUtils.parse("^(t(A), t(B))", ctx, vars);
		RewriterStatement stmt2 = RewriterUtils.parse("t(^(A, B))", ctx, vars);

		RewriterRule rule = new RewriterRuleBuilder(ctx, "testRule")
				.setUnidirectional(true)
				.completeRule(stmt1, stmt2)
				.build();

		System.out.println(RewriterCodeGen.generateClass("MRuleTest", List.of(new Tuple2<>("testRule", rule)), ctx));

		try {
			Function<Hop, Hop> f = RewriterCodeGen.compileRewrites("MRuleTest", List.of(new Tuple2<>("testRule", rule)), ctx);
			HashMap<String, Hop> inputParams = new HashMap<>();
			inputParams.put(DataExpression.RAND_ROWS, new LiteralOp(100));
			inputParams.put(DataExpression.RAND_COLS, new LiteralOp(100));
			inputParams.put(DataExpression.RAND_MIN, new LiteralOp(0.0));
			inputParams.put(DataExpression.RAND_MAX, new LiteralOp(1.0));
			Hop A = new DataGenOp(Types.OpOpDG.RAND, new DataIdentifier("A", Types.DataType.MATRIX, Types.ValueType.FP64), inputParams);
			Hop B = new DataGenOp(Types.OpOpDG.RAND, new DataIdentifier("B", Types.DataType.MATRIX, Types.ValueType.FP64), inputParams);
			Hop tA = new ReorgOp("t", Types.DataType.MATRIX, Types.ValueType.FP64, Types.ReOrgOp.TRANS, A);
			Hop tB = new ReorgOp("t", Types.DataType.MATRIX, Types.ValueType.FP64, Types.ReOrgOp.TRANS, B);
			Hop pow = new BinaryOp("test", Types.DataType.MATRIX, Types.ValueType.FP64, Types.OpOp2.POW, tA, tB);
			Hop result = f.apply(pow);

			assert result instanceof ReorgOp && result.getInput().size() == 1 && result.getInput(0) instanceof BinaryOp;
		} catch (Exception e) {
			e.printStackTrace();
			assert false;
		}
	}

	@Test
	public void test4() {
		HashMap<String, RewriterStatement> vars = new HashMap<>();
		vars.put("A", RewriterUtils.parse("A", ctx, "MATRIX:A"));
		vars.put("B", RewriterUtils.parse("B", ctx, "MATRIX:B"));
		RewriterStatement stmt1 = RewriterUtils.parse("%*%(t(A), t(B))", ctx, vars);
		RewriterStatement stmt2 = RewriterUtils.parse("t(%*%(B, A))", ctx, vars);

		RewriterRule rule = new RewriterRuleBuilder(ctx, "testRule")
				.setUnidirectional(true)
				.completeRule(stmt1, stmt2)
				.build();

		System.out.println(RewriterCodeGen.generateClass("MRuleTest", List.of(new Tuple2<>("testRule", rule)), ctx));

		try {
			Function<Hop, Hop> f = RewriterCodeGen.compileRewrites("MRuleTest", List.of(new Tuple2<>("testRule", rule)), ctx);
			HashMap<String, Hop> inputParams = new HashMap<>();
			inputParams.put(DataExpression.RAND_ROWS, new LiteralOp(100));
			inputParams.put(DataExpression.RAND_COLS, new LiteralOp(100));
			inputParams.put(DataExpression.RAND_MIN, new LiteralOp(0.0));
			inputParams.put(DataExpression.RAND_MAX, new LiteralOp(1.0));
			Hop A = new DataGenOp(Types.OpOpDG.RAND, new DataIdentifier("A", Types.DataType.MATRIX, Types.ValueType.FP64), inputParams);
			Hop B = new DataGenOp(Types.OpOpDG.RAND, new DataIdentifier("B", Types.DataType.MATRIX, Types.ValueType.FP64), inputParams);
			Hop tA = new ReorgOp("t", Types.DataType.MATRIX, Types.ValueType.FP64, Types.ReOrgOp.TRANS, A);
			Hop tB = new ReorgOp("t", Types.DataType.MATRIX, Types.ValueType.FP64, Types.ReOrgOp.TRANS, B);
			Hop matmul = HopRewriteUtils.createMatrixMultiply(tA, tB);
			Hop result = f.apply(matmul);

			assert result instanceof ReorgOp && result.getInput().size() == 1 && HopRewriteUtils.isMatrixMultiply(result.getInput(0));
		} catch (Exception e) {
			e.printStackTrace();
			assert false;
		}
	}

}
