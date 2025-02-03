/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.test.component.codegen.rewrite.functions;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.DataGenOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.ReorgOp;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.hops.rewriter.codegen.RewriterCodeGen;
import org.apache.sysds.hops.rewriter.rule.RewriterRule;
import org.apache.sysds.hops.rewriter.rule.RewriterRuleBuilder;
import org.apache.sysds.hops.rewriter.rule.RewriterRuleSet;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.test.component.codegen.rewrite.RewriterTopologySortTests;
import org.junit.BeforeClass;
import org.junit.Test;
import scala.Tuple2;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.function.Function;

public class CodeGenTests {
	protected static final Log LOG = LogFactory.getLog(CodeGenTests.class.getName());

	private static RuleContext ctx;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
	}

	@Test
	public void test1() {
		RewriterStatement stmt1 = RewriterUtils.parse("+(1, 1)", ctx, "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("2", ctx, "LITERAL_INT:2");

		RewriterRule rule = new RewriterRuleBuilder(ctx, "testRule")
				.setUnidirectional(true)
				.completeRule(stmt1, stmt2)
				.build();

		LOG.info(RewriterCodeGen.generateClass("MRuleTest", List.of(new Tuple2<>("testRule", rule)), false, false, ctx, false, false));

		try {
			Function<Hop, Hop> f = RewriterCodeGen.compileRewrites("MRuleTest", List.of(new Tuple2<>("testRule", rule)), ctx, false, false);
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

		LOG.info(RewriterCodeGen.generateClass("MRuleTest", List.of(new Tuple2<>("testRule", rule)), false, false, ctx, false, false));

		try {
			Function<Hop, Hop> f = RewriterCodeGen.compileRewrites("MRuleTest", List.of(new Tuple2<>("testRule", rule)), ctx, false, false);
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

		LOG.info(RewriterCodeGen.generateClass("MRuleTest", List.of(new Tuple2<>("testRule", rule)), false, false, ctx, false, false));

		try {
			Function<Hop, Hop> f = RewriterCodeGen.compileRewrites("MRuleTest", List.of(new Tuple2<>("testRule", rule)), ctx, false, false);
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

		LOG.info(RewriterCodeGen.generateClass("MRuleTest", List.of(new Tuple2<>("testRule", rule)), false, false, ctx, false, false));

		try {
			Function<Hop, Hop> f = RewriterCodeGen.compileRewrites("MRuleTest", List.of(new Tuple2<>("testRule", rule)), ctx, false, false);
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

	@Test
	public void test5() {
		HashMap<String, RewriterStatement> vars = new HashMap<>();
		vars.put("A", RewriterUtils.parse("A", ctx, "MATRIX:A"));
		vars.put("B", RewriterUtils.parse("B", ctx, "MATRIX:B"));
		RewriterStatement stmt1 = RewriterUtils.parse("rowSums(t(A))", ctx, vars);
		RewriterStatement stmt2 = RewriterUtils.parse("t(colSums(A))", ctx, vars);

		RewriterRule rule = new RewriterRuleBuilder(ctx, "testRule")
				.setUnidirectional(true)
				.completeRule(stmt1, stmt2)
				.build();

		LOG.info(RewriterCodeGen.generateClass("MRuleTest", List.of(new Tuple2<>("testRule", rule)), false, false, ctx, false, false));

		try {
			Function<Hop, Hop> f = RewriterCodeGen.compileRewrites("MRuleTest", List.of(new Tuple2<>("testRule", rule)), ctx, false, false);
			HashMap<String, Hop> inputParams = new HashMap<>();
			inputParams.put(DataExpression.RAND_ROWS, new LiteralOp(100));
			inputParams.put(DataExpression.RAND_COLS, new LiteralOp(100));
			inputParams.put(DataExpression.RAND_MIN, new LiteralOp(0.0));
			inputParams.put(DataExpression.RAND_MAX, new LiteralOp(1.0));
			Hop A = new DataGenOp(Types.OpOpDG.RAND, new DataIdentifier("A", Types.DataType.MATRIX, Types.ValueType.FP64), inputParams);
			Hop tA = new ReorgOp("t", Types.DataType.MATRIX, Types.ValueType.FP64, Types.ReOrgOp.TRANS, A);
			Hop rowSums = HopRewriteUtils.createAggUnaryOp(tA, Types.AggOp.SUM, Types.Direction.Row);
			Hop result = f.apply(rowSums);

			assert result instanceof ReorgOp && result.getInput().size() == 1 && result.getInput(0) instanceof AggUnaryOp;
		} catch (Exception e) {
			e.printStackTrace();
			assert false;
		}
	}

	@Test
	public void generateExample() {
		String ruleStr = "MATRIX:B\nFLOAT:a,c\n+(a,-(B,c))\n=>\n+(-(a,c),B)";
		RewriterRule rule = RewriterUtils.parseRule(ruleStr, ctx);
		RewriterRuleSet rs = new RewriterRuleSet(ctx, List.of(rule));
		RewriterCodeGen.DEBUG = false;
		String code = rs.toJavaCode("Test", false, false, true, false);
		LOG.info(code);
	}

	@Test
	public void generateExample2() {
		String ruleStr = "MATRIX:A\n+(A,A)\n=>\n*2(A)";
		RewriterRule rule = RewriterUtils.parseRule(ruleStr, ctx);
		RewriterRuleSet rs = new RewriterRuleSet(ctx, List.of(rule));
		RewriterCodeGen.DEBUG = false;
		String code = rs.toJavaCode("Test", false, false, true, false);
		LOG.info(code);
	}

	@Test
	public void testConditional() {
		String ruleStr = "MATRIX:Xm,tmp852\n" +
				"FLOAT:tmp65855\n" +
				"\n" +
				"%*%(t(/(Xm,tmp65855)),tmp852)\n" +
				"=>\n" +
				"{\n" +
				"%*%(t(Xm),/(tmp852,tmp65855))\n" +
				"/(%*%(t(Xm),tmp852),tmp65855)\n" +
				"t(/(%*%(t(tmp852),Xm),tmp65855))\n" +
				"}";
		RewriterRule rule = RewriterUtils.parseRule(ruleStr, ctx);
		RewriterRuleSet rs = new RewriterRuleSet(ctx, List.of(rule));
		rs.determineConditionalApplicability();
		RewriterCodeGen.DEBUG = false;
		String code = rs.toJavaCode("GeneratedRewriteClass", false, true, true, false);
		LOG.info(code);
	}

	@Test
	public void testLiteral() {
		String ruleStr = "MATRIX:A\n" +
				"\n" +
				"-(+(A, $1:literal.FLOAT()), $2:literal.FLOAT())\n" +
				"=>\n" +
				"+(A, -($1, $2))";
		RewriterRule rule = RewriterUtils.parseRule(ruleStr, ctx);
		RewriterRuleSet rs = new RewriterRuleSet(ctx, List.of(rule));
		rs.determineConditionalApplicability();
		RewriterCodeGen.DEBUG = false;
		String code = rs.toJavaCode("GeneratedRewriteClass", false, true, true, false);
		LOG.info(code);
	}

	@Test
	public void testCFold() {
		String ruleStr = "LITERAL_FLOAT:1.0,2.0\n" +
				"\n" +
				"+(1.0,1.0)\n" +
				"=>\n" +
				"2.0";
		RewriterRule rule = RewriterUtils.parseRule(ruleStr, ctx);
		RewriterRuleSet rs = new RewriterRuleSet(ctx, List.of(rule));
		rs.determineConditionalApplicability();
		RewriterCodeGen.DEBUG = false;
		String code = rs.toJavaCode("GeneratedRewriteClass", false, true, true, false);
		LOG.info(code);
	}

	//@Test
	public void codeGen() {
		List<String> files = List.of("/Users/janniklindemann/Dev/Rewrite-Generator-Reproducibility/data/rules_end_to_end.dml");
		//List<String> files = List.of(RewriteAutomaticallyGenerated.FILE_PATH_MB);
		String targetPath = "/Users/janniklindemann/Dev/MScThesis/other/GeneratedRewriteClass.java";

		try {
			// This is to specify that the generated code should print to the console if it modifies the DAG
			// This should be disabled when generating production code
			RewriterCodeGen.DEBUG = false;
			RewriterCodeGen.generateRewritesFromFiles(files, targetPath, true, 3, true, false, ctx);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
