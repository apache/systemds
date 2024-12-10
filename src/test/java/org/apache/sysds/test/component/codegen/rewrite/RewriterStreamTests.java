package org.apache.sysds.test.component.codegen.rewrite;

import org.apache.sysds.hops.rewriter.estimators.RewriterCostEstimator;
import org.apache.sysds.hops.rewriter.RewriterDatabase;
import org.apache.sysds.hops.rewriter.RewriterHeuristic;
import org.apache.sysds.hops.rewriter.RewriterRule;
import org.apache.sysds.hops.rewriter.RewriterRuleBuilder;
import org.apache.sysds.hops.rewriter.RewriterRuleSet;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

public class RewriterStreamTests {

	private static RuleContext ctx;
	//private static Function<RewriterStatement, RewriterStatement> converter;
	private static Function<RewriterStatement, RewriterStatement> canonicalConverter;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
		//converter = RewriterUtils.buildFusedOperatorCreator(ctx, true);
		canonicalConverter = RewriterUtils.buildCanonicalFormConverter(ctx, true);
	}

	@Test
	public void testAdditionFloat1() {
		RewriterStatement stmt = RewriterUtils.parse("+(+(a, b), 1)", ctx, "MATRIX:A,B,C", "FLOAT:a,b", "LITERAL_INT:0,1");
		stmt = canonicalConverter.apply(stmt);
		System.out.println(stmt.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, RewriterUtils.parse("+(argList(a, b, 1))", ctx, "FLOAT:a,b", "LITERAL_INT:0,1"), stmt));
	}

	@Test
	public void testAdditionFloat2() {
		RewriterStatement stmt = RewriterUtils.parse("+(1, +(a, b))", ctx, "FLOAT:a,b", "LITERAL_INT:0,1");
		stmt = canonicalConverter.apply(stmt);
		System.out.println(stmt.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, RewriterUtils.parse("+(argList(a, b, 1))", ctx, "FLOAT:a,b", "LITERAL_INT:0,1"), stmt));
	}

	@Test
	public void testAdditionMatrix1() {
		RewriterStatement stmt1 = RewriterUtils.parse("+(+(A, B), 1)", ctx, "MATRIX:A,B", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("+(+(B, 1), A)", ctx, "MATRIX:A,B", "LITERAL_INT:1");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testSubtractionFloat1() {
		RewriterStatement stmt = RewriterUtils.parse("+(-(a, b), 1)", ctx, "MATRIX:A,B,C", "FLOAT:a,b", "LITERAL_INT:0,1");
		stmt = canonicalConverter.apply(stmt);
		System.out.println(stmt.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, RewriterUtils.parse("+(argList(-(b), a, 1))", ctx, "FLOAT:a,b", "LITERAL_INT:0,1"), stmt));
	}

	@Test
	public void testSubtractionFloat2() {
		RewriterStatement stmt = RewriterUtils.parse("+(1, -(a, -(b, c)))", ctx, "MATRIX:A,B,C", "FLOAT:a,b,c", "LITERAL_INT:0,1");
		stmt = canonicalConverter.apply(stmt);
		System.out.println(stmt.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, RewriterUtils.parse("+(argList(-(b), a, c, 1))", ctx, "FLOAT:a,b, c", "LITERAL_INT:0,1"), stmt));
	}

	// Fusion will no longer be pursued
	/*@Test
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
	}*/

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
		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testTraceEquivalence1() {
		RewriterStatement stmt = RewriterUtils.parse("trace(%*%(A, B))", ctx, "MATRIX:A,B");
		RewriterStatement stmt2 = RewriterUtils.parse("sum(*(t(A), B))", ctx, "MATRIX:A,B");
		stmt = canonicalConverter.apply(stmt);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt));
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
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt));
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
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt));
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
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt));
	}

	@Test
	public void testSumEquality6() {
		RewriterStatement stmt = RewriterUtils.parse("sum(+(B, sum(*(a, A))))", ctx, "MATRIX:A,B", "FLOAT:a");
		RewriterStatement stmt2 = RewriterUtils.parse("sum(+(B, *(a, sum(A))))", ctx, "MATRIX:A,B", "FLOAT:a");
		stmt = canonicalConverter.apply(stmt);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt));
	}

	@Test
	public void testSumEquality() {
		RewriterStatement stmt = RewriterUtils.parse("sum(+(B, sum(*(a, A))))", ctx, "MATRIX:A,B", "FLOAT:a");
		RewriterStatement stmt2 = RewriterUtils.parse("+(*(a, length(A)), sum(+(B, sum(A))))", ctx, "MATRIX:A,B", "FLOAT:a");
		RewriterStatement stmt3 = RewriterUtils.parse("sum(+(B, *(a, sum(A))))", ctx, "MATRIX:A,B", "FLOAT:a");
		stmt = canonicalConverter.apply(stmt);
		stmt2 = canonicalConverter.apply(stmt2);
		stmt3 = canonicalConverter.apply(stmt3);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt3.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt3, stmt));
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt));
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
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt));
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
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt));
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
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt));
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
		assert stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt));
	}

	@Test
	public void testRealExamples1() {
		RewriterStatement stmt1 = RewriterUtils.parse("t(%*%(t(U),V))", ctx, "MATRIX:U,V");
		RewriterStatement stmt2 = RewriterUtils.parse("%*%(t(V), U)", ctx, "MATRIX:U,V");
		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);
		//TopologicalSort.sort(stmt1, ctx);
		//TopologicalSort.sort(stmt2, ctx);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
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
		assert !stmt.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt));
	}

	@Test
	public void test2() {
		RewriterStatement stmt = RewriterUtils.parse("+(0.0,*(2,%*%(t(X),T)))", ctx, "MATRIX:T,X", "FLOAT:0.0", "INT:2");
		stmt = canonicalConverter.apply(stmt);

		System.out.println(stmt.toParsableString(ctx));
	}

	@Test
	public void mTest() {
		List<RewriterRule> rules = new ArrayList<>();
		rules.add(new RewriterRuleBuilder(ctx, "?")
				.setUnidirectional(true)
				.parseGlobalVars("FLOAT:a")
				.withParsedStatement("a")
				.toParsedStatement("f(a, a)")
						.iff(match -> {
							return !match.getExpressionRoot().isInstruction() || !match.getExpressionRoot().trueInstruction().equals("f");
						}, true)
				.build()
		);

		RewriterHeuristic heur = new RewriterHeuristic(new RewriterRuleSet(ctx, rules));
		RewriterStatement stmt = RewriterUtils.parse("A", ctx, "FLOAT:A");
		stmt = heur.apply(stmt);
		System.out.println(stmt);
	}

	@Test
	public void test3() {
		RewriterStatement stmt = RewriterUtils.parse("+(+(A,X),t(X))", ctx, "MATRIX:X,A");
		stmt = canonicalConverter.apply(stmt);

		System.out.println(stmt.toParsableString(ctx));
	}

	@Test
	public void test4() {
		RewriterDatabase db = new RewriterDatabase();
		RewriterStatement stmt = RewriterUtils.parse("trace(%*%(A, B))", ctx, "MATRIX:A,B");
		RewriterStatement stmt2 = RewriterUtils.parse("sum(*(A, t(B)))", ctx, "MATRIX:A,B");
		stmt = canonicalConverter.apply(stmt);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		db.insertEntry(ctx, stmt);

		assert !db.insertEntry(ctx, stmt2);
	}

	@Test
	public void testForFailure() {
		RewriterStatement stmt = RewriterUtils.parse("[](hIndex,i,i,1,1)", ctx, "MATRIX:hIndex", "INT:i", "LITERAL_INT:1");
		stmt = canonicalConverter.apply(stmt);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
	}

	@Test
	public void testTypeConversions() {
		RewriterStatement stmt1 = RewriterUtils.parse("+(TRUE, 1)", ctx, "LITERAL_BOOL:TRUE", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("+(1, 1)", ctx, "LITERAL_INT:1");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testCSE() {
		RewriterStatement stmt1 = RewriterUtils.parse("+(*(a, b), *(b, a))", ctx, "FLOAT:a,b");
		RewriterStatement stmt2 = RewriterUtils.parse("+($1:*(a, b), $1)", ctx, "FLOAT:a,b");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		RewriterDatabase db = new RewriterDatabase();

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		db.insertEntry(ctx, stmt1);

		assert !db.insertEntry(ctx, stmt2);
	}

	@Test
	public void testExactMatch() {
		RewriterStatement stmt1 = RewriterUtils.parse("+(*(a, b), *(b, a))", ctx, "FLOAT:a,b");
		RewriterStatement stmt2 = RewriterUtils.parse("+($1:*(a, b), $1)", ctx, "FLOAT:a,b");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert !stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
		assert !stmt2.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt1, stmt2));
	}

	@Test
	public void testMinEquivalence() {
		RewriterStatement stmt1 = RewriterUtils.parse("min(min(A), min(B))", ctx, "MATRIX:A,B");
		RewriterStatement stmt2 = RewriterUtils.parse("min(A, B)", ctx, "MATRIX:A,B");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testSumEquivalence() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(A)", ctx, "MATRIX:A,B");
		RewriterStatement stmt2 = RewriterUtils.parse("sum(t(A))", ctx, "MATRIX:A,B");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testSimpleAlgebra1() {
		RewriterStatement stmt1 = RewriterUtils.parse("-(X, *(Y, X))", ctx, "MATRIX:X,Y");
		RewriterStatement stmt2 = RewriterUtils.parse("*(-(1, Y), X)", ctx, "MATRIX:X,Y", "LITERAL_INT:1");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testSimpleAlgebra2() {
		RewriterStatement stmt1 = RewriterUtils.parse("diag(*(X, 7))", ctx, "MATRIX:X,Y", "LITERAL_INT:7");
		RewriterStatement stmt2 = RewriterUtils.parse("*(diag(X), 7)", ctx, "MATRIX:X,Y", "LITERAL_INT:7");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testSimpleAlgebra3() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(+(+(X, 7), Y))", ctx, "MATRIX:X,Y", "LITERAL_INT:7");
		RewriterStatement stmt2 = RewriterUtils.parse("+(+(sum(X), 7), sum(Y))", ctx, "MATRIX:X,Y", "LITERAL_INT:7");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testSimpleAlgebra4() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(-(+(+(X, 7), Y)))", ctx, "MATRIX:X,Y", "LITERAL_INT:7");

		RewriterStatement matX = RewriterUtils.parse("X", ctx, "MATRIX:X");
		RewriterStatement matY = RewriterUtils.parse("Y", ctx, "MATRIX:Y");
		Map<String, RewriterStatement> vars = new HashMap<>();
		vars.put("X", matX);
		vars.put("Y", matY);
		RewriterStatement stmt2 = RewriterUtils.parse("-(+(sum(+(X, 7)), sum(Y)))", ctx, vars, "LITERAL_INT:7");
		stmt2.givenThatEqual(vars.get("X").getNCol(), vars.get("Y").getNCol(), stmt2, ctx);
		stmt2.givenThatEqual(vars.get("X").getNRow(), vars.get("Y").getNRow(), stmt2, ctx);
		stmt2 = stmt2.recomputeAssertions();

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testSimpleSumPullOut() {
		RewriterStatement stmt1 = RewriterUtils.parse("-(sum(+(A, 7)))", ctx, "MATRIX:A", "LITERAL_INT:7");
		RewriterStatement stmt2 = RewriterUtils.parse("sum(+(-(A), -7))", ctx, "MATRIX:A", "LITERAL_INT:-7");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testSimpleInverseEquivalence() {
		RewriterStatement stmt1 = RewriterUtils.parse("inv(A)", ctx, "MATRIX:A,B", "LITERAL_INT:7");
		RewriterStatement stmt2 = RewriterUtils.parse("-(inv(-(A)))", ctx, "MATRIX:A,B", "LITERAL_INT:7");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testBackrefInequality() {
		// TODO
		// Some example where _backRef() is not the same as another one
		// As we need to compare to the meta-data
		assert false;
	}

	@Test
	public void myTest() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(-(X, 7))", ctx, "MATRIX:X,Y", "LITERAL_INT:1,7", "INT:a", "LITERAL_FLOAT:7.0");
		stmt1 = canonicalConverter.apply(stmt1);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
	}

	@Test
	public void myTest2() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(_idxExpr(_idx(1, 7), -(a)))", ctx, "MATRIX:X,Y", "LITERAL_INT:1,7", "INT:a");
		stmt1 = canonicalConverter.apply(stmt1);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
	}

	@Test
	public void myTest3() {
		RewriterStatement stmt = RewriterUtils.parse("%*%(X,[](B,1,ncol(X),1,ncol(B)))", ctx, "MATRIX:X,B,intercept", "LITERAL_INT:1");
		stmt = canonicalConverter.apply(stmt);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
	}

	@Test
	public void myTest4() {
		RewriterStatement stmt = RewriterUtils.parse("*(CBind(t(KM),KM_cols_select),KM_cols_select)", ctx, "MATRIX:KM,KM_cols_select");
		stmt = canonicalConverter.apply(stmt);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
	}

	@Test
	public void myTest5() {
		RewriterStatement stmt = RewriterUtils.parse("*(CBind(A, A),A)", ctx, "MATRIX:A");
		stmt = canonicalConverter.apply(stmt);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
	}

	@Test
	public void myTest6() {
		RewriterStatement stmt = RewriterUtils.parse("rowSums(<=(D,minD))", ctx, "MATRIX:D,minD");
		stmt = canonicalConverter.apply(stmt);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
	}

	@Test
	public void myTest7() {
		String stmtStr = "MATRIX:combined\n" +
				"FLOAT:int0,int496,int236,int618\n" +
				"LITERAL_INT:1,2\n" +
				"INT:parsertemp71754,int497,int280\n" +
				"&(RBind(!=([](combined,1,-(parsertemp71754,int497),1,ncol(combined)),[](combined,2,nrow(combined),1,ncol(combined))),rand(1,1,int0,int496)),RBind(rand(1,1,int618,int236),!=([](combined,1,-(parsertemp71754,int280),1,ncol(combined)),[](combined,2,nrow(combined),1,ncol(combined)))))";

		RewriterStatement stmt = RewriterUtils.parse(stmtStr, ctx);
		stmt = canonicalConverter.apply(stmt);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
	}

	@Test
	public void myTest8() {
		String stmtStr = "MATRIX:prec_chol,X,mu\n" +
				"INT:i,k\n" +
				"LITERAL_INT:1,5\n" +
				"%*%(X,[](prec_chol,1,*(i,ncol(X)),1,5))";

		RewriterStatement stmt = RewriterUtils.parse(stmtStr, ctx);
		stmt = canonicalConverter.apply(stmt);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
	}

	@Test
	public void myTest9() {
		String stmtStr = "MATRIX:A,scale_X,shift_X,parsertemp282257,parsertemp282256,parsertemp282259,parsertemp282258\n" +
				"INT:m_ext\n" +
				"LITERAL_INT:1\n" +
				"+(%*%(diag(scale_X),t(+(%*%(parsertemp282256,A),%*%(shift_X,A)))),%*%(shift_X,[](t(+(parsertemp282257,parsertemp282258)),m_ext,m_ext,1,nrow(parsertemp282259))))";

		RewriterStatement stmt = RewriterUtils.parse(stmtStr, ctx);
		stmt = canonicalConverter.apply(stmt);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
	}

	@Test
	public void myTest10() {
		String stmtStr = "MATRIX:P,minD,D,X\n" +
				"/(%*%(t(/(<=(D,minD),rowSums(P))),X),t(colSums(/(<=(D,minD),rowSums(P)))))";

		RewriterStatement stmt = RewriterUtils.parse(stmtStr, ctx);
		stmt = canonicalConverter.apply(stmt);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
	}

	@Test
	public void testConstantFolding1() {
		RewriterStatement stmt1 = RewriterUtils.parse("*(1, A)", ctx, "MATRIX:A", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("A", ctx, "MATRIX:A");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testConstantFolding2() {
		RewriterStatement stmt1 = RewriterUtils.parse("+(A, 0)", ctx, "MATRIX:A", "LITERAL_INT:0");
		RewriterStatement stmt2 = RewriterUtils.parse("A", ctx, "MATRIX:A");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testConstantFolding3() {
		RewriterStatement stmt1 = RewriterUtils.parse("+(A, *(3, -(1, 1)))", ctx, "MATRIX:A", "LITERAL_INT:1,3");
		RewriterStatement stmt2 = RewriterUtils.parse("A", ctx, "MATRIX:A");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testConstantFolding4() {
		RewriterStatement stmt1 = RewriterUtils.parse("*(A, 0)", ctx, "MATRIX:A", "LITERAL_FLOAT:0");
		RewriterStatement stmt2 = RewriterUtils.parse("rand(nrow(A), ncol(A), 0, 0)", ctx, "MATRIX:A", "LITERAL_FLOAT:0");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testAdvancedEquivalence1() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(+(A, -7))", ctx, "MATRIX:A", "LITERAL_FLOAT:-7");
		RewriterStatement stmt2 = RewriterUtils.parse("sum(-(A, 7))", ctx, "MATRIX:A", "LITERAL_FLOAT:7");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testInequality() {
		RewriterStatement stmt1 = RewriterUtils.parse("/(*(A, A), B)", ctx, "MATRIX:A,B");
		RewriterStatement stmt2 = RewriterUtils.parse("/(*(A, A), sum(B))", ctx, "MATRIX:A,B");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert !stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testDiagEquivalence() {
		RewriterStatement stmt1 = RewriterUtils.parse("diag(diag(A))", ctx, "MATRIX:A,B");
		RewriterStatement stmt2 = RewriterUtils.parse("diag(A)", ctx, "MATRIX:A,B");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testRIXInequality() {
		RewriterStatement stmt1 = RewriterUtils.parse("+(A, [](B, 1, nrow(A), 1, ncol(A)))", ctx, "MATRIX:A,B", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("+(A, B)", ctx, "MATRIX:A,B");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert !stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void convergenceTest() {
		String stmtStr = "MATRIX:dl_matrix\n" +
				"INT:i,j,46307663-5c68-48ba-aa86-c1c36de45dbe\n" +
				"LITERAL_INT:1,2\n" +
				"[](dl_matrix,+(i,-(2)),-(i,2),1,1)";

		RewriterStatement stmt = RewriterUtils.parse(stmtStr, ctx);
		stmt = canonicalConverter.apply(stmt);

		System.out.println("==========");
		System.out.println(stmt.toParsableString(ctx, true));
	}

	@Test
	public void someTest() {
		RewriterStatement stmt1 = RewriterUtils.parse("+([](%*%(A,B),151,151,1,ncol(B)),C)", ctx, "MATRIX:A,B,C", "LITERAL_INT:1,151");
		RewriterStatement stmt2 = RewriterUtils.parse("+([](C,151,151,1,ncol(B)),%*%(A,B))", ctx, "MATRIX:A,B,C", "LITERAL_INT:1,151");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert !stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void my_Test() {
		RewriterStatement stmt1 = RewriterUtils.parse("[](A, 1, 1, 151, 151)", ctx, "MATRIX:A,B,C", "LITERAL_INT:1,151");

		stmt1 = canonicalConverter.apply(stmt1);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
	}

	@Test
	public void testSumEquality2() {
		RewriterStatement stmt1 = RewriterUtils.parse("rowSums(colSums(A))", ctx, "MATRIX:A");
		RewriterStatement stmt2 = RewriterUtils.parse("as.matrix(sum(A))", ctx, "MATRIX:A");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testSumEquality3() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(%*%(A, B))", ctx, "MATRIX:A,B");
		RewriterStatement stmt2 = RewriterUtils.parse("as.scalar(%*%(colSums(A), rowSums(B)))", ctx, "MATRIX:A,B");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testSumEquality4() {
		RewriterStatement stmt1 = RewriterUtils.parse("%*%([](A, 1, 1, 1, ncol(A)), [](B, 1, nrow(B), 1, 1))", ctx, "MATRIX:A,B", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("as.matrix(sum(*(t([](A, 1, 1, 1, ncol(A))), [](B, 1, nrow(B), 1, 1))))", ctx, "MATRIX:A,B", "LITERAL_INT:1");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testSumEquality5() {
		RewriterStatement stmt1 = RewriterUtils.parse("rowSums([](A, 1, nrow(A), 1, 1))", ctx, "MATRIX:A,B", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("[](A, 1, nrow(A), 1, 1)", ctx, "MATRIX:A,B", "LITERAL_INT:1");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testSimpleConvergence() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(a)", ctx, "FLOAT:a");

		stmt1 = canonicalConverter.apply(stmt1);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
	}

	@Test
	public void testImplicitInequality() {
		RewriterStatement stmt1 = RewriterUtils.parse("+([](A,1, nrow(A), 1, 1), B)", ctx, "MATRIX:A,B", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("+([](A,1, nrow(A), 1, 1), [](B, 1, nrow(B), 1, 1))", ctx, "MATRIX:A,B", "LITERAL_INT:1");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert !stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testTraceEquivalence() {
		RewriterStatement stmt1 = RewriterUtils.parse("trace(%*%(t(S),R))", ctx, "MATRIX:S,R", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("sum(*(S,R))", ctx, "MATRIX:S,R", "LITERAL_INT:1");

		System.out.println("Cost1: " + RewriterCostEstimator.estimateCost(stmt1, ctx));
		System.out.println("Cost2: " + RewriterCostEstimator.estimateCost(stmt2, ctx));

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testMMEquivalence() {
		RewriterStatement stmt1 = RewriterUtils.parse("%*%(A,*(b, B))", ctx, "MATRIX:A,B", "FLOAT:b", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("*(b, %*%(A, B))", ctx, "MATRIX:A,B", "FLOAT:b", "LITERAL_INT:1");

		System.out.println("Cost1: " + RewriterCostEstimator.estimateCost(stmt1, ctx));
		System.out.println("Cost2: " + RewriterCostEstimator.estimateCost(stmt2, ctx));

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println(stmt1.getAssertions(ctx));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));
		System.out.println(stmt2.getAssertions(ctx));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testMMEquivalence2() {
		RewriterStatement stmt1 = RewriterUtils.parse("cast.MATRIX(sum(*(t([](A, 1, 1, 1, ncol(A))), [](B, 1, nrow(B), 1, 1))))", ctx, "MATRIX:A,B", "FLOAT:b", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("%*%([](A, 1, 1, 1, ncol(A)), [](B, 1, nrow(B), 1, 1))", ctx, "MATRIX:A,B", "FLOAT:b", "LITERAL_INT:1");

		System.out.println("Cost1: " + RewriterCostEstimator.estimateCost(stmt1, ctx));
		System.out.println("Cost2: " + RewriterCostEstimator.estimateCost(stmt2, ctx));

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testColSumEquivalence4() {
		RewriterStatement stmt1 = RewriterUtils.parse("colSums(*(A, b))", ctx, "MATRIX:A,B", "FLOAT:b", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("*(b, colSums(A))", ctx, "MATRIX:A,B", "FLOAT:b", "LITERAL_INT:1");

		System.out.println("Cost1: " + RewriterCostEstimator.estimateCost(stmt1, ctx));
		System.out.println("Cost2: " + RewriterCostEstimator.estimateCost(stmt2, ctx));

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testColSumEquivalence5() {
		RewriterStatement stmt1 = RewriterUtils.parse("colSums(*(A, b))", ctx, "MATRIX:A,B", "FLOAT:b", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("*(b, colSums(A))", ctx, "MATRIX:A,B", "FLOAT:b", "LITERAL_INT:1");

		System.out.println("Cost1: " + RewriterCostEstimator.estimateCost(stmt1, ctx));
		System.out.println("Cost2: " + RewriterCostEstimator.estimateCost(stmt2, ctx));

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testZeroElimination() {
		RewriterStatement stmt1 = RewriterUtils.parse("*(A,0.0)", ctx, "MATRIX:A,B", "FLOAT:b", "LITERAL_INT:1", "LITERAL_FLOAT:0.0");
		RewriterStatement stmt2 = RewriterUtils.parse("const(A, 0.0)", ctx, "MATRIX:A,B", "FLOAT:b", "LITERAL_INT:1", "LITERAL_FLOAT:0.0");

		System.out.println("Cost1: " + RewriterCostEstimator.estimateCost(stmt1, ctx));
		System.out.println("Cost2: " + RewriterCostEstimator.estimateCost(stmt2, ctx));

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testMMScalarPullout() {
		RewriterStatement stmt1 = RewriterUtils.parse("%*%(*(A, b), B)", ctx, "MATRIX:A,B", "FLOAT:b", "LITERAL_INT:1", "LITERAL_FLOAT:0.0");
		RewriterStatement stmt2 = RewriterUtils.parse("*(b, %*%(A, B))", ctx, "MATRIX:A,B", "FLOAT:b", "LITERAL_INT:1", "LITERAL_FLOAT:0.0");

		long cost1 = RewriterCostEstimator.estimateCost(stmt1, ctx);
		long cost2 = RewriterCostEstimator.estimateCost(stmt2, ctx);

		System.out.println("Cost1: " + cost1);
		System.out.println("Cost2: " + cost2);

		assert cost2 == cost1;

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testWrong() {
		RewriterStatement stmt1 = RewriterUtils.parse("*(sum(colVec(A)),colSums(B))", ctx, "MATRIX:A,B", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("%*%(colVec(A),colSums(B))", ctx, "MATRIX:A,B", "LITERAL_INT:1");

		System.out.println("Cost1: " + RewriterCostEstimator.estimateCost(stmt1, ctx));
		System.out.println("Cost2: " + RewriterCostEstimator.estimateCost(stmt2, ctx));

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert !stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testWrong2() {
		RewriterStatement stmt1 = RewriterUtils.parse("*(a,1.0)", ctx, "FLOAT:a", "LITERAL_FLOAT:1.0");
		RewriterStatement stmt2 = RewriterUtils.parse("a", ctx, "FLOAT:a", "LITERAL_INT:1");

		System.out.println("Cost1: " + RewriterCostEstimator.estimateCost(stmt1, ctx));
		System.out.println("Cost2: " + RewriterCostEstimator.estimateCost(stmt2, ctx));

		RewriterStatement newStmt = canonicalConverter.apply(stmt1);
		System.out.println(newStmt);
		System.out.println(stmt1);
		//stmt2 = canonicalConverter.apply(stmt2);

		/*System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));*/
	}

	@Test
	public void testRev() {
		RewriterStatement stmt1 = RewriterUtils.parse("rev(rev(A))", ctx, "MATRIX:A,B", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("A", ctx, "MATRIX:A,B", "LITERAL_INT:1");

		System.out.println("Cost1: " + RewriterCostEstimator.estimateCost(stmt1, ctx));
		System.out.println("Cost2: " + RewriterCostEstimator.estimateCost(stmt2, ctx));

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testTrace() {
		RewriterStatement stmt1 = RewriterUtils.parse("trace(%*%(B,B))", ctx, "MATRIX:A,B", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("sum(*(A, t(B)))", ctx, "MATRIX:A,B", "LITERAL_INT:1");

		System.out.println("Cost1: " + RewriterCostEstimator.estimateCost(stmt1, ctx));
		System.out.println("Cost2: " + RewriterCostEstimator.estimateCost(stmt2, ctx));

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		stmt1.compress();
		stmt2.compress();

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert !stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testFused1() {
		RewriterStatement stmt1 = RewriterUtils.parse("1-*(A, B)", ctx, "MATRIX:A,B");
		RewriterStatement stmt2 = RewriterUtils.parse("-(1.0, *(A, B))", ctx, "MATRIX:A,B", "LITERAL_FLOAT:1.0");

		System.out.println("Cost1: " + RewriterCostEstimator.estimateCost(stmt1, ctx));
		System.out.println("Cost2: " + RewriterCostEstimator.estimateCost(stmt2, ctx));

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testFused2() {
		RewriterStatement stmt1 = RewriterUtils.parse("+(a, 1-*(A, B))", ctx, "MATRIX:A,B", "FLOAT:a");
		RewriterStatement stmt2 = RewriterUtils.parse("-(1.0, -(*(A, B), a))", ctx, "MATRIX:A,B", "FLOAT:a", "LITERAL_FLOAT:1.0");

		System.out.println("Cost1: " + RewriterCostEstimator.estimateCost(stmt1, ctx));
		System.out.println("Cost2: " + RewriterCostEstimator.estimateCost(stmt2, ctx));

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testFused3() {
		RewriterStatement stmt1 = RewriterUtils.parse("log_nz(A)", ctx, "MATRIX:A,B", "FLOAT:a");
		RewriterStatement stmt2 = RewriterUtils.parse("*(!=(0.0, A), log(A))", ctx, "MATRIX:A,B", "LITERAL_FLOAT:0.0");

		System.out.println("Cost1: " + RewriterCostEstimator.estimateCost(stmt1, ctx));
		System.out.println("Cost2: " + RewriterCostEstimator.estimateCost(stmt2, ctx));

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testFused4() {
		RewriterStatement stmt1 = RewriterUtils.parse("log_nz(A, a)", ctx, "MATRIX:A,B", "FLOAT:a");
		RewriterStatement stmt2 = RewriterUtils.parse("*(!=(0.0, A), log(A, a))", ctx, "MATRIX:A,B", "FLOAT:a", "LITERAL_FLOAT:0.0");

		System.out.println("Cost1: " + RewriterCostEstimator.estimateCost(stmt1, ctx));
		System.out.println("Cost2: " + RewriterCostEstimator.estimateCost(stmt2, ctx));

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testFused5() {
		RewriterStatement stmt1 = RewriterUtils.parse("sq(1-*(A,A))", ctx, "MATRIX:A,B", "FLOAT:a");

		System.out.println("Cost1: " + RewriterCostEstimator.estimateCost(stmt1, ctx));

		stmt1 = canonicalConverter.apply(stmt1);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
	}

	@Test
	public void testFused6() {
		RewriterStatement stmt1 = RewriterUtils.parse("/(A,A)", ctx, "MATRIX:A,B", "FLOAT:a");
		RewriterStatement stmt2 = RewriterUtils.parse("/(A,rev(A))", ctx, "MATRIX:A,B", "FLOAT:a", "LITERAL_FLOAT:0.0");

		System.out.println("Cost1: " + RewriterCostEstimator.estimateCost(stmt1, ctx));
		System.out.println("Cost2: " + RewriterCostEstimator.estimateCost(stmt2, ctx));

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert !stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testFused7() {
		RewriterStatement stmt1 = RewriterUtils.parse("+*(A,a,B)", ctx, "MATRIX:A,B", "FLOAT:a");
		RewriterStatement stmt2 = RewriterUtils.parse("+(*(a, B), A)", ctx, "MATRIX:A,B", "FLOAT:a");

		System.out.println("Cost1: " + RewriterCostEstimator.estimateCost(stmt1, ctx));
		System.out.println("Cost2: " + RewriterCostEstimator.estimateCost(stmt2, ctx));

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testFused8() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(!=(0.0, A))", ctx, "MATRIX:A,B", "LITERAL_FLOAT:0.0");
		RewriterStatement stmt2 = RewriterUtils.parse("_nnz(A)", ctx, "MATRIX:A,B", "FLOAT:a");

		System.out.println("Cost1: " + RewriterCostEstimator.estimateCost(stmt1, ctx));
		System.out.println("Cost2: " + RewriterCostEstimator.estimateCost(stmt2, ctx));

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testFusedCompilation() {
		RewriterStatement stmt1 = RewriterUtils.parse("+(a,*2(1-*(B,B)))", ctx, "MATRIX:A,B", "FLOAT:a");

		System.out.println("Cost1: " + RewriterCostEstimator.estimateCost(stmt1, ctx));

		stmt1 = canonicalConverter.apply(stmt1);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
	}

	@Test
	public void testSum() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(+(a,A))", ctx, "MATRIX:A,B", "FLOAT:a");
		RewriterStatement stmt2 = RewriterUtils.parse("+(*(a, cast.FLOAT(length(A))), sum(A))", ctx, "MATRIX:A,B", "FLOAT:a", "LITERAL_FLOAT:0.0");

		System.out.println("Cost1: " + RewriterCostEstimator.estimateCost(stmt1, ctx));
		System.out.println("Cost2: " + RewriterCostEstimator.estimateCost(stmt2, ctx));

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testRowSums() {
		RewriterStatement stmt1 = RewriterUtils.parse("*(rowSums(/(a,C)),b)", ctx, "MATRIX:A,B,C", "FLOAT:a,b");
		RewriterStatement stmt2 = RewriterUtils.parse("rowSums(/(*(a,b),C))", ctx, "MATRIX:A,B,C", "FLOAT:a,b", "LITERAL_FLOAT:0.0");

		System.out.println("Cost1: " + RewriterCostEstimator.estimateCost(stmt1, ctx));
		System.out.println("Cost2: " + RewriterCostEstimator.estimateCost(stmt2, ctx));

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testRowSums2() {
		RewriterStatement stmt1 = RewriterUtils.parse("rowSums(*(A,+(B,1.0)))", ctx, "MATRIX:A,B,C", "FLOAT:a,b", "LITERAL_FLOAT:1.0");
		RewriterStatement stmt2 = RewriterUtils.parse("+(rowSums(A), rowSums(*(B,A)))", ctx, "MATRIX:A,B,C", "FLOAT:a,b", "LITERAL_FLOAT:1.0");

		System.out.println("Cost1: " + RewriterCostEstimator.estimateCost(stmt1, ctx));
		System.out.println("Cost2: " + RewriterCostEstimator.estimateCost(stmt2, ctx));

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testDistrib3() {
		RewriterStatement stmt1 = RewriterUtils.parse("*(A,+(B,1.0))", ctx, "MATRIX:A,B,C", "FLOAT:a,b", "LITERAL_FLOAT:1.0");
		RewriterStatement stmt2 = RewriterUtils.parse("+(A, *(B,A))", ctx, "MATRIX:A,B,C", "FLOAT:a,b", "LITERAL_FLOAT:1.0");

		System.out.println("Cost1: " + RewriterCostEstimator.estimateCost(stmt1, ctx));
		System.out.println("Cost2: " + RewriterCostEstimator.estimateCost(stmt2, ctx));

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testRev2() {
		RewriterStatement stmt1 = RewriterUtils.parse("trace(rev(A))", ctx, "MATRIX:A,B,C", "FLOAT:a,b", "LITERAL_FLOAT:1.0");
		RewriterStatement stmt2 = RewriterUtils.parse("trace(A)", ctx, "MATRIX:A,B,C", "FLOAT:a,b", "LITERAL_FLOAT:1.0");

		System.out.println("Cost1: " + RewriterCostEstimator.estimateCost(stmt1, ctx));
		System.out.println("Cost2: " + RewriterCostEstimator.estimateCost(stmt2, ctx));

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert !stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testSumInequality() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(+(a,*(B,c)))", ctx, "MATRIX:B", "FLOAT:a,c");
		RewriterStatement stmt2 = RewriterUtils.parse("*(a, sum(+(B,c)))", ctx, "MATRIX:B", "FLOAT:a,c", "LITERAL_FLOAT:0.0");

		System.out.println("Cost1: " + RewriterCostEstimator.estimateCost(stmt1, ctx));
		System.out.println("Cost2: " + RewriterCostEstimator.estimateCost(stmt2, ctx));

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert !stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}

	@Test
	public void testDiag1() {
		RewriterStatement stmt1 = RewriterUtils.parse("diag(+(A, B))", ctx, "MATRIX:A,B");
		RewriterStatement stmt2 = RewriterUtils.parse("+(diag(A), diag(B))", ctx, "MATRIX:A,B");

		System.out.println("Cost1: " + RewriterCostEstimator.estimateCost(stmt1, ctx));
		System.out.println("Cost2: " + RewriterCostEstimator.estimateCost(stmt2, ctx));

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		System.out.println("==========");
		System.out.println(stmt1.toParsableString(ctx, true));
		System.out.println("==========");
		System.out.println(stmt2.toParsableString(ctx, true));

		assert stmt1.match(RewriterStatement.MatcherContext.exactMatch(ctx, stmt2, stmt1));
	}
}
