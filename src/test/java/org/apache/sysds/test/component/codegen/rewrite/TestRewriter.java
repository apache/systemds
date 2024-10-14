package org.apache.sysds.test.component.codegen.rewrite;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.mutable.MutableObject;
import org.apache.spark.internal.config.R;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.rewriter.MetaPropagator;
import org.apache.sysds.hops.rewriter.RewriterContextSettings;
import org.apache.sysds.hops.rewriter.RewriterDatabase;
import org.apache.sysds.hops.rewriter.RewriterHeuristic;
import org.apache.sysds.hops.rewriter.RewriterHeuristics;
import org.apache.sysds.hops.rewriter.RewriterInstruction;
import org.apache.sysds.hops.rewriter.RewriterRule;
import org.apache.sysds.hops.rewriter.RewriterRuleBuilder;
import org.apache.sysds.hops.rewriter.RewriterRuleCollection;
import org.apache.sysds.hops.rewriter.RewriterRuleSet;
import org.apache.sysds.hops.rewriter.RewriterRuntimeUtils;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.StatementBlock;
import org.junit.Test;
import scala.Tuple2;
import scala.Tuple3;
import scala.Tuple6;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;

public class TestRewriter {

	private enum DataType {
		FLOAT, INT, MATRIX
	};

	private ExecutedRule currentExecution;

	/*private int currentHopCount = 0;
	private RewriterStatement lastStatement = null;
	private List<Hop> lastHops = null;
	private String lastProg = null;
	private RewriterStatement nextStatement = null;
	private List<Hop> nextHops = null;
	private String nextProg = null;*/
	private List<ExecutedRule> costIncreasingTransformations = new ArrayList<>();

	private Function<DMLProgram, Boolean> interceptor = prog -> {
		//int hopCtr = 0;
		for (StatementBlock sb : prog.getStatementBlocks()) {
			int hopCount = sb.getHops() == null ? 0 : sb.getHops().stream().mapToInt(this::countHops).sum();
			System.out.println("HopCount: " + hopCount);
			currentExecution.to.hops = sb.getHops();
			currentExecution.to.hopCount = hopCount;

			if (currentExecution.from.hopCount < currentExecution.to.hopCount)
				costIncreasingTransformations.add(currentExecution);

			/*if (lastStatement == null) {
				currentHopCount = hopCount;
			} else if (hopCount > currentHopCount) {
				costIncreasingTransformations.add(new Tuple6<>(lastStatement, lastProg, lastHops, nextStatement, nextProg, nextHops));
				currentHopCount = hopCount;
			}*/

			//System.out.println(phase + "-Size: " + hopCount);
			//System.out.println("==> " + sb);
			return true;
		}
		return true;
	};

	private int countHops(List<Hop> hops) {
		return hops.stream().mapToInt(this::countHops).sum();
	}

	private int countHops(Hop hop) {
		if (hop instanceof LiteralOp)
			return 0;
		int curr = 1;
		for (Hop child : hop.getInput())
			curr += countHops(child);
		return curr;
	}

	private String toDMLString(RewriterStatement stmt, final RuleContext ctx) {
		List<String> execStr = stmt.toExecutableString(ctx);
		boolean isMatrix = stmt.getResultingDataType(ctx).equals("MATRIX");
		String resString;
		if (isMatrix)
			resString = "print(toString(" + execStr.get(execStr.size()-1) + "))";
		else
			resString = "print(" + execStr.get(execStr.size()-1) + ")";
		execStr.set(execStr.size()-1, resString);
		return String.join("\n", execStr);
	}

	@Test
	public void myTest() {
		RewriterRuntimeUtils.setupIfNecessary();
		/*RuleContext ctx = RewriterUtils.buildDefaultContext();
		Function<RewriterStatement, RewriterStatement> converter = RewriterUtils.buildCanonicalFormConverter(ctx, true);
		RewriterStatement stmt = RewriterUtils.parse("sum(A)", ctx, "MATRIX:A");
		converter.apply(stmt);*/
		RewriterRuntimeUtils.executeScript("X=as.matrix(1)\nprint(sum(X))");
		//RewriterRuntimeUtils.executeScript("X=rand(rows=10,cols=5)\nY=rand(rows=5,cols=10)\nprint(sum(X%*%Y))");
		//RewriterRuntimeUtils.executeScript("X=rand(rows=10,cols=5)\nY=rand(rows=5,cols=10)\nprint(sum(colSums(X) * colSums(t(Y))))");
	}

	//@Test
	public void interceptionTest() {
		System.out.println("OptLevel:" + OptimizerUtils.getOptLevel().toString());
		System.out.println("AllowOpFusion: " + OptimizerUtils.ALLOW_OPERATOR_FUSION);
		System.out.println("AllowSumProductRewrites: " + OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES);
		System.out.println("AllowConstantFolding: " + OptimizerUtils.ALLOW_CONSTANT_FOLDING);

		// Setup default context
		RuleContext ctx = RewriterUtils.buildDefaultContext();
		Function<RewriterStatement, RewriterStatement> converter = RewriterUtils.buildCanonicalFormConverter(ctx, false);

		RewriterDatabase db = new RewriterDatabase();
		RewriterDatabase exactExprDB = new RewriterDatabase();
		List<RewriterStatement> equivalentStatements = new ArrayList<>();

		RewriterRuntimeUtils.attachHopInterceptor(prog -> {
			RewriterRuntimeUtils.forAllUniqueTranslatableStatements(prog, 10, stmt -> {
				RewriterStatement cpy = stmt.nestedCopyOrInject(new HashMap<>(), el -> null);
				System.out.println("Stmt: " + stmt);
				stmt = converter.apply(stmt);

				RewriterStatement oldEntry = db.insertOrReturn(ctx, stmt);

				if (oldEntry == null) {
					List<RewriterStatement> expr = new ArrayList<>();
					expr.add(cpy);
					stmt.unsafePutMeta("equivalentExpressions", expr);
				} else {
					List<RewriterStatement> eStmts = (List<RewriterStatement>) oldEntry.getMeta("equivalentExpressions");
					eStmts.add(cpy);

					if (eStmts.size() == 2)
						equivalentStatements.add(oldEntry);

					System.out.println("Found equivalent statement!");
				}

				//System.out.println("Canonical form:");
				//System.out.println(stmt.toString(ctx));
			}, exactExprDB, ctx);
			return false;
		});

		RewriterRuntimeUtils.executeScript("X=rand(rows=10,cols=5)\nY=rand(rows=5,cols=10)\nprint(sum(X%*%Y))");
		RewriterRuntimeUtils.executeScript("X=rand(rows=10,cols=5)\nY=rand(rows=5,cols=10)\nprint(sum(colSums(X) * colSums(t(Y))))");

		System.out.println("===== ALL EQUIVALENCES =====");

		for (RewriterStatement eStmt : equivalentStatements) {
			System.out.println("Canonical form: " + eStmt.toString(ctx));
			((List<RewriterStatement>)eStmt.getMeta("equivalentExpressions")).forEach(stmt -> System.out.println(stmt.toString(ctx)));
		}
	}

	/*@Test
	public void test() {
		System.out.println("OptLevel:" + OptimizerUtils.getOptLevel().toString());
		System.out.println("AllowOpFusion: " + OptimizerUtils.ALLOW_OPERATOR_FUSION);
		System.out.println("AllowSumProductRewrites: " + OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES);
		System.out.println("AllowConstantFolding: " + OptimizerUtils.ALLOW_CONSTANT_FOLDING);

		createRules((ex) -> {
			try {
				testDMLStmt(ex);
				return true;
			} catch (Exception e) {
				e.printStackTrace();
				return false;
			}
		});

		System.out.println("===== FOUND TRANSFORMATIONS =====");

		for (ExecutedRule incTransforms : costIncreasingTransformations) {
			System.out.println("==========");
			System.out.println("Rule: " + incTransforms.appliedRule.rule);
			System.out.println("Dir: " + incTransforms.appliedRule.forward);
			System.out.println("MatchRoot: " + incTransforms.match.getMatchRoot().toString(incTransforms.ctx));
			System.out.println(incTransforms.from.executableString);
			System.out.println("=>");
			System.out.println(incTransforms.to.executableString);
			System.out.println("HopCount: " + incTransforms.from.hopCount + " => " + incTransforms.to.hopCount);
		}
	}*/

	private static RewriterHeuristic mHeur;

	private void testDMLStmt(ExecutedRule ex) {
		final RuleContext ctx = ex.ctx;
		RewriterStatement stmt = ex.to.stmt;
		currentExecution = ex;

		if (mHeur == null)
			mHeur = RewriterRuleCollection.getHeur(ctx);
		try {
			RewriterHeuristic heur = mHeur;
			stmt = heur.apply(stmt);

			RewriterRuntimeUtils.attachHopInterceptor(interceptor);
			//System.setOut(new PrintStream(new CustomOutputStream(System.out, line -> System.err.println("INTERCEPT: " + line))));



			/*StringBuilder sb = new StringBuilder();
			sb.append(createVar("A", DataType.MATRIX, "random", Map.of("rows", 1, "cols", 10)));
			sb.append(createVar("B", DataType.MATRIX, "random", Map.of("rows", 1, "cols", 10)));
			sb.append("if(max(A) > 1) {print(lineage(A))} else {print(lineage(B))}\n");*/

			long timeMillis = System.currentTimeMillis();

			String str = toDMLString(stmt, ctx);
			ex.to.executableString = str;
			System.out.println("Executing:\n" + str);
			DMLScript.executeScript(new String[]{"-s", str});

			System.err.println("Done in " + (System.currentTimeMillis() - timeMillis) + "ms");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private String createVar(String name, DataType dType, String genType, Map<String, Object> meta) {
		switch (dType) {
			case INT:
				throw new NotImplementedException();
			case FLOAT:
				throw new NotImplementedException();
			case MATRIX:
				return name + "=" + createMatrixVar(genType, meta) + "\n";
			default:
				throw new IllegalArgumentException("Unknown data type");
		}
	}

	private String createMatrixVar(String genType, Map<String, Object> meta) {
		switch (genType) {
			case "random":
				return createRandomMatrix((Integer)meta.get("rows"), (Integer)meta.get("cols"));
			default:
				throw new IllegalArgumentException("Unknown matrix generation type");
		}
	}

	private String createRandomMatrix(int nrows, int ncols) {
		//return String.format("matrix(1,rows=%d,cols=%d)", nrows, ncols);
		return String.format("RAND(rows=%d,cols=%d)", nrows, ncols);
	}








	private RewriterRuleSet createRules(Function<ExecutedRule, Boolean> handler) {
		RuleContext ctx = RewriterContextSettings.getDefaultContext(new Random());
		ctx.metaPropagator = new MetaPropagator(ctx);

		ArrayList<RewriterRule> rules = new ArrayList<>();

		RewriterRuleCollection.addEqualitySubstitutions(rules, ctx);
		RewriterRuleCollection.addBooleAxioms(rules, ctx);
		RewriterRuleCollection.addImplicitBoolLiterals(rules, ctx);

		ArrayList<RewriterRule> expRules = new ArrayList<>();
		RewriterRuleCollection.expandStreamingExpressions(expRules, ctx);
		RewriterHeuristic streamExpansion = new RewriterHeuristic(new RewriterRuleSet(ctx, expRules));

		ArrayList<RewriterRule> pd = new ArrayList<>();
		RewriterRuleCollection.pushdownStreamSelections(pd, ctx);
		RewriterHeuristic streamSelectPushdown = new RewriterHeuristic(new RewriterRuleSet(ctx, pd));

		ArrayList<RewriterRule> flatten = new ArrayList<>();
		RewriterRuleCollection.flattenOperations(pd, ctx);
		RewriterHeuristic flattenOperations = new RewriterHeuristic(new RewriterRuleSet(ctx, pd));

		RewriterHeuristics canonicalFormCreator = new RewriterHeuristics();
		canonicalFormCreator.add("EXPAND STREAMING EXPRESSIONS", streamExpansion);
		canonicalFormCreator.add("PUSHDOWN STREAM SELECTIONS", streamSelectPushdown);
		canonicalFormCreator.add("FLATTEN OPERATIONS", flattenOperations);
		// TODO: Constant folding
		// TODO: CSE

		ArrayList<RewriterRule> colRules = new ArrayList<>();
		RewriterRuleCollection.collapseStreamingExpressions(colRules, ctx);
		RewriterHeuristic streamCollapse = new RewriterHeuristic(new RewriterRuleSet(ctx, colRules));

		ArrayList<RewriterRule> assertCollapsedRules = new ArrayList<>();
		RewriterRuleCollection.assertCollapsed(colRules, ctx);
		RewriterHeuristic assertCollapsed = new RewriterHeuristic(new RewriterRuleSet(ctx, assertCollapsedRules));

		/*rules.add(new RewriterRuleBuilder(ctx)
				.parseGlobalVars("LITERAL_BOOL:TRUE")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("TRUE")
				.toParsedStatement("<(_lower(1), 1)")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.parseGlobalVars("INT:a")
				.parseGlobalVars("LITERAL_INT:1,0")
				.parseGlobalStatementAsVariable("LOWER", "_lower(a)")
				.withParsedStatement("LOWER")
				.toParsedStatement("as.scalar(rand(1, 1, +(LOWER, _lower(0)), LOWER))")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.parseGlobalVars("INT:a")
				.parseGlobalVars("LITERAL_INT:1,0")
				.parseGlobalStatementAsVariable("LOWER", "_lower(a)")
				.parseGlobalStatementAsVariable("p1", "_posInt()")
				.parseGlobalStatementAsVariable("p2", "_posInt()")
				.withParsedStatement("LOWER")
				.toParsedStatement("mean(rand(p1, p2, +(LOWER, _lower(0)), LOWER))")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.parseGlobalVars("MATRIX:A")
				.withParsedStatement("mean(A)")
				.toParsedStatement("/(sum(A),*(ncol(A),nrow(A)))")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.parseGlobalVars("MATRIX:A")
				.withParsedStatement("mean(A)")
				.toParsedStatement("/(sum(A),length(A))")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.parseGlobalVars("MATRIX:A")
				.parseGlobalVars("LITERAL_INT:1")
				.parseGlobalStatementAsVariable("DIFF", "-(A, mean(A))")
				.withParsedStatement("var(A)")
				.toParsedStatement("*(/(1, length(A)), sum(*(DIFF, DIFF)))")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.parseGlobalVars("MATRIX:A")
				.withParsedStatement("length(A)")
				.toParsedStatement("*(ncol(A),nrow(A))")
				.build()
		);*/

		Random rd = new Random();
		RewriterRuleSet ruleSet = new RewriterRuleSet(ctx, rules);
		ruleSet.accelerate();
		RewriterDatabase db = new RewriterDatabase();

		String matrixDef = "MATRIX:A,B,C";
		String intDef = "LITERAL_INT:1,10,20";
		String floatDef = "LITERAL_FLOAT:0,1.0,-0.0001,0.0001,-1.0";
		String boolDef = "LITERAL_BOOL:TRUE,FALSE";

		String floatVarDef = "FLOAT:a";

		//String startStr = "trace(*(rand(10, 10, 0, 1), rand(10, 10, 0, 1)))";
		//String startStr = "t(t(t(rand(10, 10, 0, 1))))";
		//String startStr = "t(t(t(rand(10, 10, 0, 1))))";
		//String startStr = "trace(%*%(rand(10, 10, 0, 1), rand(10, 10, 0, 1)))";
		//String startStr = "sum(*(colSums(rand(10, 10, 0, 1.0)), t(rowSums(rand(10, 10, 0, 1.0)))))";
		//String startStr = "t(rowSums(t(rand(10, 10, 0, 1.0))))";
		//String startStr = "colSums(rand(10, 10, 0, 1.0))";
		//String startStr = "_idx(1, 1)";

		//String startStr = "sum(*(colSums(rand(10, 20, 0, 1.0)), colSums(t(rand(20, 10, 0, 1.0)))))";
		String startStr = "sum(*(colSums(*(rand(10, 20, 0, 1.0), a)), colSums(rand(10, 20, 0, 1.0))))";
		//String startStr = "+(+(A,B),C)";
		String startStr2 = "sum(%*%(*(rand(10, 20, 0, 1.0), a), t(rand(10, 20, 0, 1.0))))";
		//String startStr2 = "+(A,+(B,C))";
		RewriterStatement stmt = RewriterUtils.parse(startStr, ctx, matrixDef, intDef, floatDef, boolDef, floatVarDef);

		stmt = canonicalFormCreator.apply(stmt, (t, r) -> {
			if (r != null)
				System.out.println("Applying rule: " + r.getName());
			System.out.println(t);
			return true;
		});

		RewriterUtils.mergeArgLists(stmt, ctx);
		System.out.println("PRE1: " + stmt.toString(ctx));
		RewriterUtils.topologicalSort(stmt, ctx, (el, parent) -> el.isArgumentList() && parent != null && Set.of("+", "-", "*", "_idxExpr").contains(parent.trueInstruction()));
		System.out.println("FINAL1: " + stmt.toString(ctx));

		db.insertEntry(ctx, stmt);

		RewriterStatement toCompare = RewriterUtils.parse(startStr2, ctx, matrixDef, intDef, floatDef, boolDef, floatVarDef);

		toCompare = canonicalFormCreator.apply(toCompare, (t, r) -> {
			if (r != null)
				System.out.println("Applying rule: " + r.getName());
			System.out.println(t);
			return true;
		});

		RewriterUtils.mergeArgLists(toCompare, ctx);
		System.out.println("PRE2: " + toCompare.toString(ctx));
		RewriterUtils.topologicalSort(toCompare, ctx, (el, parent) -> el.isArgumentList() && parent != null && Set.of("+", "-", "*", "_idxExpr").contains(parent.trueInstruction()));
		System.out.println("FINAL2: " + toCompare.toString(ctx));

		System.out.println("Hash1: " + stmt.hashCode());
		System.out.println("Hash2: " + toCompare.hashCode());

		if (db.insertEntry(ctx, toCompare))
			System.out.println("No match!");
		else
			System.out.println("Match!");

		if (true)
			return null;

		//handler.apply(RewriterUtils.parse("+(2, 2)", ctx, "LITERAL_INT:2"), ctx);
		db.insertEntry(ctx, stmt);

		//RewriterRuleSet.ApplicableRule match = ruleSet.findFirstApplicableRule(stmt);
		long millis = System.currentTimeMillis();
		//ArrayList<RewriterRuleSet.ApplicableRule> applicableRules = ruleSet.findApplicableRules(stmt);
		List<RewriterRuleSet.ApplicableRule> applicableRules = ruleSet.acceleratedRecursiveMatch(stmt, false);

		RewriterStatement newStmt = stmt;

		ExecutionRecord initialRecord = new ExecutionRecord(stmt);
		ExecutedRule ex = ExecutedRule.create(ctx, null, null, initialRecord, initialRecord);

		PriorityQueue<ExecutionRecord> queue = new PriorityQueue<>(Comparator.comparingInt(r -> r.statementSize));
		int MAX_PARALLEL_INVESTIGATIONS = 1;
		List<Tuple2<ExecutionRecord, List<RewriterRuleSet.ApplicableRule>>> investigatedStatements = new ArrayList<>(List.of(new Tuple2<>(initialRecord, applicableRules)));

		if (!handler.apply(ex))
			return ruleSet;

		MutableObject<Tuple3<RewriterStatement, RewriterStatement, Integer>> modificationHandle = new MutableObject<>(null);

		for (int i = 0; i < 10000 && !investigatedStatements.isEmpty() && costIncreasingTransformations.size() < 100; i++) {
			MAX_PARALLEL_INVESTIGATIONS = i / 100 + 1;
			// Choose investigated statement
			int rdInvestigation = rd.nextInt(investigatedStatements.size());
			initialRecord = investigatedStatements.get(rdInvestigation)._1;
			applicableRules = investigatedStatements.get(rdInvestigation)._2;

			if (applicableRules == null) {
				applicableRules = ruleSet.acceleratedRecursiveMatch(initialRecord.stmt, false);
				if (applicableRules.isEmpty()) {
					replaceInvestigationList(investigatedStatements, queue, rdInvestigation);
					continue;
				}
			}

			int ruleIndex = rd.nextInt(applicableRules.size());
			RewriterRuleSet.ApplicableRule next = applicableRules.get(ruleIndex);

			int matchIdx = rd.nextInt(next.matches.size());
			RewriterStatement.MatchingSubexpression match = next.matches.remove(matchIdx);

			if (next.forward)
				newStmt = next.rule.applyForward(match, stmt, false, modificationHandle);
			else
				newStmt = next.rule.applyBackward(match, stmt, false, modificationHandle);

			ExecutionRecord newRcrd = new ExecutionRecord(newStmt);

			ex = ExecutedRule.create(ctx, next, match, initialRecord, newRcrd);


			if (ex.from.hopCount < ex.to.hopCount) {
				// Then we erase knowledge from the modified state as the compiler would not track it from there
				// This eliminates repetitive patterns
				// We assume that the compiler loses track of any property (this is too strong but works for now)
				RewriterStatement cpy = ex.to.stmt.nestedCopyOrInject(new HashMap<>(), x -> null);
				if (db.insertEntry(ctx, cpy)) {
					RewriterStatement newRoot = ex.to.stmt;
					ex.to.stmt = cpy;

					System.out.println("Rewrite took " + (System.currentTimeMillis() - millis) + "ms");
					System.out.println("DB-size: " + db.size());

					if (!handler.apply(ex))
						return ruleSet;

					RewriterStatement root = modificationHandle.getValue()._1();
					RewriterStatement parent = modificationHandle.getValue()._2();
					Integer pIdx = modificationHandle.getValue()._3();

					RewriterStatement replacement = RewriterUtils.parse("_rd" + root.getResultingDataType(ctx), ctx);

					if (parent != null) {
						parent.getOperands().set(pIdx, replacement);
					}

					newRcrd = new ExecutionRecord(newRoot);

					queue.add(newRcrd);
				}
			} else {
				if (db.insertEntry(ctx, newStmt)) {
					System.out.println("Rewrite took " + (System.currentTimeMillis() - millis) + "ms");
					System.out.println("DB-size: " + db.size());

					if (!handler.apply(ex))
						return ruleSet;

					queue.add(newRcrd);
				}
			}

			millis = System.currentTimeMillis();

			if (next.matches.isEmpty())
				applicableRules.remove(ruleIndex);


			while (investigatedStatements.size() < MAX_PARALLEL_INVESTIGATIONS && !queue.isEmpty()) {
				ExecutionRecord nextRec = queue.poll();
				investigatedStatements.add(new Tuple2<>(nextRec, null));
			}

			if (applicableRules.isEmpty())
				replaceInvestigationList(investigatedStatements, queue, rdInvestigation);
		}


		/*System.out.println(stmt.toString(ctx));
		System.out.println(next.toString(ctx));
		System.out.println(next.rule.applyForward(next.matches.get(0), stmt, true));*/

		return ruleSet;
	}

	private void replaceInvestigationList(List<Tuple2<ExecutionRecord, List<RewriterRuleSet.ApplicableRule>>> investigatedStatements, PriorityQueue<ExecutionRecord> q, int idx) {
		if (!q.isEmpty()) {
			ExecutionRecord nextRec = q.poll();
			investigatedStatements.set(idx, new Tuple2<>(nextRec, null));
		} else {
			investigatedStatements.remove(idx);
		}
	}





	private class CustomOutputStream extends OutputStream {
		private PrintStream ps;
		private StringBuilder buffer = new StringBuilder();
		private Consumer<String> lineHandler;

		public CustomOutputStream(PrintStream actualPrintStream, Consumer<String> lineHandler) {
			this.ps = actualPrintStream;
			this.lineHandler = lineHandler;
		}

		@Override
		public void write(int b) {
			char c = (char) b;
			if (c == '\n') {
				lineHandler.accept(buffer.toString());
				buffer.setLength(0); // Clear the buffer after handling the line
			} else {
				buffer.append(c); // Accumulate characters until newline
			}
			// Handle the byte 'b', or you can write to any custom destination
			ps.print((char) b); // Example: redirect to System.err
		}

		@Override
		public void write(byte[] b, int off, int len) {
			for (int i = off; i < off + len; i++) {
				write(b[i]);
			}
		}
	}

	private static class ExecutedRule {
		RuleContext ctx;
		RewriterRuleSet.ApplicableRule appliedRule;
		RewriterStatement.MatchingSubexpression match;
		ExecutionRecord from;
		ExecutionRecord to;

		static ExecutedRule create(RuleContext ctx, RewriterRuleSet.ApplicableRule appliedRule, RewriterStatement.MatchingSubexpression match, ExecutionRecord from, ExecutionRecord to) {
			ExecutedRule r = new ExecutedRule();
			r.ctx = ctx;
			r.appliedRule = appliedRule;
			r.match = match;
			r.from = from;
			r.to = to;
			return r;
		}
	}

	private static class ExecutionRecord {
		RewriterStatement stmt;
		String executableString;
		List<Hop> hops;
		int hopCount;
		int statementSize;

		public ExecutionRecord(RewriterStatement stmt) {
			this(stmt, null, null, -1);
		}

		public ExecutionRecord(RewriterStatement stmt, String executableString, List<Hop> hops, int hopCount) {
			this.stmt = stmt;
			this.statementSize = 0;

			this.stmt.forEachPreOrder((el, parent, pIdx) -> {
				this.statementSize++;
				return true;
			});

			this.executableString = executableString;
			this.hops = hops;
			this.hopCount = hopCount;
		}
	}
}