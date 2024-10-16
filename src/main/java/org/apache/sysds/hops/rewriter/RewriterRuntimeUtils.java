package org.apache.sysds.hops.rewriter;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.DataGenOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.ReorgOp;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.FunctionStatement;
import org.apache.sysds.parser.FunctionStatementBlock;
import org.apache.sysds.parser.IfStatement;
import org.apache.sysds.parser.IfStatementBlock;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.WhileStatement;
import org.apache.sysds.parser.WhileStatementBlock;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.function.Function;

public class RewriterRuntimeUtils {
	public static final boolean interceptAll = true;
	public static final boolean printUnknowns = true;


	private static final String matrixDefs = "MATRIX:A,B,C";
	private static final String floatDefs = "FLOAT:q,r,s,t,f1,f2,f3,f4,f5";
	private static final String intDefs = "INT:i1,i2,i3,i4,i5";
	private static final String boolDefs = "BOOL:b1,b2,b3";

	private static boolean setupComplete = false;

	private static long totalCPUTime = 0L;
	private static long evaluatedExpressions = 0;

	public static void setupIfNecessary() {
		if (setupComplete)
			return;

		setupComplete = true;
		System.out.println("INTERCEPTOR");
		if (interceptAll) {
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
				long startMillis = System.currentTimeMillis();
				RewriterRuntimeUtils.forAllUniqueTranslatableStatements(prog, 5, mstmt -> {
					List<RewriterStatement> subtrees = RewriterUtils.generateSubtrees(mstmt, new HashMap<>(), ctx);
					for (RewriterStatement stmt : subtrees) {
						try {
							stmt = ctx.metaPropagator.apply(stmt);
							System.out.println("RawStmt: " + stmt.toString(ctx));
							RewriterStatement cpy = stmt.nestedCopyOrInject(new HashMap<>(), el -> null);
							if (!exactExprDB.insertEntry(ctx, cpy))
								continue;
							evaluatedExpressions++;
							System.out.println("Stmt: " + stmt.toString(ctx));
							stmt = converter.apply(stmt);
							System.out.println("Canonical form: " + stmt.toString(ctx));

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
						} catch (Exception e) {
							e.printStackTrace();
						}
					}
				}, exactExprDB, ctx);
				totalCPUTime += System.currentTimeMillis() - startMillis;
				return false;
			});

			Runtime.getRuntime().addShutdownHook(new Thread(() -> {
				System.out.println("===== ALL EQUIVALENCES =====");

				for (RewriterStatement eStmt : equivalentStatements) {
					System.out.println("Canonical form: " + eStmt.toString(ctx));
					List<RewriterStatement> equivalences = (List<RewriterStatement>)eStmt.getMeta("equivalentExpressions");
					equivalences.forEach(stmt -> System.out.println(stmt.toString(ctx) + "\t" + stmt.hashCode()));

					if (equivalences.size() == 0)
						System.out.println("All statements were actually equivalent!");
					//System.out.println(equivalences.get(0).match(new RewriterStatement.MatcherContext(ctx, equivalences.get(0))));
				}

				System.out.println();
				System.out.println("Total rewriter CPU time: " + totalCPUTime + "ms");
				System.out.println("Total evaluated unique expressions: " + evaluatedExpressions);
			}));
		}
	}

	public static void attachHopInterceptor(Function<DMLProgram, Boolean> interceptor) {
		DMLScript.hopInterceptor = interceptor;
	}

	// TODO: Make more flexible regarding program structure
	public static void forAllHops(DMLProgram program, Consumer<Hop> consumer) {
		for (StatementBlock sb : program.getStatementBlocks())
			sb.getHops().forEach(consumer);
	}

	public static RewriterStatement buildDAGFromHop(Hop hop, int maxDepth, final RuleContext ctx) {
		return buildDAGRecursively(hop, new HashMap<>(), 0, maxDepth, ctx);
	}

	public static void forAllUniqueTranslatableStatements(DMLProgram program, int maxDepth, Consumer<RewriterStatement> stmt, RewriterDatabase db, final RuleContext ctx) {
		Set<Hop> visited = new HashSet<>();

		for (String namespaceKey : program.getNamespaces().keySet()) {
			for (String fname : program.getFunctionStatementBlocks(namespaceKey).keySet()) {
				FunctionStatementBlock fsblock = program.getFunctionStatementBlock(namespaceKey, fname);
				handleStatementBlock(fsblock, maxDepth, stmt, visited, db, ctx);
			}
		}

		for (StatementBlock sb : program.getStatementBlocks()) {
			handleStatementBlock(sb, maxDepth, stmt, visited, db, ctx);
		}
	}

	private static void handleStatementBlock(StatementBlock sb, int maxDepth, Consumer<RewriterStatement> consumer, Set<Hop> visited, RewriterDatabase db, final RuleContext ctx) {
		if (sb instanceof FunctionStatementBlock)
		{
			FunctionStatementBlock fsb = (FunctionStatementBlock) sb;
			FunctionStatement fstmt = (FunctionStatement) fsb.getStatement(0);
			fstmt.getBody().forEach(s -> handleStatementBlock(s, maxDepth, consumer, visited, db, ctx));
		}
		else if (sb instanceof WhileStatementBlock)
		{
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement wstmt = (WhileStatement)wsb.getStatement(0);
			forAllUniqueTranslatableStatements(wsb.getPredicateHops(), maxDepth, consumer, visited, db, ctx);
			wstmt.getBody().forEach(s -> handleStatementBlock(s, maxDepth, consumer, visited, db, ctx));
		}
		else if (sb instanceof IfStatementBlock)
		{
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement istmt = (IfStatement)isb.getStatement(0);
			forAllUniqueTranslatableStatements(isb.getPredicateHops(), maxDepth, consumer, visited, db, ctx);
			istmt.getIfBody().forEach(s -> handleStatementBlock(s, maxDepth, consumer, visited, db, ctx));
			istmt.getElseBody().forEach(s -> handleStatementBlock(s, maxDepth, consumer, visited, db, ctx));
		}
		else if (sb instanceof ForStatementBlock)
		{
			ForStatementBlock fsb = (ForStatementBlock) sb;
			ForStatement fstmt = (ForStatement)fsb.getStatement(0);
			forAllUniqueTranslatableStatements(fsb.getFromHops(), maxDepth, consumer, visited, db, ctx);
			forAllUniqueTranslatableStatements(fsb.getToHops(), maxDepth, consumer, visited, db, ctx);
			forAllUniqueTranslatableStatements(fsb.getIncrementHops(), maxDepth, consumer, visited, db, ctx);
			fstmt.getBody().forEach(s -> handleStatementBlock(s, maxDepth, consumer, visited, db, ctx));
		}
		else
		{
			if (sb.getHops() != null)
				sb.getHops().forEach(hop -> forAllUniqueTranslatableStatements(hop, maxDepth, consumer, visited, db, ctx));
		}
	}

	private static void forAllUniqueTranslatableStatements(Hop currentHop, int maxDepth, Consumer<RewriterStatement> consumer, Set<Hop> visited, RewriterDatabase db, final RuleContext ctx) {
		if (currentHop == null || visited.contains(currentHop))
			return;

		visited.add(currentHop);
		RewriterStatement stmt = buildDAGRecursively(currentHop, new HashMap<>(), 0, maxDepth, ctx);

		if (stmt != null)
			stmt = ctx.metaPropagator.apply(stmt);

		if (stmt != null && db.insertEntry(ctx, stmt)) {
			RewriterStatement cpy = stmt.nestedCopyOrInject(new HashMap<>(), el -> null);
			consumer.accept(cpy);
		}

		if (currentHop.getInput() != null)
			currentHop.getInput().forEach(child -> forAllUniqueTranslatableStatements(child, maxDepth, consumer, visited, db, ctx));
	}

	private static RewriterStatement buildDAGRecursively(Hop next, Map<Hop, RewriterStatement> cache, int depth, int maxDepth, final RuleContext ctx) {
		if (depth == maxDepth)
			return buildLeaf(next, ctx);

		if (cache.containsKey(next))
			return checkForCorrectTypes(cache.get(next), next, ctx);

		if (next instanceof LiteralOp) {
			RewriterStatement literal = buildLiteral((LiteralOp)next, ctx);
			literal = checkForCorrectTypes(literal, next, ctx);
			cache.put(next, literal);
			return literal;
		}

		if (next instanceof AggBinaryOp) {
			RewriterStatement stmt = buildAggBinaryOp((AggBinaryOp) next, ctx);
			stmt = checkForCorrectTypes(stmt, next, ctx);

			if (stmt == null)
				return buildLeaf(next, ctx);

			if (buildInputs(stmt, next.getInput(), cache, true, depth, maxDepth, ctx))
				return stmt;

			return null;
		}

		if (next instanceof AggUnaryOp) {
			RewriterStatement stmt = buildAggUnaryOp((AggUnaryOp) next, ctx);
			stmt = checkForCorrectTypes(stmt, next, ctx);

			if (stmt == null)
				return buildLeaf(next, ctx);

			if (buildInputs(stmt, next.getInput(), cache, true, depth, maxDepth, ctx))
				return stmt;

			return null;
		}

		if (next instanceof BinaryOp) {
			RewriterStatement stmt = buildBinaryOp((BinaryOp) next, ctx);
			stmt = checkForCorrectTypes(stmt, next, ctx);

			if (stmt == null)
				return buildLeaf(next, ctx);

			if (buildInputs(stmt, next.getInput(), cache, true, depth, maxDepth, ctx))
				return stmt;

			return null;
		}

		if (next instanceof ReorgOp) {
			RewriterStatement stmt = buildReorgOp((ReorgOp) next, ctx);
			stmt = checkForCorrectTypes(stmt, next, ctx);

			if (stmt == null)
				return buildLeaf(next, ctx);

			if (buildInputs(stmt, next.getInput(), cache, true, depth, maxDepth, ctx))
				return stmt;

			return null;
		}

		if (next instanceof DataGenOp) {
			List<Hop> interestingHops = new ArrayList<>();
			RewriterStatement stmt = buildDataGenOp((DataGenOp)next, ctx, interestingHops);
			stmt = checkForCorrectTypes(stmt, next, ctx);

			if (stmt == null)
				return buildLeaf(next, ctx);

			if (buildInputs(stmt, interestingHops, cache, true, depth, maxDepth, ctx))
				return stmt;

			return null;
		}

		if (next instanceof DataOp) {
			DataOp dop = (DataOp) next;

			if (dop.isRead())
				return buildLeaf(next, ctx);
		}

		if (printUnknowns) {
			System.out.println("Unknown Op: " + next);
			System.out.println("Class: " + next.getClass().getSimpleName());
			System.out.println("OPString: " + next.getOpString());
		}

		return null;
	}

	private static RewriterStatement checkForCorrectTypes(RewriterStatement stmt, Hop hop, final RuleContext ctx) {
		if (stmt == null)
			return null;

		String actualType = resolveExactDataType(hop);

		if (actualType == null)
			return null;

		if (actualType.equals(stmt.getResultingDataType(ctx)))
			return stmt;

		if (actualType.equals("MATRIX")) {
			HashMap<String, RewriterStatement> oldTypes = new HashMap<>();
			oldTypes.put("A", stmt);
			RewriterStatement newStmt = RewriterUtils.parseExpression("as.matrix(A)", new HashMap<>(), oldTypes, ctx);
			return newStmt;
		}

		return null;
	}

	private static RewriterStatement buildLeaf(Hop hop, final RuleContext ctx) {
		switch (hop.getDataType()) {
			case SCALAR:
				return buildScalarLeaf(hop, ctx);
			case MATRIX:
				return RewriterUtils.parse(hop.getName(), ctx, "MATRIX:" + hop.getName());
		}

		return null; // Not supported then
	}

	private static RewriterStatement buildScalarLeaf(Hop hop, final RuleContext ctx) {
		switch (hop.getValueType()) {
			case FP64:
			case FP32:
				return RewriterUtils.parse(hop.getName(), ctx, "FLOAT:" + hop.getName());
			case INT64:
			case INT32:
				return RewriterUtils.parse(hop.getName(), ctx, "INT:" + hop.getName());
			case BOOLEAN:
				return RewriterUtils.parse(hop.getName(), ctx, "BOOL:" + hop.getName());
		}

		return null; // Not supported then
	}

	private static boolean buildInputs(RewriterStatement stmt, List<Hop> inputs, Map<Hop, RewriterStatement> cache, boolean fixedSize, int depth, int maxDepth, final RuleContext ctx) {
		List<RewriterStatement> children = new ArrayList<>();
		for (Hop in : inputs) {
			RewriterStatement childStmt = buildDAGRecursively(in, cache, depth + 1, maxDepth, ctx);

			if (childStmt == null) {
				//System.out.println("Could not build child: " + in);
				return false;
			}

			children.add(childStmt);
		}

		if (fixedSize && stmt.getOperands().size() != children.size())
			return false;

		stmt.getOperands().clear();
		stmt.getOperands().addAll(children);
		stmt.consolidate(ctx);
		return true;
	}

	private static RewriterStatement buildAggBinaryOp(AggBinaryOp op, final RuleContext ctx) {
		// Some placeholder definitions
		switch(op.getOpString()) {
			case "ba(+*)": // Matrix multiplication
				return RewriterUtils.parse("%*%(A, B)", ctx, matrixDefs, floatDefs, intDefs, boolDefs);
		}

		if (printUnknowns)
			System.out.println("Unknown AggBinaryOp: " + op.getOpString());
		return null;
	}

	private static RewriterStatement buildAggUnaryOp(AggUnaryOp op, final RuleContext ctx) {
		// Some placeholder definitions
		switch(op.getOpString()) {
			case "ua(+C)": // Matrix multiplication
				return RewriterUtils.parse("colSums(A)", ctx, matrixDefs, floatDefs, intDefs, boolDefs);
			case "ua(+R)":
				return RewriterUtils.parse("rowSums(A)", ctx, matrixDefs, floatDefs, intDefs, boolDefs);
			case "ua(+RC)":
				return RewriterUtils.parse("sum(A)", ctx, matrixDefs, floatDefs, intDefs, boolDefs);
		}

		if (printUnknowns)
			System.out.println("Unknown AggUnaryOp: " + op.getOpString());
		return null;
	}

	private static RewriterStatement buildBinaryOp(BinaryOp op, final RuleContext ctx) {
		String t1 = resolveExactDataType(op.getInput().get(0));
		String t2 = resolveExactDataType(op.getInput().get(1));

		if (t1 == null || t2 == null)
			return null;

		t1 += ":a";
		t2 += ":b";

		switch(op.getOpString()) {
			case "b(+)": // Addition
				System.out.println("Sparsity: " + op.getSparsity());
				return RewriterUtils.parse("+(a, b)", ctx, t1, t2);
			case "b(*)": // Matrix multiplication
				return RewriterUtils.parse("*(a, b)", ctx, t1, t2);
			case "b(-)":
				return RewriterUtils.parse("-(a, b)", ctx, t1, t2);
			case "b(/)":
				return RewriterUtils.parse("/(a, b)", ctx, t1, t2);
			case "b(||)":
				return RewriterUtils.parse("|(a, b)", ctx, t1, t2);
			case "b(!=)":
				return RewriterUtils.parse("!=(a, b)", ctx, t1, t2);
			case "b(==)":
				return RewriterUtils.parse("==(a, b)", ctx, t1, t2);
			case "b(&&)":
				return RewriterUtils.parse("&(a, b)", ctx, t1, t2);
			case "b(<)":
				return RewriterUtils.parse("<(a, b)", ctx, t1, t2);
			case "b(>)":
				// TODO: Add heuristic to transform > to <
				return RewriterUtils.parse(">(a, b)", ctx, t1, t2);
		}

		if (printUnknowns)
			System.out.println("Unknown BinaryOp: " + op.getOpString());
		return null;
	}

	private static String resolveExactDataType(Hop hop) {
		if (hop.getDataType() == Types.DataType.MATRIX)
			return "MATRIX";

		switch (hop.getValueType()) {
			case FP64:
			case FP32:
				return "FLOAT";
			case INT64:
			case INT32:
				return "INT";
			case BOOLEAN:
				return "BOOL";
		}

		if (printUnknowns)
			System.out.println("Unknown type: " + hop + " -> " + hop.getDataType() + " : " + hop.getValueType());

		return null;
	}

	private static RewriterStatement buildReorgOp(ReorgOp op, final RuleContext ctx) {
		switch(op.getOpString()) {
			case "r(r')": // Matrix multiplication
				return RewriterUtils.parse("t(A)", ctx, matrixDefs, floatDefs, intDefs, boolDefs);
		}

		//System.out.println("Unknown BinaryOp: " + op.getOpString());
		return null;
	}

	private static RewriterStatement buildDataGenOp(DataGenOp op, final RuleContext ctx, List<Hop> interestingHops) {
		System.out.println("Sparsity of " + op + ": " + op.getSparsity());
		// TODO:
		switch(op.getOpString()) {
			case "dg(rand)":
				interestingHops.add(op.getParam("rows"));
				interestingHops.add(op.getParam("cols"));
				interestingHops.add(op.getParam("min"));
				interestingHops.add(op.getParam("max"));
				return RewriterUtils.parse("rand(i1, i2, f1, f2)", ctx, matrixDefs, floatDefs, intDefs, boolDefs);
		}
		return null;
	}

	private static RewriterStatement buildLiteral(LiteralOp literal, final RuleContext ctx) {
		if (literal.getDataType() != Types.DataType.SCALAR)
			return null; // Then it is not supported yet

		switch (literal.getValueType()) {
			case FP64:
			case FP32:
				return new RewriterDataType().as(UUID.randomUUID().toString()).ofType("FLOAT").asLiteral(literal.getDoubleValue()).consolidate(ctx);
			case INT32:
			case INT64:
				return new RewriterDataType().as(UUID.randomUUID().toString()).ofType("INT").asLiteral(literal.getLongValue()).consolidate(ctx);
			case BOOLEAN:
				return new RewriterDataType().as(UUID.randomUUID().toString()).ofType("BOOL").asLiteral(literal.getBooleanValue()).consolidate(ctx);
			default:
				return null; // Not supported yet
		}
	}

	public static boolean executeScript(String script) {
		try {
			return DMLScript.executeScript(new String[]{"-s", script});
		} catch (Exception ex) {
			ex.printStackTrace();
			return false;
		}
	}
}
