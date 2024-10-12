package org.apache.sysds.hops.rewriter;

import org.apache.commons.collections4.bidimap.DualHashBidiMap;
import org.apache.commons.lang3.mutable.MutableBoolean;
import org.apache.commons.lang3.mutable.MutableObject;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;

public class RewriterMain2 {

	public static void main(String[] args) {
		DualHashBidiMap<String, String> equivalentRowColAggregations = new DualHashBidiMap<>() {
			{
				put("rowSums(MATRIX)", "colSums(MATRIX)");
				put("rowMeans(MATRIX)", "colMeans(MATRIX)");
				put("rowVars(MATRIX)", "colVars(MATRIX)");
			}
		};

		StringBuilder builder = new StringBuilder();

		builder.append("dtype ANY\n");
		builder.append("dtype COLLECTION::ANY\n");
		builder.append("dtype NUMERIC::ANY\n");
		builder.append("dtype INT::NUMERIC\n");
		builder.append("dtype FLOAT::NUMERIC\n");
		builder.append("dtype MATRIX::COLLECTION\n");

		builder.append("argList(MATRIX)::MATRIX...\n"); // This is a meta function that can take any number of MATRIX arguments

		builder.append("IdxSelectPushableBinaryInstruction(MATRIX,MATRIX)::MATRIX\n");
		builder.append("impl +\n");
		builder.append("impl -\n");
		builder.append("impl *\n");
		builder.append("impl /\n");
		builder.append("impl min\n");
		builder.append("impl max\n");

		builder.append("RowSelectPushableBinaryInstruction(MATRIX,MATRIX)::MATRIX\n");
		builder.append("impl IdxSelectPushableBinaryInstruction\n");
		builder.append("impl CBind\n");

		builder.append("ColSelectPushableBinaryInstruction(MATRIX,MATRIX)::MATRIX\n");
		builder.append("impl IdxSelectPushableBinaryInstruction\n");
		builder.append("impl RBind\n");

		builder.append("IdxSelectMMPushableBinaryInstruction(MATRIX,MATRIX)::MATRIX\n");
		builder.append("impl %*%\n");

		builder.append("RowSelectMMPushableBinaryInstruction(MATRIX,MATRIX)::MATRIX\n");
		builder.append("impl IdxSelectMMPushableBinaryInstruction\n");

		builder.append("ColSelectMMPushableBinaryInstruction(MATRIX,MATRIX)::MATRIX\n");
		builder.append("impl IdxSelectMMPushableBinaryInstruction\n");

		builder.append("IdxSelectTransposePushableBinaryInstruction(MATRIX,MATRIX)::MATRIX\n");
		builder.append("impl t\n");

		builder.append("RowSelectTransposePushableBinaryInstruction(MATRIX,MATRIX)::MATRIX\n");
		builder.append("impl IdxSelectTransposePushableBinaryInstruction\n");

		builder.append("ColSelectTransposePushableBinaryInstruction(MATRIX,MATRIX)::MATRIX\n");
		builder.append("impl IdxSelectTransposePushableBinaryInstruction\n");

		// Aggregation functions

		builder.append("FullAggregationInstruction(MATRIX)::MATRIX\n");
		builder.append("impl FullAdditiveAggregationInstruction(MATRIX)\n");
		builder.append("impl mean\n");
		builder.append("impl var\n");

		builder.append("RowAggregationInstruction(MATRIX)::MATRIX\n"); // Assumes that rowAggregation of a row vector is itself
		builder.append("impl RowAdditiveAggregationInstruction\n");
		builder.append("impl rowMeans\n");
		builder.append("impl rowVars\n");

		builder.append("ColAggregationInstruction(MATRIX)::MATRIX\n"); // Assumes that colAggregation of a column vector is itself
		builder.append("impl ColAdditiveAggregationInstruction\n");
		builder.append("impl colMeans\n");
		builder.append("impl colVars\n");



		builder.append("FullAdditiveAggregationInstruction(MATRIX)::MATRIX\n");
		builder.append("impl sum\n");

		builder.append("RowAdditiveAggregationInstruction(MATRIX)::MATRIX\n");
		builder.append("impl rowSums\n");

		builder.append("ColAdditiveAggregationInstruction(MATRIX)::MATRIX\n");
		builder.append("impl colSums\n");



		// Function aggregation properties

		builder.append("FullAdditiveAggregationPushableInstruction(MATRIX,MATRIX)::MATRIX\n");
		builder.append("impl ElementWiseAdditiveInstruction\n");

		builder.append("RowAdditiveAggregationPushableInstruction(MATRIX,MATRIX)::MATRIX\n");
		builder.append("impl ElementWiseAdditiveInstruction\n");

		builder.append("ColAdditiveAggregationPushableInstruction(MATRIX,MATRIX)::MATRIX\n");
		builder.append("impl ElementWiseAdditiveInstruction\n");


		// Permutation functions

		builder.append("Rearrangement(MATRIX)::MATRIX\n"); // An operation that keeps all elements but can reformat the matrix
		builder.append("impl Permutation\n");
		builder.append("impl t\n"); // Transposition

		builder.append("RowPermutation(MATRIX)::MATRIX\n");

		builder.append("ColPermutation(MATRIX)::MATRIX\n");

		builder.append("Permutation(MATRIX)::MATRIX\n");
		builder.append("impl RowPermutation\n");
		builder.append("impl ColPermutation\n");
		//builder.append("impl t\n"); // Transpose matrix



		// Matrix extending operations

		builder.append("CBind(MATRIX,MATRIX)::MATRIX\n");
		builder.append("RBind(MATRIX,MATRIX)::MATRIX\n");


		// Meta preserving instructions

		builder.append("SizePreservingInstruction(MATRIX,MATRIX)::MATRIX\n"); // Maintains the size information of the matrix
		builder.append("impl +\n");
		builder.append("impl -\n");
		builder.append("impl *\n");
		builder.append("impl /\n");
		builder.append("impl min\n");
		builder.append("impl max\n");

		builder.append("SizeSwappingInstruction(MATRIX)::MATRIX\n");
		builder.append("impl t\n");

		builder.append("SizeInstruction(MATRIX)::INT\n");
		builder.append("impl nrows\n");
		builder.append("impl ncols\n");



		// Element-wise instruction
		builder.append("ElementWiseInstruction(MATRIX,MATRIX)::MATRIX\n");
		builder.append("impl ElementWiseAdditiveInstruction\n");
		builder.append("impl *\n");
		builder.append("impl /\n");
		builder.append("impl max\n");
		builder.append("impl min\n");

		//
		builder.append("ElementWiseAdditiveInstruction(MATRIX,MATRIX)::MATRIX\n");
		builder.append("impl +\n");
		builder.append("impl -\n");

		//

		builder.append("rowSelect(MATRIX,INT,INT)::MATRIX\n");
		builder.append("colSelect(MATRIX,INT,INT)::MATRIX\n");
		builder.append("min(INT,INT)::INT\n");
		builder.append("max(INT,INT)::INT\n");

		builder.append("index(MATRIX,INT,INT,INT,INT)::MATRIX\n");

		builder.append("FusableBinaryOperator(MATRIX,MATRIX)::MATRIX\n");
		builder.append("impl +\n");
		builder.append("impl -\n");
		builder.append("impl *\n");
		builder.append("impl %*%\n");

		builder.append("FusedOperator(MATRIX...)::MATRIX\n");
		builder.append("impl +\n");
		builder.append("impl -\n");
		builder.append("impl *\n");
		builder.append("impl %*%\n");

		builder.append("ncols(MATRIX)::INT\n");
		builder.append("nrows(MATRIX)::INT\n");

		RewriterUtils.buildBinaryAlgebraInstructions(builder, "+", List.of("INT", "FLOAT", "MATRIX"));
		RewriterUtils.buildBinaryAlgebraInstructions(builder, "-", List.of("INT", "FLOAT", "MATRIX"));
		RewriterUtils.buildBinaryAlgebraInstructions(builder, "*", List.of("INT", "FLOAT", "MATRIX"));
		RewriterUtils.buildBinaryAlgebraInstructions(builder, "/", List.of("INT", "FLOAT", "MATRIX"));

		/*builder.append("-(INT,INT)::INT\n");
		builder.append("+(INT,INT)::INT\n");
		builder.append("*(INT,INT)::INT\n");
		builder.append("/(INT,INT)::INT\n");

		builder.append("-(FLOAT,FLOAT)::FLOAT\n");
		builder.append("+(FLOAT,FLOAT)::FLOAT\n");
		builder.append("*(FLOAT,FLOAT)::FLOAT\n");
		builder.append("/(FLOAT,FLOAT)::FLOAT\n");

		builder.append("-(INT,FLOAT)::FLOAT\n");
		builder.append("+(INT,FLOAT)::FLOAT\n");
		builder.append("*(INT,FLOAT)::FLOAT\n");
		builder.append("/(INT,FLOAT)::FLOAT\n");

		builder.append("-(FLOAT,INT)::FLOAT\n");
		builder.append("+(FLOAT,INT)::FLOAT\n");
		builder.append("*(FLOAT,INT)::FLOAT\n");
		builder.append("/(FLOAT,INT)::FLOAT\n");

		builder.append("/(MATRIX,INT)::FLOAT\n");*/

		// Some bool algebra
		builder.append("<=(INT,INT)::INT\n");
		builder.append("==(INT,INT)::INT\n");
		builder.append("&&(INT,INT)::INT\n");

		builder.append("if(INT,MATRIX,MATRIX)::MATRIX\n");

		// Some others
		builder.append("asMatrix(INT)::MATRIX\n");
		builder.append("asMatrix(FLOAT)::MATRIX\n");

		// Compile time functions
		builder.append("_compileTimeIsEqual(MATRIX,MATRIX)::INT\n");
		builder.append("_compileTimeIsEqual(INT,INT)::INT\n");
		builder.append("_compileTimeSelectLeastExpensive(MATRIX,MATRIX)::MATRIX\n"); // Selects the least expensive of the two matrices to obtain
		builder.append("_compileTimeSelectLeastExpensive(INT,INT)::INT\n");
		builder.append("_compileTimeSelectLeastExpensive(FLOAT,FLOAT)::FLOAT\n");

		RuleContext ctx = RuleContext.createContext(builder.toString());
		ctx.customStringRepr.put("+(INT,INT)", RewriterUtils.binaryStringRepr(" + "));
		ctx.customStringRepr.put("+(FLOAT,FLOAT)", RewriterUtils.binaryStringRepr(" + "));
		ctx.customStringRepr.put("+(INT,FLOAT)", RewriterUtils.binaryStringRepr(" + "));
		ctx.customStringRepr.put("+(FLOAT,INT)", RewriterUtils.binaryStringRepr(" + "));
		ctx.customStringRepr.put("-(INT,INT)", RewriterUtils.binaryStringRepr(" - "));
		ctx.customStringRepr.put("-(FLOAT,INT)", RewriterUtils.binaryStringRepr(" - "));
		ctx.customStringRepr.put("-(INT,FLOAT)", RewriterUtils.binaryStringRepr(" - "));
		ctx.customStringRepr.put("-(FLOAT,FLOAT)", RewriterUtils.binaryStringRepr(" - "));
		ctx.customStringRepr.put("/(INT,INT)", RewriterUtils.binaryStringRepr(" / "));
		ctx.customStringRepr.put("/(FLOAT,FLOAT)", RewriterUtils.binaryStringRepr(" / "));
		ctx.customStringRepr.put("/(INT,FLOAT)", RewriterUtils.binaryStringRepr(" / "));
		ctx.customStringRepr.put("/(FLOAT,INT)", RewriterUtils.binaryStringRepr(" / "));
		ctx.customStringRepr.put("/(MATRIX,INT)", RewriterUtils.binaryStringRepr(" / "));
		ctx.customStringRepr.put("*(INT,INT)", RewriterUtils.binaryStringRepr(" * "));
		ctx.customStringRepr.put("*(FLOAT,INT)", RewriterUtils.binaryStringRepr(" * "));
		ctx.customStringRepr.put("*(INT,FLOAT)", RewriterUtils.binaryStringRepr(" * "));
		ctx.customStringRepr.put("*(FLOAT,FLOAT)", RewriterUtils.binaryStringRepr(" * "));


		ctx.customStringRepr.put("+(MATRIX,MATRIX)", RewriterUtils.binaryStringRepr(" + "));
		ctx.customStringRepr.put("-(MATRIX,MATRIX)", RewriterUtils.binaryStringRepr(" - "));
		ctx.customStringRepr.put("*(MATRIX,MATRIX)", RewriterUtils.binaryStringRepr(" * "));
		ctx.customStringRepr.put("/(MATRIX,MATRIX)", RewriterUtils.binaryStringRepr(" / "));
		ctx.customStringRepr.put("%*%(MATRIX,MATRIX)", RewriterUtils.binaryStringRepr(" %*% "));
		ctx.customStringRepr.put("<=(INT,INT)", RewriterUtils.binaryStringRepr(" <= "));
		ctx.customStringRepr.put("==(INT,INT)", RewriterUtils.binaryStringRepr(" == "));
		ctx.customStringRepr.put("&&(INT,INT)", RewriterUtils.binaryStringRepr(" && "));
		ctx.customStringRepr.put("index(MATRIX,INT,INT,INT,INT)", (stmt, ctx2) -> {
			String out;
			RewriterInstruction mInstr = (RewriterInstruction) stmt;
			List<RewriterStatement> ops = mInstr.getOperands();
			RewriterStatement op1 = ops.get(0);

			if (op1 instanceof RewriterDataType)
				out = op1.toString(ctx2);
			else
				out = "(" + op1.toString(ctx2) + ")";

			out += "[" + ops.get(1).toString(ctx2) + " : " + ops.get(2).toString(ctx2) + ", " + ops.get(3).toString(ctx2) + " : " + ops.get(4).toString(ctx2) + "]";
			return out;
		});
		ctx.customStringRepr.put("argList(MATRIX)", (stmt, ctx2) -> {
			RewriterInstruction mInstr = (RewriterInstruction) stmt;
			String out = mInstr.getOperands().get(0).toString(ctx2);

			for (int i = 1; i < mInstr.getOperands().size(); i++)
				out += ", " + mInstr.getOperands().get(i).toString(ctx2);

			return out;
		});
		ctx.customStringRepr.put("if(INT,MATRIX,MATRIX)", (stmt, ctx2) -> {
			RewriterInstruction mInstr = (RewriterInstruction) stmt;
			StringBuilder sb = new StringBuilder();
			sb.append("if (");
			sb.append(mInstr.getOperands().get(0));
			sb.append(")\n");
			sb.append("{\n");
			sb.append(mInstr.getOperands().get(1));
			sb.append("\n}\nelse\n{\n");
			sb.append(mInstr.getOperands().get(2));
			sb.append("\n}");
			return sb.toString();
		});

		/*HashMap<Integer, RewriterStatement> mHooks = new HashMap<>();
		RewriterRule rule = new RewriterRuleBuilder(ctx)
				.parseGlobalVars("MATRIX:A")
				.parseGlobalVars("INT:h,i,j,k")
				.withParsedStatement("index(A,h,i,j,k)", mHooks)
				.toParsedStatement("rowSelect(colSelect(A,j,k),h,i)", mHooks)
				.build();
		System.out.println(rule);
		if (true)
			return;

		HashMap<String, RewriterStatement> vars = new HashMap<>();
		HashMap<Integer, RewriterStatement> hooks = new HashMap<>();
		RewriterUtils.parseDataTypes("INT:test", vars, ctx);

		RewriterStatement stmt = RewriterUtils.parseExpression(new MutableObject<>("$2:+(test,$1:test2())"), hooks, vars, ctx);
		System.out.println(hooks);
		System.out.println(stmt.toString(ctx));*/

		System.out.println(ctx.instrTypes);
		System.out.println(ctx.instrProperties);
		System.out.println(RewriterUtils.mapToImplementedFunctions(ctx));

		//RewriterRuleSet ruleSet = RewriterRuleSet.selectionPushdown;

		// Assumptions: Semantic correctness (e.g. CBind(A, B), A, B already have the same number of rows)

		// TODO: Adapt matcher such that for instance RBind(A, A) matches RBind(A, B); BUT: Not the other way round

		RewriterHeuristic selectionBreakup = new RewriterHeuristic(RewriterRuleSet.buildSelectionBreakup(ctx));

		RewriterHeuristic selectionPushdown = new RewriterHeuristic(RewriterRuleSet.buildSelectionPushdownRuleSet(ctx));
		RewriterHeuristic rbindcbindPushdown = new RewriterHeuristic(RewriterRuleSet.buildRbindCbindSelectionPushdown(ctx));

		RewriterHeuristic rbindElimination = new RewriterHeuristic(RewriterRuleSet.buildRBindElimination(ctx));

		RewriterHeuristic prepareCBindElimination = new RewriterHeuristic(RewriterRuleSet.buildReorderColRowSelect("colSelect", "rowSelect", ctx));
		RewriterHeuristic cbindElimination = new RewriterHeuristic(RewriterRuleSet.buildCBindElimination(ctx));

		RewriterHeuristic prepareSelectionSimplification = new RewriterHeuristic(RewriterRuleSet.buildReorderColRowSelect("rowSelect", "colSelect", ctx));
		RewriterHeuristic selectionSimplification = new RewriterHeuristic(RewriterRuleSet.buildSelectionSimplification(ctx));

		// TODO: Eliminate e.g. colSums(colSums(A))

		RewriterHeuristic aggregationPushdown = new RewriterHeuristic(RewriterRuleSet.buildAggregationPushdown(ctx, equivalentRowColAggregations));

		// TODO: This is still narrow and experimental
		RewriterHeuristic elementWiseInstructionPushdown = new RewriterHeuristic(RewriterRuleSet.buildElementWiseInstructionPushdown(ctx));

		RewriterHeuristic transposeElimination = new RewriterHeuristic(RewriterRuleSet.buildTransposeElimination(ctx));

		RewriterHeuristic metaInstructionSimplification = new RewriterHeuristic(RewriterRuleSet.buildMetaInstructionSimplification(ctx));

		RewriterHeuristic compileTimeFolding = new RewriterHeuristic(RewriterRuleSet.buildCompileTimeFolding(ctx));

		RewriterHeuristic operatorFusion = new RewriterHeuristic(RewriterRuleSet.buildDynamicOpInstructions(ctx));

		final HashMap<String, Set<String>> mset = RewriterUtils.mapToImplementedFunctions(ctx);

		RewriterHeuristics heur = new RewriterHeuristics();
		heur.add("UNFOLD AGGREGATIONS", new RewriterHeuristic(RewriterRuleSet.buildUnfoldAggregations(ctx)));
		heur.add("SELECTION BREAKUP", selectionBreakup);
		heur.add("SELECTION PUSHDOWN", selectionPushdown);
		heur.add("RBINDCBIND", rbindcbindPushdown);
		heur.add("PREPARE SELECTION SIMPLIFICATION", prepareSelectionSimplification);
		heur.add("SELECTION SIMPLIFICATION", selectionSimplification);
		heur.add("AGGREGATION PUSHDOWN", aggregationPushdown);
		heur.add("ELEMENT-WISE INSTRUCTION PUSHDOWN", elementWiseInstructionPushdown);
		heur.add("TRANSPOSITION ELIMINATION", transposeElimination);
		heur.add("META-INSTRUCTION SIMPLIFICATION", metaInstructionSimplification);
		heur.add("COMPILE-TIME FOLDING", compileTimeFolding);
		heur.add("AGGREGATION FOLDING", new RewriterHeuristic(RewriterRuleSet.buildAggregationFolding(ctx)));
		/*heur.add("OPERATOR FUSION", operatorFusion);
		heur.add("OPERATOR MERGE", (stmt, func, bool) -> {
			if (stmt instanceof RewriterInstruction)
				RewriterUtils.mergeArgLists((RewriterInstruction) stmt, ctx);
			func.apply(stmt);
			return stmt;
		});*/

		//System.out.println(heur);

		heur.forEachRuleSet(rs -> {
			rs.forEachRule((rule, mctx) -> {
				rule.createNonGenericRules(mset).forEach(r -> System.out.println(r));
			});
		}, true);

		String matrixDef = "MATRIX:A,B,C";
		String intDef = "INT:q,r,s,t,i,j,k,l";
		//String expr = "colSelect(CBind(index(A, q, r, s, t), B), a, b)";
		//String expr = "RBind(CBind(index(A,q,r,s,t), index(A,i,j,k,l)), A)";
		//String expr = "colSelect(RBind(index(CBind(colSums(-(t(rowSums(t(+(A,B)))), t(C))), rowSelect(C, q, r)), q, r, s, t), rowSelect(B, k, l)), i, j)";
		//String expr = "mean(RowPermutation(A))";
		//String expr = "rowSums(+(A,B))";
		//String expr = "t(%*%(colSums(t(+(rowSums(A), rowSums(C)))), t(B)))";
		//String expr = "colSums(+(colSums(A), colSums(B)))";
		//String expr = "colSums(+(colMeans(A), colMeans(B)))";
		//String expr = "CBind(colSelect(A, q, r), colSelect(A, +(r, i), s))";
		//String expr = "nrows(rowSums(A))";
		//String expr = "argList(+(t(A), t(B)), -(t(B), t(C)))";
		//String expr = "mean(+(A, B)))";
		//String expr = "+(max(A, B), max(A, C))";
		String expr = "colSelect(%*%(A, B), i, j)";
		RewriterStatement instr = RewriterUtils.parse(expr, ctx, matrixDef, intDef);

		long millis = System.currentTimeMillis();

		heur.apply(instr, (current, r) -> {
			println(current);
			println("<<<");
			println();
			return true;
		});

		millis = System.currentTimeMillis() - millis;
		System.out.println("Finished in " + millis + "ms");
	}

	public static boolean doPrint = true;

	public static void println() {
		if (doPrint)
			System.out.println();
	}

	public static void println(Object o) {
		if (doPrint)
			System.out.println(o);
	}
}
