package org.apache.sysds.hops.rewriter;

import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class RewriterContextSettings {

	public static final List<String> ALL_TYPES = List.of("FLOAT", "INT", "BOOL", "MATRIX");

	public static String getDefaultContextString() {
		StringBuilder builder = new StringBuilder();
		ALL_TYPES.forEach(t -> {
			builder.append("argList(" + t + ")::" + t + "...\n");
			builder.append("argList(" + t + "...)::" + t + "...\n");
		}); // This is a meta function that can take any number of arguments

		builder.append("IdxSelectPushableBinaryInstruction(MATRIX,MATRIX)::MATRIX\n");
		builder.append("impl +\n");
		//builder.append("impl -\n");
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

		builder.append("FullAggregationInstruction(MATRIX)::FLOAT\n");
		builder.append("impl FullAdditiveAggregationInstruction\n"); // TODO
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



		builder.append("FullAdditiveAggregationInstruction(MATRIX)::FLOAT\n");
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
		//builder.append("impl -\n");
		builder.append("impl *\n");
		builder.append("impl /\n");
		builder.append("impl min\n");
		builder.append("impl max\n");

		builder.append("SizeSwappingInstruction(MATRIX)::MATRIX\n");
		builder.append("impl t\n");

		builder.append("SizeInstruction(MATRIX)::INT\n");
		builder.append("impl nrow\n");
		builder.append("impl ncol\n");
		builder.append("impl length\n");

		RewriterUtils.buildBinaryPermutations(List.of("INT", "FLOAT", "BOOL"), (t1, t2) -> {
			builder.append("BinaryScalarInstruction(" + t1 + ","  + t2 + ")::" + RewriterUtils.defaultTypeHierarchy(t1, t2) + "\n");
			builder.append("impl ElementWiseInstruction\n");
		});

		RewriterUtils.buildBinaryPermutations(List.of("MATRIX...", "MATRIX", "INT", "FLOAT", "BOOL"), (t1, t2) -> {
			builder.append("ElementWiseInstruction(" + t1 + "," + t2 + ")::" + RewriterUtils.defaultTypeHierarchy(t1, t2) + "\n");
			builder.append("impl ElementWiseSumExpandableInstruction\n");
			builder.append("impl /\n");
			builder.append("impl max\n");
			builder.append("impl min\n");
		});

		builder.append("ElementWiseInstruction(MATRIX...)::MATRIX\n");
		builder.append("impl ElementWiseSumExpandableInstruction\n");
		builder.append("impl /\n");
		builder.append("impl max\n");
		builder.append("impl min\n");

		RewriterUtils.buildBinaryPermutations(List.of("MATRIX...", "MATRIX", "INT", "FLOAT", "BOOL"), (t1, t2) -> {
			builder.append("ElementWiseSumExpandableInstruction(" + t1 + "," + t2 + ")::" + RewriterUtils.defaultTypeHierarchy(t1, t2) + "\n"); // Any instruction that allows op(sum(A*), sum(B*)) = sum(op(A, B))
			builder.append("impl ElementWiseAdditiveInstruction\n");
			builder.append("impl *\n");

			builder.append("ElementWiseAdditiveInstruction(" + t1 + "," + t2 + ")::" + RewriterUtils.defaultTypeHierarchy(t1, t2) + "\n");
			builder.append("impl +\n");
			//builder.append("impl -\n");
		});

		builder.append("ElementWiseAdditiveInstruction(MATRIX...)::MATRIX\n");
		builder.append("impl +\n");
		//builder.append("impl -\n");


		ALL_TYPES.forEach(t -> {
			builder.append("UnaryOperator(" + t + ")::" + t + "\n");
			builder.append("impl -\n");
		});

		//

		builder.append("rowSelect(MATRIX,INT,INT)::MATRIX\n");
		builder.append("colSelect(MATRIX,INT,INT)::MATRIX\n");
		builder.append("min(INT,INT)::INT\n");
		builder.append("max(INT,INT)::INT\n");

		builder.append("index(MATRIX,INT,INT,INT,INT)::MATRIX\n");

		RewriterUtils.buildBinaryPermutations(List.of("MATRIX...", "MATRIX", "INT", "FLOAT", "BOOL"), (t1, t2) -> {
			builder.append("FusableBinaryOperator(" + t1 + "," + t2 + ")::" + RewriterUtils.defaultTypeHierarchy(t1, t2) + "\n");
			builder.append("impl +\n");
			//builder.append("impl -\n");
			builder.append("impl *\n");
			builder.append("impl %*%\n");
		});

		List.of("MATRIX", "INT", "FLOAT", "BOOL").forEach(t -> {
			builder.append("FusedOperator(" + t + "...)::" + t + "\n");
			builder.append("impl +\n");
			//builder.append("impl -\n");
			builder.append("impl *\n");
			builder.append("impl %*%\n");
		});

		builder.append("ncol(MATRIX)::INT\n");
		builder.append("nrow(MATRIX)::INT\n");
		builder.append("length(MATRIX)::INT\n");

		RewriterUtils.buildBinaryAlgebraInstructions(builder, "+", List.of("INT", "FLOAT", "BOOL", "MATRIX"));
		//RewriterUtils.buildBinaryAlgebraInstructions(builder, "-", List.of("INT", "FLOAT", "BOOL", "MATRIX"));
		RewriterUtils.buildBinaryAlgebraInstructions(builder, "*", List.of("INT", "FLOAT", "BOOL", "MATRIX"));
		ALL_TYPES.forEach(t -> builder.append("-(" + t + ")::" + t + "\n"));
		ALL_TYPES.forEach(t -> builder.append("inv(" + t + ")::" + t + "\n"));
		//RewriterUtils.buildBinaryAlgebraInstructions(builder, "/", List.of("INT", "FLOAT", "BOOL", "MATRIX"));

		builder.append("if(INT,MATRIX,MATRIX)::MATRIX\n");

		// Compile time functions
		builder.append("_compileTimeIsEqual(MATRIX,MATRIX)::INT\n");
		builder.append("_compileTimeIsEqual(INT,INT)::INT\n");
		builder.append("_compileTimeSelectLeastExpensive(MATRIX,MATRIX)::MATRIX\n"); // Selects the least expensive of the two matrices to obtain
		builder.append("_compileTimeSelectLeastExpensive(INT,INT)::INT\n");
		builder.append("_compileTimeSelectLeastExpensive(FLOAT,FLOAT)::FLOAT\n");


		// Custom implementation starts here
		builder.append("as.matrix(INT)::MATRIX\n");
		builder.append("as.matrix(FLOAT)::MATRIX\n");
		builder.append("as.matrix(BOOL)::MATRIX\n");
		builder.append("as.scalar(MATRIX)::FLOAT\n");

		builder.append("max(MATRIX)::FLOAT\n");
		builder.append("min(MATRIX)::FLOAT\n");

		builder.append("rand(INT,INT,FLOAT,FLOAT)::MATRIX\n"); // Args: min, max, rows, cols
		builder.append("rand(INT,INT)::FLOAT\n"); // Just to make it possible to say that random is dependent on both matrix indices
		builder.append("matrix(INT,INT,INT)::MATRIX\n");

		builder.append("trace(MATRIX)::FLOAT\n");

		// Boole algebra

		RewriterUtils.buildBinaryPermutations(List.of("MATRIX", "FLOAT", "INT", "BOOL"), (t1, t2) -> {
			String ret = t1.equals("MATRIX") ^ t2.equals("MATRIX") ? "MATRIX" : "BOOL";
			builder.append("==(" + t1 + "," + t2 + ")::" + ret + "\n");
			builder.append("!=(" + t1 + "," + t2 + ")::" + ret + "\n");
			builder.append("<(" + t1 + "," + t2 + ")::" + ret + "\n");
			builder.append("<=(" + t1 + "," + t2 + ")::" + ret + "\n");
			builder.append(">(" + t1 + "," + t2 + ")::" + ret + "\n");
			builder.append(">=(" + t1 + "," + t2 + ")::" + ret + "\n");
			builder.append("&(" + t1 + "," + t2 + ")::" + ret + "\n");
			builder.append("|(" + t1 + "," + t2 + ")::" + ret + "\n");
		});

		List.of("MATRIX", "FLOAT", "INT", "BOOL").forEach(t -> {
			builder.append("!(" + t + ")::" + (t.equals("MATRIX") ? "MATRIX" : "BOOL") + "\n");
		});

		// Expressions that will be rewritten to an equivalent expression
		RewriterUtils.buildBinaryPermutations(ALL_TYPES, (t1, t2) -> {
			builder.append("-(" + t1+ "," + t2 + ")::" + RewriterUtils.defaultTypeHierarchy(t1, t2) + "\n");
			builder.append("/(" + t1+ "," + t2 + ")::" + RewriterUtils.defaultTypeHierarchy(t1, t2) + "\n");
		});


		// Meta-Instruction
		builder.append("_lower(INT)::FLOAT\n");
		builder.append("_lower(FLOAT)::FLOAT\n");
		builder.append("_higher(INT)::FLOAT\n");
		builder.append("_higher(FLOAT)::FLOAT\n");
		builder.append("_posInt()::INT\n");

		builder.append("_rdFloat()::FLOAT\n");
		builder.append("_rdBool()::BOOL\n");
		builder.append("_anyBool()::BOOL\n");

		builder.append("_rdFLOAT()::FLOAT\n");
		builder.append("_rdBOOL()::BOOL\n");
		builder.append("_rdINT()::INT\n");
		builder.append("_rdMATRIX()::MATRIX\n");
		builder.append("_rdMATRIX(INT,INT)::MATRIX\n");

		List.of("INT", "FLOAT", "BOOL", "MATRIX").forEach(t -> builder.append("_asVar(" + t + ")::" + t + "\n"));

		builder.append("[](MATRIX,INT,INT)::FLOAT\n");
		builder.append("[](MATRIX,INT,INT,INT,INT)::MATRIX\n");
		builder.append("diag(MATRIX)::MATRIX\n");
		builder.append("sum(FLOAT...)::FLOAT\n");
		builder.append("sum(FLOAT*)::FLOAT\n");
		builder.append("sum(FLOAT)::FLOAT\n");

		builder.append("_m(INT,INT,FLOAT)::MATRIX\n");
		builder.append("_idxExpr(INT,FLOAT)::FLOAT*\n");
		builder.append("_idxExpr(INT,FLOAT*)::FLOAT*\n");
		builder.append("_idxExpr(INT...,FLOAT)::FLOAT*\n");
		builder.append("_idxExpr(INT...,FLOAT*)::FLOAT*\n");
		//builder.append("_idxExpr(INT,FLOAT...)::FLOAT*\n");
		builder.append("_idx(INT,INT)::INT\n");
		builder.append("_nrow()::INT\n");
		builder.append("_ncol()::INT\n");

		ALL_TYPES.forEach(t -> builder.append("_map(FLOAT...," + t + ")::" + t + "\n"));
		ALL_TYPES.forEach(t -> builder.append("_reduce(FLOAT...," + t + ")::" + t + "\n"));
		builder.append("_v(MATRIX)::FLOAT\n");
		builder.append("_cur()::FLOAT\n");

		/*builder.append("_map(INT,INT,FLOAT)::MATRIX\n");
		builder.append("_matIdx(MATRIX)::IDX[MATRIX]\n");
		builder.append("_nextRowIdx(MATRIX)::INT\n");
		builder.append("_nextColIdx(MATRIX)::INT\n");
		builder.append("_next(IDX[MATRIX])::FLOAT\n");

		builder.append("_get(MATRIX,INT,INT)::FLOAT\n");*/

		return builder.toString();
	}
	public static RuleContext getDefaultContext(Random rd) {
		String ctxString = getDefaultContextString();

		RuleContext ctx = RuleContext.createContext(ctxString);

		/*ctx.customStringRepr.put("_idx(INT,INT)", (stmt, mctx) -> {
			return stmt.trueInstruction() + "(" + String.join(", ", stmt.getOperands().stream().map(el -> el.toString(mctx)).collect(Collectors.toList())) + ") [" + stmt.getMeta("idxId") + "]";
		});

		ctx.customStringRepr.put("_m(INT,INT,FLOAT)", (stmt, mctx) -> {
			return stmt.trueInstruction() + "["  + stmt.getMeta("ownerId") + "](" + String.join(", ", stmt.getOperands().stream().map(el -> el.toString(mctx)).collect(Collectors.toList())) + ")";
		});*/

		// Meta instruction resolver
		ctx.customStringRepr.put("_lower(INT)", (stmt, mctx) -> {
			double mrd = 1F - rd.nextDouble();
			if (stmt.getMeta("MetaInstrRdFloatValue") != null)
				mrd = (double)stmt.getMeta("MetaInstrRdFloatValue");
			else
				stmt.unsafePutMeta("MetaInstrRdFloatValue", mrd);
			if (stmt.getOperands().get(0).isLiteral())
				return "(" + (((long) stmt.getOperands().get(0).getLiteral()) - mrd) + ")";
			else
				return stmt.getOperands().get(0).toString(ctx) + " - " + mrd;
		});
		ctx.customStringRepr.put("_lower(FLOAT)", (stmt, mctx) -> {
			double mrd = 1F - rd.nextDouble();
			if (stmt.getMeta("MetaInstrRdFloatValue") != null)
				mrd = (double)stmt.getMeta("MetaInstrRdFloatValue");
			else
				stmt.unsafePutMeta("MetaInstrRdFloatValue", mrd);
			if (stmt.getOperands().get(0).isLiteral())
				return "(" + (((double) stmt.getOperands().get(0).getLiteral()) - mrd) + ")";
			else
				return stmt.getOperands().get(0).toString(ctx) + " - " + mrd;
		});
		ctx.customStringRepr.put("_higher(INT)", (stmt, mctx) -> {
			double mrd = rd.nextDouble();
			if (stmt.getMeta("MetaInstrRdFloatValue") != null)
				mrd = (double)stmt.getMeta("MetaInstrRdFloatValue");
			else
				stmt.unsafePutMeta("MetaInstrRdFloatValue", mrd);
			if (stmt.getOperands().get(0).isLiteral())
				return "(" + (((long) stmt.getOperands().get(0).getLiteral()) + mrd) + ")";
			else
				return stmt.getOperands().get(0).toString(ctx) + " + " + mrd;
		});
		ctx.customStringRepr.put("_higher(FLOAT)", (stmt, mctx) -> {
			double mrd = rd.nextDouble();
			if (stmt.getMeta("MetaInstrRdFloatValue") != null)
				mrd = (double)stmt.getMeta("MetaInstrRdFloatValue");
			else
				stmt.unsafePutMeta("MetaInstrRdFloatValue", mrd);
			if (stmt.getOperands().get(0).isLiteral())
				return "(" + (((double) stmt.getOperands().get(0).getLiteral()) + mrd) + ")";
			else
				return stmt.getOperands().get(0).toString(ctx) + " + " + mrd;
		});

		ctx.customStringRepr.put("_posInt()", (stmt, mctx) -> {
			long i = 1 + rd.nextInt(100);
			if (stmt.getMeta("MetaInstrRdIntValue") != null)
				i = (long)stmt.getMeta("MetaInstrRdIntValue");
			else
				stmt.unsafePutMeta("MetaInstrRdIntValue", i);
			return String.valueOf(i);
		});

		ctx.customStringRepr.put("_rdFloat()", (stmt, mctx) -> {
			double f = (rd.nextDouble() - 0.5f) * (rd.nextInt(100000) + 1);
			if (stmt.getMeta("MetaInstrRdFloatValue") != null)
				f = (double)stmt.getMeta("MetaInstrRdFloatValue");
			else
				stmt.unsafePutMeta("MetaInstrRdFloatValue", f);
			return String.valueOf(f);
		});

		ctx.customStringRepr.put("_rdBool()", (stmt, mctx) -> {
			/*boolean b = rd.nextBoolean();
			if (stmt.getMeta("MetaInstrRdBoolValue") != null)
				b = (boolean)stmt.getMeta("MetaInstrRdBoolValue");
			else
				stmt.unsafePutMeta("MetaInstrRdBoolValue", b);
			return String.valueOf(b).toUpperCase();*/
			return "as.scalar(rand() < 0.5)";
		});

		ctx.customStringRepr.put("_rdFLOAT()", ctx.customStringRepr.get("_rdFloat()"));
		ctx.customStringRepr.put("_rdBOOL()", ctx.customStringRepr.get("_rdBool()"));

		ctx.customStringRepr.put("_rdMATRIX()", (stmt, mctx) -> "rand(cols=100, rows=100, min=-1000, max=1000)");
		ctx.customStringRepr.put("_rdMATRIX(INT,INT)", (stmt, mctx) -> "rand(rows=" + stmt.getOperands().get(0).toString(mctx) + ", cols=" + stmt.getOperands().get(1).toString(mctx) + ", min=-1000, max=1000)");

		ctx.customStringRepr.put("_rdINT()", (stmt, mctx) -> "as.scalar(floor(rand(min=-1000, max=1000)))");

		// TODO: This should later also be able to inject references to existing bool values
		ctx.customStringRepr.put("_anyBool()", ctx.customStringRepr.get("_rdBool()"));

		ALL_TYPES.forEach(t -> ctx.customStringRepr.put("_asVar(" + t + ")", (stmt, mctx) -> ((RewriterInstruction)stmt).getOperands().get(0).toString(ctx)));

		ctx.customStringRepr.put("rand(INT,INT,FLOAT,FLOAT)", (stmt, mctx) -> {
			List<RewriterStatement> ops = stmt.getOperands();
			return "rand(rows=(" + ops.get(0) + "), cols=(" + ops.get(1) + "), min=(" + ops.get(2) + "), max=(" + ops.get(3) + "))";
		});
		ctx.customStringRepr.put("rand(INT,INT,INT,INT)", ctx.customStringRepr.get("rand(INT,INT,FLOAT,FLOAT)"));
		ctx.customStringRepr.put("rand(INT,INT,FLOAT,INT)", ctx.customStringRepr.get("rand(INT,INT,FLOAT,FLOAT)"));
		ctx.customStringRepr.put("rand(INT,INT,INT,FLOAT)", ctx.customStringRepr.get("rand(INT,INT,FLOAT,FLOAT)"));

		RewriterUtils.putAsDefaultBinaryPrintable(List.of("<", "<=", ">", ">=", "==", "!=", "&", "|"), List.of("INT", "FLOAT", "BOOL", "MATRIX"), ctx.customStringRepr);

		/*RewriterUtils.putAsBinaryPrintable("<", List.of("INT", "FLOAT", "BOOL", "MATRIX"), ctx.customStringRepr, RewriterUtils.binaryStringRepr(" < "));
		RewriterUtils.putAsBinaryPrintable("<=", List.of("INT", "FLOAT", "BOOL", "MATRIX"), ctx.customStringRepr, RewriterUtils.binaryStringRepr(" <= "));
		RewriterUtils.putAsBinaryPrintable(">", List.of("INT", "FLOAT", "BOOL", "MATRIX"), ctx.customStringRepr, RewriterUtils.binaryStringRepr(" > "));
		RewriterUtils.putAsBinaryPrintable(">=", List.of("INT", "FLOAT", "BOOL", "MATRIX"), ctx.customStringRepr, RewriterUtils.binaryStringRepr(" >= "));
		RewriterUtils.putAsBinaryPrintable("==", List.of("INT", "FLOAT", "MATRIX", "BOOL"), ctx.customStringRepr, RewriterUtils.binaryStringRepr(" == "));
		RewriterUtils.putAsBinaryPrintable("!=", List.of("INT", "FLOAT", "MATRIX", "BOOL"), ctx.customStringRepr, RewriterUtils.binaryStringRepr(" != "));*/

		RewriterUtils.putAsBinaryPrintable("*", List.of("INT", "FLOAT", "MATRIX", "BOOL"), ctx.customStringRepr, RewriterUtils.binaryStringRepr(" * "));
		//RewriterUtils.putAsBinaryPrintable("/", List.of("INT", "FLOAT", "MATRIX", "BOOL"), ctx.customStringRepr, RewriterUtils.binaryStringRepr(" / "));
		//RewriterUtils.putAsBinaryPrintable("-", List.of("INT", "FLOAT", "MATRIX", "BOOL"), ctx.customStringRepr, RewriterUtils.binaryStringRepr(" - "));
		RewriterUtils.putAsBinaryPrintable("+", List.of("INT", "FLOAT", "MATRIX", "BOOL"), ctx.customStringRepr, RewriterUtils.binaryStringRepr(" + "));

		ctx.customStringRepr.put("%*%(MATRIX,MATRIX)", RewriterUtils.binaryStringRepr(" %*% "));
		//ctx.customStringRepr.put("<=(INT,INT)", RewriterUtils.binaryStringRepr(" <= "));
		//ctx.customStringRepr.put("==(INT,INT)", RewriterUtils.binaryStringRepr(" == "));
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
		/*ctx.customStringRepr.put("argList(MATRIX)", (stmt, ctx2) -> {
			RewriterInstruction mInstr = (RewriterInstruction) stmt;
			String out = mInstr.getOperands().get(0).toString(ctx2);

			for (int i = 1; i < mInstr.getOperands().size(); i++)
				out += ", " + mInstr.getOperands().get(i).toString(ctx2);

			return out;
		});*/
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

		return ctx;
	}
}
