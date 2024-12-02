package org.apache.sysds.hops.rewriter.estimators;

import org.apache.sysds.hops.rewriter.ConstantFoldingFunctions;
import org.apache.sysds.hops.rewriter.RewriterDataType;
import org.apache.sysds.hops.rewriter.RewriterInstruction;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.hops.rewriter.utils.StatementUtils;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

public class RewriterSparsityEstimator {

	/*public static RewriterStatement getCanonicalized(RewriterStatement instr, final RuleContext ctx) {
		RewriterStatement cpy = instr.copyNode();
		Map<RewriterStatement, RewriterStatement> mmap = new HashMap<>();

		for (int i = 0; i < cpy.getOperands().size(); i++) {
			RewriterStatement existing = mmap.get(cpy.getOperands().get(i));

			if (existing != null) {
				cpy.getOperands().set(i, existing);
			} else {
				RewriterStatement mDat = new RewriterDataType().as(UUID.randomUUID().toString()).ofType(cpy.getOperands().get(i).getResultingDataType(ctx)).consolidate(ctx);
			}
		}

		RewriterUtils.prepareForSparsityEstimation();
	}*/

	public static RewriterStatement rollupSparsities(RewriterStatement sparsityEstimate, Map<RewriterStatement, RewriterStatement> sparsityMap, final RuleContext ctx) {
		sparsityEstimate.forEachPreOrder(cur -> {
			for (int i = 0; i < cur.getOperands().size(); i++) {
				RewriterStatement child = cur.getChild(i);

				if (child.isInstruction() && child.trueInstruction().equals("_nnz")) {
					RewriterStatement subEstimate = sparsityMap.get(child.getChild(0));

					if (subEstimate != null) {
						cur.getOperands().set(i, subEstimate);
					}
				}
			}
			return true;
		}, false);

		return sparsityEstimate;
	}

	public static Map<RewriterStatement, RewriterStatement> estimateAllNNZ(RewriterStatement stmt, final RuleContext ctx) {
		Map<RewriterStatement, RewriterStatement> map = new HashMap<>();
		stmt.forEachPostOrder((cur, pred) -> {
			RewriterStatement estimation = estimateNNZ(cur, ctx);
			if (estimation != null)
				map.put(cur, estimation);
		}, false);

		return map;
	}

	public static RewriterStatement estimateNNZ(RewriterStatement stmt, final RuleContext ctx) {
		if (!stmt.isInstruction())
			return null;
		switch (stmt.trueInstruction()) {
			case "%*%":
				return new RewriterInstruction("*", ctx, StatementUtils.min(ctx, new RewriterInstruction("*", ctx, stmt.getNRow(), stmt.getNCol()), RewriterStatement.nnz(stmt.getChild(1), ctx)), new RewriterInstruction("*", ctx, stmt.getNRow(), stmt.getNCol()), RewriterStatement.nnz(stmt.getChild(0), ctx));
		}

		switch (stmt.trueTypedInstruction(ctx)) {
			case "*(MATRIX,MATRIX)":
				return StatementUtils.min(ctx, RewriterStatement.nnz(stmt.getChild(0), ctx), RewriterStatement.nnz(stmt.getChild(1), ctx));
			case "*(MATRIX,FLOAT)":
				if (stmt.getChild(1).isLiteral() && ConstantFoldingFunctions.overwritesLiteral(((Float) stmt.getChild(1).getLiteral()), "*", ctx) != null)
					return RewriterStatement.literal(ctx, 0L);
				return RewriterStatement.nnz(stmt.getChild(0), ctx);
			case "*(FLOAT,MATRIX)":
				if (stmt.getChild(0).isLiteral() && ConstantFoldingFunctions.overwritesLiteral(((Float) stmt.getChild(0).getLiteral()), "*", ctx) != null)
					return RewriterStatement.literal(ctx, 0L);
				return RewriterStatement.nnz(stmt.getChild(1), ctx);
			case "+(MATRIX,MATRIX)":
			case "-(MATRIX,MATRIX)":
				return StatementUtils.min(ctx, new RewriterInstruction("+", ctx, RewriterStatement.nnz(stmt.getChild(0), ctx), RewriterStatement.nnz(stmt.getChild(1), ctx)), StatementUtils.length(ctx, stmt));
			case "+(MATRIX,FLOAT)":
			case "-(MATRIX,FLOAT)":
				if (stmt.getChild(1).isLiteral() && ConstantFoldingFunctions.isNeutralElement(stmt.getChild(1).getLiteral(), "+"))
					return RewriterStatement.nnz(stmt.getChild(0), ctx);
				return StatementUtils.length(ctx, stmt);
			case "+(FLOAT,MATRIX)":
			case "-(FLOAT,MATRIX)":
				if (stmt.getChild(0).isLiteral() && ConstantFoldingFunctions.isNeutralElement(stmt.getChild(0).getLiteral(), "+"))
					return RewriterStatement.nnz(stmt.getChild(1), ctx);
				return StatementUtils.length(ctx, stmt);
			case "!=(MATRIX,MATRIX)":
				if (stmt.getChild(0).equals(stmt.getChild(1)))
					return RewriterStatement.literal(ctx, 0L);
				return StatementUtils.length(ctx, stmt);

			case "sqrt(MATRIX)":
				return RewriterStatement.nnz(stmt.getChild(0), ctx);

			case "diag(MATRIX)":
				return StatementUtils.min(ctx, stmt.getNRow(), RewriterStatement.nnz(stmt.getChild(0), ctx));

			case "/(MATRIX,FLOAT)":
			case "/(MATRIX,MATRIX)":
				return RewriterStatement.nnz(stmt.getChild(0), ctx);
			case "/(FLOAT,MATRIX)":
				if (stmt.getChild(0).isLiteral() && ConstantFoldingFunctions.isNeutralElement(stmt.getChild(0).getLiteral(), "+"))
					return RewriterStatement.literal(ctx, 0L);
				return StatementUtils.length(ctx, stmt);


			// Fused operators
			case "log_nz(MATRIX)":
			case "*2(MATRIX)":
			case "sq(MATRIX)":
				return RewriterStatement.nnz(stmt.getChild(0), ctx);
			case "1-*(MATRIX,MATRIX)":
				return StatementUtils.length(ctx, stmt);
			case "+*(MATRIX,FLOAT,MATRIX)":
			case "-*(MATRIX,FLOAT,MATRIX)":
				if (stmt.getChild(1).isLiteral() && ConstantFoldingFunctions.isNeutralElement(stmt.getChild(1).getLiteral(), "+"))
					return RewriterStatement.nnz(stmt.getChild(0), ctx);
				return StatementUtils.min(ctx, new RewriterInstruction("+", ctx, RewriterStatement.nnz(stmt.getChild(0), ctx), RewriterStatement.nnz(stmt.getChild(2), ctx)), StatementUtils.length(ctx, stmt));
		}

		return StatementUtils.length(ctx, stmt);
	}
}
