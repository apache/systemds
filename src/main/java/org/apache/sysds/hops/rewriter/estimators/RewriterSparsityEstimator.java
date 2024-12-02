package org.apache.sysds.hops.rewriter.estimators;

import org.apache.sysds.hops.rewriter.ConstantFoldingFunctions;
import org.apache.sysds.hops.rewriter.RewriterInstruction;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.hops.rewriter.utils.StatementUtils;

import java.util.Map;
import java.util.UUID;

public class RewriterSparsityEstimator {
	public static RewriterStatement estimateNNZ(RewriterStatement stmt, Map<RewriterStatement, Long> matrixNNZs, final RuleContext ctx) {
		long[] nnzs = stmt.getOperands().stream().mapToLong(matrixNNZs::get).toArray();

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
		}

		return StatementUtils.length(ctx, stmt);
	}
}
