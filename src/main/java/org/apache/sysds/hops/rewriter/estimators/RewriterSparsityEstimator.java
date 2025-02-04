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

package org.apache.sysds.hops.rewriter.estimators;

import org.apache.sysds.hops.rewriter.utils.ConstantFoldingUtils;
import org.apache.sysds.hops.rewriter.RewriterInstruction;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.hops.rewriter.utils.StatementUtils;

import java.util.HashMap;
import java.util.Map;

public class RewriterSparsityEstimator {
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
		if (!stmt.isInstruction() || !stmt.getResultingDataType(ctx).equals("MATRIX"))
			return null;
		switch (stmt.trueInstruction()) {
			case "%*%":
				RewriterStatement min1 = StatementUtils.min(ctx, RewriterStatement.multiArgInstr(ctx, "*", RewriterStatement.nnz(stmt.getChild(0), ctx), new RewriterInstruction("inv", ctx, stmt.getChild(0).getNRow())), RewriterStatement.literal(ctx, 1.0D));
				RewriterStatement min2 = StatementUtils.min(ctx, RewriterStatement.multiArgInstr(ctx, "*", RewriterStatement.nnz(stmt.getChild(1), ctx), new RewriterInstruction("inv", ctx, stmt.getChild(1).getNCol())), RewriterStatement.literal(ctx, 1.0D));
				return RewriterStatement.multiArgInstr(ctx, "*", min1, min2, stmt.getNRow(), stmt.getNCol());
		}

		switch (stmt.trueTypedInstruction(ctx)) {
			case "*(MATRIX,MATRIX)":
				return StatementUtils.min(ctx, RewriterStatement.nnz(stmt.getChild(0), ctx), RewriterStatement.nnz(stmt.getChild(1), ctx));
			case "*(MATRIX,FLOAT)":
				if (stmt.getChild(1).isLiteral() && ConstantFoldingUtils.overwritesLiteral(((Double) stmt.getChild(1).getLiteral()), "*", ctx) != null)
					return RewriterStatement.literal(ctx, 0L);
				return RewriterStatement.nnz(stmt.getChild(0), ctx);
			case "*(FLOAT,MATRIX)":
				if (stmt.getChild(0).isLiteral() && ConstantFoldingUtils.overwritesLiteral(((Double) stmt.getChild(0).getLiteral()), "*", ctx) != null)
					return RewriterStatement.literal(ctx, 0L);
				return RewriterStatement.nnz(stmt.getChild(1), ctx);
			case "+(MATRIX,MATRIX)":
			case "-(MATRIX,MATRIX)":
				return StatementUtils.min(ctx, RewriterStatement.multiArgInstr(ctx, "+", RewriterStatement.nnz(stmt.getChild(0), ctx), RewriterStatement.nnz(stmt.getChild(1), ctx)), StatementUtils.length(ctx, stmt));
			case "+(MATRIX,FLOAT)":
			case "-(MATRIX,FLOAT)":
				if (stmt.getChild(1).isLiteral() && ConstantFoldingUtils.isNeutralElement(stmt.getChild(1).getLiteral(), "+"))
					return RewriterStatement.nnz(stmt.getChild(0), ctx);
				return StatementUtils.length(ctx, stmt);
			case "+(FLOAT,MATRIX)":
			case "-(FLOAT,MATRIX)":
				if (stmt.getChild(0).isLiteral() && ConstantFoldingUtils.isNeutralElement(stmt.getChild(0).getLiteral(), "+"))
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
				if (stmt.getChild(0).isLiteral() && ConstantFoldingUtils.isNeutralElement(stmt.getChild(0).getLiteral(), "+"))
					return RewriterStatement.literal(ctx, 0L);
				return StatementUtils.length(ctx, stmt);

			case "RBind(MATRIX,MATRIX)":
			case "CBind(MATRIX,MATRIX)":
				return StatementUtils.add(ctx, RewriterStatement.nnz(stmt.getChild(0), ctx), RewriterStatement.nnz(stmt.getChild(1), ctx));

			// Fused operators
			case "log_nz(MATRIX)":
			case "*2(MATRIX)":
			case "sq(MATRIX)":
			case "t(MATRIX)":
				return RewriterStatement.nnz(stmt.getChild(0), ctx);
			case "1-*(MATRIX,MATRIX)":
				return StatementUtils.length(ctx, stmt);
			case "+*(MATRIX,FLOAT,MATRIX)":
			case "-*(MATRIX,FLOAT,MATRIX)":
				if (stmt.getChild(1).isLiteral() && ConstantFoldingUtils.isNeutralElement(stmt.getChild(1).getLiteral(), "+"))
					return RewriterStatement.nnz(stmt.getChild(0), ctx);
				return StatementUtils.min(ctx, RewriterStatement.multiArgInstr(ctx, "+", RewriterStatement.nnz(stmt.getChild(0), ctx), RewriterStatement.nnz(stmt.getChild(2), ctx)), StatementUtils.length(ctx, stmt));
			case "const(MATRIX,FLOAT)":
				if (stmt.getChild(1).isLiteral() && ConstantFoldingUtils.isNeutralElement(stmt.getChild(1).getLiteral(), "+"))
					return RewriterStatement.literal(ctx, 0L);
			case "rowSums(MATRIX)":
			case "colSums(MATRIX)":
				StatementUtils.min(ctx, RewriterStatement.nnz(stmt.getChild(0), ctx), StatementUtils.length(ctx, stmt));
		}

		return StatementUtils.length(ctx, stmt);
	}
}
