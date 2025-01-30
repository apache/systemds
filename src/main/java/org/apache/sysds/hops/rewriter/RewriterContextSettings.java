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

package org.apache.sysds.hops.rewriter;

import org.apache.sysds.hops.rewriter.utils.RewriterUtils;

import java.util.List;
import java.util.Random;

public class RewriterContextSettings {

	public static final List<String> ALL_TYPES = List.of("FLOAT", "INT", "BOOL", "MATRIX");
	public static final List<String> SCALARS = List.of("FLOAT", "INT", "BOOL");

	public static String getDefaultContextString() {
		StringBuilder builder = new StringBuilder();
		ALL_TYPES.forEach(t -> {
			builder.append("argList(" + t + ")::" + t + "...\n");
			builder.append("argList(" + t + "...)::" + t + "...\n");
		}); // This is a meta function that can take any number of arguments

		builder.append("CBind(MATRIX,MATRIX)::MATRIX\n"); // This instruction is not really supported
		builder.append("RBind(MATRIX,MATRIX)::MATRIX\n"); // This instruction is not really supported

		builder.append("sum(MATRIX)::FLOAT\n");
		builder.append("rowSums(MATRIX)::MATRIX\n");
		builder.append("colSums(MATRIX)::MATRIX\n");

		builder.append("max(MATRIX)::FLOAT\n"); // Support for min/max is limited
		builder.append("min(MATRIX)::FLOAT\n"); // Support for min/max is limited

		builder.append("%*%(MATRIX,MATRIX)::MATRIX\n");

		builder.append("rev(MATRIX)::MATRIX\n");
		builder.append("t(MATRIX)::MATRIX\n");

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
			builder.append("impl ^\n");
			builder.append("impl >\n");
			builder.append("impl <\n");
			builder.append("impl >=\n");
			builder.append("impl <=\n");
			builder.append("impl ==\n");
			builder.append("impl |\n");
			builder.append("impl &\n");
			builder.append("impl /\n");
			builder.append("impl !=\n");
		});

		builder.append("ElementWiseInstruction(MATRIX...)::MATRIX\n");
		builder.append("impl ElementWiseSumExpandableInstruction\n");
		builder.append("impl /\n");
		builder.append("impl max\n");
		builder.append("impl min\n");
		builder.append("impl ^\n");
		builder.append("impl >\n");
		builder.append("impl <\n");
		builder.append("impl >=\n");
		builder.append("impl <=\n");
		builder.append("impl ==\n");
		builder.append("impl |\n");
		builder.append("impl &\n");

		RewriterUtils.buildBinaryPermutations(List.of("MATRIX...", "MATRIX", "INT", "FLOAT", "BOOL"), (t1, t2) -> {
			builder.append("ElementWiseSumExpandableInstruction(" + t1 + "," + t2 + ")::" + RewriterUtils.defaultTypeHierarchy(t1, t2) + "\n"); // Any instruction that allows op(sum(A*), sum(B*)) = sum(op(A, B))
			builder.append("impl ElementWiseAdditiveInstruction\n");
			builder.append("impl *\n");

			builder.append("ElementWiseAdditiveInstruction(" + t1 + "," + t2 + ")::" + RewriterUtils.defaultTypeHierarchy(t1, t2) + "\n");
			builder.append("impl +\n");
			builder.append("impl -\n");
		});

		builder.append("ElementWiseAdditiveInstruction(MATRIX...)::MATRIX\n");
		builder.append("impl +\n");
		//builder.append("impl -\n");


		ALL_TYPES.forEach(t -> {
			builder.append("UnaryElementWiseOperator(" + t + ")::" + t + "\n");
			builder.append("impl -\n");
			builder.append("impl abs\n");
			builder.append("impl !\n");
			builder.append("impl round\n");
		});

		builder.append("rowSelect(MATRIX,INT,INT)::MATRIX\n");
		builder.append("colSelect(MATRIX,INT,INT)::MATRIX\n");
		builder.append("min(INT,INT)::INT\n");
		builder.append("max(INT,INT)::INT\n");

		builder.append("index(MATRIX,INT,INT,INT,INT)::MATRIX\n");

		RewriterUtils.buildBinaryPermutations(List.of("MATRIX...", "MATRIX", "INT", "FLOAT", "BOOL"), (t1, t2) -> {
			builder.append("FusableBinaryOperator(" + t1 + "," + t2 + ")::" + RewriterUtils.defaultTypeHierarchy(t1, t2) + "\n");
			builder.append("impl +\n");
			builder.append("impl *\n");
		});

		List.of("MATRIX", "INT", "FLOAT", "BOOL").forEach(t -> {
			builder.append("FusedOperator(" + t + "...)::" + t + "\n");
			builder.append("impl +\n");
			builder.append("impl *\n");
		});

		builder.append("ncol(MATRIX)::INT\n");
		builder.append("nrow(MATRIX)::INT\n");
		builder.append("length(MATRIX)::INT\n");

		RewriterUtils.buildBinaryAlgebraInstructions(builder, "+", List.of("INT", "FLOAT", "BOOL", "MATRIX"));
		RewriterUtils.buildBinaryAlgebraInstructions(builder, "*", List.of("INT", "FLOAT", "BOOL", "MATRIX"));
		RewriterUtils.buildBinaryAlgebraInstructions(builder, "^", ALL_TYPES);
		ALL_TYPES.forEach(t -> builder.append("-(" + t + ")::" + t + "\n"));
		ALL_TYPES.forEach(t -> builder.append("inv(" + t + ")::" + t + "\n"));


		builder.append("as.matrix(INT)::MATRIX\n");
		builder.append("as.matrix(FLOAT)::MATRIX\n");
		builder.append("as.matrix(BOOL)::MATRIX\n");
		builder.append("as.scalar(MATRIX)::FLOAT\n");
		builder.append("as.scalar(FLOAT)::FLOAT\n");
		builder.append("as.float(INT)::FLOAT\n");
		builder.append("as.float(BOOL)::FLOAT\n");
		builder.append("as.int(BOOL)::INT\n");

		RewriterUtils.buildBinaryPermutations(ALL_TYPES, (tFrom, tTo) -> {
			builder.append("cast." + tTo + "(" + tFrom + ")::" + tTo + "\n");
		});

		builder.append("rand(INT,INT,FLOAT,FLOAT)::MATRIX\n"); // Args: rows, cols, min, max
		builder.append("rand(INT,INT)::FLOAT\n"); // Just to make it possible to say that random is dependent on both matrix indices
		builder.append("rand(INT...)::FLOAT\n");
		builder.append("matrix(INT,INT,INT)::MATRIX\n");

		builder.append("trace(MATRIX)::FLOAT\n");

		// Boole algebra

		RewriterUtils.buildBinaryPermutations(List.of("MATRIX", "FLOAT", "INT", "BOOL"), (t1, t2) -> {
			String ret = t1.equals("MATRIX") || t2.equals("MATRIX") ? "MATRIX" : "BOOL";
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

		// Unary ops
		ALL_TYPES.forEach(t -> {
			builder.append("ElementWiseUnary.FLOAT(" + t + ")::" + (t.equals("MATRIX") ? "MATRIX" : "FLOAT") + "\n");
			builder.append("impl sqrt\n");
			builder.append("impl exp\n");
			builder.append("impl log\n");
			builder.append("impl inv\n");
		});

		builder.append("[](MATRIX,INT,INT)::FLOAT\n");
		builder.append("[](MATRIX,INT,INT,INT,INT)::MATRIX\n");
		builder.append("diag(MATRIX)::MATRIX\n");
		builder.append("replace(MATRIX,FLOAT,FLOAT)::MATRIX\n"); // This is not supported
		builder.append("_nnz(MATRIX)::INT\n");
		builder.append("sumSq(MATRIX)::FLOAT\n");
		builder.append("sq(MATRIX)::MATRIX\n");
		builder.append("+*(MATRIX,FLOAT,MATRIX)::MATRIX\n");
		builder.append("-*(MATRIX,FLOAT,MATRIX)::MATRIX\n");
		builder.append("*2(MATRIX)::MATRIX\n");

		for (String t : SCALARS) {
			for (String t2 : SCALARS)
				builder.append("ifelse(BOOL," + t + "," + t2 + ")::" + RewriterUtils.convertibleType(t, t2) + "\n");
		}


		List.of("INT", "FLOAT", "BOOL").forEach(t -> {
			String newType = t.equals("BOOL") ? "INT" : t;
			builder.append("sum(" + t + "...)::" + newType + "\n");
			builder.append("sum(" + t + "*)::" + newType + "\n");
			builder.append("sum(" + t + ")::" + newType + "\n");

			builder.append("min(" + t + "...)::" + t + "\n");
			builder.append("min(" + t + "*)::" + t + "\n");
			builder.append("min(" + t + ")::" + t + "\n");

			builder.append("max(" + t + "...)::" + t + "\n");
			builder.append("max(" + t + "*)::" + t + "\n");
			builder.append("max(" + t + ")::" + t + "\n");
		});

		// Some fused operators
		builder.append("1-*(MATRIX,MATRIX)::MATRIX\n"); 		// OpOp2.MINUS1_MULT
		builder.append("log_nz(MATRIX)::MATRIX\n");				// OpOp1.LOG_NZ
		SCALARS.forEach(t -> {
			builder.append("log(MATRIX," + t + ")::MATRIX\n");
			builder.append("log_nz(MATRIX," + t + ")::MATRIX\n");
		});

		builder.append("const(MATRIX,FLOAT)::MATRIX\n");

		builder.append("rowVec(MATRIX)::MATRIX\n");
		builder.append("colVec(MATRIX)::MATRIX\n");
		builder.append("cellMat(MATRIX)::MATRIX\n");

		builder.append("_m(INT,INT,FLOAT)::MATRIX\n");
		builder.append("_m(INT,INT,BOOL)::MATRIX\n");
		builder.append("_m(INT,INT,INT)::MATRIX\n");
		List.of("FLOAT", "INT", "BOOL").forEach(t -> {
			builder.append("_idxExpr(INT," + t + ")::" + t + "*\n");
			builder.append("_idxExpr(INT," + t + "*)::" + t + "*\n");
			builder.append("_idxExpr(INT...," + t + ")::" + t + "*\n");
			builder.append("_idxExpr(INT...," + t + "*)::" + t + "*\n");
		});
		builder.append("_idx(INT,INT)::INT\n");

		ALL_TYPES.forEach(t -> builder.append("_EClass(" + t + "...)::" + t + "\n"));
		ALL_TYPES.forEach(t -> builder.append("_backRef." + t + "()::" + t + "\n"));

		for (String s : SCALARS)
			builder.append("literal." + s + "()::" + s + "\n");

		return builder.toString();
	}
	public static RuleContext getDefaultContext() {
		String ctxString = getDefaultContextString();

		RuleContext ctx = RuleContext.createContext(ctxString);

		ctx.customStringRepr.put("rand(INT,INT,FLOAT,FLOAT)", (stmt, mctx) -> {
			List<RewriterStatement> ops = stmt.getOperands();
			return "rand(rows=(" + ops.get(0) + "), cols=(" + ops.get(1) + "), min=(" + ops.get(2) + "), max=(" + ops.get(3) + "))";
		});
		ctx.customStringRepr.put("rand(INT,INT,INT,INT)", ctx.customStringRepr.get("rand(INT,INT,FLOAT,FLOAT)"));
		ctx.customStringRepr.put("rand(INT,INT,FLOAT,INT)", ctx.customStringRepr.get("rand(INT,INT,FLOAT,FLOAT)"));
		ctx.customStringRepr.put("rand(INT,INT,INT,FLOAT)", ctx.customStringRepr.get("rand(INT,INT,FLOAT,FLOAT)"));

		RewriterUtils.putAsDefaultBinaryPrintable(List.of("<", "<=", ">", ">=", "==", "!=", "&", "|"), List.of("INT", "FLOAT", "BOOL", "MATRIX"), ctx.customStringRepr);

		RewriterUtils.putAsBinaryPrintable("*", List.of("INT", "FLOAT", "MATRIX", "BOOL"), ctx.customStringRepr, RewriterUtils.binaryStringRepr(" * "));
		RewriterUtils.putAsBinaryPrintable("+", List.of("INT", "FLOAT", "MATRIX", "BOOL"), ctx.customStringRepr, RewriterUtils.binaryStringRepr(" + "));

		ctx.customStringRepr.put("%*%(MATRIX,MATRIX)", RewriterUtils.binaryStringRepr(" %*% "));
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

		return ctx;
	}
}
