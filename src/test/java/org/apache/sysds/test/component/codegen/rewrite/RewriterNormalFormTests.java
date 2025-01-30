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

package org.apache.sysds.test.component.codegen.rewrite;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.hops.rewriter.TopologicalSort;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.function.Function;

public class RewriterNormalFormTests {
	protected static final Log LOG = LogFactory.getLog(RewriterNormalFormTests.class.getName());

	private static RuleContext ctx;
	private static Function<RewriterStatement, RewriterStatement> canonicalConverter;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
		canonicalConverter = RewriterUtils.buildCanonicalFormConverter(ctx, false);
	}

	//e.g., matrix(1,nrow(X),ncol(X))/X -> 1/X
	@Test
	public void testUnnecessaryVectorize() {
		RewriterStatement stmt1 = RewriterUtils.parse("/(const(A, 1.0), A)", ctx, "MATRIX:A", "LITERAL_FLOAT:1.0");
		RewriterStatement stmt2 = RewriterUtils.parse("/(1.0, A)", ctx, "MATRIX:A", "LITERAL_FLOAT:1.0");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testRemoveUnnecessaryBinaryOperation() {
		RewriterStatement stmt1 = RewriterUtils.parse("*(1.0, A)", ctx, "MATRIX:A", "LITERAL_FLOAT:1.0");
		RewriterStatement stmt2 = RewriterUtils.parse("A", ctx, "MATRIX:A", "LITERAL_FLOAT:1.0");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testFuseDatagenAndBinaryOperation() {
		RewriterStatement stmt1 = RewriterUtils.parse("*(rand(nrow(A), ncol(A), -1.0, 1.0), a)", ctx, "MATRIX:A", "FLOAT:a", "LITERAL_FLOAT:1.0,-1.0");
		RewriterStatement stmt2 = RewriterUtils.parse("rand(nrow(A), ncol(A), -(a), a)", ctx, "MATRIX:A", "FLOAT:a");

		assert match(stmt1, stmt2);
	}

	//@Test
	public void testFuseDatagenAndMinusOperation() {
		RewriterStatement stmt1 = RewriterUtils.parse("-(rand(nrow(A), ncol(A), -2.0, 1.0))", ctx, "MATRIX:A", "LITERAL_FLOAT:1.0,-2.0");
		RewriterStatement stmt2 = RewriterUtils.parse("rand(nrow(A), ncol(A), -1.0, 2.0)", ctx, "MATRIX:A", "LITERAL_FLOAT:-1.0,2.0");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testCanonicalizeMatrixMultScalarAdd() {
		RewriterStatement stmt1 = RewriterUtils.parse("+(eps, %*%(A, t(B)))", ctx, "MATRIX:A,B", "FLOAT:eps");
		RewriterStatement stmt2 = RewriterUtils.parse("+(%*%(A, t(B)), eps)", ctx, "MATRIX:A,B", "FLOAT:eps");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testCanonicalizeMatrixMultScalarAdd2() {
		RewriterStatement stmt1 = RewriterUtils.parse("-(%*%(A, t(B)), eps)", ctx, "MATRIX:A,B", "FLOAT:eps");
		RewriterStatement stmt2 = RewriterUtils.parse("+(%*%(A, t(B)), -(eps))", ctx, "MATRIX:A,B", "FLOAT:eps");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testSimplifyMultiBinaryToBinaryOperation() {
		RewriterStatement stmt1 = RewriterUtils.parse("-(1.0, *(A,B))", ctx, "MATRIX:A,B", "LITERAL_FLOAT:1.0");
		RewriterStatement stmt2 = RewriterUtils.parse("1-*(A, B)", ctx, "MATRIX:A,B", "LITERAL_FLOAT:1.0");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testSimplifyDistributiveBinaryOperation() {
		RewriterStatement stmt1 = RewriterUtils.parse("-(A, *(B,A))", ctx, "MATRIX:A,B,C", "LITERAL_FLOAT:1.0");
		RewriterStatement stmt2 = RewriterUtils.parse("*(-(1.0,B), A)", ctx, "MATRIX:A,B,C", "LITERAL_FLOAT:1.0");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testSimplifyBushyBinaryOperation() {
		RewriterStatement stmt1 = RewriterUtils.parse("*(A,*(B, %*%(C, colVec(D))))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0");
		RewriterStatement stmt2 = RewriterUtils.parse("*(*(A,B), %*%(C, colVec(D)))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0");

		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		LOG.info("==========");
		LOG.info(stmt1.toParsableString(ctx, true));
		LOG.info("==========");
		LOG.info(stmt2.toParsableString(ctx, true));
		assert RewriterStatement.MatcherContext.exactMatch(ctx, stmt1, stmt2).match();
	}

	@Test
	public void testSimplifyUnaryAggReorgOperation() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(t(A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0");
		RewriterStatement stmt2 = RewriterUtils.parse("sum(A)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testRemoveUnnecessaryAggregates() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(rowSums(A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0");
		RewriterStatement stmt2 = RewriterUtils.parse("sum(A)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testSimplifyBinaryMatrixScalarOperation() {
		RewriterStatement stmt1 = RewriterUtils.parse("as.scalar(*(A,a))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0");
		RewriterStatement stmt2 = RewriterUtils.parse("*(as.scalar(A),a)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testPushdownUnaryAggTransposeOperation() {
		RewriterStatement stmt1 = RewriterUtils.parse("colSums(t(A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0");
		RewriterStatement stmt2 = RewriterUtils.parse("t(rowSums(A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testPushdownCSETransposeScalarOperation() {
		// Introduce a dummy instruction * as I don't support the assignment operator
		RewriterStatement stmt1 = RewriterUtils.parse("*(t(A), t(sq(A)))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0");
		RewriterStatement stmt2 = RewriterUtils.parse("*(t(A), sq(t(A)))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testPushdownSumBinaryMult() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(*(a,A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0");
		RewriterStatement stmt2 = RewriterUtils.parse("*(a, sum(A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testSimplifyTraceMatrixMult() {
		RewriterStatement stmt1 = RewriterUtils.parse("trace(%*%(A,B))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0");
		RewriterStatement stmt2 = RewriterUtils.parse("sum(*(A, t(B)))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testSimplifySlicedMatrixMult() {
		RewriterStatement stmt1 = RewriterUtils.parse("[](%*%(A,B), 1, 1)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("as.scalar(%*%(rowVec(A), colVec(B)))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0", "LITERAL_INT:1");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testRemoveUnnecessaryReorgOperation() {
		RewriterStatement stmt1 = RewriterUtils.parse("t(t(A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("A", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0", "LITERAL_INT:1");

		assert match(stmt1, stmt2);
	}

	//@Test
	public void testRemoveUnnecessaryReorgOperation2() {
		RewriterStatement stmt1 = RewriterUtils.parse("rev(rev(A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("A", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0", "LITERAL_INT:1");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testSimplifyTransposeAggBinBinaryChains() {
		RewriterStatement stmt1 = RewriterUtils.parse("t(+(%*%(t(A),t(B)), C))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("+(%*%(B,A), t(C))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0", "LITERAL_INT:1");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testRemoveUnnecessaryMinus() {
		RewriterStatement stmt1 = RewriterUtils.parse("-(-(A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("A", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:1.0,2.0", "LITERAL_INT:1");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testFuseLogNzUnaryOperation() {
		RewriterStatement stmt1 = RewriterUtils.parse("*(!=(A,0.0), log(A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("log_nz(A)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testFuseLogNzBinaryOperation() {
		RewriterStatement stmt1 = RewriterUtils.parse("*(!=(A,0.0), log(A, a))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("log_nz(A, a)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1");

		assert match(stmt1, stmt2);
	}

	//@Test
	public void testSimplifyNotOverComparisons() {
		RewriterStatement stmt1 = RewriterUtils.parse("!(>(A,B))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("<=(A,B)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1");

		assert match(stmt1, stmt2);
	}

	///// DYNAMIC SIMPLIFICATIONS //////

	@Test
	public void testRemoveEmptyRightIndexing() {
		// We do not directly support the specification of nnz, but we can emulate such a matrix by multiplying with 0
		RewriterStatement stmt1 = RewriterUtils.parse("[](*(A, 0.0), 1, nrow(A), 1, 1)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("const(colVec(A), 0.0)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testRemoveUnnecessaryRightIndexing() {
		RewriterStatement stmt1 = RewriterUtils.parse("[](colVec(A), 1, nrow(A), 1, 1)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("colVec(A)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testRemoveUnnecessaryReorgOperation3() {
		RewriterStatement stmt1 = RewriterUtils.parse("t(cellMat(A)))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("cellMat(A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1");

		assert match(stmt1, stmt2);
	}

	//@Test
	public void testRemoveUnnecessaryOuterProduct() {
		RewriterStatement stmt1 = RewriterUtils.parse("*(A, %*%(colVec(B), const(t(colVec(B)), 1.0)))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1");
		RewriterStatement stmt2 = RewriterUtils.parse("*(A, colVec(B))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testRemoveUnnecessaryIfElseOperation() {
		// Ifelse is not directly supported yet but only on scalars. Thus, we will our index expression syntax to reflect that statement
		// Note that we "cheated" here a bit as we index using nrow(A) and ncol(A). We would not get a match if we used nrow(B)...
		RewriterStatement stmt1 = RewriterUtils.parse("_m($1:_idx(1, nrow(A)), $2:_idx(1, ncol(A)), ifelse(TRUE, [](A, $1, $2), [](B, $1, $2)))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE");
		RewriterStatement stmt2 = RewriterUtils.parse("A", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testFuseDatagenAndReorgOperation() {
		RewriterStatement stmt1 = RewriterUtils.parse("t(rand(i, 1, 0.0, 1.0))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("rand(1, i, 0.0, 1.0)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testSimplifyColwiseAggregate() {
		RewriterStatement stmt1 = RewriterUtils.parse("colSums(rowVec(A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("rowVec(A)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testSimplifyRowwiseAggregate() {
		RewriterStatement stmt1 = RewriterUtils.parse("rowSums(colVec(A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("colVec(A)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	// We don't have broadcasting semantics
	@Test
	public void testSimplifyColSumsMVMult() {
		RewriterStatement stmt1 = RewriterUtils.parse("colSums(*(colVec(A), colVec(B)))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("%*%(t(colVec(B)), colVec(A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	// We don't have broadcasting semantics
	@Test
	public void testSimplifyRowSumsMVMult() {
		RewriterStatement stmt1 = RewriterUtils.parse("rowSums(*(rowVec(A), rowVec(B)))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("%*%(rowVec(A), t(rowVec(B)))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testSimplifyUnnecessaryAggregate() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(cellMat(A)))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("as.scalar(A)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testSimplifyEmptyAggregate() {
		// We emulate an empty matrix by multiplying by zero
		RewriterStatement stmt1 = RewriterUtils.parse("sum(*(A, 0.0))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("0.0", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testSimplifyEmptyReorgOperation() {
		// We emulate an empty matrix by multiplying by zero
		RewriterStatement stmt1 = RewriterUtils.parse("t(*(A, 0.0))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("const(t(A), 0.0)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	// This is a hacky workaround
	@Test
	public void testSimplifyEmptyMatrixMult() {
		// We emulate an empty matrix by multiplying by zero
		// Note that we pass the dimension info of the matrix multiply to get the same e-class assertions
		RewriterStatement stmt1 = RewriterUtils.parse("%*%(*(A, 0.0), B)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("const(%*%(A, B), 0.0)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		// We need to explicitly assert A and B
		stmt2.givenThatEqual(stmt2.getChild(0, 1).getNRow(), stmt2.getChild(0, 0).getNCol(), ctx);
		stmt2.recomputeAssertions();

		assert match(stmt1, stmt2, true);
	}

	@Test
	public void testSimplifyEmptyMatrixMult2() {
		RewriterStatement stmt1 = RewriterUtils.parse("%*%(colVec(A), cast.MATRIX(1.0))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("colVec(A)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testSimplifyScalarMatrixMult() {
		RewriterStatement stmt1 = RewriterUtils.parse("%*%(colVec(A), cast.MATRIX(a))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("*(colVec(A), as.scalar(cast.MATRIX(a)))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testSimplifyDistributiveMatrixMult() {
		RewriterStatement stmt1 = RewriterUtils.parse("+(%*%(A, B), %*%(A, C))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("%*%(A, +(B, C)))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	// Note that we did not implement the overloaded diag(A) operation as we defined diag(A) as setting all other entries to zero (which is not how it is actually handled by SystemDS)
	// In this case, we obtain the same rewrite, even though the diag operation is different
	@Test
	public void testSimplifySumDiagToTrace() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(diag(A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("trace(A)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	// Note that we did not implement the overloaded diag(A) operation as we defined diag(A) as setting all other entries to zero (which is not how it is actually handled by SystemDS)
	// In this case, we obtain the same equivalence, but in case of our implementation the rewrite would not be beneficial
	@Test
	public void testPushdownBinaryOperationOnDiag() {
		RewriterStatement stmt1 = RewriterUtils.parse("*(diag(A), a)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("diag(*(A, a))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testPushdownSumOnAdditiveBinary() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(+(A, B))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("+(sum(A), sum(B))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		// We need to assert that the dimensions are the same, which we currently cannot do implicitly through an expression
		stmt2.givenThatEqualDimensions(stmt2.getChild(0, 0), stmt2.getChild(1, 0), ctx);

		assert match(stmt1, stmt2);
	}

	@Test
	public void testSimplifyDotProductSum() {
		RewriterStatement stmt1 = RewriterUtils.parse("cast.MATRIX(sum(sq(colVec(A))))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("%*%(t(colVec(A)), colVec(A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testFuseSumSquared() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(sq(A))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("sumSq(A)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testFuseAxpyBinaryOperationChain() {
		RewriterStatement stmt1 = RewriterUtils.parse("+(A, *(a, B))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("+*(A, a, B)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testFuseAxpyBinaryOperationChain2() {
		RewriterStatement stmt1 = RewriterUtils.parse("-(A, *(a, B))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("-*(A, a, B)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testReorderMinusMatrixMult() {
		RewriterStatement stmt1 = RewriterUtils.parse("%*%(-(t(A)), B)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("-(%*%(t(A), B))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testSimplifySumMatrixMult() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(%*%(A, B))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("sum(*(t(colSums(A)), rowSums(B)))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testSimplifyEmptyBinaryOperation() {
		RewriterStatement stmt1 = RewriterUtils.parse("*(A, const(A, 0.0))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("const(A, 0.0)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testSimplifyEmptyBinaryOperation2() {
		RewriterStatement stmt1 = RewriterUtils.parse("+(A, const(A, 0.0))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("A", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testSimplifyEmptyBinaryOperation3() {
		RewriterStatement stmt1 = RewriterUtils.parse("-(A, const(A, 0.0))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("A", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	//@Test
	public void testSimplifyScalarMVBinaryOperation() {
		RewriterStatement stmt1 = RewriterUtils.parse("*(A, colVec(colVec(B)))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("*(A, as.scalar(B))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	@Test
	public void testSimplifyNnzComputation() {
		RewriterStatement stmt1 = RewriterUtils.parse("sum(!=(A, 0.0))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("_nnz(A)", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	// We only support concrete literals (which is a current limitation of this framework)
	@Test
	public void testSimplifyNrowNcolComputation() {
		// We simulate a matrix with known dimensions by doing a concrete left-indexing
		RewriterStatement stmt1 = RewriterUtils.parse("nrow([](A, 1, 5, 1, 5))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1,5", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("5", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1,5", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	// We only support concrete literals (which is a current limitation of this framework)
	@Test
	public void testSimplifyNrowNcolComputation2() {
		// We simulate a matrix with known dimensions by doing a concrete left-indexing
		RewriterStatement stmt1 = RewriterUtils.parse("ncol([](A, 1, 5, 1, 5))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1,5", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("5", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1,5", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	// We only support concrete literals (which is a current limitation of this framework)
	@Test
	public void testSimplifyNrowNcolComputation3() {
		// We simulate a matrix with known dimensions by doing a concrete left-indexing
		RewriterStatement stmt1 = RewriterUtils.parse("length([](A, 1, 5, 1, 5))", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1,5", "LITERAL_BOOL:TRUE,FALSE", "INT:i");
		RewriterStatement stmt2 = RewriterUtils.parse("25", ctx, "MATRIX:A,B,C,D", "FLOAT:a,b,c", "LITERAL_FLOAT:0.0,1.0,2.0", "LITERAL_INT:1,25", "LITERAL_BOOL:TRUE,FALSE", "INT:i");

		assert match(stmt1, stmt2);
	}

	private boolean match(RewriterStatement stmt1, RewriterStatement stmt2) {
		return match(stmt1, stmt2, false);
	}

	private boolean match(RewriterStatement stmt1, RewriterStatement stmt2, boolean debug) {
		stmt1 = canonicalConverter.apply(stmt1);
		stmt2 = canonicalConverter.apply(stmt2);

		LOG.info("==========");
		LOG.info(stmt1.toParsableString(ctx, true));
		LOG.info("==========");
		LOG.info(stmt2.toParsableString(ctx, true));
		return RewriterStatement.MatcherContext.exactMatch(ctx, stmt1, stmt2).debug(debug).match();
	}
}
