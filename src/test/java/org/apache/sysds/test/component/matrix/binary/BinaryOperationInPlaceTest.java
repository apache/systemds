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

package org.apache.sysds.test.component.matrix.binary;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.LessThan;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Or;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class BinaryOperationInPlaceTest {
	@Test
	public void testPlus() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(10, 10, 0, 10, 1.0, 1);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(10, 10, 0, 10, 1.0, 2);
		executePlus(m1, m2);
	}

	@Test
	public void testPlus_emptyInplace() {
		MatrixBlock m1 = new MatrixBlock(10, 10, false);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(10, 10, 0, 10, 1.0, 2);
		executePlus(m1, m2);
	}

	@Test
	public void testPlus_emptyOther() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(10, 10, 0, 10, 1.0, 1);
		MatrixBlock m2 = new MatrixBlock(10, 10, false);
		executePlus(m1, m2);
	}

	@Test
	public void testPlus_emptyInplace_butAllocatedDense() {
		MatrixBlock m1 = new MatrixBlock(10, 10, false);
		m1.allocateDenseBlock();
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(10, 10, 0, 10, 1.0, 2);
		executePlus(m1, m2);
	}

	@Test
	public void testDivide() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(10, 10, 0, 10, 1.0, 1);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(10, 10, 0, 10, 1.0, 2);
		executeDivide(m1, m2);
	}

	@Test
	public void testDivide_matrixVector() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(100, 10, 0, 10, 1.0, 1);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(1, 10, 0, 10, 1.0, 2);
		executeDivide(m1, m2);
	}

	@Test(expected = Exception.class)
	public void testDivide_Invalid_1() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(100, 10, 0, 10, 1.0, 1);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(1, 11, 0, 10, 1.0, 2);
		executeDivide(m1, m2);
	}

	@Test(expected = Exception.class)
	public void testDivide_Invalid_2() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(100, 10, 0, 10, 1.0, 1);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(1, 9, 0, 10, 1.0, 2);
		executeDivide(m1, m2);
	}

	@Test
	public void testDivide_matrixVector_emptyVector() {
		try {

			MatrixBlock m1 = TestUtils.generateTestMatrixBlock(100, 10, 0, 10, 1.0, 1);
			MatrixBlock m2 = new MatrixBlock(1, 10, 0.0);
			executeDivide(m1, m2);
		}
		catch(DMLRuntimeException e) {
			throw e;
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testDivide_matrixVector_sparseBoth() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(100, 1000, 0, 10, 0.2, 1);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(1, 1000, 0, 10, 0.2, 2);
		m1.examSparsity();
		m2.examSparsity();
		executeDivide(m1, m2);
	}

	@Test
	public void testDivide_matrixVector_oneEmpty() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(10, 10, 0, 10, 0.2, 1);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(1, 10, 0, 10, 0.0, 2);
		m1.examSparsity();
		m2.examSparsity();
		executeDivide(m1, m2);
	}

	@Test
	public void testOr_matrixMatrix_denseDense() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(10, 10, 0, 10, 1.0, 1);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(10, 10, 0, 10, 1.0, 2);
		executeOr(m1, m2);
	}

	@Test
	public void testOr_matrixMatrix_denseSparse() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 1.0, 1);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.1, 2);
		executeOr(m1, m2);
	}

	@Test
	public void testLT_matrixMatrix_denseDense() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(10, 10, 0, 10, 1.0, 1);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(10, 10, 0, 10, 1.0, 2);
		executeLT(m1, m2);
	}

	@Test
	public void testLT_matrixMatrix_denseSparse() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 1.0, 1);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.1, 2);
		executeLT(m1, m2);
	}

	@Test
	public void testLT_matrixMatrix_denseEmpty() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 1.0, 1);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.0, 2);
		executeLT(m1, m2);
	}

	@Test
	public void testLT_matrixMatrix_EmptyDense() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.0, 1);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 1.0, 2);
		executeLT(m1, m2);
	}

	@Test
	public void testLT_matrixMatrix_EmptySparse() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.0, 1);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, .1, 2);
		executeLT(m1, m2);
	}

	@Test
	public void testLT_matrixMatrix_EmptyEmpty() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.0, 1);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.0, 2);
		executeLT(m1, m2);
	}

	@Test
	public void testPlus_matrixMatrix_DenseSparse() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 1.0, 1);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.1, 2);
		executePlus(m1, m2);
	}

	@Test
	public void testMinus_matrixMatrix_DenseSparse() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 1.0, 1);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.1, 2);
		executeMinus(m1, m2);
	}

	@Test
	public void testMinus_matrixMatrix_DenseDense() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 1.0, 1);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 1.0, 2);
		executeMinus(m1, m2);
	}

	@Test
	public void testLT_matrixMatrix_DenseDense() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 1.0, 1);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 1.0, 2);
		executeLT(m1, m2);
	}

	@Test
	public void testLT_matrixMatrix_DenseSparse() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 1.0, 1);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 1.0, 2);
		executeLT(m1, m2);
	}

	@Test
	public void testLT_matrixMatrix_SparseDense() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.1, 1);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 1.0, 2);
		assertTrue(m1.isInSparseFormat());
		executeLT(m1, m2);
	}

	@Test
	public void testPlus_matrixMatrix_SparseDense() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.1, 1);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 1.0, 2);
		assertTrue(m1.isInSparseFormat());
		executePlus(m1, m2);
	}

	@Test
	public void testPlus_matrixMatrix_SparseSparse() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.1, 1);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.1, 2);
		assertTrue(m1.isInSparseFormat());
		executePlus(m1, m2);
	}

	@Test
	public void testDiv_matrixMatrix_SparseSparse() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.1, 1);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.1, 2);
		assertTrue(m1.isInSparseFormat());
		executeDivide(m1, m2);
	}

	private void executeDivide(MatrixBlock m1, MatrixBlock m2) {
		BinaryOperator op = new BinaryOperator(Divide.getDivideFnObject());
		testInplace(m1, m2, op);
	}

	private void executePlus(MatrixBlock m1, MatrixBlock m2) {
		BinaryOperator op = new BinaryOperator(Plus.getPlusFnObject());
		testInplace(m1, m2, op);
	}

	private void executeMinus(MatrixBlock m1, MatrixBlock m2) {
		BinaryOperator op = new BinaryOperator(Minus.getMinusFnObject());
		testInplace(m1, m2, op);
	}

	private void executeOr(MatrixBlock m1, MatrixBlock m2) {
		BinaryOperator op = new BinaryOperator(Or.getOrFnObject());
		testInplace(m1, m2, op);
	}

	private void executeLT(MatrixBlock m1, MatrixBlock m2) {
		BinaryOperator op = new BinaryOperator(LessThan.getLessThanFnObject());
		testInplace(m1, m2, op);
	}

	private void testInplace(MatrixBlock m1, MatrixBlock m2, BinaryOperator op) {
		MatrixBlock ret1 = m1.binaryOperations(op, m2);
		m1.binaryOperationsInPlace(op, m2);
		TestUtils.compareMatricesBitAvgDistance(ret1, m1, 0, 0, "Result is incorrect for inplace op");
	}
}
