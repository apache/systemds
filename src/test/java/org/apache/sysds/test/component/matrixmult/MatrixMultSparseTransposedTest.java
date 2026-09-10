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

package org.apache.sysds.test.component.matrixmult;

import static org.junit.Assert.assertTrue;

import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.Random;

public class MatrixMultSparseTransposedTest {
	private static final double SPARSITY = 0.05;

	@Test
	public void testNoTransANoTransB() {
		runTests(false, false);
	}

	@Test
	public void testTransANoTransB() {
		runTests(true, false);
	}

	@Test
	public void testNoTransATransB() {
		runTests(false, true);
	}

	@Test
	public void testTransATransB() {
		runTests(true, true);
	}

	private void runTests(boolean transA, boolean transB) {
		Random rand = new Random(7);
		for( int i=0; i<10; i++ )
			runTest(transA, transB, rand.nextInt(300)+1, rand.nextInt(300)+1, rand.nextInt(300)+1, i);

		runTest(transA, transB, 1, 17, 23);
		runTest(transA, transB, 17, 1, 23);
		runTest(transA, transB, 31, 29, 7);
		runTest(transA, transB, 127, 83, 61);
		runTest(transA, transB, 300, 200, 120);
	}

	private void runTest(boolean transA, boolean transB, int m, int n, int cd) {
		runTest(transA, transB, m, n, cd, 0);
	}

	private void runTest(boolean transA, boolean transB, int m, int n, int cd, int seedOffset) {
		int rowsA = transA ? cd : m;
		int colsA = transA ? m : cd;
		int rowsB = transB ? n : cd;
		int colsB = transB ? cd : n;

		MatrixBlock a = sparseRand(rowsA, colsA, 7+seedOffset);
		MatrixBlock b = sparseRand(rowsB, colsB, 3+seedOffset);
		MatrixBlock c = new MatrixBlock(m, n, false);
		c.allocateDenseBlock();

		LibMatrixMult.matrixMultSparseSparseMM(a.getSparseBlock(), b.getSparseBlock(), c.getDenseBlock(),
			transA, transB, m, n, cd, 0, m);
		c.recomputeNonZeros();

		MatrixBlock aIn = transA ? LibMatrixReorg.transpose(a) : a;
		MatrixBlock bIn = transB ? LibMatrixReorg.transpose(b) : b;
		MatrixBlock expected = LibMatrixMult.matrixMult(aIn, bIn);

		TestUtils.compareMatrices(expected, c, 1e-8);
	}

	private MatrixBlock sparseRand(int rows, int cols, int seed) {
		MatrixBlock ret = MatrixBlock.randOperations(rows, cols, SPARSITY, -1, 1, "uniform", seed);
		ret.recomputeNonZeros();
		if( !ret.isInSparseFormat() )
			ret.denseToSparse(true);
		assertTrue("Expected sparse test input", ret.isInSparseFormat());
		return ret;
	}
}
