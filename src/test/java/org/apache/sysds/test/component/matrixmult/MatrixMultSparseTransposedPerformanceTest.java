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

import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

public class MatrixMultSparseTransposedPerformanceTest {
	private static final int M = 500;
	private static final int N = 500;
	private static final int CD = 500;
	private static final double SPARSITY = 0.01;
	private static final int REP = 50;

	@Test
	public void testPerfNoTransANoTransB() {
		runTest(false, false, "C = A %*% B");
	}

	@Test
	public void testPerfTransANoTransB() {
		runTest(true, false, "C = t(A) %*% B");
	}

	@Test
	public void testPerfNoTransATransB() {
		runTest(false, true, "C = A %*% t(B)");
	}

	@Test
	public void testPerfTransATransB() {
		runTest(true, true, "C = t(A) %*% t(B)");
	}

	private void runTest(boolean transA, boolean transB, String name) {
		int rowsA = transA ? CD : M;
		int colsA = transA ? M : CD;
		int rowsB = transB ? N : CD;
		int colsB = transB ? CD : N;
		MatrixBlock a = sparseRand(rowsA, colsA, 7);
		MatrixBlock b = sparseRand(rowsB, colsB, 3);
		MatrixBlock c = new MatrixBlock(M, N, false);
		c.allocateDenseBlock();

		for( int i=0; i<10; i++ ) {
			runOldMethod(a, b, transA, transB);
			runNewKernel(a, b, c, transA, transB);
		}

		long startTimeOld = System.nanoTime();
		for( int i=0; i<REP; i++ )
			runOldMethod(a, b, transA, transB);
		double avgTimeOld = (System.nanoTime() - startTimeOld) / 1e6 / REP;

		long startTimeNew = System.nanoTime();
		for( int i=0; i<REP; i++ )
			runNewKernel(a, b, c, transA, transB);
		double avgTimeNew = (System.nanoTime() - startTimeNew) / 1e6 / REP;

		System.out.printf("%s | Old Method: %.3f ms | New Kernel: %.3f ms%n", name, avgTimeOld, avgTimeNew);
	}

	private void runNewKernel(MatrixBlock a, MatrixBlock b, MatrixBlock c, boolean transA, boolean transB) {
		c.getDenseBlock().reset();
		LibMatrixMult.matrixMultSparseSparseMM(a.getSparseBlock(), b.getSparseBlock(), c.getDenseBlock(),
			transA, transB, M, N, CD, 0, M);
	}

	private void runOldMethod(MatrixBlock a, MatrixBlock b, boolean transA, boolean transB) {
		MatrixBlock aIn = transA ? LibMatrixReorg.transpose(a) : a;
		MatrixBlock bIn = transB ? LibMatrixReorg.transpose(b) : b;
		LibMatrixMult.matrixMult(aIn, bIn);
	}

	private MatrixBlock sparseRand(int rows, int cols, int seed) {
		MatrixBlock ret = MatrixBlock.randOperations(rows, cols, SPARSITY, -1, 1, "uniform", seed);
		ret.recomputeNonZeros();
		if( !ret.isInSparseFormat() )
			ret.denseToSparse(true);
		return ret;
	}
}
