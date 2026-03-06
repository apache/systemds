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

public class MatrixMultTransposedPerformanceTest {
	// could be adjusted, as it takes a lot of runtime with higher dimensions
	private final int m = 200;
	private final int n = 200;
	private final int k = 200;

	@Test
	public void testPerf_1_NoTransA_TransB() {
		System.out.println("Case: C = A %*% t(B)");
		runTest(false, true);
		System.out.println();
	}

	@Test
	public void testPerf_2_TransA_NoTransB() {
		System.out.println("Case: C = t(A) %*% B");
		runTest(true, false);
		System.out.println();
	}

	@Test
	public void testPerf_3_TransA_TransB() {
		System.out.println("Case: C = t(A) %*% t(B)");
		runTest(true, true);
	}

	private void runTest(boolean tA, boolean tB) {
		int REP = 100;

		// setup Dimensions
		int rowsA = tA ? k : m;
		int colsA = tA ? m : k;
		int rowsB = tB ? n : k;
		int colsB = tB ? k : n;

		// generate random matrices
		MatrixBlock A = MatrixBlock.randOperations(rowsA, colsA, 1.0, -1, 1, "uniform", 7);
		MatrixBlock B = MatrixBlock.randOperations(rowsB, colsB, 1.0, -1, 1, "uniform", 3);
		MatrixBlock C = new MatrixBlock(m, n, false);
		C.allocateDenseBlock();

		for(int i=0; i<50; i++) {
			runOldMethod(A, B, tA, tB);
			runNewKernel(A, B, C, tA, tB);
		}

		// Measure Old Method
		long startTimeOld = System.nanoTime();
		for(int i = 0; i < REP; i++) {
			runOldMethod(A, B, tA, tB);
		}
		double avgTimeOld = (System.nanoTime() - startTimeOld) / 1e6 / REP;

		// Measure New Kernel
		double startTimeNew = System.nanoTime();
		for(int i = 0; i < REP; i++) {
			runNewKernel(A, B, C, tA, tB);
		}
		double avgTimeNew = (System.nanoTime() - startTimeNew) / 1e6 / REP;

		// print results comparison
		System.out.printf("Old Method: %.3f ms | New Kernel: %.3f ms%n", avgTimeOld, avgTimeNew);
	}

	private void runNewKernel(MatrixBlock A, MatrixBlock B, MatrixBlock C, boolean tA, boolean tB) {
		C.reset();
		LibMatrixMult.matrixMultDenseDenseMM(A.getDenseBlock(), B.getDenseBlock(), C.getDenseBlock(), tA, tB, m, k, 0, m, 0, n);
	}

	private void runOldMethod(MatrixBlock A, MatrixBlock B, boolean tA, boolean tB) {
		// do transpose if needed
		MatrixBlock A_in = tA ? LibMatrixReorg.transpose(A) : A;
		MatrixBlock B_in = tB ? LibMatrixReorg.transpose(B) : B;

		MatrixBlock C = new MatrixBlock(m, n, false);
		C.allocateDenseBlock();

		LibMatrixMult.matrixMultDenseDenseMM(A_in.getDenseBlock(), B_in.getDenseBlock(), C.getDenseBlock(), false,
			false, m, k, 0, m, 0, n);
	}
}
