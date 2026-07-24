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

	@Test
	public void testPerfDenseDense() throws Exception {
		System.out.println("==============================================================");
		System.out.println("BENCHMARK: Dense-Dense Transposed Kernels");
		System.out.println("==============================================================");
		runBenchmarkSuite(1.0, 1.0);
	}

	@Test
	public void testPerfSparseDense() throws Exception {
		System.out.println("==============================================================");
		System.out.println("BENCHMARK: Sparse-Dense Transposed Kernels");
		System.out.println("==============================================================");
		for (double sp : new double[]{0.01, 0.05}) {
			System.out.println(">>> Sparsity A: " + sp);
			runBenchmarkSuite(sp, 1.0);
		}
	}

	@Test
	public void testPerfDenseSparse() throws Exception {
		System.out.println("==============================================================");
		System.out.println("BENCHMARK: Dense-Sparse Transposed Kernels");
		System.out.println("==============================================================");
		for (double sp : new double[]{0.01, 0.05}) {
			System.out.println(">>> Sparsity B: " + sp);
			runBenchmarkSuite(1.0, sp);
		}
	}

	private void runBenchmarkSuite(double spA, double spB) throws Exception {
		boolean[][] transConfigs = {
			{true, false},
			{false, true},
			{true, true}
		};

		int[] sizes = {200, 500};

		for (boolean[] tc : transConfigs) {
			boolean tA = tc[0];
			boolean tB = tc[1];
			String exprName = (tA && tB) ? "t(A) %*% t(B)" : tA ? "t(A) %*% B" : "A %*% t(B)";

			System.out.printf("--- Case: C = %s ---%n", exprName);

			for (int size : sizes) {
				System.out.printf("Size: %d%n", size);
				runTest(spA, spB, tA, tB, size, size, size);
			}
			System.out.println();
		}
	}

	private void runTest(double spA, double spB, boolean tA, boolean tB, int m, int n, int k) throws Exception {
		final int REP = 100;
		final int WARMUP = 50;

		int rowsA = tA ? k : m;
		int colsA = tA ? m : k;
		int rowsB = tB ? n : k;
		int colsB = tB ? k : n;

		MatrixBlock ma = generateInput(rowsA, colsA, spA, 7);
		MatrixBlock mb = generateInput(rowsB, colsB, spB, 3);
		MatrixBlock mc = new MatrixBlock(m, n, false);
		mc.allocateDenseBlock();

		for (int i = 0; i < WARMUP; i++) {
			runOldMethod(ma, mb, tA, tB);
			runNewKernel(ma, mb, mc, tA, tB);
		}

		long startOld = System.nanoTime();
		for (int i = 0; i < REP; i++) {
			runOldMethod(ma, mb, tA, tB);
		}
		double avgOld = (System.nanoTime() - startOld) / 1e6 / REP;

		long startNew = System.nanoTime();
		for (int i = 0; i < REP; i++) {
			runNewKernel(ma, mb, mc, tA, tB);
		}
		double avgNew = (System.nanoTime() - startNew) / 1e6 / REP;

		System.out.printf("  Old Method: %.3f ms | New Kernel: %.3f ms | Speedup: %.2fx%n", 
			avgOld, avgNew, avgOld / Math.max(1e-9, avgNew));
	}

	private MatrixBlock generateInput(int rows, int cols, double sparsity, long seed) {
		MatrixBlock mb = MatrixBlock.randOperations(rows, cols, sparsity, -1, 1, "uniform", seed);
		mb.examSparsity();
		if (sparsity < 1.0) {
			if (!mb.isInSparseFormat())
				mb.denseToSparse(true);
			if (mb.getSparseBlock() == null)
				mb.allocateSparseRowsBlock();
		}
		return mb;
	}

	private void runNewKernel(MatrixBlock ma, MatrixBlock mb, MatrixBlock mc, boolean tA, boolean tB) throws Exception {
		mc.reset();
		mc.allocateDenseBlock();

		boolean sparseA = ma.isInSparseFormat();
		boolean sparseB = mb.isInSparseFormat();

		int m = tA ? ma.getNumColumns() : ma.getNumRows();
		int n = tB ? mb.getNumRows() : mb.getNumColumns();
		int k = tA ? ma.getNumRows() : ma.getNumColumns();

		if (!sparseA && !sparseB) {
			LibMatrixMult.matrixMultDenseDenseMM(
				ma.getDenseBlock(),
				mb.getDenseBlock(),
				mc.getDenseBlock(),
				tA, tB, n, k, 0, m, 0, n
			);
		}
		else if (sparseA && !sparseB) {
			int cd = tB ? mb.getNumColumns() : mb.getNumRows();
			long xsp = (long) m * cd / Math.max(1L, ma.getNonZeros());
			LibMatrixMult.matrixMultSparseDenseMM(
				ma.getSparseBlock(),
				mb.getDenseBlock(),
				mc.getDenseBlock(),
				tA, tB, n, cd, xsp, 0, m
			);
		}
		else if (!sparseA && sparseB) {
			long xsp = (long) ma.getNumRows() * ma.getNumColumns() / Math.max(1L, ma.getNonZeros());
			LibMatrixMult.matrixMultDenseSparseMM(
				ma.getDenseBlock(),
				mb.getSparseBlock(),
				mc.getDenseBlock(),
				tA, tB, n, k, xsp, 0, m
			);
		}
		mc.recomputeNonZeros();
	}

	private void runOldMethod(MatrixBlock ma, MatrixBlock mb, boolean tA, boolean tB) throws Exception {
		MatrixBlock A_in = tA ? LibMatrixReorg.transpose(ma) : ma;
		MatrixBlock B_in = tB ? LibMatrixReorg.transpose(mb) : mb;

		boolean sparseA = A_in.isInSparseFormat();
		boolean sparseB = B_in.isInSparseFormat();

		int m = A_in.getNumRows();
		int n = B_in.getNumColumns();
		int k = A_in.getNumColumns();

		MatrixBlock mc = new MatrixBlock(m, n, false);
		mc.allocateDenseBlock();

		if (!sparseA && !sparseB) {
			LibMatrixMult.matrixMultDenseDenseMM(
				A_in.getDenseBlock(),
				B_in.getDenseBlock(),
				mc.getDenseBlock(),
				false, false, n, k, 0, m, 0, n
			);
		}
		else if (sparseA && !sparseB) {
			long xsp = (long) m * k / Math.max(1L, A_in.getNonZeros());
			LibMatrixMult.matrixMultSparseDenseMM(
				A_in.getSparseBlock(),
				B_in.getDenseBlock(),
				mc.getDenseBlock(),
				false, false, n, k, xsp, 0, m
			);
		}
		else if (!sparseA && sparseB) {
			long xsp = (long) A_in.getNumRows() * A_in.getNumColumns() / Math.max(1L, A_in.getNonZeros());
			LibMatrixMult.matrixMultDenseSparseMM(
				A_in.getDenseBlock(),
				B_in.getSparseBlock(),
				mc.getDenseBlock(),
				false, false, n, k, xsp, 0, m
			);
		}
	}
}
