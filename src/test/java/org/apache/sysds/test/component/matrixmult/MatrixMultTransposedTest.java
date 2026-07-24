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

import java.util.Random;

import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class MatrixMultTransposedTest {

	@Test
	public void testDenseDenseTransA() throws Exception {
		for(int i = 0; i < 10; i++)
			runRandomTest(false, false, true, false);
	}

	@Test
	public void testDenseDenseTransB() throws Exception {
		for(int i = 0; i < 10; i++)
			runRandomTest(false, false, false, true);
	}

	@Test
	public void testDenseDenseTransATransB() throws Exception {
		for(int i = 0; i < 10; i++)
			runRandomTest(false, false, true, true);
	}

	@Test
	public void testSparseDenseTransA() throws Exception {
		for(int i = 0; i < 10; i++)
			runRandomTest(true, false, true, false);
	}

	@Test
	public void testSparseDenseTransB() throws Exception {
		for(int i = 0; i < 10; i++)
			runRandomTest(true, false, false, true);
	}

	@Test
	public void testSparseDenseTransATransB() throws Exception {
		for(int i = 0; i < 10; i++)
			runRandomTest(true, false, true, true);
	}

	@Test
	public void testDenseSparseTransA() throws Exception {
		for(int i = 0; i < 10; i++)
			runRandomTest(false, true, true, false);
	}

	@Test
	public void testDenseSparseTransB() throws Exception {
		for(int i = 0; i < 10; i++)
			runRandomTest(false, true, false, true);
	}

	@Test
	public void testDenseSparseTransATransB() throws Exception {
		for(int i = 0; i < 10; i++)
			runRandomTest(false, true, true, true);
	}

	private void runRandomTest(boolean sparseA, boolean sparseB, boolean tA, boolean tB) throws Exception {
		Random rand = new Random();
		int m = rand.nextInt(300) + 1;
		int n = rand.nextInt(300) + 1;
		int k = rand.nextInt(300) + 1;

		double spA = sparseA ? 0.05 : 1.0;
		double spB = sparseB ? 0.05 : 1.0;

		runTest(spA, spB, tA, tB, m, n, k);
	}

	private void runTest(double spA, double spB, boolean tA, boolean tB, int m, int n, int k) throws Exception {
		int rowsA = tA ? k : m;
		int colsA = tA ? m : k;
		int rowsB = tB ? n : k;
		int colsB = tB ? k : n;

		MatrixBlock ma = generateInput(rowsA, colsA, spA, 7);
		MatrixBlock mb = generateInput(rowsB, colsB, spB, 3);
		MatrixBlock mc = new MatrixBlock(m, n, false);
		mc.allocateDenseBlock();

		runNewKernel(ma, mb, mc, tA, tB);

		MatrixBlock A_in = tA ? LibMatrixReorg.transpose(ma) : ma;
		MatrixBlock B_in = tB ? LibMatrixReorg.transpose(mb) : mb;
		MatrixBlock expected = LibMatrixMult.matrixMult(A_in, B_in);

		TestUtils.compareMatrices(expected, mc, 1e-8);
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
}
