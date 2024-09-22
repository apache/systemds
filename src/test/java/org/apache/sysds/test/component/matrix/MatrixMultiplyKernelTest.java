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

package org.apache.sysds.test.component.matrix;

import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;
import org.junit.Test;

public class MatrixMultiplyKernelTest {
	private static final int MIN_PAR = (int)LibMatrixMult.PAR_MINFLOP_THRESHOLD1+1;
	private static final int MIN_PAR_SQRT = (int)Math.sqrt(MIN_PAR);
	
	// dense-dense kernels
	
	@Test
	public void testDenseDenseDotProduct() {
		testMatrixMultiply(1, MIN_PAR, 1, 1, 1);
	}
	
	@Test
	public void testDenseDenseOuterProduct() {
		testMatrixMultiply(MIN_PAR_SQRT, 1, MIN_PAR_SQRT, 1, 1);
	}
	
	@Test
	public void testDenseDenseVectorScalar() {
		testMatrixMultiply(MIN_PAR, 1, 1, 1, 1);
	}
	
	@Test
	public void testDenseDenseMatrixSmallVector() {
		testMatrixMultiply(MIN_PAR, 16, 1, 1, 1);
	}
	
	@Test //parallelization over rows in lhs
	public void testDenseDenseMatrixLargeVector() {
		testMatrixMultiply(4000, 3000, 1, 1, 1);
	}
	
	@Test //parallelization over rows in rhs
	public void testDenseDenseMatrixLargeVectorPm2() {
		testMatrixMultiply(16, MIN_PAR, 1, 1, 1);
	}
	
	@Test
	public void testDenseDenseVectorMatrix() {
		testMatrixMultiply(1, MIN_PAR, 16, 1, 1);
	}
	
	@Test
	public void testDenseDenseSmallMatrixMatrix() {
		testMatrixMultiply(16, MIN_PAR, 16, 1, 1);
	}
	
	@Test
	public void testDenseDenseMatrixSmallMatrix() {
		testMatrixMultiply(MIN_PAR_SQRT, MIN_PAR_SQRT, 4, 1, 1);
	}
	
	@Test
	public void testDenseDenseMatrixMatrix() {
		testMatrixMultiply(MIN_PAR_SQRT, MIN_PAR_SQRT, MIN_PAR_SQRT, 1, 1);
	}
	
	// dense-sparse kernels
	
	@Test
	public void testDenseSparseVectorMatrix() {
		testMatrixMultiply(1, MIN_PAR, 12, 1, 0.1);
	}
	
	@Test
	public void testDenseSparseMatrixMatrix() {
		testMatrixMultiply(MIN_PAR_SQRT, MIN_PAR_SQRT, MIN_PAR_SQRT, 1, 0.1);
	}
	
	// sparse-dense kernels
	
	@Test
	public void testSparseDenseDotProduct() {
		testMatrixMultiply(1, MIN_PAR, 1, 0.1, 1);
	}
	
	@Test
	public void testSparseDenseMatrixSmallVector() {
		testMatrixMultiply(MIN_PAR_SQRT, 1024, 1, 0.1, 1);
	}
	
	@Test // see SYSTEMDS-3769
	public void testSparseDenseMatrixLargeVector() {
		testMatrixMultiply(13, 8000, 1, 0.1, 1);
	}
	
	@Test
	public void testSparseDenseVectorMatrix() {
		testMatrixMultiply(1, MIN_PAR_SQRT, MIN_PAR_SQRT, 0.1, 1);
	}
	
	@Test
	public void testSparseDenseSmallMatrixMatrix() {
		testMatrixMultiply(9, MIN_PAR_SQRT, MIN_PAR_SQRT, 0.1, 1);
	}
	
	@Test
	public void testSparseDenseMatrixSmallMatrix() {
		testMatrixMultiply(MIN_PAR_SQRT, MIN_PAR_SQRT, 9, 0.1, 1);
	}
	
	@Test
	public void testSparseDenseMatrixMatrix() {
		testMatrixMultiply(MIN_PAR_SQRT, MIN_PAR_SQRT, MIN_PAR_SQRT, 0.1, 1);
	}
	
	// sparse-sparse kernels
	@Test
	public void testSparseSparseVectorMatrix() {
		testMatrixMultiply(1, MIN_PAR_SQRT, MIN_PAR_SQRT, 0.1, 0.1);
	}
	
	@Test //w/ sparse output
	public void testSparseSparseSparseMatrixMatrix() {
		testMatrixMultiply(MIN_PAR_SQRT, 2, MIN_PAR_SQRT, 0.1, 0.1);
	}
	
	@Test
	public void testSparseSparseMatrixSmallMatrix() {
		testMatrixMultiply(MIN_PAR_SQRT, 15, 1000, 0.1, 0.1);
	}
	
	@Test
	public void testSparseSparseMatrixMatrix() {
		testMatrixMultiply(MIN_PAR_SQRT, MIN_PAR_SQRT, MIN_PAR_SQRT, 0.1, 0.1);
	}
	
	private void testMatrixMultiply(int n, int m, int l, double sp1, double sp2) {
		MatrixBlock mb1 = MatrixBlock.randOperations(n, m, sp1, 0, 0.1, "uniform", 3);
		MatrixBlock mb2 = MatrixBlock.randOperations(m, l, sp2, 0, 0.1, "uniform", 7);
		//run single- and multi-threaded kernels and compare
		MatrixBlock ret1 = LibMatrixMult.matrixMult(mb1, mb2);
		MatrixBlock ret2 = LibMatrixMult.matrixMult(mb1, mb2,
			InfrastructureAnalyzer.getLocalParallelism());
		TestUtils.compareMatrices(ret1, ret2, 1e-8);
	}
}
