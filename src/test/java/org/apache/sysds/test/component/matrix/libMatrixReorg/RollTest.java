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

package org.apache.sysds.test.component.matrix.libMatrixReorg;

import org.apache.sysds.runtime.functionobjects.IndexFunction;
import org.apache.sysds.runtime.functionobjects.RollIndex;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import static org.junit.Assert.fail;

/**
 * Test class for the roll function in MatrixBlock.
 * <p>
 * This test verifies that the roll function produces identical results
 * when applied to both sparse and dense representations of the same matrix.
 */
@RunWith(Parameterized.class)
public class RollTest {
	private final int shift;

	// Input matrices
	private MatrixBlock inputSparse;
	private MatrixBlock inputDense;

	/**
	 * Constructor for parameterized test cases.
	 *
	 * @param rows	 Number of rows in the test matrix.
	 * @param cols	 Number of columns in the test matrix.
	 * @param sparsity Sparsity level of the test matrix (0.0 to 1.0).
	 * @param shift	Shift value for the roll operation.
	 */
	public RollTest(int rows, int cols, double sparsity, int shift) {
		this.shift = shift;

		// Generate a MatrixBlock with the given parameters
		inputSparse = TestUtils.generateTestMatrixBlock(rows, cols, 0, 10, sparsity, 1);
		inputSparse.recomputeNonZeros();

		inputDense = new MatrixBlock(rows, cols, false); // false indicates dense
		inputDense.copy(inputSparse, false); // Copy without maintaining sparsity
		inputDense.recomputeNonZeros();
	}

	/**
	 * Defines the parameters for the test cases.
	 * Each Object[] contains {rows, cols, sparsity, shift}.
	 *
	 * @return Collection of test parameters.
	 */
	@Parameters(name = "Rows: {0}, Cols: {1}, Sparsity: {2}, Shift: {3}")
	public static Collection<Object[]> data() {
		List<Object[]> tests = new ArrayList<>();

		// Define various sizes, sparsity levels, and shift values
		int[] rows = {1, 19, 1001, 2017};
		int[] cols = {1, 17, 1001, 2017};
		double[] sparsities = {0.01, 0.1, 0.7, 1.0};
		int[] shifts = {0, 1, 5, 10, 15};

		// Generate all combinations of sizes, sparsities, and shifts
		for (int row : rows) {
			for (int col : cols) {
				for (double sparsity : sparsities) {
					for (int shift : shifts) {
						tests.add(new Object[]{row, col, sparsity, shift});
					}
				}
			}
		}
		return tests;
	}

	/**
	 * The actual test method that performs the roll operation on both
	 * sparse and dense matrices and compares the results.
	 * This test will execute the single threaded operation
	 */
	@Test
	public void testSingleThreadedOperation() {
		int numThreads = 1;
		compareDenseAndSparseRepresentation(numThreads);
	}


	/**
	 * The actual test method that performs the roll operation on both
	 * sparse and dense matrices and compares the results.
	 * This test will execute the multithreaded operation
	 */
	@Test
	public void testMultiThreadedOperation() {
		// number of threads should be at least two to invoke multithreaded operation
		int cores = Runtime.getRuntime().availableProcessors();
		int numThreads = Math.max(2, cores);

		compareDenseAndSparseRepresentation(numThreads);
	}

	private void compareDenseAndSparseRepresentation(int numThreads) {
		try {
			IndexFunction op = new RollIndex(shift);
			MatrixBlock outputDense = inputDense.reorgOperations(
					new ReorgOperator(op, numThreads), new MatrixBlock(), 0, 0, 0);
			MatrixBlock outputSparse = inputSparse.reorgOperations(
					new ReorgOperator(op, numThreads), new MatrixBlock(), 0, 0, 0);
			outputSparse.sparseToDense();

			// Compare the dense representations of both outputs
			TestUtils.compareMatrices(outputSparse, outputDense, 1e-9,
					"Compare Sparse and Dense Roll Results");

		} catch (Exception e) {
			e.printStackTrace();
			fail("Exception occurred during roll function test: " + e.getMessage());
		}
	}
}
