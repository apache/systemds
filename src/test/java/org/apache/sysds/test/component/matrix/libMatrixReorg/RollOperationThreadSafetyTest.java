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

import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;

import org.apache.sysds.runtime.functionobjects.IndexFunction;
import org.apache.sysds.runtime.functionobjects.RollIndex;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.stats.Timing;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(Parameterized.class)
public class RollOperationThreadSafetyTest {

	private static final int MIN_ROWS = 2017;
	private static final int MIN_COLS = 1001;
	private static final int MIN_SHIFT = -50;
	private static final int MAX_SHIFT = 1022;
	private static final int NUM_TESTS = 100;
	private static final double TEST_SPARSITY = 0.01;
	private final int rows;
	private final int cols;
	private final int shift;
	private final MatrixBlock inputDense;
	private final MatrixBlock inputSparse;

	public RollOperationThreadSafetyTest(int rows, int cols, int shift) {
		this.rows = rows;
		this.cols = cols;
		this.shift = shift;

		MatrixBlock tempInput = TestUtils.generateTestMatrixBlock(rows, cols, -100, 100, TEST_SPARSITY, 42);

		this.inputSparse = tempInput;

		this.inputDense = new MatrixBlock(rows, cols, false);
		this.inputDense.copy(tempInput, false);
		this.inputDense.recomputeNonZeros();
	}

	/**
	 * Defines the parameters for the test cases (Random Rows, Random Cols, Random Shift).
	 *
	 * @return Collection of test parameters.
	 */
	@Parameters(name = "Case {index}: Size={0}x{1}, Shift={2}")
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		Random rand = new Random(42);

		for(int i = 0; i < NUM_TESTS; i++) {
			// Generate random dimensions (adding random buffer to the minimums)
			int r = MIN_ROWS + rand.nextInt(500);
			int c = MIN_COLS + rand.nextInt(500);

			int s = rand.nextInt((MAX_SHIFT - MIN_SHIFT) + 1) + MIN_SHIFT;

			tests.add(new Object[] {r, c, s});
		}
		return tests;
	}

	@Test
	public void denseRollOperationSingleAndMultiThreadedShouldReturnSameResult() {
		int numThreads = getNumThreads();

		// Single-threaded timing
		Timing tSingle = new Timing(true);
		MatrixBlock outSingle = rollOperation(inputDense, 1);
		double timeSingle = tSingle.stop();

		// Multithreaded timing
		Timing tMulti = new Timing(true);
		MatrixBlock outMulti = rollOperation(inputDense, numThreads);
		double timeMulti = tMulti.stop();

		logTiming("Dense", numThreads, timeSingle, timeMulti);

		TestUtils.compareMatrices(outSingle, outMulti, 1e-12,
			"Dense Mismatch (numThreads=1 vs numThreads>1) for Size=" + rows + "x" + cols + " Shift=" + shift);
	}

	@Test
	public void sparseRollOperationSingleAndMultiThreadedShouldReturnSameResult() {
		int numThreads = getNumThreads();

		// Single-threaded timing
		Timing tSingle = new Timing(true);
		MatrixBlock outSingle = rollOperation(inputSparse, 1);
		double timeSingle = tSingle.stop();

		// Multithreaded timing
		Timing tMulti = new Timing(true);
		MatrixBlock outMulti = rollOperation(inputSparse, numThreads);
		double timeMulti = tMulti.stop();

		logTiming("Sparse", numThreads, timeSingle, timeMulti);

		TestUtils.compareMatrices(outSingle, outMulti, 1e-12,
			"Sparse Mismatch (numThreads=1 vs numThreads>1) for Size=" + rows + "x" + cols + " Shift=" + shift);
	}

	private MatrixBlock rollOperation(MatrixBlock inBlock, int numThreads) {
		IndexFunction op = new RollIndex(shift);
		ReorgOperator reorgOperator = new ReorgOperator(op, numThreads);

		MatrixBlock outBlock = new MatrixBlock(rows, cols, inBlock.isInSparseFormat());

		return inBlock.reorgOperations(reorgOperator, outBlock, 0, 0, 0);
	}

	private static int getNumThreads() {
		// number of threads should be at least two to invoke multithreaded operation
		int cores = Runtime.getRuntime().availableProcessors();
		return Math.max(2, cores);
	}

	private void logTiming(String type, int numThreads, double timeSingle, double timeMulti) {
		double speedup = timeSingle / timeMulti;

		System.out.println("\n--- " + type + " Roll Operation Timing for " + rows + "x" + cols + ", Shift=" + shift + " ---");
		System.out.printf("Single-threaded (1 core) took: %.3f ms\n", timeSingle);
		System.out.printf("Multithreaded (%d cores) took: %.3f ms\n", numThreads, timeMulti);
		System.out.printf("Speedup: %.2f\n", speedup);
		System.out.println("--------------------------------------------------------------------------------");
	}
}
