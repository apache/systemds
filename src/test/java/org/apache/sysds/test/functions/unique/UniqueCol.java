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

package org.apache.sysds.test.functions.unique;

import org.apache.sysds.common.Types;
import org.junit.Test;

public class UniqueCol extends UniqueBase {
	private final static String TEST_NAME = "uniqueCol";
	private final static String TEST_DIR = "functions/unique/";
	private static final String TEST_CLASS_DIR = TEST_DIR + UniqueCol.class.getSimpleName() + "/";

	@Override
	protected String getTestName() {
		return TEST_NAME;
	}

	@Override
	protected String getTestDir() {
		return TEST_DIR;
	}

	@Override
	protected String getTestClassDir() {
		return TEST_CLASS_DIR;
	}

	@Test
	public void testBaseCaseCP() {
		double[][] inputMatrix = {{0}};
		double[][] expectedMatrix = {{0}};
		uniqueTest(inputMatrix, expectedMatrix, Types.ExecType.CP, 0.0);
	}

	@Test
	public void testSingleColumnCP() {
		double[][] inputMatrix = {{1}, {1}, {6}, {9}, {4}, {2}, {0}, {9}, {0}, {0}, {4}, {4}};
		double[][] expectedMatrix = {{1}, {6}, {9}, {4}, {2}, {0}};
		uniqueTest(inputMatrix, expectedMatrix, Types.ExecType.CP, 0.0);
	}

	@Test
	public void testConstantColumnsCP() {
		// every column holds a single distinct value, so the result has one row
		double[][] inputMatrix = {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};
		double[][] expectedMatrix = {{1, 2, 3}};
		uniqueTestOrdered(inputMatrix, expectedMatrix, Types.ExecType.CP, 0.0);
	}

	@Test
	public void testNoDuplicatesCP() {
		double[][] inputMatrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
		double[][] expectedMatrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
		uniqueTest(inputMatrix, expectedMatrix, Types.ExecType.CP, 0.0);
	}

	/**
	 * Large enough to take the multi-threaded path. Every column holds a single distinct value, so the expected result
	 * is one row and independent of any hash set iteration order.
	 */
	@Test
	public void testMultiThreadedCP() {
		int rlen = 64, clen = 400; // 25,600 cells, above the multi-threading threshold
		double[][] inputMatrix = new double[rlen][clen];
		double[][] expectedMatrix = new double[1][clen];
		for(int j = 0; j < clen; j++) {
			for(int i = 0; i < rlen; i++)
				inputMatrix[i][j] = j;
			expectedMatrix[0][j] = j;
		}
		uniqueTestOrdered(inputMatrix, expectedMatrix, Types.ExecType.CP, 0.0);
	}

	/**
	 * Same input under a heavily reduced local memory budget. Column-wise workers reuse a single set that is cleared
	 * per column, so only one live set per thread is charged and the parallel path stays applicable; this guards
	 * against needlessly falling back to batched or sequential execution.
	 */
	@Test
	public void testReducedMemoryBudgetCP() {
		int rlen = 64, clen = 400;
		double[][] inputMatrix = new double[rlen][clen];
		double[][] expectedMatrix = new double[1][clen];
		for(int j = 0; j < clen; j++) {
			for(int i = 0; i < rlen; i++)
				inputMatrix[i][j] = j;
			expectedMatrix[0][j] = j;
		}
		uniqueTestConstrainedMemory(inputMatrix, expectedMatrix, Types.ExecType.CP, 0.0, 16 * 1024 * 1024);
	}
}
