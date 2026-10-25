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
		uniqueTest(inputMatrix, expectedMatrix, Types.ExecType.CP, 0.0);
	}

	@Test
	public void testNoDuplicatesCP() {
		double[][] inputMatrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
		double[][] expectedMatrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
		uniqueTest(inputMatrix, expectedMatrix, Types.ExecType.CP, 0.0);
	}

	/**
	 * The cases above stay below the 16,384 cell threshold of the multi-threaded implementation and therefore only
	 * exercise the sequential path. This one is above it, so the column partitions are actually deduplicated in
	 * parallel.
	 */
	@Test
	public void testMultiThreadedCP() {
		uniqueTest(constantColumns(64, 400), expectedConstantColumns(400), Types.ExecType.CP, 0.0);
	}

	/**
	 * Selects the batched path: with 8 threads and a 4 MB local memory budget, one live column set per thread no longer
	 * fits, while the budget still holds several columns worth of values so batching remains applicable.
	 */
	@Test
	public void testBatchedCP() {
		uniqueTestConstrainedMemory(constantColumns(4096, 8), expectedConstantColumns(8), Types.ExecType.CP, 0.0,
			4 * 1024 * 1024, 8);
	}

	/**
	 * Same as the multi-threaded case, but with an input that is read in sparse format, so unique() reaches the values
	 * through the sparse block lookup rather than the dense array. Only every eighth column is populated, and every
	 * column still holds a single distinct value, either its filler or zero.
	 */
	@Test
	public void testSparseMultiThreadedCP() {
		int rlen = 64, clen = 400;
		double[][] inputMatrix = new double[rlen][clen];
		double[][] expectedMatrix = new double[1][clen];
		for(int j = 0; j < clen; j++) {
			double value = (j % 8 == 0) ? j + 1 : 0;
			for(int i = 0; i < rlen; i++)
				inputMatrix[i][j] = value;
			expectedMatrix[0][j] = value;
		}
		uniqueTest(inputMatrix, expectedMatrix, Types.ExecType.CP, 0.0);
	}

	/** Every column holds a single distinct value, so the expected result is one row. */
	private static double[][] constantColumns(int rlen, int clen) {
		double[][] ret = new double[rlen][clen];
		for(int j = 0; j < clen; j++)
			for(int i = 0; i < rlen; i++)
				ret[i][j] = j + 1;
		return ret;
	}

	private static double[][] expectedConstantColumns(int clen) {
		double[][] ret = new double[1][clen];
		for(int j = 0; j < clen; j++)
			ret[0][j] = j + 1;
		return ret;
	}
}
