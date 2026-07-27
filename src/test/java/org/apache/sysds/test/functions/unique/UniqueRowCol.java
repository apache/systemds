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

public class UniqueRowCol extends UniqueBase {
	private final static String TEST_NAME = "uniqueRowCol";
	private final static String TEST_DIR = "functions/unique/";
	private static final String TEST_CLASS_DIR = TEST_DIR + UniqueRowCol.class.getSimpleName() + "/";


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
	public void testBaseCase1CP() {
		double[][] inputMatrix = {{0}};
		double[][] expectedMatrix = {{0}};
		uniqueTest(inputMatrix, expectedMatrix, Types.ExecType.CP, 0.0);
	}

	@Test
	public void testBaseCase2CP() {
		double[][] inputMatrix = {{1}};
		double[][] expectedMatrix = {{1}};
		uniqueTest(inputMatrix, expectedMatrix, Types.ExecType.CP, 0.0);
	}

	@Test
	public void testSkinnySmallCP() {
		double[][] inputMatrix = {{1},{1},{6},{9},{4},{2},{0},{9},{0},{0},{4},{4}};
		double[][] expectedMatrix = {{1},{6},{9},{4},{2},{0}};
		uniqueTest(inputMatrix, expectedMatrix, Types.ExecType.CP, 0.0);
	}

	@Test
	public void testWideSmallCP() {
		double[][] inputMatrix = {{1},{1},{6},{9},{4},{2},{0},{9},{0},{0},{4},{4}};
		double[][] expectedMatrix = {{1,6,9,4,2,0}};
		uniqueTest(inputMatrix, expectedMatrix, Types.ExecType.CP, 0.0);
	}

	/**
	 * Large enough to take the multi-threaded path. The result is a single column, so the comparison is unaffected by
	 * the order in which the merged hash set is iterated.
	 */
	@Test
	public void testMultiThreadedCP() {
		uniqueTest(cyclicValues(500, 40, 37), expectedCyclicValues(37), Types.ExecType.CP, 0.0);
	}

	/**
	 * Same input, but with a local memory budget that is too small to hold the thread-local sets and the merged set at
	 * once. This selects the batched path from an end-to-end script run.
	 */
	@Test
	public void testBatchedCP() {
		uniqueTestConstrainedMemory(cyclicValues(500, 40, 37), expectedCyclicValues(37), Types.ExecType.CP, 0.0,
			8 * 1024 * 1024);
	}

	private static double[][] cyclicValues(int rlen, int clen, int distinct) {
		double[][] ret = new double[rlen][clen];
		for(int i = 0; i < rlen; i++)
			for(int j = 0; j < clen; j++)
				ret[i][j] = ((long) i * clen + j) % distinct;
		return ret;
	}

	private static double[][] expectedCyclicValues(int distinct) {
		double[][] ret = new double[distinct][1];
		for(int i = 0; i < distinct; i++)
			ret[i][0] = i;
		return ret;
	}
}
