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

package org.apache.sysds.test.functions.builtin.part1;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.apache.sysds.runtime.matrix.data.MatrixValue;

import java.util.HashMap;

public class BuiltinKNNGraphTest extends AutomatedTestBase {
	private final static String TEST_NAME = "knnGraph";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinKNNGraphTest.class.getSimpleName() + "/";

	private final static String OUTPUT_NAME_KNN_GRAPH = "KNN_GRAPH";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME));
	}

	@Test
	public void basicTest() {
		double[][] X = {{1, 0}, {2, 2}, {2, 2.5}, {10, 10}, {15, 15}};
		double[][] refMatrix = {{0., 1., 1., 0., 0.}, {1., 0., 1., 0., 0.}, {1., 1., 0., 0., 0.}, {0., 0., 1., 0., 1.},
			{0., 0., 1., 1., 0.}};
		HashMap<MatrixValue.CellIndex, Double> refHMMatrix = TestUtils.convert2DDoubleArrayToHashMap(refMatrix);

		runKNNGraphTest(ExecMode.SINGLE_NODE, 2, X, refHMMatrix);
	}

	private void runKNNGraphTest(ExecMode exec_mode, Integer k, double[][] X,
		HashMap<MatrixValue.CellIndex, Double> refHMMatrix) {
		ExecMode platform_old = setExecMode(exec_mode);

		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;

		// create Test Input
		writeInputMatrixWithMTD("X", X, true);

		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-stats", "-nvargs", "in_X=" + input("X"), "in_k=" + Integer.toString(k),
			"out_G=" + output(OUTPUT_NAME_KNN_GRAPH)};

		// execute tests
		runTest(true, false, null, -1);

		// read result
		HashMap<MatrixValue.CellIndex, Double> resultGraph = readDMLMatrixFromOutputDir(OUTPUT_NAME_KNN_GRAPH);

		// compare result with reference
		TestUtils.compareMatrices(resultGraph, refHMMatrix, 0, "ResGraph", "RefGraph");

		// restore execution mode
		setExecMode(platform_old);
	}

}
