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

package org.apache.sysds.test.functions.builtin;

import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class BuiltinALSTest extends AutomatedTestBase {

	private final static String TEST_NAME = "als";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinALSTest.class.getSimpleName() + "/";

	private final static double eps = 0.00001;
	private final static int rows = 6;
	private final static int cols = 6;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
	}

	@Test
	public void testALSCG() {
		runtestALS("alsCG");
	}
	
	@Test
	public void testALSDS() {
		runtestALS("alsDS");
	}
	
	@Test
	public void testALS() {
		runtestALS("als");
	}

	private void runtestALS(String alg) {
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		List<String> proArgs = new ArrayList<>();
		proArgs.add("-args");
		proArgs.add(input("X"));
		proArgs.add(alg);
		proArgs.add(output("U"));
		proArgs.add(output("V"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);

		double[][] X = {
			{7,1,1,2,2,1},{7,2,2,3,2,1},
			{7,3,1,4,1,1},{7,4,2,5,3,1},
			{7,5,3,6,5,1}, {7,6,5,1,4,1}};
		writeInputMatrixWithMTD("X", X, true);

		runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

		//compare expected results
		HashMap<MatrixValue.CellIndex, Double> matrixV = readDMLMatrixFromHDFS("V");
		HashMap<MatrixValue.CellIndex, Double> matrixU = readDMLMatrixFromHDFS("U");
		double[][] doubleV = TestUtils.convertHashMapToDoubleArray(matrixV);
		double[][] doubleU = TestUtils.convertHashMapToDoubleArray(matrixU);
		double[][] result = TestUtils.performMatrixMultiplication(doubleU, doubleV);

		TestUtils.compareMatrices(X, result, rows, cols, eps);
	}
}
