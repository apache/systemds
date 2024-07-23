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

package org.apache.sysds.test.functions.builtin.part2;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;

public class BuiltinRaGroupbyTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "raGroupby";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinRaGroupbyTest.class.getSimpleName() + "/";
	private final static double eps = 1e-8;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"result"}));
	}

	@Test
	public void testRaGroupbyTest() {
		//generate actual dataset and variables
		double[][] X = {
				{1, 2, 3},
				{4, 7, 8},
				{1, 3, 6},
				{4, 7, 8},
				{4, 8, 9}};
		int select_col = 1;

		// Expected output matrix
		double[][] Y = {
				{1, 2, 3, 3, 6, 0, 0},
				{4, 7, 8, 7, 8, 8, 9}
		};

		runRaGroupbyTest(X, select_col, Y);
	}

	@Test
	public void testRaGroupbyTestwithDifferentColumn() {
		//generate actual dataset and variables
		double[][] X = {
				{1, 2, 3},
				{4, 7, 8},
				{1, 3, 6},
				{4, 7, 8},
				{4, 8, 9}};
		int select_col = 2;

		// Expected output matrix
		double[][] Y = {
				{2, 1, 3, 0, 0},
				{8, 4, 9, 0, 0},
				{3, 1, 6, 0, 0},
				{7, 4, 8, 4, 8}
		};

		runRaGroupbyTest(X, select_col, Y);
	}

	@Test
	public void testRaGroupbyTestwithNoGroup() {
		// Test case with different values in select_col
		double[][] X = {
				{1, 1, 1},
				{2, 2, 2},
				{3, 1, 3},
				{4, 2, 4},
				{5, 1, 5}};
		int select_col = 3;

		// Expected output matrix
		double[][] Y = {
				{1, 1, 1},
				{2, 2, 2},
				{4, 4, 2},
				{5, 5, 1},
				{3, 3, 1}
		};

		runRaGroupbyTest(X, select_col, Y);
	}

	@Test
	public void testRaGroupbyTestwithOneGroup() {
		//generate actual dataset and variables
		double[][] X = {
				{1, 2, 3, 8, 2},
				{4, 7, 8, 8, 3},
				{1, 3, 6, 8, 4},
				{4, 7, 8, 8, 5},
				{4, 8, 9, 8, 6}};
		int select_col = 4;

		// Expected output matrix
		double[][] Y = {
				{8, 1, 2, 3, 2, 4, 7, 8, 3, 1, 3, 6, 4, 4, 7, 8, 5, 4, 8, 9, 6},
		};

		runRaGroupbyTest(X, select_col, Y);
	}

	private void runRaGroupbyTest(double [][] X, int col, double [][] Y)
	{
		ExecMode platformOld = setExecMode(ExecMode.SINGLE_NODE);
		
		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;

			fullDMLScriptName = HOME + TEST_NAME + ".dml";

			//test single groupby methods
			//programArgs = new String[]{"-stats", "-args",
			//	input("X"), String.valueOf(col), output("result") };

			//test all groupby methods
			programArgs = new String[]{"-stats", "-args",
					input("X"), String.valueOf(col), "nested-loop", "permutation-matrix", output("result1"), output("result2") };

			//fullRScriptName = HOME + TEST_NAME + ".R";
			//rCmd = "Rscript" + " " + fullRScriptName + " "
			//	+ inputDir() + " " + col + " "  + expectedDir();

			writeInputMatrixWithMTD("X", X, true);
			System.out.println(Arrays.deepToString(X));
			//writeExpectedMatrix("result", Y);

			// run dmlScript and RScript
			runTest(true, false, null, -1);
			//runRScript(true);

			//compare matrices
			HashMap<CellIndex, Double> dmlfile1 = readDMLMatrixFromOutputDir("result1");
			HashMap<CellIndex, Double> dmlfile2 = readDMLMatrixFromOutputDir("result2");
			HashMap<CellIndex, Double> expectedOutput = TestUtils.convert2DDoubleArrayToHashMap(Y);
			//HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("result");
			TestUtils.compareMatrices(dmlfile1, expectedOutput, eps, "Stat-DML", "Expected");
			TestUtils.compareMatrices(dmlfile2, expectedOutput, eps, "Stat-DML", "Expected");
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
