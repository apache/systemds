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
	public void testRaGroupbyTest1() {
		testRaGroupbyTest("nested-loop");
	}

	@Test
	public void testRaGroupbyTest2() {
		testRaGroupbyTest("permutation-matrix");
	}

	@Test
	public void testRaGroupbyTestwithDifferentColumn1() {
		testRaGroupbyTestwithDifferentColumn("nested-loop");
	}

	@Test
	public void testRaGroupbyTestwithDifferentColumn2() {
		testRaGroupbyTestwithDifferentColumn("permutation-matrix");
	}

	@Test
	public void testRaGroupbyTestwithNoGroup1() {
		testRaGroupbyTestwithNoGroup("nested-loop");
	}

	@Test
	public void testRaGroupbyTestwithNoGroup2() {
		testRaGroupbyTestwithNoGroup("permutation-matrix");
	}

	@Test
	public void testRaGroupbyTestwithOneGroup1() {
		testRaGroupbyTestwithOneGroup("nested-loop");
	}

	@Test
	public void testRaGroupbyTestwithOneGroup2() {
		testRaGroupbyTestwithOneGroup("permutation-matrix");
	}

	@Test
	public void testRaGroupbyTestwithMultipleGroupRows1() {
		testRaGroupbyTestwithMultipleGroupRows("nested-loop");
	}

	@Test
	public void testRaGroupbyTestwithMultipleGroupRows2() {
		testRaGroupbyTestwithMultipleGroupRows("permutation-matrix");
	}

	public void testRaGroupbyTest(String method) {
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

		runRaGroupbyTest(X, select_col, Y, method);
	}

	public void testRaGroupbyTestwithDifferentColumn(String method) {
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

		runRaGroupbyTest(X, select_col, Y, method);
	}

	public void testRaGroupbyTestwithNoGroup(String method) {
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

		runRaGroupbyTest(X, select_col, Y, method);
	}

	public void testRaGroupbyTestwithOneGroup(String method) {
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

		runRaGroupbyTest(X, select_col, Y, method);
	}

	public void testRaGroupbyTestwithMultipleGroupRows(String method) {
		// Test case with multiple groups having different numbers of rows
		// 10 rows x 5 columns, grouping by column 2
		// Groups: 1->3 rows, 2->2 rows, 3->2 rows, 4->2 rows, 5->1 row
		double[][] X = {
				{1, 1, 11, 12, 13},
				{1, 2, 21, 22, 23},
				{1, 3, 31, 32, 33},
				{1, 4, 41, 42, 43},
				{2, 1, 14, 15, 16},
				{2, 2, 24, 25, 26},
				{2, 3, 34, 35, 36},
				{2, 4, 44, 45, 46},
				{2, 5, 54, 55, 56},
				{3, 1, 17, 18, 19}};
		int select_col = 2;

		// Expected output matrix (grouping by column 2, removing column 2)
		// Note: Groups are ordered as they appear in the unique() function output
		// Group 1: 3 rows -> [1,11,12,13], [2,14,15,16], [3,17,18,19]
		// Group 2: 2 rows -> [1,21,22,23], [2,24,25,26]
		// Group 4: 2 rows -> [1,41,42,43], [2,44,45,46]
		// Group 5: 1 row  -> [2,54,55,56]
		// Group 3: 2 rows -> [1,31,32,33], [2,34,35,36]
		double[][] Y = {
				{1, 1, 11, 12, 13, 2, 14, 15, 16, 3, 17, 18, 19},
				{2, 1, 21, 22, 23, 2, 24, 25, 26, 0, 0, 0, 0},
				{4, 1, 41, 42, 43, 2, 44, 45, 46, 0, 0, 0, 0},
				{5, 2, 54, 55, 56, 0, 0, 0, 0, 0, 0, 0, 0},
				{3, 1, 31, 32, 33, 2, 34, 35, 36, 0, 0, 0, 0}
		};

		runRaGroupbyTest(X, select_col, Y, method);
	}

	private void runRaGroupbyTest(double [][] X, int col, double [][] Y, String method)
	{
		ExecMode platformOld = setExecMode(ExecMode.SINGLE_NODE);
		
		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;

			fullDMLScriptName = HOME + TEST_NAME + ".dml";

			//test groupby methods
			programArgs = new String[]{"-stats", "-args",
				input("X"), String.valueOf(col), method, output("result") };

			//fullRScriptName = HOME + TEST_NAME + ".R";
			//rCmd = "Rscript" + " " + fullRScriptName + " "
			//	+ inputDir() + " " + col + " "  + expectedDir();

			writeInputMatrixWithMTD("X", X, true);
			//writeExpectedMatrix("result", Y);

			// run dmlScript and RScript
			runTest(true, false, null, -1);
			//runRScript(true);

			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("result");
			HashMap<CellIndex, Double> expectedOutput = TestUtils.convert2DDoubleArrayToHashMap(Y);
			//HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("result");
			TestUtils.compareMatrices(dmlfile, expectedOutput, eps, "Stat-DML", "Expected");
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
