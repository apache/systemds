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

public class BuiltinRaJoinTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "raJoin";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinRaJoinTest.class.getSimpleName() + "/";
	private final static double eps = 1e-8;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"result"}));
	}

	// TODO test all join methods
	
	@Test
	public void testRaJoinTest() {
		//generate actual dataset and variables
		double[][] A = {
				{1, 2, 3},
				{4, 7, 8},
				{1, 3, 6},
				{4, 3, 5},
				{5, 8, 9}
		};
		double[][] B = {
				{1, 2, 9},
				{3, 7, 6},
				{2, 8, 5},
				{4, 7, 8},
				{4, 5, 10}
		};
		int colA = 1;
		int colB = 1;

		// Expected output matrix
		double[][] Y = {
				{1, 2, 3, 1, 2, 9},
				{1, 3, 6, 1, 2, 9},
				{4, 7, 8, 4, 7, 8},
				{4, 7, 8, 4, 5, 10},
				{4, 3, 5, 4, 7, 8},
				{4, 3, 5, 4, 5, 10},
		};
		runRaJoinTest(A, colA, B, colB, Y);
	}

	@Test
	public void testRaJoinTestwithDifferentColumn() {
		// Generate actual dataset and variables
		double[][] A = {
				{1, 5, 3},
				{2, 6, 8},
				{3, 7, 6},
				{4, 8, 5},
				{5, 9, 9}
		};
		double[][] B = {
				{1, 9, 2},
				{2, 8, 7},
				{3, 7, 6},
				{4, 5, 4},
				{5, 6, 1}
		};
		int colA = 2;
		int colB = 3;

		// Expected output matrix
		double[][] Y = {
				{2, 6, 8, 3, 7, 6},
				{3, 7, 6, 2, 8, 7}
		};
		runRaJoinTest(A, colA, B, colB, Y);
	}

	@Test
	public void testRaJoinTestwithDifferentColumn2() {
		// Generate actual dataset and variables
		double[][] A = {
				{1, 2, 3, 4, 5},
				{6, 7, 8, 9, 10},
				{11, 12, 13, 14, 8},
				{16, 17, 18, 19, 20},
				{21, 22, 23, 24, 25}
		};
		double[][] B = {
				{3, 5, 100},
				{1, 10, 200},
				{50, 25, 500}
		};
		int colA = 5; // Joining on the 5th column of A
		int colB = 2; // Joining on the 1st column of B

		// Expected output matrix
		double[][] Y = {
				{1, 2, 3, 4, 5, 3, 5, 100},
				{6, 7, 8, 9, 10, 1, 10, 200},
				{21, 22, 23, 24, 25, 50, 25, 500}
		};
		runRaJoinTest(A, colA, B, colB, Y);
	}

	@Test
	public void testRaJoinTestwithNoMatchingRows() {
		// Generate actual dataset and variables
		double[][] A = {
				{1, 2, 3},
				{2, 3, 4},
				{3, 4, 5}
		};
		double[][] B = {
				{4, 5, 6},
				{5, 6, 7},
				{6, 7, 8}
		};
		int colA = 1;
		int colB = 1;

		// Expected output matrix (no matching rows)
		double[][] Y = {};
		runRaJoinTest(A, colA, B, colB, Y);
	}

	@Test
	public void testRaJoinTestwithAllMatchingRows() {
		// Generate actual dataset and variables
		double[][] A = {
				{1, 2, 3},
				{2, 3, 4},
				{3, 4, 5}
		};
		double[][] B = {
				{1, 2, 6},
				{2, 3, 7},
				{3, 4, 8}
		};
		int colA = 1;
		int colB = 1;

		// Expected output matrix (all rows match)
		double[][] Y = {
				{1, 2, 3, 1, 2, 6},
				{2, 3, 4, 2, 3, 7},
				{3, 4, 5, 3, 4, 8}
		};
		runRaJoinTest(A, colA, B, colB, Y);
	}

	private void runRaJoinTest(double [][] A, int colA, double [][] B, int colB, double [][] Y)
	{
		ExecMode platformOld = setExecMode(ExecMode.SINGLE_NODE);
		
		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;

			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats", "-args",
				input("A"), String.valueOf(colA), input("B"), String.valueOf(colB), output("result") };
			System.out.println(Arrays.deepToString(A));
			System.out.println(colA);
			//fullRScriptName = HOME + TEST_NAME + ".R";
			//rCmd = "Rscript" + " " + fullRScriptName + " "
			//	+ inputDir() + " " + col + " "  + expectedDir();

			writeInputMatrixWithMTD("A", A, true);
			writeInputMatrixWithMTD("B", B, true);

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
