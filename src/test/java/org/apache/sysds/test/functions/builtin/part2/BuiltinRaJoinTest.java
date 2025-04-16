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
import org.junit.Ignore;
import org.junit.Test;

import java.util.HashMap;

@Ignore //FIXME
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

	@Test
	public void testRaJoinTest1() {
		testRaJoinTest("nested-loop");
	}
	
	@Test
	public void testRaJoinTest2() {
		testRaJoinTest("sort-merge");
	}
	
	@Test
	public void testRaJoinTestwithDifferentColumn1() {
		testRaJoinTestwithDifferentColumn("nested-loop");
	}
	
	@Test
	public void testRaJoinTestwithDifferentColumn2() {
		testRaJoinTestwithDifferentColumn("sort-merge");
	}
	@Test
	public void testRaJoinTestwithDifferentColumn3() {
		testRaJoinTestwithDifferentColumn("hash2");
	}
	
	@Test
	public void testRaJoinTestwithDifferentColumn21() {
		testRaJoinTestwithDifferentColumn2("nested-loop");
	}
	
	@Test
	public void testRaJoinTestwithDifferentColumn22() {
		testRaJoinTestwithDifferentColumn2("sort-merge");
	}
	@Test
	public void testRaJoinTestwithDifferentColumn23() {
		testRaJoinTestwithDifferentColumn2("hash2");
	}
	
	@Test
	public void testRaJoinTestwithNoMatchingRows1() {
		testRaJoinTestwithNoMatchingRows("nested-loop");
	}
	
	@Test
	public void testRaJoinTestwithNoMatchingRows2() {
		testRaJoinTestwithNoMatchingRows("sort-merge");
	}

	@Test
	public void testRaJoinTestwithNoMatchingRows3() {
		testRaJoinTestwithNoMatchingRows("hash2");
	}
	
	@Test
	public void testRaJoinTestwithAllMatchingRows1() {
		testRaJoinTestwithAllMatchingRows("nested-loop");
	}
	
	@Test
	public void testRaJoinTestwithAllMatchingRows2() {
		testRaJoinTestwithAllMatchingRows("sort-merge");
	}
	
	@Test
	public void testRaJoinTestwithAllMatchingRows3() {
		testRaJoinTestwithAllMatchingRows("hash");
	}

	@Test
	public void testRaJoinTestwithAllMatchingRows4() {
		testRaJoinTestwithAllMatchingRows("hash2");
	}
	
	@Test
	public void testRaJoinTestwithOneToMany1() {
		testRaJoinTestwithOneToMany("nested-loop");
	}
	
	@Test
	public void testRaJoinTestwithOneToMany2() {
		testRaJoinTestwithOneToMany("sort-merge");
	}
	
	@Test
	public void testRaJoinTestwithOneToMany3() {
		testRaJoinTestwithOneToMany("hash");
	}

	@Test
	public void testRaJoinTestwithOneToMany4() {
		testRaJoinTestwithOneToMany("hash2");
	}
	
	
	private void testRaJoinTest(String method) {
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
		runRaJoinTest(A, colA, B, colB, Y, method);
	}

	private void testRaJoinTestwithDifferentColumn(String method) {
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
		runRaJoinTest(A, colA, B, colB, Y, method);
	}

	private void testRaJoinTestwithDifferentColumn2(String method) {
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
		runRaJoinTest(A, colA, B, colB, Y, method);
	}

	private void testRaJoinTestwithNoMatchingRows(String method) {
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
		runRaJoinTest(A, colA, B, colB, Y, method);
	}

	private void testRaJoinTestwithAllMatchingRows(String method) {
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
		runRaJoinTest(A, colA, B, colB, Y, method);
	}
	
	private void testRaJoinTestwithOneToMany(String method) {
		// Generate actual dataset and variables
		double[][] A = {
				{2, 2, 2},
				{3, 3, 3},
				{4, 4, 4}
		};
		double[][] B = {
				{2, 1, 1},
				{2, 2, 2},
				{3, 1, 1},
				{3, 2, 2},
				{3, 3, 3},
				{4, 1, 1}
		};
		int colA = 1;
		int colB = 1;

		double[][] Y = {
				{2, 2, 2, 2, 1, 1},
				{2, 2, 2, 2, 2, 2},
				{3, 3, 3, 3, 1, 1},
				{3, 3, 3, 3, 2, 2},
				{3, 3, 3, 3, 3, 3},
				{4, 4, 4, 4, 1, 1}
		};
		runRaJoinTest(A, colA, B, colB, Y, method);
	}

	private void runRaJoinTest(double [][] A, int colA, double [][] B, int colB, double [][] Y, String method)
	{
		ExecMode platformOld = setExecMode(ExecMode.SINGLE_NODE);
		
		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;

			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats", "-args",
				input("A"), String.valueOf(colA), input("B"),
				String.valueOf(colB), method, output("result") };
			writeInputMatrixWithMTD("A", A, true);
			writeInputMatrixWithMTD("B", B, true);

			// run dmlScript 
			runTest(null);

			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("result");
			HashMap<CellIndex, Double> expectedOutput = TestUtils.convert2DDoubleArrayToHashMap(Y);
			TestUtils.compareMatrices(dmlfile, expectedOutput, eps, "Stat-DML", "Expected");
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
