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

package org.apache.sysml.test.integration.functions.binary.matrix_full_other;

import java.util.HashMap;

import org.junit.Test;

import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class MatrixMultShortLhsTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "MatrixMultShortLhs";
	private final static String TEST_DIR = "functions/binary/matrix_full_other/";
	private final static String TEST_CLASS_DIR = TEST_DIR + MatrixMultShortLhsTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rowsA = 10;
	private final static int colsA = 2023;
	private final static int rowsB = 2023;
	private final static int colsB = 1997;
	
	private final static double sparsity1 = 0.9;
	private final static double sparsity2 = 0.1;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "C" })); 
	}
	
	@Test
	public void testMMDenseDenseCP() {
		runMatrixMatrixMultiplicationTest(false, false, ExecType.CP);
	}
	
	@Test
	public void testMMDenseSparseCP() {
		runMatrixMatrixMultiplicationTest(false, true, ExecType.CP);
	}
	
	@Test
	public void testMMSparseDenseCP() {
		runMatrixMatrixMultiplicationTest(true, false, ExecType.CP);
	}
	
	@Test
	public void testMMSparseSparseCP() {
		runMatrixMatrixMultiplicationTest(true, true, ExecType.CP);
	}

	private void runMatrixMatrixMultiplicationTest( boolean sparseM1, boolean sparseM2, ExecType instType)
	{	
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		double sparsityA = sparseM1?sparsity2:sparsity1; 
		double sparsityB = sparseM2?sparsity2:sparsity1; 
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args",
			input("A"), input("B"), output("C") };
		
		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " 
			+ inputDir() + " " + expectedDir();

		//generate datasets
		double[][] A = getRandomMatrix(rowsA, colsA, 0, 1, sparsityA, 7); 
		writeInputMatrixWithMTD("A", A, true);
		double[][] B = getRandomMatrix(rowsB, colsB, 0, 1, sparsityB, 3); 
		writeInputMatrixWithMTD("B", B, true);

		//run tests
		runTest(true, false, null, -1); 
		runRScript(true); 
		
		//compare matrices 
		HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("C");
		HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("C");
		TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
	}
}
