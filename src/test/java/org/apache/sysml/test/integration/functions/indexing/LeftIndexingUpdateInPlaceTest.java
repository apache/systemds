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

package org.apache.sysml.test.integration.functions.indexing;

import java.util.HashMap;

import org.junit.Test;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class LeftIndexingUpdateInPlaceTest extends AutomatedTestBase
{
	private final static String TEST_DIR = "functions/indexing/";
	private final static String TEST_NAME = "LeftIndexingUpdateInPlaceTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + LeftIndexingUpdateInPlaceTest.class.getSimpleName() + "/";
	
	private final static int rows1 = 1281;
	private final static int cols1 = 1102;
	private final static int cols2 = 226;
	private final static int cols3 = 1;
	
	private final static double sparsity1 = 0.05;
	private final static double sparsity2 = 0.9;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
	}
	
	@Test
	public void testSparseMatrixSparseMatrix() {
		runLeftIndexingUpdateInPlaceTest(true, true, false, false);
	}
	
	@Test
	public void testSparseMatrixSparseVector() {
		runLeftIndexingUpdateInPlaceTest(true, true, true, false);
	}
	
	@Test
	public void testSparseMatrixDenseMatrix() {
		runLeftIndexingUpdateInPlaceTest(true, false, false, false);
	}
	
	@Test
	public void testSparseMatrixDenseVector() {
		runLeftIndexingUpdateInPlaceTest(true, false, true, false);
	}
	
	@Test
	public void testDenseMatrixSparseMatrix() {
		runLeftIndexingUpdateInPlaceTest(false, true, false, false);
	}
	
	@Test
	public void testDenseMatrixSparseVector() {
		runLeftIndexingUpdateInPlaceTest(false, true, true, false);
	}
	
	@Test
	public void testDenseMatrixDenseMatrix() {
		runLeftIndexingUpdateInPlaceTest(false, false, false, false);
	}
	
	@Test
	public void testDenseMatrixDenseVector() {
		runLeftIndexingUpdateInPlaceTest(false, false, true, false);
	}
	
	@Test
	public void testSparseMatrixEmptyMatrix() {
		runLeftIndexingUpdateInPlaceTest(true, true, false, true);
	}
	
	@Test
	public void testSparseMatrixEmptyVector() {
		runLeftIndexingUpdateInPlaceTest(true, true, true, true);
	}
	
	@Test
	public void testDenseMatrixEmptyMatrix() {
		runLeftIndexingUpdateInPlaceTest(false, true, false, true);
	}
	
	@Test
	public void testDenseMatrixEmptyVector() {
		runLeftIndexingUpdateInPlaceTest(false, true, true, true);
	}

	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param vectorM2
	 */
	public void runLeftIndexingUpdateInPlaceTest(boolean sparseM1, boolean sparseM2, boolean vectorM2, boolean emptyM2) 
	{
		RUNTIME_PLATFORM oldRTP = rtplatform;
		rtplatform = RUNTIME_PLATFORM.HYBRID;
		
		try {
		    TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			double spM1 = sparseM1 ? sparsity1 : sparsity2;
			double spM2 = emptyM2 ? 0 : (sparseM2 ? sparsity1 : sparsity2);
			int colsM2 = vectorM2 ? cols3 : cols2;
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-args", input("A"), input("B"), output("R")};
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
				inputDir() + " " + expectedDir();
			
			//generate input data sets
			double[][] A = getRandomMatrix(rows1, cols1, -1, 1, spM1, 1234);
	        writeInputMatrixWithMTD("A", A, true);
			double[][] B = getRandomMatrix(rows1, colsM2, -1, 1, spM2, 5678);
	        writeInputMatrixWithMTD("B", B, true);
	        
	        //run dml and r script
	        runTest(true, false, null, 1); 
			runRScript(true);
			
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, 0, "DML", "R");
			checkDMLMetaDataFile("R", new MatrixCharacteristics(rows1,cols1,1,1));
		}
		finally {
			rtplatform = oldRTP;
		}
	}
}
