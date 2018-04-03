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

package org.apache.sysml.test.integration.functions.aggregate;

import java.util.HashMap;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class RowColProdsAggregateTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "RowProds";
	private final static String TEST_NAME2 = "ColProds";
	
	private final static String TEST_DIR = "functions/aggregate/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RowColProdsAggregateTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int dim1 = 1079;
	private final static int dim2 = 15;
	private final static double sparsity1 = 0.1;
	private final static double sparsity2 = 1.0; //otherwise 0 output
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{"B"})); 
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[]{"B"})); 
		if (TEST_CACHE_ENABLED) {
			setOutAndExpectedDeletionDisabled(true);
		}
	}

	@BeforeClass
	public static void init() {
		TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
	}

	@AfterClass
	public static void cleanUp() {
		if (TEST_CACHE_ENABLED) {
			TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
		}
	}
	
	@Test
	public void testRowProdsDenseMatrixCP() {
		runProdsAggregateTest(TEST_NAME1, false, true, ExecType.CP);
	}
	
	@Test
	public void testRowProdsSparseMatrixCP() {
		runProdsAggregateTest(TEST_NAME1, true, true, ExecType.CP);
	}
	
	@Test
	public void testRowProdsDenseMatrixSP() {
		runProdsAggregateTest(TEST_NAME1, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testRowProdsSparseMatrixSP() {
		runProdsAggregateTest(TEST_NAME1, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testColProdsDenseMatrixCP() {
		runProdsAggregateTest(TEST_NAME2, false, true, ExecType.CP);
	}
	
	@Test
	public void testColProdsSparseMatrixCP() {
		runProdsAggregateTest(TEST_NAME2, true, true, ExecType.CP);
	}
	
	@Test
	public void testColProdsDenseMatrixSP() {
		runProdsAggregateTest(TEST_NAME2, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testColProdsSparseMatrixSP() {
		runProdsAggregateTest(TEST_NAME2, true, true, ExecType.SPARK);
	}

	private void runProdsAggregateTest(String TEST_NAME, boolean sparse, boolean rewrites, ExecType instType)
	{
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( instType ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		boolean oldRewritesFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
		
		try
		{
			double sparsity = (sparse) ? sparsity1 : sparsity2;
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			String TEST_CACHE_DIR = "";
			if (TEST_CACHE_ENABLED) {
				TEST_CACHE_DIR = TEST_NAME + "_" + sparsity + "/";
			}
			
			loadTestConfiguration(config, TEST_CACHE_DIR);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain", "-args", input("A"),  output("B") };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
			
			int rows = TEST_NAME.equals(TEST_NAME1) ? dim1 : dim2;
			int cols = TEST_NAME.equals(TEST_NAME1) ? dim2 : dim1;
			double[][] A = getRandomMatrix(rows, cols, 0.9, 1, sparsity, 1234);
			writeInputMatrixWithMTD("A", A, true);
			
			runTest(true, false, null, -1);
			runRScript(true);
			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("B");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("B");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldRewritesFlag;
		}
	}
}
