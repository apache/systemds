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

package org.apache.sysds.test.functions.binary.matrix;

import java.util.HashMap;

import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.AggBinaryOp.MMultMethod;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

/**
 * 
 */
public class ZipMMSparkMatrixMultiplicationTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME1 = "ZipMMTest";
	private final static String TEST_DIR = "functions/binary/matrix/";
	private final static String TEST_CLASS_DIR = TEST_DIR + 
		ZipMMSparkMatrixMultiplicationTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	//singleblock (zipmm applied)
	private final static int rowsA = 2407;
	private final static int colsA = 312;
	private final static int rowsB = 2407;
	private final static int colsB1 = 73;
	private final static int colsB2 = 1;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration( TEST_NAME1, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "C" }) );
	}

	@Test
	public void testZipMMDenseDenseSP() 
	{
		runZipMMMatrixMultiplicationTest(false, false, ExecType.SPARK, false);
	}
	
	@Test
	public void testZipMMDenseSparseSP() 
	{
		runZipMMMatrixMultiplicationTest(false, true, ExecType.SPARK, false);
	}
	
	@Test
	public void testZipMMSparseDenseSP() 
	{
		runZipMMMatrixMultiplicationTest(true, false, ExecType.SPARK, false);
	}
	
	@Test
	public void testZipMMSparseSparseSP() 
	{
		runZipMMMatrixMultiplicationTest(true, true, ExecType.SPARK, false);
	}
	
	@Test
	public void testZipMMDenseDenseMVSP() 
	{
		runZipMMMatrixMultiplicationTest(false, false, ExecType.SPARK, true);
	}
	
	@Test
	public void testZipMMDenseSparseMVSP() 
	{
		runZipMMMatrixMultiplicationTest(false, true, ExecType.SPARK, true);
	}
	
	@Test
	public void testZipMMSparseDenseMVSP() 
	{
		runZipMMMatrixMultiplicationTest(true, false, ExecType.SPARK, true);
	}
	
	@Test
	public void testZipMMSparseSparseMVSP() 
	{
		runZipMMMatrixMultiplicationTest(true, true, ExecType.SPARK, true);
	}
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runZipMMMatrixMultiplicationTest( boolean sparseM1, boolean sparseM2, ExecType instType, boolean vectorM2)
	{
		//rtplatform for MR
		ExecMode platformOld = rtplatform;
		switch( instType ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
	
		//force zipmm execution
		MMultMethod methodOld = AggBinaryOp.FORCED_MMULT_METHOD;
		AggBinaryOp.FORCED_MMULT_METHOD = MMultMethod.ZIPMM;
		
		int colsB = vectorM2 ? colsB2 : colsB1;
		String TEST_NAME = TEST_NAME1;
		
		try
		{
			getAndLoadTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-args", input("A"), input("B"), output("C")};
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			//generate actual dataset
			double[][] A = getRandomMatrix(rowsA, colsA, 0, 1, sparseM1?sparsity2:sparsity1, 7); 
			writeInputMatrixWithMTD("A", A, true);
			double[][] B = getRandomMatrix(rowsB, colsB, 0, 1, sparseM2?sparsity2:sparsity1, 3); 
			writeInputMatrixWithMTD("B", B, true);
	
			//run test case
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("C");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("C");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			AggBinaryOp.FORCED_MMULT_METHOD = methodOld;
		}
	}

}