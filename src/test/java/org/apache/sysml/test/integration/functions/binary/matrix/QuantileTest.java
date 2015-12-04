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

package org.apache.sysml.test.integration.functions.binary.matrix;

import java.util.HashMap;

import org.junit.Test;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

/**
 * 
 */
public class QuantileTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME1 = "Quantile";
	private final static String TEST_NAME2 = "Median";
	private final static String TEST_NAME3 = "IQM";
	
	private final static String TEST_DIR = "functions/binary/matrix/";
	private final static String TEST_CLASS_DIR = TEST_DIR + QuantileTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rows = 1973;
	private final static int maxVal = 7; 
	private final static double sparsity1 = 0.9;
	private final static double sparsity2 = 0.3;
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) ); 
		addTestConfiguration(TEST_NAME2, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }) ); 
		addTestConfiguration(TEST_NAME3, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "R" }) ); 		
	}
	
	@Test
	public void testQuantile1DenseCP() 
	{
		runQuantileTest(TEST_NAME1, 0.25, false, ExecType.CP);
	}
	
	@Test
	public void testQuantile2DenseCP() 
	{
		runQuantileTest(TEST_NAME1, 0.50, false, ExecType.CP);
	}
	
	@Test
	public void testQuantile3DenseCP() 
	{
		runQuantileTest(TEST_NAME1, 0.75, false, ExecType.CP);
	}
	
	@Test
	public void testQuantile1SparseCP() 
	{
		runQuantileTest(TEST_NAME1, 0.25, true, ExecType.CP);
	}
	
	@Test
	public void testQuantile2SparseCP() 
	{
		runQuantileTest(TEST_NAME1, 0.50, true, ExecType.CP);
	}
	
	@Test
	public void testQuantile3SparseCP() 
	{
		runQuantileTest(TEST_NAME1, 0.75, true, ExecType.CP);
	}
	
	@Test
	public void testQuantile1DenseMR() 
	{
		runQuantileTest(TEST_NAME1, 0.25, false, ExecType.MR);
	}
	
	@Test
	public void testQuantile2DenseMR() 
	{
		runQuantileTest(TEST_NAME1, 0.50, false, ExecType.MR);
	}
	
	@Test
	public void testQuantile3DenseMR() 
	{
		runQuantileTest(TEST_NAME1, 0.75, false, ExecType.MR);
	}
	
	@Test
	public void testQuantile1SparseMR() 
	{
		runQuantileTest(TEST_NAME1, 0.25, true, ExecType.MR);
	}
	
	@Test
	public void testQuantile2SparseMR() 
	{
		runQuantileTest(TEST_NAME1, 0.50, true, ExecType.MR);
	}
	
	@Test
	public void testQuantile3SparseMR() 
	{
		runQuantileTest(TEST_NAME1, 0.75, true, ExecType.MR);
	}

	@Test
	public void testQuantile1DenseSP() 
	{
		runQuantileTest(TEST_NAME1, 0.25, false, ExecType.SPARK);
	}
	
	@Test
	public void testQuantile2DenseSP() 
	{
		runQuantileTest(TEST_NAME1, 0.50, false, ExecType.SPARK);
	}
	
	@Test
	public void testQuantile3DenseSP() 
	{
		runQuantileTest(TEST_NAME1, 0.75, false, ExecType.SPARK);
	}
	
	@Test
	public void testQuantile1SparseSP() 
	{
		runQuantileTest(TEST_NAME1, 0.25, true, ExecType.SPARK);
	}
	
	@Test
	public void testQuantile2SparseSP() 
	{
		runQuantileTest(TEST_NAME1, 0.50, true, ExecType.SPARK);
	}
	
	@Test
	public void testQuantile3SparseSP() 
	{
		runQuantileTest(TEST_NAME1, 0.75, true, ExecType.SPARK);
	}

	@Test
	public void testMedianDenseCP() 
	{
		runQuantileTest(TEST_NAME2, -1, false, ExecType.CP);
	}
	
	@Test
	public void testMedianSparseCP() 
	{
		runQuantileTest(TEST_NAME2, -1, true, ExecType.CP);
	}
	
	@Test
	public void testMedianDenseMR() 
	{
		runQuantileTest(TEST_NAME2, -1, false, ExecType.MR);
	}
	
	@Test
	public void testMedianSparseMR() 
	{
		runQuantileTest(TEST_NAME2, -1, true, ExecType.MR);
	}
	
	@Test
	public void testMedianDenseSP() 
	{
		runQuantileTest(TEST_NAME2, -1, false, ExecType.SPARK);
	}

	@Test
	public void testMedianSparseSP() 
	{
		runQuantileTest(TEST_NAME2, -1, true, ExecType.SPARK);
	}

	@Test
	public void testIQMDenseCP() 
	{
		runQuantileTest(TEST_NAME3, -1, false, ExecType.CP);
	}
	
	@Test
	public void testIQMSparseCP() 
	{
		runQuantileTest(TEST_NAME3, -1, true, ExecType.CP);
	}
	
	@Test
	public void testIQMDenseMR() 
	{
		runQuantileTest(TEST_NAME3, -1, false, ExecType.MR);
	}
	
	@Test
	public void testIQMSparseMR() 
	{
		runQuantileTest(TEST_NAME3, -1, true, ExecType.MR);
	}
	
	@Test
	public void testIQMDenseSP() 
	{
		runQuantileTest(TEST_NAME3, -1, false, ExecType.SPARK);
	}

	@Test
	public void testIQMSparseSP() 
	{
		runQuantileTest(TEST_NAME3, -1, true, ExecType.SPARK);
	}
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runQuantileTest( String TEST_NAME, double p, boolean sparse, ExecType et)
	{
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( et ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		try
		{
			getAndLoadTestConfiguration(TEST_NAME);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("A"), Double.toString(p), output("R")};
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + p + " "+ expectedDir();
	
			//generate actual dataset (always dense because values <=0 invalid)
			double sparsitya = sparse ? sparsity2 : sparsity1;
			double[][] A = getRandomMatrix(rows, 1, 1, maxVal, sparsitya, 1236); 
			writeInputMatrixWithMTD("A", A, true);
			
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}

}