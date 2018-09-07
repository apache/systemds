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

/**
 * NOTES:
 *  * the output definition of DML and R differs for col*; R always returns a column vector
 *    while DML returns a row vector.
 *  * the R package Matrix does not support colMins and colMaxs; hence, we use the matrixStats package 
 * 
 */
public class FullColAggregateTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "ColSums";
	private final static String TEST_NAME2 = "ColMeans";
	private final static String TEST_NAME3 = "ColMaxs";
	private final static String TEST_NAME4 = "ColMins";
	
	private final static String TEST_DIR = "functions/aggregate/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FullColAggregateTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rows = 1005;
	private final static int cols1 = 1;
	private final static int cols2 = 1079;
	private final static double sparsity1 = 0.1;
	private final static double sparsity2 = 0.7;
	
	private enum OpType{
		COL_SUMS,
		COL_MEANS,
		COL_MAX,
		COL_MIN
	}
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{"B"})); 
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[]{"B"})); 
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[]{"B"})); 
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[]{"B"})); 

		if (TEST_CACHE_ENABLED) {
			setOutAndExpectedDeletionDisabled(true);
		}
	}

	@BeforeClass
	public static void init()
	{
		TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
	}

	@AfterClass
	public static void cleanUp()
	{
		if (TEST_CACHE_ENABLED) {
			TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
		}
	}
	
	@Test
	public void testColSumsDenseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, false, false, ExecType.CP);
	}
	
	@Test
	public void testColMeansDenseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, false, false, ExecType.CP);
	}	
	
	@Test
	public void testColMaxDenseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, false, false, ExecType.CP);
	}
	
	@Test
	public void testColMinDenseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, false, false, ExecType.CP);
	}
	
	@Test
	public void testColSumsDenseVectorCP() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, false, true, ExecType.CP);
	}
	
	@Test
	public void testColMeansDenseVectorCP() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, false, true, ExecType.CP);
	}	
	
	@Test
	public void testColMaxDenseVectorCP() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, false, true, ExecType.CP);
	}
	
	@Test
	public void testColMinDenseVectorCP() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, false, true, ExecType.CP);
	}
	
	@Test
	public void testColSumsSparseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, true, false, ExecType.CP);
	}
	
	@Test
	public void testColMeansSparseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, true, false, ExecType.CP);
	}	
	
	@Test
	public void testColMaxSparseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, true, false, ExecType.CP);
	}
	
	@Test
	public void testColMinSparseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, true, false, ExecType.CP);
	}
	
	@Test
	public void testColSumsSparseVectorCP() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, true, true, ExecType.CP);
	}
	
	@Test
	public void testColMeansSparseVectorCP() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, true, true, ExecType.CP);
	}	
	
	@Test
	public void testColMaxSparseVectorCP() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, true, true, ExecType.CP);
	}
	
	@Test
	public void testColMinSparseVectorCP() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, true, true, ExecType.CP);
	}
	
	@Test
	public void testColSumsDenseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, false, false, ExecType.MR);
	}
	
	@Test
	public void testColMeansDenseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, false, false, ExecType.MR);
	}	
	
	@Test
	public void testColMaxDenseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, false, false, ExecType.MR);
	}
	
	@Test
	public void testColMinDenseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, false, false, ExecType.MR);
	}
	
	@Test
	public void testColSumsDenseVectorMR() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, false, true, ExecType.MR);
	}
	
	@Test
	public void testColMeansDenseVectorMR() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, false, true, ExecType.MR);
	}	
	
	@Test
	public void testColMaxDenseVectorMR() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, false, true, ExecType.MR);
	}
	
	@Test
	public void testColMinDenseVectorMR() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, false, true, ExecType.MR);
	}
	
	@Test
	public void testColSumsSparseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, true, false, ExecType.MR);
	}
	
	@Test
	public void testColMeansSparseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, true, false, ExecType.MR);
	}	
	
	@Test
	public void testColMaxSparseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, true, false, ExecType.MR);
	}
	
	@Test
	public void testColMinSparseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, true, false, ExecType.MR);
	}
	
	@Test
	public void testColSumsSparseVectorMR() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, true, true, ExecType.MR);
	}
	
	@Test
	public void testColMeansSparseVectorMR() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, true, true, ExecType.MR);
	}	
	
	@Test
	public void testColMaxSparseVectorMR() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, true, true, ExecType.MR);
	}
	
	@Test
	public void testColMinSparseVectorMR() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, true, true, ExecType.MR);
	}

	@Test
	public void testColSumsDenseMatrixNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, false, false, ExecType.CP, false);
	}
	
	@Test
	public void testColMeansDenseMatrixNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, false, false, ExecType.CP, false);
	}	
	
	@Test
	public void testColMaxDenseMatrixNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, false, false, ExecType.CP, false);
	}
	
	@Test
	public void testColMinDenseMatrixNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, false, false, ExecType.CP, false);
	}
	
	@Test
	public void testColSumsDenseVectorNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, false, true, ExecType.CP, false);
	}
	
	@Test
	public void testColMeansDenseVectorNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, false, true, ExecType.CP, false);
	}	
	
	@Test
	public void testColMaxDenseVectorNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, false, true, ExecType.CP, false);
	}
	
	@Test
	public void testColMinDenseVectorNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, false, true, ExecType.CP, false);
	}
	
	@Test
	public void testColSumsSparseMatrixNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, true, false, ExecType.CP, false);
	}
	
	@Test
	public void testColMeansSparseMatrixNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, true, false, ExecType.CP, false);
	}	
	
	@Test
	public void testColMaxSparseMatrixNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, true, false, ExecType.CP, false);
	}
	
	@Test
	public void testColMinSparseMatrixNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, true, false, ExecType.CP, false);
	}
	
	@Test
	public void testColSumsSparseVectorNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, true, true, ExecType.CP, false);
	}
	
	@Test
	public void testColMeansSparseVectorNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, true, true, ExecType.CP, false);
	}	
	
	@Test
	public void testColMaxSparseVectorNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, true, true, ExecType.CP, false);
	}
	
	@Test
	public void testColMinSparseVectorNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, true, true, ExecType.CP, false);
	}
	
	@Test
	public void testColSumsDenseMatrixNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, false, false, ExecType.MR, false);
	}
	
	@Test
	public void testColMeansDenseMatrixNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, false, false, ExecType.MR, false);
	}	
	
	@Test
	public void testColMaxDenseMatrixNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, false, false, ExecType.MR, false);
	}
	
	@Test
	public void testColMinDenseMatrixNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, false, false, ExecType.MR, false);
	}
	
	@Test
	public void testColSumsDenseVectorNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, false, true, ExecType.MR, false);
	}
	
	@Test
	public void testColMeansDenseVectorNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, false, true, ExecType.MR, false);
	}	
	
	@Test
	public void testColMaxDenseVectorNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, false, true, ExecType.MR, false);
	}
	
	@Test
	public void testColMinDenseVectorNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, false, true, ExecType.MR, false);
	}
	
	@Test
	public void testColSumsSparseMatrixNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, true, false, ExecType.MR, false);
	}
	
	@Test
	public void testColMeansSparseMatrixNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, true, false, ExecType.MR, false);
	}	
	
	@Test
	public void testColMaxSparseMatrixNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, true, false, ExecType.MR, false);
	}
	
	@Test
	public void testColMinSparseMatrixNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, true, false, ExecType.MR, false);
	}
	
	@Test
	public void testColSumsSparseVectorNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, true, true, ExecType.MR, false);
	}
	
	@Test
	public void testColMeansSparseVectorNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, true, true, ExecType.MR, false);
	}	
	
	@Test
	public void testColMaxSparseVectorNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, true, true, ExecType.MR, false);
	}
	
	@Test
	public void testColMinSparseVectorNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, true, true, ExecType.MR, false);
	}
	
	@Test
	public void testColSumsDenseMatrixNoRewritesSP() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, false, false, ExecType.SPARK, false);
	}
	
	@Test
	public void testColMeansDenseMatrixNoRewritesSP() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, false, false, ExecType.SPARK, false);
	}	
	
	@Test
	public void testColMaxDenseMatrixNoRewritesSP() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, false, false, ExecType.SPARK, false);
	}
	
	@Test
	public void testColMinDenseMatrixNoRewritesSP() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, false, false, ExecType.SPARK, false);
	}
	
	@Test
	public void testColSumsDenseVectorNoRewritesSP() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, false, true, ExecType.SPARK, false);
	}
	
	@Test
	public void testColMeansDenseVectorNoRewritesSP() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, false, true, ExecType.SPARK, false);
	}	
	
	@Test
	public void testColMaxDenseVectorNoRewritesSP() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, false, true, ExecType.SPARK, false);
	}
	
	@Test
	public void testColMinDenseVectorNoRewritesSP() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, false, true, ExecType.SPARK, false);
	}
	
	@Test
	public void testColSumsSparseMatrixNoRewritesSP() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, true, false, ExecType.SPARK, false);
	}
	
	@Test
	public void testColMeansSparseMatrixNoRewritesSP() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, true, false, ExecType.SPARK, false);
	}	
	
	@Test
	public void testColMaxSparseMatrixNoRewritesSP() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, true, false, ExecType.SPARK, false);
	}
	
	@Test
	public void testColMinSparseMatrixNoRewritesSP() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, true, false, ExecType.SPARK, false);
	}
	
	@Test
	public void testColSumsSparseVectorNoRewritesSP() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, true, true, ExecType.SPARK, false);
	}
	
	@Test
	public void testColMeansSparseVectorNoRewritesSP() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, true, true, ExecType.SPARK, false);
	}	
	
	@Test
	public void testColMaxSparseVectorNoRewritesSP() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, true, true, ExecType.SPARK, false);
	}
	
	@Test
	public void testColMinSparseVectorNoRewritesSP() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, true, true, ExecType.SPARK, false);
	}
	
	/**
	 * 
	 * @param type
	 * @param sparse
	 * @param vector
	 * @param instType
	 */
	private void runColAggregateOperationTest( OpType type, boolean sparse, boolean vector, ExecType instType)
	{
		runColAggregateOperationTest(type, sparse, vector, instType, true); //by default apply algebraic simplification
	}
	
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runColAggregateOperationTest( OpType type, boolean sparse, boolean vector, ExecType instType, boolean rewrites)
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
			String TEST_NAME = null;
			switch( type )
			{
				case COL_SUMS: TEST_NAME = TEST_NAME1; break;
				case COL_MEANS: TEST_NAME = TEST_NAME2; break;
				case COL_MAX: TEST_NAME = TEST_NAME3; break;
				case COL_MIN: TEST_NAME = TEST_NAME4; break;
			}
			
			int cols = (vector) ? cols1 : cols2;
			double sparsity = (sparse) ? sparsity1 : sparsity2;
			
			String TEST_CACHE_DIR = "";
			if (TEST_CACHE_ENABLED)
			{
				TEST_CACHE_DIR = type.ordinal() + "_" + cols + "_" + sparsity + "/";
			}
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config, TEST_CACHE_DIR);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("A"),
				Integer.toString(rows), Integer.toString(cols), output("B") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			//generate actual dataset 
			double[][] A = getRandomMatrix(rows, cols, -0.05, 1, sparsity, 7); 
			writeInputMatrix("A", A, true);
	
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
		
			runRScript(true); 
		
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("B");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("B");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldRewritesFlag;
		}
	}
	
		
}