/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.test.integration.functions.unary.matrix;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;
import com.ibm.bi.dml.utils.Statistics;

/**
 * 
 * 
 */
public class FullCummaxTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "Cummax";
	private final static String TEST_DIR = "functions/unary/matrix/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FullCummaxTest.class.getSimpleName() + "/";
	
	private final static double eps = 1e-10;
	
	private final static int rowsMatrix = 1201;
	private final static int colsMatrix = 1103;
	private final static double spSparse = 0.1;
	private final static double spDense = 0.9;
	
	private enum InputType {
		COL_VECTOR,
		ROW_VECTOR,
		MATRIX
	}
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"})); 
	}

	@Test
	public void testCummaxColVectorDenseCP() 
	{
		runColAggregateOperationTest(InputType.COL_VECTOR, false, ExecType.CP);
	}
	
	@Test
	public void testCummaxRowVectorDenseCP() 
	{
		runColAggregateOperationTest(InputType.ROW_VECTOR, false, ExecType.CP);
	}
	
	@Test
	public void testCummaxRowVectorDenseNoRewritesCP() 
	{
		runColAggregateOperationTest(InputType.ROW_VECTOR, false, ExecType.CP, false);
	}
	
	@Test
	public void testCummaxMatrixDenseCP() 
	{
		runColAggregateOperationTest(InputType.MATRIX, false, ExecType.CP);
	}
	
	@Test
	public void testCummaxColVectorSparseCP() 
	{
		runColAggregateOperationTest(InputType.COL_VECTOR, true, ExecType.CP);
	}
	
	@Test
	public void testCummaxRowVectorSparseCP() 
	{
		runColAggregateOperationTest(InputType.ROW_VECTOR, true, ExecType.CP);
	}
	
	@Test
	public void testCummaxRowVectorSparseNoRewritesCP() 
	{
		runColAggregateOperationTest(InputType.ROW_VECTOR, true, ExecType.CP, false);
	}
	
	@Test
	public void testCummaxMatrixSparseCP() 
	{
		runColAggregateOperationTest(InputType.MATRIX, true, ExecType.CP);
	}
	
	@Test
	public void testCummaxColVectorDenseMR() 
	{
		runColAggregateOperationTest(InputType.COL_VECTOR, false, ExecType.MR);
	}
	
	@Test
	public void testCummaxRowVectorDenseMR() 
	{
		runColAggregateOperationTest(InputType.ROW_VECTOR, false, ExecType.MR);
	}
	
	@Test
	public void testCummaxRowVectorDenseNoRewritesMR() 
	{
		runColAggregateOperationTest(InputType.ROW_VECTOR, false, ExecType.MR, false);
	}
	
	@Test
	public void testCummaxMatrixDenseMR() 
	{
		runColAggregateOperationTest(InputType.MATRIX, false, ExecType.MR);
	}
	
	@Test
	public void testCummaxColVectorSparseMR() 
	{
		runColAggregateOperationTest(InputType.COL_VECTOR, true, ExecType.MR);
	}
	
	@Test
	public void testCummaxRowVectorSparseNoRewritesMR() 
	{
		runColAggregateOperationTest(InputType.ROW_VECTOR, true, ExecType.MR, false);
	}
	
	@Test
	public void testCummaxMatrixSparseMR() 
	{
		runColAggregateOperationTest(InputType.MATRIX, true, ExecType.MR);
	}

	@Test
	public void testCummaxColVectorDenseSP() 
	{
		runColAggregateOperationTest(InputType.COL_VECTOR, false, ExecType.SPARK);
	}
	
	@Test
	public void testCummaxRowVectorDenseSP() 
	{
		runColAggregateOperationTest(InputType.ROW_VECTOR, false, ExecType.SPARK);
	}
	
	@Test
	public void testCummaxRowVectorDenseNoRewritesSP() 
	{
		runColAggregateOperationTest(InputType.ROW_VECTOR, false, ExecType.SPARK, false);
	}
	
	@Test
	public void testCummaxMatrixDenseSP() 
	{
		runColAggregateOperationTest(InputType.MATRIX, false, ExecType.SPARK);
	}
	
	@Test
	public void testCummaxColVectorSparseSP() 
	{
		runColAggregateOperationTest(InputType.COL_VECTOR, true, ExecType.SPARK);
	}
	
	@Test
	public void testCummaxRowVectorSparseSP() 
	{
		runColAggregateOperationTest(InputType.ROW_VECTOR, true, ExecType.SPARK);
	}
	
	@Test
	public void testCummaxRowVectorSparseNoRewritesSP() 
	{
		runColAggregateOperationTest(InputType.ROW_VECTOR, true, ExecType.SPARK, false);
	}
	
	@Test
	public void testCummaxMatrixSparseSP() 
	{
		runColAggregateOperationTest(InputType.MATRIX, true, ExecType.SPARK);
	}

	
	/**
	 * 
	 * @param type
	 * @param sparse
	 * @param instType
	 */
	private void runColAggregateOperationTest( InputType type, boolean sparse, ExecType instType)
	{
		//by default we apply algebraic simplification rewrites
		runColAggregateOperationTest(type, sparse, instType, true);
	}
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runColAggregateOperationTest( InputType type, boolean sparse, ExecType instType, boolean rewrites)
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
		
		//rewrites
		boolean oldFlagRewrites = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
		
		try
		{
			int cols = (type==InputType.COL_VECTOR) ? 1 : colsMatrix;
			int rows = (type==InputType.ROW_VECTOR) ? 1 : rowsMatrix;
			double sparsity = (sparse) ? spSparse : spDense;
			
			getAndLoadTestConfiguration(TEST_NAME);
			
			// This is for running the junit test the new way, i.e., construct the arguments directly
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain", "-args", input("A"), output("B") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			//generate actual dataset 
			double[][] A = getRandomMatrix(rows, cols, -0.05, 1, sparsity, 7); 
			writeInputMatrixWithMTD("A", A, true);
	
			runTest(true, false, null, -1); 
			if( instType==ExecType.CP || instType==ExecType.SPARK ) //in CP no MR jobs should be executed
				Assert.assertEquals("Unexpected number of executed MR jobs.", 0, Statistics.getNoOfExecutedMRJobs());
			
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
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlagRewrites;
		}
	}	
}