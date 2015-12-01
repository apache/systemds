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

package org.apache.sysml.test.integration.functions.unary.matrix;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.apache.sysml.utils.Statistics;

/**
 * 
 * 
 */
public class MLUnaryBuiltinTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "SProp";
	private final static String TEST_NAME2 = "Sigmoid";
	
	private final static String TEST_DIR = "functions/unary/matrix/";
	private static final String TEST_CLASS_DIR = TEST_DIR + MLUnaryBuiltinTest.class.getSimpleName() + "/";
	
	private final static double eps = 1e-10;
	
	private final static int rowsMatrix = 1201;
	private final static int colsMatrix = 1103;
	private final static double spSparse = 0.05;
	private final static double spDense = 0.5;
	
	private enum InputType {
		COL_VECTOR,
		MATRIX
	}
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1,
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{"B"}));
		addTestConfiguration(TEST_NAME2,
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[]{"B"}));
	}

	
	@Test
	public void testSampleProportionVectorDenseCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, InputType.COL_VECTOR, false, ExecType.CP);
	}
	
	@Test
	public void testSampleProportionVectorSparseCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, InputType.COL_VECTOR, true, ExecType.CP);
	}
	
	@Test
	public void testSampleProportionMatrixDenseCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, InputType.MATRIX, false, ExecType.CP);
	}
	
	@Test
	public void testSampleProportionMatrixSparseCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, InputType.MATRIX, true, ExecType.CP);
	}
	
	@Test
	public void testSampleProportionVectorDenseMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, InputType.COL_VECTOR, false, ExecType.MR);
	}
	
	@Test
	public void testSampleProportionVectorSparseMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, InputType.COL_VECTOR, true, ExecType.MR);
	}
	
	@Test
	public void testSampleProportionMatrixDenseMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, InputType.MATRIX, false, ExecType.MR);
	}
	
	@Test
	public void testSampleProportionMatrixSparseMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, InputType.MATRIX, true, ExecType.MR);
	}
	
	@Test
	public void testSampleProportionVectorDenseSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, InputType.COL_VECTOR, false, ExecType.SPARK);
	}
	
	@Test
	public void testSampleProportionVectorSparseSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, InputType.COL_VECTOR, true, ExecType.SPARK);
	}
	
	@Test
	public void testSampleProportionMatrixDenseSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, InputType.MATRIX, false, ExecType.SPARK);
	}
	
	@Test
	public void testSampleProportionMatrixSparseSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, InputType.MATRIX, true, ExecType.SPARK);
	}
	

	@Test
	public void testSigmoidVectorDenseCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, InputType.COL_VECTOR, false, ExecType.CP);
	}
	
	@Test
	public void testSigmoidVectorSparseCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, InputType.COL_VECTOR, true, ExecType.CP);
	}
	
	@Test
	public void testSigmoidMatrixDenseCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, InputType.MATRIX, false, ExecType.CP);
	}
	
	@Test
	public void testSigmoidMatrixSparseCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, InputType.MATRIX, true, ExecType.CP);
	}
	
	@Test
	public void testSigmoidVectorDenseSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, InputType.COL_VECTOR, false, ExecType.SPARK);
	}
	
	@Test
	public void testSigmoidVectorSparseSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, InputType.COL_VECTOR, true, ExecType.SPARK);
	}
	
	@Test
	public void testSigmoidMatrixDenseSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, InputType.MATRIX, false, ExecType.SPARK);
	}
	
	@Test
	public void testSigmoidMatrixSparseSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, InputType.MATRIX, true, ExecType.SPARK);
	}
	
	@Test
	public void testSigmoidVectorDenseMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, InputType.COL_VECTOR, false, ExecType.MR);
	}
	
	@Test
	public void testSigmoidVectorSparseMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, InputType.COL_VECTOR, true, ExecType.MR);
	}
	
	@Test
	public void testSigmoidMatrixDenseMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, InputType.MATRIX, false, ExecType.MR);
	}
	
	@Test
	public void testSigmoidMatrixSparseMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, InputType.MATRIX, true, ExecType.MR);
	}

	
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runMLUnaryBuiltinTest( String testname, InputType type, boolean sparse, ExecType instType)
	{
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( instType ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		try
		{
			int rows = rowsMatrix;
			int cols = (type==InputType.COL_VECTOR) ? 1 : colsMatrix;
			double sparsity = (sparse) ? spSparse : spDense;
			String TEST_NAME = testname;
			
			getAndLoadTestConfiguration(TEST_NAME);
			
			// This is for running the junit test the new way, i.e., construct the arguments directly
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats", "-args", input("A"), output("B") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			//generate actual dataset 
			double[][] A = getRandomMatrix(rows, cols, -0.05, 1, sparsity, 7); 
			writeInputMatrixWithMTD("A", A, true);
	
			runTest(true, false, null, -1); 
			if( instType==ExecType.CP ) //in CP no MR jobs should be executed
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
		}
	}	
}