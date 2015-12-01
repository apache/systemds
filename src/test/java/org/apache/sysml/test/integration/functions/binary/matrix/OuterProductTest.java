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

package org.apache.sysml.test.integration.functions.binary.matrix;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;

public class OuterProductTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME = "OuterProduct";
	private final static String TEST_DIR = "functions/binary/matrix/";
	private final static String TEST_CLASS_DIR = TEST_DIR + OuterProductTest.class.getSimpleName() + "/";
	
	private final static int rows = 41456;
	private final static int cols = 9703;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "C" }) ); 
	}

	
	@Test
	public void testMMDenseDenseMR() 
	{
		runMatrixMatrixMultiplicationTest(false, false, ExecType.MR);
	}
	
	@Test
	public void testMMSparseSparseMR() 
	{
		runMatrixMatrixMultiplicationTest(true, true, ExecType.MR);
	}	

	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runMatrixMatrixMultiplicationTest( boolean sparseM1, boolean sparseM2, ExecType instType)
	{
		//setup exec type, rows, cols

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
			getAndLoadTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-args", input("A"), input("B"), output("C") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			//generate actual dataset
			double[][] A = getRandomMatrix(rows, 1, -1, 1, sparseM1?sparsity2:sparsity1, 7); 
			writeInputMatrixWithMTD("A", A, true);
			double[][] B = getRandomMatrix(1, cols, -1, 1, sparseM2?sparsity2:sparsity1, 3); 
			writeInputMatrixWithMTD("B", B, true);
	
			//run tests
			runTest(true, false, null, -1); 
			//runRScript(true); R fails here with out-of-memory 
			
			//compare matrices (single minimum)
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("C");
			//HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("C");
			//TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			Double dmlret = dmlfile.get(new CellIndex(1,1));
			Double compare = computeMinOuterProduct(A, B, rows, cols);
			Assert.assertEquals("Wrong result value.", compare, dmlret);
			
			int expectedNumCompiled = 4; //REBLOCK, MMRJ, GMR, GMR write
			int expectedNumExecuted = 4; //REBLOCK, MMRJ, GMR, GMR write
			
			checkNumCompiledMRJobs(expectedNumCompiled); 
			checkNumExecutedMRJobs(expectedNumExecuted); 
		
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
	
	/**
	 * Min over outer product for comparison because R runs out of memory.
	 * 
	 * @param A
	 * @param B
	 * @param rows
	 * @param cols
	 * @return
	 */
	public double computeMinOuterProduct( double[][] A, double[][] B, int rows, int cols )
	{
		double min = Double.MAX_VALUE;
		
		for( int i=0; i<rows; i++ )
		{
			double val1 = A[i][0];
			if( val1!=0 || min <= 0 )
				for( int j=0; j<cols; j++ )
				{
					double val2 = B[0][j];
					double val3 = val1 * val2;
					min = Math.min(min, val3);
				}
		}
		
		return min;
	}
}