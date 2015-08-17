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

package com.ibm.bi.dml.test.integration.functions.aggregate;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

/**
 * 
 * 
 */
public class AggregateInfTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME = "InfSum";

	private final static String TEST_DIR = "functions/aggregate/";
	private final static double eps = 1e-10;
	
	private final static int rows = 1205;
	private final static int cols = 1179;
	private final static double sparsity1 = 0.1;
	private final static double sparsity2 = 0.7;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_DIR, TEST_NAME,new String[]{"B"})); 
	}

	
	@Test
	public void testSumPosInfDenseCP() 
	{
		runInfAggregateOperationTest(true, false, ExecType.CP);
	}
	
	@Test
	public void testSumNegInfDenseCP() 
	{
		runInfAggregateOperationTest(false, false, ExecType.CP);
	}
	
	@Test
	public void testSumPosInfSparseCP() 
	{
		runInfAggregateOperationTest(true, true, ExecType.CP);
	}
	
	@Test
	public void testSumNegInfSparseCP() 
	{
		runInfAggregateOperationTest(false, true, ExecType.CP);
	}
	
	@Test
	public void testSumPosInfDenseMR() 
	{
		runInfAggregateOperationTest(true, false, ExecType.MR);
	}
	
	@Test
	public void testSumNegInfDenseMR() 
	{
		runInfAggregateOperationTest(false, false, ExecType.MR);
	}
	
	@Test
	public void testSumPosInfSparseMR() 
	{
		runInfAggregateOperationTest(true, true, ExecType.MR);
	}
	
	@Test
	public void testSumNegInfSparseMR() 
	{
		runInfAggregateOperationTest(false, true, ExecType.MR);
	}

	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runInfAggregateOperationTest( boolean pos, boolean sparse, ExecType instType)
	{
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = (instType==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
	
		try
		{
			double sparsity = (sparse) ? sparsity1 : sparsity2;
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "A",
					                        HOME + OUTPUT_DIR + "B"    };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			//generate actual dataset 
			double[][] A = getRandomMatrix(rows, cols, -0.05, 1, sparsity, 7); 
			double infval = pos ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
			A[7][7] = infval;
			writeInputMatrixWithMTD("A", A, false);
	
			//run test
			runTest(true, false, null, -1); 
		
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("B");
			HashMap<CellIndex, Double> compfile  = new HashMap<CellIndex, Double>();
			compfile.put(new CellIndex(1,1), infval);
			TestUtils.compareMatrices(dmlfile, compfile, eps, "Stat-DML", "Stat-R");
		}
		finally
		{
			rtplatform = platformOld;
		}
	}
}