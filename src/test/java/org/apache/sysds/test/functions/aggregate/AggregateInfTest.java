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

package org.apache.sysds.test.functions.aggregate;

import java.util.HashMap;

import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

/**
 * 
 * 
 */
public class AggregateInfTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "InfSum";

	private final static String TEST_DIR = "functions/aggregate/";
	private static final String TEST_CLASS_DIR = TEST_DIR + AggregateInfTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rows = 1205;
	private final static int cols = 1179;
	private final static double sparsity1 = 0.1;
	private final static double sparsity2 = 0.7;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME,
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"B"})); 
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

	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runInfAggregateOperationTest( boolean pos, boolean sparse, ExecType instType)
	{
		//rtplatform for MR
		ExecMode platformOld = rtplatform;
		rtplatform = ExecMode.HYBRID;
	
		try
		{
			double sparsity = (sparse) ? sparsity1 : sparsity2;
			
			getAndLoadTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("A"), output("B") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			//generate actual dataset 
			double[][] A = getRandomMatrix(rows, cols, -0.05, 1, sparsity, 7); 
			double infval = pos ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
			A[7][7] = infval;
			writeInputMatrixWithMTD("A", A, false);
	
			//run test
			runTest(true, false, null, -1); 
		
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("B");
			HashMap<CellIndex, Double> compfile  = new HashMap<>();
			compfile.put(new CellIndex(1,1), infval);
			TestUtils.compareMatrices(dmlfile, compfile, eps, "Stat-DML", "Stat-R");
		}
		finally
		{
			rtplatform = platformOld;
		}
	}
}