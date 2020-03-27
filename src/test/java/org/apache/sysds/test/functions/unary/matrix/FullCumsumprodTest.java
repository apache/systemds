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

package org.apache.sysds.test.functions.unary.matrix;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

public class FullCumsumprodTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "Cumsumprod";
	private final static String TEST_DIR = "functions/unary/matrix/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FullCumsumprodTest.class.getSimpleName() + "/";
	
	private final static int rows = 1201;
	private final static double spDense = 1.0;
	private final static double spSparse = 0.3;
	
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"})); 
	}
	
	@Test
	public void testCumsumprodForwardDenseCP() {
		runCumsumprodTest(false, false, ExecType.CP);
	}
	
	@Test
	public void testCumsumprodForwardSparseCP() {
		runCumsumprodTest(false, true, ExecType.CP);
	}
	
	@Test
	public void testCumsumprodBackwardDenseCP() {
		runCumsumprodTest(true, false, ExecType.CP);
	}
	
	@Test
	public void testCumsumprodBackwardSparseCP() {
		runCumsumprodTest(true, true, ExecType.CP);
	}
	
	@Test
	public void testCumsumprodForwardDenseSP() {
		runCumsumprodTest(false, false, ExecType.SPARK);
	}
	
	@Test
	public void testCumsumprodForwardSparseSP() {
		runCumsumprodTest(false, true, ExecType.SPARK);
	}
	
	@Test
	public void testCumsumprodBackwardDenseSP() {
		runCumsumprodTest(true, false, ExecType.SPARK);
	}
	
	@Test
	public void testCumsumprodBackwardSparseSP() {
		runCumsumprodTest(true, true, ExecType.SPARK);
	}
	
	private void runCumsumprodTest(boolean reverse, boolean sparse, ExecType instType)
	{
		ExecMode platformOld = rtplatform;
		switch( instType ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK || rtplatform == ExecMode.HYBRID )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		try
		{
			double sparsity = sparse ? spSparse : spDense;
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			// This is for running the junit test the new way, i.e., construct the arguments directly
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain", "-stats", "-args", input("A"), input("B"),
				String.valueOf(reverse).toUpperCase(), output("C") };
			
			double[][] A = getRandomMatrix(rows, 1, -10, 10, sparsity, 3);
			double[][] B = getRandomMatrix(rows, 1, -1, 1, 0.9, 7);
			writeInputMatrixWithMTD("A", A, false);
			writeInputMatrixWithMTD("B", B, false);
			
			runTest(true, false, null, -1); 
			
			Assert.assertEquals(new Double(rows),
				readDMLMatrixFromHDFS("C").get(new CellIndex(1,1)));
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
