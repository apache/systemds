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

package org.apache.sysml.test.integration.functions.misc;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.apache.sysml.utils.Statistics;

/**
 * Regression test for function recompile-once issue with literal replacement.
 * 
 */
public class RewriteAggBinaryWithConstMultBin extends AutomatedTestBase 
{
	
	private static final String TEST_NAME1 = "RewritePlusBinaryWithConstMultBin";
	private static final String TEST_NAME2 = "RewriteMinusBinaryWithConstMultBin";

	private static final String TEST_DIR = "functions/misc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteAggBinaryWithConstMultBin.class.getSimpleName() + "/";
	
	//private static final int rows = 1234;
	//private static final int cols = 567;
	private static final double eps = Math.pow(10, -10);
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }) );
	}
	
	@Test
	public void testPlusBinaryWithConstMultBinNoRewrite() 
	{
		RewriteAggBinaryWithConstMultBin( TEST_NAME1, false );
	}
	
	
	@Test
	public void testPlusBinaryWithConstMultBinRewrite() 
	{
		RewriteAggBinaryWithConstMultBin( TEST_NAME1, true);
	}
	
	
	@Test
	public void testMinusBinaryWithConstMultBinNoRewrite() 
	{
		RewriteAggBinaryWithConstMultBin( TEST_NAME2, false );
	}
	
	@Test
	public void testMinusBinaryWithConstMultBinRewrite() 
	{
		RewriteAggBinaryWithConstMultBin( TEST_NAME2, true );
	}
	
	
	/**
	 * 
	 * @param condition
	 * @param branchRemoval
	 * @param IPA
	 */
	private void RewriteAggBinaryWithConstMultBin( String testname, boolean rewrites )
	{	
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-explain", "-stats","-args", output("S") };
			
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());			

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("S");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("S");
			Assert.assertTrue(TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R"));
			System.out.println("Test case passed");
			
		}
		finally
		{
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}
		
	}	
}