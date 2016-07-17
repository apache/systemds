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
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.apache.sysml.utils.Statistics;

/**
 * Regression test for function recompile-once issue with literal replacement.
 * 
 */
public class RewriteFuseBinaryOpChainTest extends AutomatedTestBase 
{
	
	private static final String TEST_NAME1 = "RewriteFuseBinaryOpChainTest1";
	private static final String TEST_NAME2 = "RewriteFuseBinaryOpChainTest2";

	private static final String TEST_DIR = "functions/misc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteFuseBinaryOpChainTest.class.getSimpleName() + "/";
	
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
	public void testFuseBinaryPlusNoRewrite() 
	{
		testFuseBinaryChain( TEST_NAME1, false, ExecType.CP );
	}
	
	
	@Test
	public void testFuseBinaryPlusRewrite() 
	{
		testFuseBinaryChain( TEST_NAME1, true, ExecType.CP);
	}
	@Test
	public void testFuseBinaryMinusNoRewrite() 
	{
		testFuseBinaryChain( TEST_NAME2, false, ExecType.CP );
	}
	
	@Test
	public void testFuseBinaryMinusRewrite() 
	{
		testFuseBinaryChain( TEST_NAME2, true, ExecType.CP );
	}
	
	
	
	@Test
	public void testSpFuseBinaryPlusNoRewrite() 
	{
		testFuseBinaryChain( TEST_NAME1, false, ExecType.SPARK );
	}
	
	
	@Test
	public void testSpFuseBinaryPlusRewrite() 
	{
		testFuseBinaryChain( TEST_NAME1, true, ExecType.SPARK );
	}
	
	
	@Test
	public void testSpFuseBinaryMinusNoRewrite() 
	{
		testFuseBinaryChain( TEST_NAME2, false, ExecType.SPARK  );
	}
	
	@Test
	public void testSpFuseBinaryMinusRewrite() 
	{
		testFuseBinaryChain( TEST_NAME2, true, ExecType.SPARK  );
	}
	
	
	/**
	 * 
	 * @param condition
	 * @param branchRemoval
	 * @param IPA
	 */
	private void testFuseBinaryChain( String testname, boolean rewrites, ExecType instType )
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
		boolean rewritesOld = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
		try
		{
			
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-explain", "-stats","-args", output("S") };
			
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());			

			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("S");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("S");
			Assert.assertTrue(TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R"));
		}
		finally
		{
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewritesOld;
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
		
	}	
}