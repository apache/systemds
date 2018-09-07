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

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Assert;
import org.junit.Test;

/**
 * Test rewriting `2*X*3*v*5*w*4*z*5*Y*2*v*2*X`, where `v` and `z` are row vectors and `w` is a column vector,
 * successfully rewrites to `Y*(X^2)*z*(v^2)*w*2400`.
 */
public class RewriteElementwiseMultChainOptimizationAllTest extends AutomatedTestBase
{
	private static final String TEST_NAME1 = "RewriteEMultChainOpAll";
	private static final String TEST_DIR = "functions/misc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteElementwiseMultChainOptimizationAllTest.class.getSimpleName() + "/";
	
	private static final int rows = 123;
	private static final int cols = 321;
	private static final double eps = Math.pow(10, -10);
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
	}

	@Test
	public void testMatrixMultChainOptNoRewritesCP() {
		testRewriteMatrixMultChainOp(TEST_NAME1, false, ExecType.CP);
	}
	
	@Test
	public void testMatrixMultChainOptNoRewritesSP() {
		testRewriteMatrixMultChainOp(TEST_NAME1, false, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixMultChainOptRewritesCP() {
		testRewriteMatrixMultChainOp(TEST_NAME1, true, ExecType.CP);
	}
	
	@Test
	public void testMatrixMultChainOptRewritesSP() {
		testRewriteMatrixMultChainOp(TEST_NAME1, true, ExecType.SPARK);
	}

	private void testRewriteMatrixMultChainOp(String testname, boolean rewrites, ExecType et)
	{	
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( et ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID_SPARK; break;
		}
		
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK || rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		boolean rewritesOld = OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES;
		OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = rewrites;
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[] { "-explain", "hops", "-stats", "-args", input("X"), input("Y"), input("v"), input("z"), input("w"), output("R") };
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());

			double Xsparsity = 0.8, Ysparsity = 0.6;
			double[][] X = getRandomMatrix(rows, cols, -1, 1, Xsparsity, 7);
			double[][] Y = getRandomMatrix(rows, cols, -1, 1, Ysparsity, 3);
			double[][] z = getRandomMatrix(1, cols, -1, 1, Ysparsity, 5);
			double[][] v = getRandomMatrix(1, cols, -1, 1, Xsparsity, 4);
			double[][] w = getRandomMatrix(rows, 1, -1, 1, Ysparsity, 6);
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("Y", Y, true);
			writeInputMatrixWithMTD("z", z, true);
			writeInputMatrixWithMTD("v", v, true);
			writeInputMatrixWithMTD("w", w, true);

			//execute tests
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			
			//check for presence of power operator, if we did a rewrite
			if( rewrites ) {
				Assert.assertTrue(heavyHittersContainsSubString("^2"));
			}
		}
		finally {
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = rewritesOld;
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
