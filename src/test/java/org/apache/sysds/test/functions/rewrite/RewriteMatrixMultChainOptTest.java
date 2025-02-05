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

package org.apache.sysds.test.functions.rewrite;

import java.util.HashMap;

import org.apache.sysds.common.Opcodes;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class RewriteMatrixMultChainOptTest extends AutomatedTestBase 
{
	private static final String TEST_NAME1 = "RewriteMatrixMultChainOp";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteMatrixMultChainOptTest.class.getSimpleName() + "/";
	
	private static final int rows = 1234;
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
		ExecMode platformOld = rtplatform;
		switch( et ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
		
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK || rtplatform == ExecMode.HYBRID )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		boolean rewritesOld = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{ "-explain", "hops", "-stats", 
				"-args", input("X"), input("Y"), output("R") };
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());			

			double[][] X = getRandomMatrix(rows, cols, -1, 1, 0.97d, 7);
			double[][] Y = getRandomMatrix(cols, 1, -1, 1, 0.9d, 3);
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("Y", Y, true);
			
			//execute tests
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			
			//check for correct matrix multiplication order, which also allows
			//the compilation of mmchain operators
			if( rewrites ) {
				Assert.assertTrue(heavyHittersContainsSubString(Opcodes.MMCHAIN.toString())
					|| heavyHittersContainsSubString("sp_mapmmchain"));
			}
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewritesOld;
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			Recompiler.reinitRecompiler();
		}
	}
}
