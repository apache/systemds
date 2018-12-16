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

public class RewriteCumulativeAggregatesTest extends AutomatedTestBase 
{
	private static final String TEST_NAME = "RewriteCumulativeAggregates";
	private static final String TEST_DIR = "functions/misc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteCumulativeAggregatesTest.class.getSimpleName() + "/";
	
	private static final int rows = 1234;
	private static final int rows2 = 876;
	private static final int cols = 7;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "R" }) );
	}

	@Test
	public void testCumAggRewrite1False() {
		testCumAggRewrite(1, false);
	}
	
	@Test
	public void testCumAggRewrite1True() {
		testCumAggRewrite(1, true);
	}
	
	@Test
	public void testCumAggRewrite2False() {
		testCumAggRewrite(2, false);
	}
	
	@Test
	public void testCumAggRewrite2True() {
		testCumAggRewrite(2, true);
	}
	
	@Test
	public void testCumAggRewrite3False() {
		testCumAggRewrite(3, false);
	}
	
	@Test
	public void testCumAggRewrite3True() {
		testCumAggRewrite(3, true);
	}
	
	@Test
	public void testCumAggRewrite4False() {
		testCumAggRewrite(4, false);
	}
	
	@Test
	public void testCumAggRewrite4True() {
		testCumAggRewrite(4, true);
	}
	
	@Test
	public void testCumAggRewrite4SPSingleRowBlock() {
		testCumAggRewrite(4, true, ExecType.SPARK);
	}
	
	private void testCumAggRewrite(int num, boolean rewrites) {
		testCumAggRewrite(num, rewrites, ExecType.CP);
	}
	
	private void testCumAggRewrite(int num, boolean rewrites, ExecType et)
	{
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		RUNTIME_PLATFORM platformOld = setRuntimePlatform(et);
		
		try {
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{ "-explain","-stats", "-args",
				input("A"), String.valueOf(num), output("R") };
			rCmd = getRCmd(inputDir(), String.valueOf(num), expectedDir());
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			
			//generate input data
			double[][] A = getRandomMatrix((num==4)?
				et==ExecType.CP?1:rows2:rows,
				(num==1)?rows:cols, -1, 1, 0.9, 7); 
			writeInputMatrixWithMTD("A", A, true);
			
			//run performance tests
			runTest(true, false, null, -1);
			runRScript(true);
			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, 1e-7, "Stat-DML", "Stat-R");
			
			//check applied rewrites
			if( rewrites )
				Assert.assertTrue(!heavyHittersContainsString((num==2) ? "rev" : "ucumk+"));
			if( num==4 && et==ExecType.SPARK )
				Assert.assertTrue(!heavyHittersContainsString("ucumk+","ucumack+"));
		}
		finally {
			rtplatform = platformOld;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}
	}
}
