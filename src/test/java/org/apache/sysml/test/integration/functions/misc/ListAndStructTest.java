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

import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;

import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.apache.sysml.utils.Statistics;

public class ListAndStructTest extends AutomatedTestBase 
{
	private static final String TEST_NAME1 = "ListUnnamed";
	private static final String TEST_NAME2 = "ListNamed";
	private static final String TEST_NAME3 = "ListUnnamedFun";
	private static final String TEST_NAME4 = "ListNamedFun";
	private static final String TEST_NAME5 = "ListUnnamedParfor";
	private static final String TEST_NAME6 = "ListNamedParfor";
	
	private static final String TEST_DIR = "functions/misc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ListAndStructTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME6, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME6, new String[] { "R" }) );
	}
	
	@Test
	public void testListUnnamed() {
		runListStructTest(TEST_NAME1, false);
	}
	
	@Test
	public void testListUnnamedRewrites() {
		runListStructTest(TEST_NAME1, true);
	}
	
	@Test
	public void testListNamed() {
		runListStructTest(TEST_NAME2, false);
	}
	
	@Test
	public void testListNamedRewrites() {
		runListStructTest(TEST_NAME2, true);
	}
	
	@Test
	public void testListUnnamedFun() {
		runListStructTest(TEST_NAME3, false);
	}
	
	@Test
	public void testListUnnamedFunRewrites() {
		runListStructTest(TEST_NAME3, true);
	}
	
	@Test
	public void testListNamedFun() {
		runListStructTest(TEST_NAME4, false);
	}
	
	@Test
	public void testListNamedFunRewrites() {
		runListStructTest(TEST_NAME4, true);
	}
	
	@Test
	public void testListUnnamedParFor() {
		runListStructTest(TEST_NAME5, false);
	}
	
	@Test
	public void testListUnnamedParForRewrites() {
		runListStructTest(TEST_NAME5, true);
	}
	
	@Test
	public void testListNamedParFor() {
		runListStructTest(TEST_NAME6, false);
	}
	
	@Test
	public void testListNamedParForRewrites() {
		runListStructTest(TEST_NAME6, true);
	}
	
	private void runListStructTest(String testname, boolean rewrites)
	{
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		
		try {
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{ "-stats","-explain","-args", output("R") };
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(expectedDir());
			
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			
			//run test
			runTest(true, false, null, -1);
			runRScript(true);
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
			Assert.assertEquals(dmlfile.get(new CellIndex(1,1)), rfile.get(new CellIndex(1,1)));
			
			//check for properly compiled CP operations
			Assert.assertTrue(Statistics.getNoOfExecutedMRJobs()==0);
			Assert.assertTrue(Statistics.getNoOfExecutedSPInst()==0);
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}
	}
}
