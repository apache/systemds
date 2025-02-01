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

import org.apache.sysds.common.Opcodes;
import org.junit.Assert;
import org.junit.Test;

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.utils.Statistics;

public class RewriteForLoopRemovalTest extends AutomatedTestBase
{
	private final static String TEST_NAME1 = "removal_for";
	private final static String TEST_NAME2 = "removal_parfor";
	private final static String TEST_DIR = "functions/rewrite/";
	private final static String TEST_CLASS_DIR = TEST_DIR + RewriteForLoopRemovalTest.class.getSimpleName() + "/";
	
	private final static int rows = 10;
	private final static int cols = 15;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"R"}) );
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"R"}) );
	}
	
	@Test
	public void runForLoopRemovalTest() {
		runLoopRemovalTest(TEST_NAME1, true);
	}
	
	@Test
	public void runParForLoopRemovalTest() {
		runLoopRemovalTest(TEST_NAME2, true);
	}
	
	@Test
	public void runForLoopRemovalNoRewriteTest() {
		runLoopRemovalTest(TEST_NAME1, false);
	}
	
	@Test
	public void runParForLoopRemovalNoRewriteTest() {
		runLoopRemovalTest(TEST_NAME2, false);
	}
	
	private void runLoopRemovalTest(String testname, boolean rewrites)
	{
		boolean rewritesOld = OptimizerUtils.ALLOW_FOR_LOOP_REMOVAL;
		OptimizerUtils.ALLOW_FOR_LOOP_REMOVAL = rewrites;
		
		try {
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-explain", "-stats", "-args",
				Integer.toString(rows), Integer.toString(cols)};
			
			runTest(true, false, null, -1);
			
			//check for applied rewrite (which enabled CSE of sum)
			long cnt = Statistics.getCPHeavyHitterCount(Opcodes.UAKP.toString());
			long expected = rewrites ? 1 : 2;
			Assert.assertEquals(expected, cnt);
		}
		finally {
			OptimizerUtils.ALLOW_FOR_LOOP_REMOVAL = rewritesOld;
		}
	}
}
