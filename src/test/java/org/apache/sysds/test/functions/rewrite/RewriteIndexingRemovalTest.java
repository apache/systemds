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

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.utils.Statistics;

public class RewriteIndexingRemovalTest extends AutomatedTestBase
{
	private final static String TEST_NAME1 = "removal_rix1";
	private final static String TEST_NAME2 = "removal_rix2";
	private final static String TEST_DIR = "functions/rewrite/";
	private final static String TEST_CLASS_DIR = TEST_DIR + RewriteIndexingRemovalTest.class.getSimpleName() + "/";
	
	private final static int rows = 10;
	private final static int cols = 15;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"R"}) );
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"R"}) );
	}
	
	@Test
	public void runIndexingRemovalTest() {
		runIndexingRemovalTest(TEST_NAME1);
	}
	
	@Test
	public void runDynIndexingRemovalTest() {
		runIndexingRemovalTest(TEST_NAME2);
	}
	
	private void runIndexingRemovalTest(String testname) {
		TestConfiguration config = getTestConfiguration(testname);
		loadTestConfiguration(config);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + testname + ".dml";
		programArgs = new String[]{"-explain", "-stats", "-args",
			Integer.toString(rows), Integer.toString(cols)};
		
		runTest(true, false, null, -1);
		Assert.assertEquals(0, Statistics.getCPHeavyHitterCount("rix"));
		if(testname.equals(TEST_NAME2))
			Assert.assertEquals(2, Statistics.getNoOfCompiledSPInst());
	}
}
