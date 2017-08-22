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
import org.junit.Test;
import org.junit.Assert;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class RewriteMergeBlocksTest extends AutomatedTestBase 
{
	private static final String TEST_NAME1 = "RewriteMergeIfCut"; //full merge
	private static final String TEST_NAME2 = "RewriteMergeFunctionCut"; //only input merge

	private static final String TEST_DIR = "functions/misc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteMergeBlocksTest.class.getSimpleName() + "/";
	
	private static final double eps = Math.pow(10,-10);
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{"R"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[]{"R"}));
	}
	
	@Test
	public void testIfCutMerge() {
		testRewriteMerge(TEST_NAME1);
	}
	
	@Test
	public void testFunctionCutMerge() {
		testRewriteMerge(TEST_NAME2);
	}
	
	private void testRewriteMerge(String testname)
	{	
		TestConfiguration config = getTestConfiguration(testname);
		loadTestConfiguration(config);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + testname + ".dml";
		programArgs = new String[]{ "-explain","-stats","-args", output("R") };
		
		fullRScriptName = HOME + testname + ".R";
		rCmd = getRCmd(expectedDir());

		runTest(true, false, null, -1); 
		runRScript(true); 
		
		//compare outputs and check for compiled mmchain as proof for merge blocks 
		HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
		HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
		TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		Assert.assertTrue(testname.equals(TEST_NAME1) == 
			heavyHittersContainsSubString("mmchain"));
	}	
}
