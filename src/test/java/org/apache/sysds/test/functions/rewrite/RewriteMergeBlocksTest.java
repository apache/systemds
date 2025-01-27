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
import org.junit.Test;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;

public class RewriteMergeBlocksTest extends AutomatedTestBase 
{
	private static final String TEST_NAME1 = "RewriteMergeIfCut"; //full merge
	private static final String TEST_NAME2 = "RewriteMergeFunctionCut"; //full merge
	private static final String TEST_NAME3 = "RewriteMergeFunctionCut2"; //only input merge
	private static final String TEST_NAME4 = "RewriteMergeFunctionCut3"; //only input merge
	private static final String TEST_NAME5 = "RewriteMergeFunctionCut4"; //only input merge
	

	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteMergeBlocksTest.class.getSimpleName() + "/";
	
	private static final double eps = Math.pow(10,-10);
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{"R"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[]{"R"}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[]{"R"}));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[]{"R"}));
		addTestConfiguration(TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5, new String[]{"R"}));
	}
	
	@Test
	public void testIfCutMerge() {
		testRewriteMerge(TEST_NAME1, true);
	}
	
	@Test
	public void testFunctionCutMerge() {
		testRewriteMerge(TEST_NAME2, true);
	}
	
	@Test
	public void testFunctionCutMerge2() {
		testRewriteMerge(TEST_NAME3, false);
	}
	
	@Test
	public void testFunctionCutMerge3() {
		testRewriteMerge(TEST_NAME4, false);
	}
	
	@Test
	public void testFunctionCutMerge4() {
		//note: this test primarily checks for result correctness
		//(prevent too eager merge of functions)
		testRewriteMerge(TEST_NAME5, true);
	}
	
	private void testRewriteMerge(String testname, boolean expectedMerge) {
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
		HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
		HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("R");
		TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		Assert.assertTrue(expectedMerge == 
			heavyHittersContainsSubString(Opcodes.MMCHAIN.toString()));
	}
}
