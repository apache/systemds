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
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class RewriteMatrixChainDPTest extends AutomatedTestBase {

	private static final String TEST_DIR = "functions/rewrite/mmchain/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteMatrixChainDPTest.class.getSimpleName() + "/";

	private static final String[] TEST_CASES = {
		"test1", "test2", "test3", "test4", "test5", "test6", "test7",
		"test8", "test9", "test10", "test11", "test12","test13",
		"test14", "test15", "test16", "test17", "test18", "test19",
		"test20", "test21", "test22", "test23", "test24"
	};

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		for (String testName : TEST_CASES) {
			addTestConfiguration(testName, new TestConfiguration(TEST_CLASS_DIR, testName, new String[] {"R"}));
		}
	}

	@Test
	public void testMatrixChainDPTest1() {  runTestMatrixChainDP(TEST_CASES[0]); }

	@Test
	public void testMatrixChainDPTest2() { runTestMatrixChainDP(TEST_CASES[1]); }

	@Test
	public void testMatrixChainDPTest3() { runTestMatrixChainDP(TEST_CASES[2]); }

	@Test
	public void testMatrixChainDPTest4() { runTestMatrixChainDP(TEST_CASES[3]); }

	@Test
	public void testMatrixChainDPTest5() { runTestMatrixChainDP(TEST_CASES[4]); }

	@Test
	public void testMatrixChainDPTest6() { runTestMatrixChainDP(TEST_CASES[5]); }

	@Test
	public void testMatrixChainDPTest7() { runTestMatrixChainDP(TEST_CASES[6]); }

	@Test
	public void testMatrixChainDPTest8() { runTestMatrixChainDP(TEST_CASES[7]); }

	@Test
	public void testMatrixChainDPTest9() { runTestMatrixChainDP(TEST_CASES[8]); }

	@Test
	public void testMatrixChainDPTest10() { runTestMatrixChainDP(TEST_CASES[9]); }

	@Test
	public void testMatrixChainDPTest11() { runTestMatrixChainDP(TEST_CASES[10]); }

	@Test
	public void testMatrixChainDPTest12() { runTestMatrixChainDP(TEST_CASES[11]); }

	@Test
	public void testMatrixChainDPTest13() { runTestMatrixChainDP(TEST_CASES[12]); }

	@Test
	public void testMatrixChainDPTest14() { runTestMatrixChainDP(TEST_CASES[13]); }

	@Test
	public void testMatrixChainDPTest15() { runTestMatrixChainDP(TEST_CASES[14]); }

	@Test
	public void testMatrixChainDPTest16() { runTestMatrixChainDP(TEST_CASES[15]); }

	@Test
	public void testMatrixChainDPTest17() { runTestMatrixChainDP(TEST_CASES[16]); }

	@Test
	public void testMatrixChainDPTest18() { runTestMatrixChainDP(TEST_CASES[17]); }

	@Test
	public void testMatrixChainDPTest19() { runTestMatrixChainDP(TEST_CASES[18]); }

	@Test
	public void testMatrixChainDPTest20() { runTestMatrixChainDP(TEST_CASES[19]); }

	@Test
	public void testMatrixChainDPTest21() { runTestMatrixChainDP(TEST_CASES[20]); }

	@Test
	public void testMatrixChainDPTest22() { runTestMatrixChainDP(TEST_CASES[21]); }

	@Test
	public void testMatrixChainDPTest23() { runTestMatrixChainDP(TEST_CASES[22]); }

	@Test
	public void testMatrixChainDPTest24() {runTestMatrixChainDP(TEST_CASES[23]);}


	private void runTestMatrixChainDP(String testName) {
		ExecMode platformOld = rtplatform;
		boolean rewritesOld = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean newMMchain1 = OptimizerUtils.ALLOW_ADVANCED_MMCHAIN_REWRITES;
		boolean newMMchain2 = OptimizerUtils.ALLOW_NEW_MMCHAIN_REWRITE;

		try {
			rtplatform = ExecMode.SINGLE_NODE;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = true;
			OptimizerUtils.ALLOW_ADVANCED_MMCHAIN_REWRITES = true;
			OptimizerUtils.ALLOW_NEW_MMCHAIN_REWRITE = true;

			TestConfiguration config = getTestConfiguration(testName);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testName + ".dml";

			programArgs = new String[]{ "-explain", "hops", "-stats", "-args", output("R") };

			// Execute the DML script
			setOutputBuffering(true);
			String output = runTest(true, false, null, -1).toString();

			/* the following uses the intermediate matrices dimensions to check, whether
			 * the rewrite rule has found the optimal plan, which is commented in each script
			 */
			switch(testName) {
				case "test1" -> {
					Assert.assertTrue(output.contains("[4,1"));
					Assert.assertFalse(output.contains("[2,4"));
				}
				case "test2" -> {
					Assert.assertTrue(output.contains("[10,5"));
					Assert.assertTrue(output.contains("[50,5"));
				}
				case "test3" -> {
					Assert.assertTrue(output.contains("[30,2"));
					Assert.assertTrue(output.contains("[2,5"));
				}
				case "test4" ->{
					Assert.assertTrue(output.contains("[4,5"));
					Assert.assertTrue(output.contains("[4,9"));
					Assert.assertTrue(output.contains("[4,8"));
				}
				case "test5" -> {
					Assert.assertTrue(output.contains("[8,3"));
					Assert.assertTrue(output.contains("[3,4"));
					Assert.assertFalse(output.contains("[4,8"));
					Assert.assertFalse(output.contains("[8,8"));
				}
				case "test6" -> {
					Assert.assertTrue(output.contains("[8,9"));
					Assert.assertTrue(output.contains("[6,2"));
					Assert.assertTrue(output.contains("[9,8"));
					Assert.assertTrue(output.contains("[2,2"));
					Assert.assertTrue(output.contains("[8,2"));
				}
				case "test7" -> {
					Assert.assertTrue(output.contains("[1,1000"));
					Assert.assertFalse(output.contains("[1000,2"));
				}
				case "test8" -> {
					Assert.assertTrue(output.contains("[30,2"));
					Assert.assertTrue(output.contains("[2,30"));
					Assert.assertTrue(output.contains("[30,2"));
					Assert.assertTrue(output.contains("[2,4"));
				}
				case "test9" -> {
					Assert.assertTrue(output.contains("[10,2"));
					Assert.assertTrue(output.contains("[3,2"));
					Assert.assertTrue(output.contains("[2,10"));
					Assert.assertTrue(output.contains("[2,30"));
					Assert.assertFalse(output.contains("[3,10"));
				}
				case "test10" -> {
					Assert.assertTrue(output.contains("[2,55"));
					Assert.assertTrue(output.contains("[35,2"));
					Assert.assertTrue(output.contains("[2,3"));
					Assert.assertTrue(output.contains("[3,2"));
				}
				case "test11" -> {
					Assert.assertTrue(output.contains("[3,55"));
					Assert.assertTrue(output.contains("[3,23"));
					Assert.assertTrue(output.contains("[35,3"));
					Assert.assertTrue(output.contains("[23,3"));
				}
				case "test12" -> {
					Assert.assertTrue(output.contains("[3,43"));
					Assert.assertTrue(output.contains("[3,12"));
					Assert.assertTrue(output.contains("[3,23"));
					Assert.assertTrue(output.contains("[23,3"));
					Assert.assertTrue(output.contains("[33,3"));
				}
				case "test13" -> {
					Assert.assertFalse(output.contains("[13,14"));
					Assert.assertTrue(output.contains("[14,12"));
					Assert.assertTrue(output.contains("[12,14"));
					Assert.assertTrue(output.contains("[12,16"));
					Assert.assertTrue(output.contains("[16,12"));
				}
				case "test14" -> {
					Assert.assertTrue(output.contains("[9,12"));
					Assert.assertTrue(output.contains("[12,9"));
					Assert.assertTrue(output.contains("[13,9"));
					Assert.assertTrue(output.contains("[16,9"));
					Assert.assertTrue(output.contains("[9,16"));
					Assert.assertTrue(output.contains("[9,14"));
				}
				case "test15" -> {
					Assert.assertTrue(output.contains("[12,12"));
					Assert.assertTrue(output.contains("[12,13"));
					Assert.assertTrue(output.contains("[13,16"));
					Assert.assertTrue(output.contains("[13,14"));
				}
				case "test16" -> {
					Assert.assertFalse(output.contains("[16,22"));
					Assert.assertTrue(output.contains("[13,16"));
					Assert.assertTrue(output.contains("[13,14"));
					Assert.assertTrue(output.contains("[22,13"));
					Assert.assertTrue(output.contains("[18,13"));
					Assert.assertTrue(output.contains("[12,13"));
				}
				case "test17" -> {
					Assert.assertFalse(output.contains("[23,16"));
					Assert.assertTrue(output.contains("[16,22"));
					Assert.assertTrue(output.contains("[22,16"));
					Assert.assertTrue(output.contains("[443,16"));
					Assert.assertTrue(output.contains("[124,16"));
					Assert.assertTrue(output.contains("[124,34"));
				}
				case "test18" -> {
					Assert.assertFalse(output.contains("[23,16"));
					Assert.assertTrue(output.contains("[16,22"));
					Assert.assertTrue(output.contains("[22,16"));
					Assert.assertTrue(output.contains("[33,22"));
					Assert.assertTrue(output.contains("[33,16"));
				}
				case "test19" -> {
					Assert.assertTrue(output.contains("[2,6"));
					Assert.assertTrue(output.contains("[2,4"));
					Assert.assertTrue(output.contains("[3,2"));
				}
				case "test20" -> {
					Assert.assertTrue(output.contains("[10,30"));
					Assert.assertTrue(output.contains("[10,5"));
					Assert.assertTrue(output.contains("[50,5"));
				}
				case "test21" -> {
					Assert.assertTrue(output.contains("[5,3"));
					Assert.assertTrue(output.contains("[40,3"));
					Assert.assertTrue(output.contains("[6,3"));
					Assert.assertTrue(output.contains("[3,6"));
					Assert.assertTrue(output.contains("[3,50"));
					Assert.assertFalse(output.contains("[20,6"));
				}
				case "test22" -> {
					Assert.assertTrue(output.contains("[23,34"));
					Assert.assertTrue(output.contains("[15,34"));
					Assert.assertTrue(output.contains("[15,25"));
					Assert.assertTrue(output.contains("[15,18"));
					Assert.assertTrue(output.contains("[18,15"));
					Assert.assertTrue(output.contains("[15,24"));
					Assert.assertTrue(output.contains("[15,16"));
				}
				case "test23" -> {
					Assert.assertTrue(output.contains("[10,5"));
					Assert.assertTrue(output.contains("[20,5"));
					Assert.assertTrue(output.contains("[10,6"));
					Assert.assertTrue(output.contains("[1000,6"));
				}
				case "test24" -> {
					Assert.assertTrue(output.contains("[9,5"));
					Assert.assertTrue(output.contains("[4,9"));
					Assert.assertTrue(output.contains("[5,4"));
					Assert.assertTrue(output.contains("[6,5"));
				}
			}
		} finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewritesOld;
			OptimizerUtils.ALLOW_ADVANCED_MMCHAIN_REWRITES = newMMchain1;
			OptimizerUtils.ALLOW_NEW_MMCHAIN_REWRITE = newMMchain2;
			rtplatform = platformOld;
			Recompiler.reinitRecompiler();
		}
	}
}
