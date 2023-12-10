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

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;

public class RewriteMatrixMultChainOptTransposeTest extends AutomatedTestBase
{
	private static final String TEST_NAME = "RewriteMMChainTestTranspose";
	protected static final int TEST_VARIANTS = 4;
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteMatrixMultChainOptTransposeTest.class.getSimpleName() + "/";

	private static final double eps = Math.pow(10, -10);

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		for( int i=1; i<=TEST_VARIANTS; i++ )
			addTestConfiguration(TEST_NAME+i, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME+i, new String[] {"R"}));
	}

	@Test
	public void testSameInputUsedMultipleTimes() {
		testMMChainWithTransposeOperator(TEST_NAME + "1", 4);
	}

	@Test
	public void testTwoMultiplicationsInTransposeOperator() {
		testMMChainWithTransposeOperator(TEST_NAME + "2", 2);
	}

	@Test
	public void testTransposeInTranspose() {
		testMMChainWithTransposeOperator(TEST_NAME + "3", 4);
	}

	@Test
	public void testMMChainFour() {
		testMMChainWithTransposeOperator(TEST_NAME + "4", 2);
	}

	private void testMMChainWithTransposeOperator(String testname, int numberOfTransposeOperators)
	{
		ExecMode etOld = setExecMode(ExecMode.SINGLE_NODE);

		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{ "-explain", "hops", "-stats",
				"-args", output("R") };
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());

			//execute tests
			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

			Assert.assertEquals(numberOfTransposeOperators, Statistics.getCPHeavyHitterCount(Types.ReOrgOp.TRANS.toString()));
		}
		finally {
			resetExecMode(etOld);
		}
	}
}
