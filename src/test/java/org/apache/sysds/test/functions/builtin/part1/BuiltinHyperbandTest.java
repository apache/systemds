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

package org.apache.sysds.test.functions.builtin.part1;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;


public class BuiltinHyperbandTest extends AutomatedTestBase
{
	private final static String TEST_NAME1 = "HyperbandLM";
	private final static String TEST_NAME2 = "HyperbandLM2";
	private final static String TEST_NAME3 = "HyperbandLM3";
	
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinHyperbandTest.class.getSimpleName() + "/";
	
	private final static int rows = 300;
	private final static int cols = 20;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1,new String[]{"R"}));
		addTestConfiguration(TEST_NAME2,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2,new String[]{"R"}));
		addTestConfiguration(TEST_NAME3,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3,new String[]{"R"}));
	}
	
	@Test
	public void testHyperbandCP() {
		runHyperband(TEST_NAME1, ExecType.CP);
	}

	@Test
	public void testHyperbandNoCompareCP() {
		runHyperband(TEST_NAME2, ExecType.CP);
	}
	
	@Test
	public void testHyperbandNoCompare2CP() {
		runHyperband(TEST_NAME3, ExecType.CP);
	}
	
	@Test
	public void testHyperbandSpark() {
		runHyperband(TEST_NAME2, ExecType.SPARK);
	}
	
	private void runHyperband(String testname, ExecType et) {
		ExecMode modeOld = setExecMode(et);
		int retries = ParForProgramBlock.MAX_RETRYS_ON_ERROR;
		
		try {
			loadTestConfiguration(getTestConfiguration(testname));
			String HOME = SCRIPT_DIR + TEST_DIR;
			ParForProgramBlock.MAX_RETRYS_ON_ERROR = 0;
			
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[] {"-stats","-args", input("X"), input("y"), output("R")};
			double[][] X = getRandomMatrix(rows, cols, 0, 1, 0.8, 3);
			double[][] y = getRandomMatrix(rows, 1, 0, 1, 0.8, 7);
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("y", y, true);
			
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			
			//expected loss smaller than default invocation
			if( testname.equals(TEST_NAME1) )
				Assert.assertTrue(TestUtils.readDMLBoolean(output("R")));
		}
		finally {
			ParForProgramBlock.MAX_RETRYS_ON_ERROR = retries;
			resetExecMode(modeOld);
		}
	}
}
