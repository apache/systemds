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

package org.apache.sysds.test.functions.parfor.misc;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

public class ParForListFrameResultVarsTest extends AutomatedTestBase 
{
	private final static String TEST_DIR = "functions/parfor/";
	private final static String TEST_NAME1 = "parfor_listResults";
	private final static String TEST_NAME2 = "parfor_frameResults";
	
	private final static String TEST_CLASS_DIR = TEST_DIR + ParForListFrameResultVarsTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{"R"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[]{"R"}));
	}

	@Test
	public void testParForListResult1a() {
		runListResultVarTest(TEST_NAME1, 2, 1);
	}
	
	@Test
	public void testParForListResult1b() {
		runListResultVarTest(TEST_NAME1, 35, 10);
	}
	
	@Test
	public void testParForFrameResult1a() {
		runListResultVarTest(TEST_NAME2, 2, 1);
	}
	
	@Test
	public void testParForFrameResult1b() {
		runListResultVarTest(TEST_NAME2, 35, 10);
	}
	
	private void runListResultVarTest(String testName, int rows, int cols) {
		loadTestConfiguration(getTestConfiguration(testName));
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + testName + ".dml";
		programArgs = new String[]{"-explain","-args",
			String.valueOf(rows), String.valueOf(cols), output("R") };

		runTest(true, false, null, -1);
		Assert.assertEquals(Double.valueOf(7),
			readDMLMatrixFromOutputDir("R").get(new CellIndex(1,1)));
	}
}
