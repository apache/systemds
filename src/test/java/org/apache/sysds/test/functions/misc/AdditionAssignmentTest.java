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

package org.apache.sysds.test.functions.misc;


import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class AdditionAssignmentTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "AddAssign";
	
	private final static String TEST_DIR = "functions/misc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + AdditionAssignmentTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{"R"}));
	}

	@Test
	public void testMatrixScalarAddition() {
		runExistsTest(TEST_NAME1, false, 5500);
	}
	
	@Test
	public void testScalarScalarAddition() {
		runExistsTest(TEST_NAME1, true, 55);
	}

	private void runExistsTest(String testName, boolean scalarTarget, double expected) {
		TestConfiguration config = getTestConfiguration(testName);
		loadTestConfiguration(config);
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + testName + ".dml";
		programArgs = new String[]{"-args",
			String.valueOf(scalarTarget).toUpperCase(), output("R") };
		
		//run script and compare output
		runTest(true, false, null, -1); 
		
		//compare results
		Double val = readDMLScalarFromOutputDir("R").get(new CellIndex(1,1));
		Assert.assertEquals(expected, val, 1e-12);
	}
}
