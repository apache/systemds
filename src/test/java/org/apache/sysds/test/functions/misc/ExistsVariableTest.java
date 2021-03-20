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

public class ExistsVariableTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "Exists1"; //for var names
	private final static String TEST_NAME2 = "Exists2"; //for vars
	
	private final static String TEST_DIR = "functions/misc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + ExistsVariableTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{"R"}));
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[]{"R"}));
	}

	@Test
	public void testExistsVarnamePositive() {
		runExistsTest(TEST_NAME1, true);
	}
	
	@Test
	public void testExistsVarnameNegative() {
		runExistsTest(TEST_NAME1, false);
	}
	
	@Test
	public void testExistsVarPositive() {
		runExistsTest(TEST_NAME2, true);
	}
	
	@Test
	public void testExistsVarNegative() {
		runExistsTest(TEST_NAME2, false);
	}
	
	private void runExistsTest(String testName, boolean pos) {
		TestConfiguration config = getTestConfiguration(testName);
		loadTestConfiguration(config);
		String HOME = SCRIPT_DIR + TEST_DIR;
		String param = pos ? "1" : "0";
		fullDMLScriptName = HOME + testName + ".dml";
		programArgs = new String[]{"-args", param, output("R") };
		
		//run script and compare output
		runTest(true, false, null, -1); 
		
		//compare results
		Double val = readDMLMatrixFromOutputDir("R").get(new CellIndex(1,1));
		val = (val!=null) ? val : 0;
		Assert.assertTrue("Wrong result: "+param+" vs "+val,
			val==Double.parseDouble(param));
	}
}
