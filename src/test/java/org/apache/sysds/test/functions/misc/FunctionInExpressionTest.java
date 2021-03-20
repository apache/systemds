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

public class FunctionInExpressionTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "FunInExpression1";
	private final static String TEST_NAME2 = "FunInExpression2";
	private final static String TEST_NAME3 = "FunInExpression3";
	private final static String TEST_NAME4 = "FunInExpression4";
	private final static String TEST_NAME5 = "FunInExpression5";
	private final static String TEST_NAME6 = "FunInExpression6";
	private final static String TEST_NAME7 = "FunInExpression7"; //dml-bodied builtin
	
	private final static String TEST_DIR = "functions/misc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FunctionInExpressionTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME6, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME6, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME7, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME7, new String[] { "R" }) );
	}

	@Test
	public void testFunInExpression1() {
		runFunInExpressionTest( TEST_NAME1 );
	}
	
	@Test
	public void testFunInExpression2() {
		runFunInExpressionTest( TEST_NAME2 );
	}
	
	@Test
	public void testFunInExpression3() {
		runFunInExpressionTest( TEST_NAME3 );
	}
	
	@Test
	public void testFunInExpression4() {
		runFunInExpressionTest( TEST_NAME4 );
	}

	@Test
	public void testFunInExpression5() {
		runFunInExpressionTest( TEST_NAME5 );
	}

	@Test
	public void testFunInExpression6() {
		runFunInExpressionTest( TEST_NAME6 );
	}
	
	@Test
	public void testFunInExpression7() {
		runFunInExpressionTest( TEST_NAME7 );
	}
	
	private void runFunInExpressionTest( String testName )
	{
		TestConfiguration config = getTestConfiguration(testName);
		loadTestConfiguration(config);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + testName + ".dml";
		programArgs = new String[]{"-args", output("R") };
		
		fullRScriptName = HOME + testName + ".R";
		rCmd = getRCmd(expectedDir());

		//run script and compare output
		runTest(true, false, null, -1); 
		
		//compare results
		double val = readDMLMatrixFromOutputDir("R").get(new CellIndex(1,1));
		Assert.assertTrue("Wrong result: 7 vs "+val, Math.abs(val-7)<Math.pow(10, -13));
	}
}
