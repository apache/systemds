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
import org.apache.sysds.api.DMLException;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class FunctionPotpourriTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "FunPotpourriNoReturn";
	private final static String TEST_NAME2 = "FunPotpourriComments";
	private final static String TEST_NAME3 = "FunPotpourriNoReturn2";
	private final static String TEST_NAME4 = "FunPotpourriEval";
	private final static String TEST_NAME5 = "FunPotpourriSubsetReturn";
	private final static String TEST_NAME6 = "FunPotpourriSubsetReturnDead";
	private final static String TEST_NAME7 = "FunPotpourriNamedArgsSingle";
	private final static String TEST_NAME8 = "FunPotpourriNamedArgsMulti";
	private final static String TEST_NAME9 = "FunPotpourriNamedArgsPartial";
	private final static String TEST_NAME10 = "FunPotpourriNamedArgsUnknown1";
	private final static String TEST_NAME11 = "FunPotpourriNamedArgsUnknown2";
	private final static String TEST_NAME12 = "FunPotpourriNamedArgsIPA";
	private final static String TEST_NAME13 = "FunPotpourriDefaultArgScalar";
	private final static String TEST_NAME14 = "FunPotpourriDefaultArgMatrix";
	private final static String TEST_NAME15 = "FunPotpourriDefaultArgScalarMatrix1";
	private final static String TEST_NAME16 = "FunPotpourriDefaultArgScalarMatrix2";
	private final static String TEST_NAME17 = "FunPotpourriNamedArgsQuotedAssign";
	private final static String TEST_NAME18 = "FunPotpourriMultiReturnBuiltin1";
	private final static String TEST_NAME19 = "FunPotpourriMultiReturnBuiltin2";
	
	private final static String TEST_DIR = "functions/misc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FunctionPotpourriTest.class.getSimpleName() + "/";
	
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
		addTestConfiguration( TEST_NAME8, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME8, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME9, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME9, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME10, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME10, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME11, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME11, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME12, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME12, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME13, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME13, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME14, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME14, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME15, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME15, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME16, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME16, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME17, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME17, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME18, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME18, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME19, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME19, new String[] { "R" }) );
	}

	@Test
	public void testFunctionNoReturn() {
		runFunctionTest( TEST_NAME1, false );
	}
	
	@Test
	public void testFunctionComments() {
		runFunctionTest( TEST_NAME2, false );
	}
	
	@Test
	public void testFunctionNoReturnSpec() {
		runFunctionTest( TEST_NAME3, false );
	}
	
	@Test
	public void testFunctionEval() {
		runFunctionTest( TEST_NAME4, false );
	}
	
	@Test
	public void testFunctionSubsetReturn() {
		runFunctionTest( TEST_NAME5, false );
	}
	
	@Test
	public void testFunctionSubsetReturnDead() {
		runFunctionTest( TEST_NAME6, false );
	}
	
	@Test
	public void testFunctionNamedArgsSingle() {
		runFunctionTest( TEST_NAME7, false );
	}
	
	@Test
	public void testFunctionNamedArgsSingleErr() {
		runFunctionTest( TEST_NAME7, true );
	}
	
	@Test
	public void testFunctionNamedArgsMulti() {
		runFunctionTest( TEST_NAME8, false );
	}
	
	@Test
	public void testFunctionNamedArgsMultiErr() {
		runFunctionTest( TEST_NAME8, true );
	}
	
	@Test
	public void testFunctionNamedArgsPartial() {
		runFunctionTest( TEST_NAME9, true );
	}
	
	@Test
	public void testFunctionNamedArgsUnkown1() {
		runFunctionTest( TEST_NAME10, true );
	}
	
	@Test
	public void testFunctionNamedArgsUnkown2() {
		runFunctionTest( TEST_NAME11, true );
	}
	
	@Test
	public void testFunctionNamedArgsIPA() {
		runFunctionTest( TEST_NAME12, false );
	}
	
	@Test
	public void testFunctionDefaultArgsScalar() {
		runFunctionTest( TEST_NAME13, false );
	}
	
	@Test
	public void testFunctionDefaultArgsMatrix() {
		runFunctionTest( TEST_NAME14, false );
	}
	
	@Test
	public void testFunctionDefaultArgsScalarMatrix1() {
		runFunctionTest( TEST_NAME15, false );
	}
	
	@Test
	public void testFunctionDefaultArgsScalarMatrix2() {
		runFunctionTest( TEST_NAME16, false );
	}
	
	@Test
	public void testFunctionNamedArgsQuotedAssign() {
		runFunctionTest( TEST_NAME17, false );
	}
	
	@Test
	public void testFunctionMultiReturnBuiltin1() {
		runFunctionTest( TEST_NAME18, false );
	}
	
	@Test
	public void testFunctionMultiReturnBuiltin2() {
		runFunctionTest( TEST_NAME19, false );
	}
	
	private void runFunctionTest(String testName, boolean error) {
		TestConfiguration config = getTestConfiguration(testName);
		loadTestConfiguration(config);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + testName + ".dml";
		programArgs = new String[]{"-explain","hops", "-stats",
			"-args", String.valueOf(error).toUpperCase()};
		
		//run script and compare output
		runTest(true, error, DMLException.class, -1);
		if( testName.equals(TEST_NAME18) )
			Assert.assertTrue(heavyHittersContainsString("print"));
	}
}
