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


import org.apache.sysds.common.Opcodes;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.apache.commons.lang3.ArrayUtils;

public class IPAFunctionInliningTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "IPAFunInline1"; //pos 1
	private final static String TEST_NAME2 = "IPAFunInline2"; //pos 2
	private final static String TEST_NAME3 = "IPAFunInline3"; //pos 3 (large but called once)
	private final static String TEST_NAME4 = "IPAFunInline4"; //neg 1 (control flow)
	private final static String TEST_NAME5 = "IPAFunInline5"; //neg 2 (large and called twice)
	private final static String TEST_NAME6 = "IPAFunInline6"; //pos 4 (regression op replication)
	private final static String TEST_NAME7 = "IPAFunInline7"; //pos 4 (regression op replication)
	
	private final static String TEST_DIR = "functions/misc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + IPAFunctionInliningTest.class.getSimpleName() + "/";
	
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
	public void testFunInline1NoIPA() {
		runIPAFunInlineTest( TEST_NAME1, false );
	}
	
	@Test
	public void testFunInline2NoIPA() {
		runIPAFunInlineTest( TEST_NAME2, false );
	}
	
	@Test
	public void testFunInline3NoIPA() {
		runIPAFunInlineTest( TEST_NAME3, false );
	}
	
	@Test
	public void testFunInline4NoIPA() {
		runIPAFunInlineTest( TEST_NAME4, false );
	}
	
	@Test
	public void testFunInline5NoIPA() {
		runIPAFunInlineTest( TEST_NAME5, false );
	}
	
	@Test
	public void testFunInline6NoIPA() {
		runIPAFunInlineTest( TEST_NAME6, false );
	}
	
	@Test
	public void testFunInline1IPA() {
		runIPAFunInlineTest( TEST_NAME1, true );
	}
	
	@Test
	public void testFunInline2IPA() {
		runIPAFunInlineTest( TEST_NAME2, true );
	}
	
	@Test
	public void testFunInline3IPA() {
		runIPAFunInlineTest( TEST_NAME3, true );
	}
	
	@Test
	public void testFunInline4IPA() {
		runIPAFunInlineTest( TEST_NAME4, true );
	}
	
	@Test
	public void testFunInline5IPA() {
		runIPAFunInlineTest( TEST_NAME5, true );
	}
	
	@Test
	public void testFunInline6IPA() {
		runIPAFunInlineTest( TEST_NAME6, true );
	}
	
	@Test
	public void testFunInline7IPA() {
		runIPAFunInlineTest( TEST_NAME7, true );
	}
	
	private void runIPAFunInlineTest( String testName, boolean IPA )
	{
		boolean oldFlagIPA = OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS;
		
		try
		{
			TestConfiguration config = getTestConfiguration(testName);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testName + ".dml";
			programArgs = new String[]{"-explain", "-stats", "-args", output("R") };
			
			fullRScriptName = HOME + testName + ".R";
			rCmd = getRCmd(expectedDir());

			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = IPA;

			//run script and compare output
			runTest(true, false, null, -1); 
			
			//compare results
			if( !testName.equals(TEST_NAME6) && !testName.equals(TEST_NAME7) ) {
				double val = readDMLMatrixFromOutputDir("R").get(new CellIndex(1,1));
				Assert.assertTrue("Wrong result: 7 vs "+val, Math.abs(val-7)<Math.pow(10, -14));
			}
			
			//compare inlined functions
			boolean inlined = ( IPA && ArrayUtils.contains(
				new String[]{TEST_NAME1, TEST_NAME2, TEST_NAME3, TEST_NAME6, TEST_NAME7}, testName));
			String[] probe = testName.equals(TEST_NAME7) ?
				new String[] {"lm","lmDS"} : new String[] {"foo"};
			Assert.assertTrue("Unexpected function call: "+inlined,
				!heavyHittersContainsSubString(probe)==inlined);
		
			//check for incorrect operation replication
			if( testName.equals(TEST_NAME6) )
				Assert.assertTrue(Statistics.getCPHeavyHitterCount(Opcodes.PRINT.toString())==1);
		}
		finally {
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = oldFlagIPA;
		}
	}
}
