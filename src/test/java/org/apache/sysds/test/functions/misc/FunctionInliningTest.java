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
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.utils.Statistics;

public class FunctionInliningTest extends AutomatedTestBase 
{
	private final static String TEST_DIR = "functions/misc/";
	private final static String TEST_NAME1 = "function_chain_inlining";
	private final static String TEST_NAME2 = "function_chain_non_inlining";
	private final static String TEST_NAME3 = "function_recursive_inlining";
	private final static String TEST_CLASS_DIR = TEST_DIR + FunctionInliningTest.class.getSimpleName() + "/";
	
	private final static long rows = 3400;
	private final static long cols = 2700;
	private final static double val = 1.0;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "Rout" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "Rout" }) );
		addTestConfiguration( TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "Rout" }) );
	}

	@Test
	public void testChainInliningIPA() 
	{
		runInliningTest(TEST_NAME1, true);
	}
	
	@Test
	public void testChainNoInliningIPA() 
	{
		runInliningTest(TEST_NAME2, true);
	}
	
	@Test
	public void testRecursiveInliningIPA() 
	{
		runInliningTest(TEST_NAME3, true);
	}
	
	@Test
	public void testChainInliningNoIPA() 
	{
		runInliningTest(TEST_NAME1, false);
	}
	
	@Test
	public void testChainNoInliningNoIPA() 
	{
		runInliningTest(TEST_NAME2, false);
	}
	
	@Test
	public void testRecursiveInliningNoIPA() 
	{
		runInliningTest(TEST_NAME3, false);
	}

	private void runInliningTest( String testname, boolean IPA )
	{
		boolean oldIPA = OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS;
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[] {"-args",String.valueOf(rows),
				String.valueOf(cols), String.valueOf(val), output("Rout") };

			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = IPA;
			
			//run testcase
			runTest(true, false, null, -1); 
			
			//compare output
			double ret = HDFSTool.readDoubleFromHDFSFile(output("Rout"));
			Assert.assertEquals(Double.valueOf(rows*cols*val*6), Double.valueOf(ret));
			
			//compiled spark instructions
			int expectNumCompiled = IPA ? 0 : (testname.equals(TEST_NAME1)?3: //foo1 and foo2 (not removed w/o IPA)
				(testname.equals(TEST_NAME2)?4:15));
			Assert.assertEquals("Unexpected number of compiled Spark instructions.", 
				expectNumCompiled, Statistics.getNoOfCompiledSPInst());
		
			//check executed MR jobs
			int expectNumExecuted = 0; //executed jobs should always be 0 due to dynamic recompilation
			Assert.assertEquals("Unexpected number of executed Spark instructions.", 
				expectNumExecuted, Statistics.getNoOfExecutedSPInst());
		}
		catch(Exception ex) {
			Assert.fail("Failed to run test: "+ex.getMessage());
		}
		finally {
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = oldIPA;
		}
	}
	
}