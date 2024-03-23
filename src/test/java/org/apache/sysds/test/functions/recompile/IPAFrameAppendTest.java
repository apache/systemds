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

package org.apache.sysds.test.functions.recompile;

import java.io.IOException;

import org.junit.Test;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

public class IPAFrameAppendTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "append_frame";
	private final static String TEST_DIR = "functions/recompile/";
	private final static String TEST_CLASS_DIR = TEST_DIR + IPAFrameAppendTest.class.getSimpleName() + "/";
	
	private final static int rows = 2000;
	private final static int cols = 1000;
	
	@Override
	public void setUp() {
		addTestConfiguration( TEST_NAME,
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "Y" }) );
	}
	
	@Test
	public void testAppend_NoIPA_NoRewrites() throws IOException {
		runIPAAppendTest(false, false);
	}
	
	@Test
	public void testAppend_IPA_NoRewrites() throws IOException {
		runIPAAppendTest(true, false);
	}
	
	@Test
	public void testAppend_NoIPA_Rewrites() throws IOException {
		runIPAAppendTest(false, true);
	}
	
	@Test
	public void testAppend_IPA_Rewrites() throws IOException {
		runIPAAppendTest(true, true);
	}
	
	private void runIPAAppendTest( boolean IPA, boolean rewrites ) throws IOException
	{
		boolean oldFlagIPA = OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS;
		boolean oldFlagRewrites = OptimizerUtils.ALLOW_COMMON_SUBEXPRESSION_ELIMINATION;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain", "-stats",
				"-args", String.valueOf(rows), String.valueOf(cols) };

			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = IPA;
			OptimizerUtils.ALLOW_COMMON_SUBEXPRESSION_ELIMINATION = rewrites;
			
			//run test
			runTest(true, false, null, -1); 
			
			//check expected number of compiled and executed Spark jobs
			int expectedNumCompiled = IPA ? 0 : (rewrites ? 5 : 6);
			int expectedNumExecuted = 0; 
			
			checkNumCompiledSparkInst(expectedNumCompiled);
			checkNumExecutedSparkInst(expectedNumExecuted);
		}
		finally {
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = oldFlagIPA;
			OptimizerUtils.ALLOW_COMMON_SUBEXPRESSION_ELIMINATION = oldFlagRewrites;
		}
	}
}
