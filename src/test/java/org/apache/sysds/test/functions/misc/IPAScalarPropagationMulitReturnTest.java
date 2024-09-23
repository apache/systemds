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
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class IPAScalarPropagationMulitReturnTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "ScalarPropagationMultiReturn";
	private final static String TEST_DIR = "functions/misc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + IPAScalarPropagationMulitReturnTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{"R"})   );
	}

	@Test
	public void testScalarPropagationNoIPA() {
		runIPAScalarVariablePropagationTest( TEST_NAME1, false );
	}
	
	@Test
	public void testScalarPropagationIPA() {
		runIPAScalarVariablePropagationTest( TEST_NAME1, true );
	}
	
	private void runIPAScalarVariablePropagationTest( String testname, boolean IPA )
	{
		boolean oldFlagIPA = OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS;
		
		try {
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-explain","-args", output("R") };
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = IPA;

			//run test, incl expected MR jobs (in case if IPA 0 due to scalar propagation)
			runTest(true, false, null, IPA ? 0 : 35);
			
			double[][] ret = TestUtils.convertHashMapToDoubleArray(
				readDMLMatrixFromOutputDir("R"), 1, 1);
			Assert.assertEquals(Double.valueOf(2), Double.valueOf(ret[0][0]));
		}
		finally {
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = oldFlagIPA;
		}
	}
}