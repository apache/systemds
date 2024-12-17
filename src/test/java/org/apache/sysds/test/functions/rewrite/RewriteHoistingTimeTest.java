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

package org.apache.sysds.test.functions.rewrite;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class RewriteHoistingTimeTest extends AutomatedTestBase
{
	private static final String TEST_NAME1 = "RewriteTimeHoisting";
	
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteHoistingTimeTest.class.getSimpleName() + "/";
	
	private static final int rows = 1001;
	private static final int cols = 1002;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"R"}) );
	}

	@Test
	public void testTimeHoistingCP() {
		test(TEST_NAME1, ExecType.CP);
	}
	
	@Test
	public void testTimeHoistingSpark() {
		test(TEST_NAME1, ExecType.SPARK);
	}

	private void test(String testname, ExecType et)
	{
		ExecMode platformOld = setExecMode(et);
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[] {"-args",
				String.valueOf(rows), String.valueOf(cols) };

			//test that time is not executed before 1k-by-1k rand
			setOutputBuffering(true);
			String out = runTest(true, false, null, -1).toString();
			double time = Double.parseDouble(out.split(";")[1]);
			System.out.println("Time = "+time+"s");
			Assert.assertTrue(time>0.001);
		}
		finally {
			resetExecMode(platformOld);
		}
	}
}
