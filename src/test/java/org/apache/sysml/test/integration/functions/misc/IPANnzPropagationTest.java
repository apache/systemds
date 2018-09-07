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

package org.apache.sysml.test.integration.functions.misc;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Test;


public class IPANnzPropagationTest extends AutomatedTestBase
{
	private final static String TEST_NAME1 = "IPANnzPropagation1";
	private final static String TEST_NAME2 = "IPANnzPropagation2";
	private final static String TEST_DIR = "functions/misc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + IPANnzPropagationTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[]{}));
	}

	@Test
	public void testNnzPropgationPositive() {
		runIPANnzPropgationTest(TEST_NAME1);
	}

	@Test
	public void testNnzPropgationNegative() {
		runIPANnzPropgationTest(TEST_NAME2);
	}


	private void runIPANnzPropgationTest(String testname)
	{
		// Save old settings
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		RUNTIME_PLATFORM platformOld = rtplatform;
		
		try
		{
			// Setup test
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-stats", "-explain", "recompile_hops"};
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			rtplatform = RUNTIME_PLATFORM.HYBRID_SPARK;
			
			runTest(true, false, null, -1);
			
			//check for propagated nnz
			checkNumCompiledSparkInst(testname.equals(TEST_NAME1) ? 0 : 1);
			checkNumExecutedSparkInst(0);
		}
		finally {
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			rtplatform = platformOld;
		}
	}
}
