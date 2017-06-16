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
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Test;

/**
 * Test for static rewrites + IPA second chance compilation.
 *
 * The DML scripts contain more information, but essentially
 * without a second chance of static rewrites + IPA after the initial
 * pass of IPA, there are many situations in which sizes will remain
 * unknown even after recompilation, thus leading to distributed ops.
 * With the second chance enabled, sizes in these situations can be
 * determined.  For example, the alternation of constant folding
 * (static rewrite) and scalar replacement (IPA) can allow for size
 * propagation without dynamic rewrites or recompilation due to
 * replacement of scalars with literals during IPA, which enables
 * constant folding of sub-DAGs of literals during static rewrites,
 * which in turn allows for scalar propagation during IPA.
 */
public class IPAConstantFoldingScalarVariablePropagationTest extends AutomatedTestBase
{
	private final static String TEST_NAME1 = "IPAConstantFoldingScalarVariablePropagation1";
	private final static String TEST_NAME2 = "IPAConstantFoldingScalarVariablePropagation2";
	private final static String TEST_DIR = "functions/misc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + IPAConstantFoldingScalarVariablePropagationTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		TestConfiguration conf1 = new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{});
		TestConfiguration conf2 = new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[]{});
		addTestConfiguration(TEST_NAME1, conf1);
		addTestConfiguration(TEST_NAME2, conf2);
	}

	@Test
	public void testConstantFoldingScalarPropagation1IPASecondChance() {
		runIPAScalarVariablePropagationTest(TEST_NAME1, true);
	}

	@Test
	public void testConstantFoldingScalarPropagation1NoIPASecondChance() {
		runIPAScalarVariablePropagationTest(TEST_NAME1, false);
	}

	@Test
	public void testConstantFoldingScalarPropagation2IPASecondChance() {
		runIPAScalarVariablePropagationTest(TEST_NAME2, true);
	}

	@Test
	public void testConstantFoldingScalarPropagation2NoIPASecondChance() {
		runIPAScalarVariablePropagationTest(TEST_NAME2, false);
	}

	/**
	 * Test for static rewrites + IPA second chance compilation to allow
	 * for scalar propagation (IPA) of constant-folded DAG of literals
	 * (static rewrites) made possible by an initial scalar propagation
	 * (IPA).
	 *
	 * @param testname  The name of the test.
	 * @param IPA_SECOND_CHANCE  Whether or not to use IPA second chance.
	 */
	private void runIPAScalarVariablePropagationTest(String testname, boolean IPA_SECOND_CHANCE)
	{
		// Save old settings
		boolean oldFlagIPASecondChance = OptimizerUtils.ALLOW_IPA_SECOND_CHANCE;
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
			OptimizerUtils.ALLOW_IPA_SECOND_CHANCE = IPA_SECOND_CHANCE;
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			rtplatform = RUNTIME_PLATFORM.HYBRID_SPARK;

			// Run test
			runTest(true, false, null, -1);

			// Check for correct number of compiled & executed Spark jobs
			// (MB: originally, this required a second chance, but not anymore)
			checkNumCompiledSparkInst(0);
			checkNumExecutedSparkInst(0);
		}
		finally {
			// Reset
			OptimizerUtils.ALLOW_IPA_SECOND_CHANCE = oldFlagIPASecondChance;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			rtplatform = platformOld;
		}
	}
}
