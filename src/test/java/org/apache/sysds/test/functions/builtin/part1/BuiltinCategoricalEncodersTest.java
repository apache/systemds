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
package org.apache.sysds.test.functions.builtin.part1;

import org.apache.sysds.common.Types;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Assert;
import org.junit.Test;

public class BuiltinCategoricalEncodersTest extends AutomatedTestBase {
	private final static String TEST_NAME1 = "frequencyEncode_test";
	private final static String TEST_NAME2 = "WoE_test";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinCategoricalEncodersTest.class.getSimpleName() + "/";


	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1,new String[]{"B"}));
		addTestConfiguration(TEST_NAME2,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2,new String[]{"B"}));
	}

	@Test
	public void testFreqEncodeCP() {
		runEncoderTests(TEST_NAME1, Types.ExecType.CP);
	}

	@Test
	public void testFreqEncodeSP() {
		runEncoderTests(TEST_NAME1, Types.ExecType.SPARK);
	}

	@Test
	public void testWoECP() {
		runEncoderTests(TEST_NAME2,  Types.ExecType.CP);
	}

	@Test
	public void testWoESpark() {
		runEncoderTests(TEST_NAME2,  Types.ExecType.SPARK);
	}

	private void runEncoderTests(String testname, Types.ExecType instType)
	{
		setOutputBuffering(true);
		Types.ExecMode platformOld = setExecMode(instType);

		try
		{
			loadTestConfiguration(getTestConfiguration(testname));

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-args",  output("B") };

			String out = runTest(null ).toString();
			Assert.assertTrue(out.contains("TRUE"));

		}
		finally {
			rtplatform = platformOld;
		}
	}
}
