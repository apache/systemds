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
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

public class BuiltinFixInvalidLengths extends AutomatedTestBase {
	private final static String TEST_NAME = "fixInvalidLengthstest";
	private final static String TEST_DIR = "functions/frame/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinFixInvalidLengths.class.getSimpleName() + "/";
	private final static String INPUT = DATASET_DIR+"/Salaries.csv";

	public static void init() {
		TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
	}

	public static void cleanUp() {
		if (TEST_CACHE_ENABLED) {
			TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
		}
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));
	}
	@Test
	public void fixInvalidTestCP() {
		runFixInvalidLength(Types.ExecType.CP);
	}

	// TODO fix exception "Invalid key lookup in empty list" 
	@Ignore
	public void fixInvalidTestSP() {
		runFixInvalidLength(Types.ExecType.SPARK);
	}

	private void runFixInvalidLength(Types.ExecType et)
	{
		Types.ExecMode platformOld = setExecMode(et);

		try {
			getAndLoadTestConfiguration(TEST_NAME);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			System.out.println(fullDMLScriptName);
			programArgs = new String[] {"-args", INPUT, output("B")};
			runTest(true, false, null, -1);
			boolean retCondition = HDFSTool.readBooleanFromHDFSFile(output("B"));
			Assert.assertEquals(true, retCondition);

		}
		catch (Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
