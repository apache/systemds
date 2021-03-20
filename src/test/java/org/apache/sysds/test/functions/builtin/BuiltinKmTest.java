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

package org.apache.sysds.test.functions.builtin;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class BuiltinKmTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "km";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinKmTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"C"}));
	}

	@Test
	public void testKmDefaultConfiguration() {
		runKmTest(10, 2.0, 1.5, 0.8, 2,
			1, 10, 0.05,"greenwood", "log", "none");
	}
	@Test
	public void testKmErrTypePeto() {
		runKmTest(10, 2.0, 1.5, 0.8, 2,
				1, 10, 0.05,"peto", "log", "none");
	}
	@Test
	public void testKmConfTypePlain() {
		runKmTest(10, 2.0, 1.5, 0.8, 2,
				1, 10, 0.05,"greenwood", "plain", "none");
	}
	@Test
	public void testKmConfTypeLogLog() {
		runKmTest(10, 2.0, 1.5, 0.8, 2,
				1, 10, 0.05,"greenwood", "log-log", "none");
	}

	private void runKmTest(int numRecords, double scaleWeibull, double shapeWeibull, double prob,
			int numCatFeaturesGroup, int numCatFeaturesStrat, int maxNumLevels, double alpha, String err_type,
			String conf_type, String test_type)
	{
		ExecMode platformOld = setExecMode(ExecType.CP);

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";

			programArgs = new String[]{
					"-nvargs", "O=" + output("O"), "M=" + output("M"), "T=" + output("T"),
					"T_GROUPS_OE=" + output("T_GROUPS_OE"), "n=" + numRecords, "l=" + scaleWeibull,
					"v=" + shapeWeibull, "p=" + prob, "g=" + numCatFeaturesGroup, "s=" + numCatFeaturesStrat,
					"f=" + maxNumLevels, "alpha=" + alpha, "err_type=" + err_type,
					"conf_type=" + conf_type, "test_type=" + test_type, "sd=" + 1};

			runTest(true, false, null, -1);
			//TODO output comparison
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
