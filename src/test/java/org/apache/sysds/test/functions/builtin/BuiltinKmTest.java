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

import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class BuiltinKmTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "km";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinKmTest.class.getSimpleName() + "/";

	private final static double eps = 1e-3;
	private final static int rows = 1765;
	private final static double spDense = 0.99;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"C"}));
	}

	@Test
	public void testFunction() {
		runKmTest(0.05,"greenwood", "log", "none");
	}

	private void runKmTest(Double alpha, String err_type,
						   String conf_type, String test_type) {
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		int seed = 11;

		programArgs = new String[]{
				"-nvargs", "X=" + input("X"), "TE=" + input("TE"), "GI=" + input("GI"),
				"SI=" + input("SI"), "O=" + output("O"), "M=" + output("M"), "T=" + output("T"),
				"alpha=" + alpha, "err_type=" + err_type, "err_type=" + err_type,
				"conf_type=" + conf_type, "test_type" + test_type};

		double[][] X = getRandomMatrix(rows, , , , spDense, seed);
		writeInputMatrixWithMTD("X", X, false);

		double[][] TE = getRandomMatrix(rows, , , , spDense, seed);
		writeInputMatrixWithMTD("TE", TE, false);

		double[][] GI = getRandomMatrix(rows, , , , spDense, seed);
		writeInputMatrixWithMTD("GI", GI, false);

		double[][] SI = getRandomMatrix(rows, , , , spDense, seed);
		writeInputMatrixWithMTD("SI", SI, false);

		runTest(true, false, null, -1);
	}
}
