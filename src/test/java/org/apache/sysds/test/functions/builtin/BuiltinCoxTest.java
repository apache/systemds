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
import org.junit.Test;

public class BuiltinCoxTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "cox";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinCoxTest.class.getSimpleName() + "/";

	private final static double eps = 1e-3;
	private final static int rows = 1765;
	private final static double spDense = 0.99;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
	}

	@Test
	void testFunction() {
		runCoxTest(0.05, 100, 0);
	}
	
	public void runCoxTest(double alpha, int moi, int mii)
	{
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		int seed = 11;

		programArgs = new String[]{
				"-nvargs", "X=" + input("X"), "TE=" + input("TE"), "F=" + input("F"),
				"R=" + input("R"), "M=" + output("M"), "S=" + output("S"), "T=" + output("T"),
				"COV=" + output("COV"), "RT=" + output("RT"), "XO=" + output("XO"), "MF=" + output("MF"),
				"alpha=" + alpha, "moi=" + moi, "mii=" + mii};

		double[][] X = getRandomMatrix(rows, , , , spDense, seed);
		writeInputMatrixWithMTD("X", X, false);

		double[][] TE = getRandomMatrix(rows, , , , spDense, seed);
		writeInputMatrixWithMTD("TE", TE, false);

		double[][] F = getRandomMatrix(rows, , , , spDense, seed);
		writeInputMatrixWithMTD("F", F, false);

		double[][] R = getRandomMatrix(rows, , , , spDense, seed);
		writeInputMatrixWithMTD("R", R, false);

		runTest(true, false, null, -1);

	}
}
