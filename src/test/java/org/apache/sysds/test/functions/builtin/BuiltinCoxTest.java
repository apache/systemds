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

public class BuiltinCoxTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "cox";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinCoxTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
	}

	@Test
	void testNormal() {
		runCoxTest(0.05, 0.1, 0.2);
	}
	
	public void runCoxTest(double alpha, int moi, int mii)
	{
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";

		programArgs = new String[]{
				"-nvargs", "X=" + input("X"), "TE=" + input("TE"), "F=" + input("F"),
				"R=" + input("R"), "M=" + input("M"), "S=" + input("S"), "T=" + input("T"),
				"COV=" + input("COV"), "RT=" + input("RT"), "XO=" + input("XO"),
				"MF=" + input("MF"), "M=" + output("M"), "S=" + output("S"), "T=" + output("T"),
				"COV=" + output("COV"), "RT=" + output("RT"), "alpha=" + alpha,
				"moi=" + moi, "mii=" + mii};


		runTest(true, false, null, -1);
	}
}
