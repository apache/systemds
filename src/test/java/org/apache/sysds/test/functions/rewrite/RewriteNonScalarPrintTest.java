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

import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;

public class RewriteNonScalarPrintTest extends AutomatedTestBase {

	private static final String TEST_NAME = "RewriteNonScalarPrint";
	private static final String TEST_DIR = "functions/rewrite/";

	private static final String TEST_CLASS_DIR =
		TEST_DIR + RewriteNonScalarPrintTest.class.getSimpleName() + "/";


	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
	}

	@Test
	public void testNonScalarPrintMatrix() {
		testRewriteNonScalarPrint(1);
	}

	@Test
	public void testNonScalarPrintFrame() {
		testRewriteNonScalarPrint(2);
	}

	@Test
	public void testNonScalarPrintList() {
		testRewriteNonScalarPrint(3);
	}

	@Test
	public void testNonScalarPrintMatrixRow() {
		testRewriteNonScalarPrint(4);
	}

	@Test
	public void testNonScalarPrintMatrixCol() {
		testRewriteNonScalarPrint(5);
	}

	private void testRewriteNonScalarPrint(int ID) {
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		loadTestConfiguration(config);
		setOutputBuffering(true);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-stats", "-args", String.valueOf(ID), output("R")};
		String fullOut = runTest(true, false, null, -1).toString();

		//Extract or remove "SystemDS Statistics:"
		int idxStats = fullOut.indexOf("SystemDS Statistics:");
		String userOutput = (idxStats >= 0) ? fullOut.substring(0, idxStats) : fullOut;

		String toString = "toString";
		long numtoString = Statistics.getCPHeavyHitterCount(toString);

		if(ID == 1) {
			Assert.assertTrue(
				userOutput.contains("1.000 2.000 3.000\n" + "4.000 5.000 6.000\n" + "7.000 8.000 9.000") &&
					numtoString == 1);
		}
		else if(ID == 2) {
			Assert.assertTrue(userOutput.contains(
				"# FRAME: nrow = 3, ncol = 3\n" + "# C1 C2 C3\n" + "# INT32 INT32 INT32\n" + "1 2 3\n" + "4 5 6\n" +
					"7 8 9") && numtoString == 1);
		}
		else if(ID == 3) {
			Assert.assertTrue(userOutput.contains("[1, 2, 3]") && numtoString == 1);
		}
		else if(ID == 4) {
			Assert.assertTrue(userOutput.contains("1.000 2.000 3.000\n") && numtoString == 1);
		}
		else if(ID == 5) {
			Assert.assertTrue(userOutput.contains("1.000\n" + "4.000\n" + "7.000") && numtoString == 1);
		}

		//Print the entire output
		System.out.println(fullOut);
	}
}
