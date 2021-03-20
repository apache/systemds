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
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;

public class BuiltinCoxTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "cox";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinCoxTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
	}

	@Test
	public void testFunction() {
		runCoxTest(50, 2.0, 1.5, 0.8, 100, 0.05, 1.0,0.000001, 100, 0);
	}
	
	public void runCoxTest(int numRecords, double scaleWeibull, double shapeWeibull, double prob,
		int numFeatures, double sparsity, double alpha, double tol, int moi, int mii)
	{
		ExecMode platformOld = setExecMode(ExecMode.SINGLE_NODE);

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";

			programArgs = new String[]{
					"-nvargs", "M=" + output("M"), "S=" + output("S"), "T=" + output("T"), "COV=" + output("COV"),
					"RT=" + output("RT"), "XO=" + output("XO"), "n=" + numRecords, "l=" + scaleWeibull,
					"v=" + shapeWeibull, "p=" + prob, "m=" + numFeatures, "sp=" + sparsity,
					"alpha=" + alpha, "tol=" + tol, "moi=" + moi, "mii=" + mii, "sd=" + 1};

			runTest(true, false, null, -1);
			//TODO output comparison
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
