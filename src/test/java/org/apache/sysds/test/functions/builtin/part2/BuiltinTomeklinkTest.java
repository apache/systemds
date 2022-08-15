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

package org.apache.sysds.test.functions.builtin.part2;

import org.junit.Ignore;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import java.util.HashMap;

public class BuiltinTomeklinkTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "tomeklink";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinTomeklinkTest.class.getSimpleName() + "/";

	private final static double eps = 1e-3;
	private final static int rows = 53;
	private final static int cols = 6;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
	}
	//TODO as the cran repository is no linger hosting the R package "unbalanced" these tests are failing
	// the fix needs to be either install the package hosted on Git or rewrite test/script with other R package
	@Ignore
	public void testTomeklinkCP() {
		runTomeklinkTest(ExecType.CP);
	}

	@Ignore
	public void testTomeklinkSP() {
		runTomeklinkTest(ExecType.SPARK);
	}

	private void runTomeklinkTest(ExecType instType)
	{
		ExecMode platformOld = setExecMode(instType);

		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-args", input("A"), input("B"), output("C")};

			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();

			//generate actual dataset
			double[][] A = getRandomMatrix(rows, cols, -1, 1, 0.7, 1);
			writeInputMatrixWithMTD("A", A, true);

			double[][] B = getRandomMatrix(rows, 1, 0, 1, 0.5, 1);
			B = TestUtils.round(B);
			writeInputMatrixWithMTD("B", B, true);

			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("C");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("C");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
