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

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Ignore;
import org.junit.Test;

import java.util.HashMap;

public class BuiltinMeanImputationTest extends AutomatedTestBase {
	private final static String TEST_NAME1 = "meanImputation";
	private final static String TEST_NAME2 = "medianImputation";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinMeanImputationTest.class.getSimpleName() + "/";

	private final static double eps = 1e-3;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1,new String[]{"B"}));
		addTestConfiguration(TEST_NAME2,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2,new String[]{"B"}));
	}

	@Test
	public void testMeanCP() {
		runMeanMedianTest(TEST_NAME1, true, Types.ExecType.CP);
	}

	@Ignore
	public void testMeanSP() {
		runMeanMedianTest(TEST_NAME1, true, Types.ExecType.SPARK);
	}

	@Test
	public void testMedianCP() {
		runMeanMedianTest(TEST_NAME2, true, Types.ExecType.CP);
	}

	@Ignore
	public void testMedianSP() {
		runMeanMedianTest(TEST_NAME2, true, Types.ExecType.SPARK);
	}

	private void runMeanMedianTest(String testname, boolean defaultProb, Types.ExecType instType)
	{
		Types.ExecMode platformOld = setExecMode(instType);

		try
		{
			loadTestConfiguration(getTestConfiguration(testname));

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-args", DATASET_DIR+"Salaries.csv", output("B") };
			fullRScriptName = HOME + testname + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + DATASET_DIR+"Salaries.csv" + " " + expectedDir();


			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
			HashMap<MatrixValue.CellIndex, Double> rfile  = readRMatrixFromExpectedDir("B");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
