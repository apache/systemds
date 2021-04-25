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

import java.util.HashMap;

import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class BuiltinLogSumExpTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "logsumexp";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinLogSumExpTest.class.getSimpleName() + "/";

	private final static double eps = 1e-4;
	private final static int rows = 100;
	private final static double spDense = 0.7;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
	}

	@Test
	public void testrowlogSumExpCP() {
		runlogSumExpTest("rows", ExecType.CP);
	}

	@Test
	public void testrowlogSumExpSP() {
		runlogSumExpTest("rows", ExecType.SPARK);
	}

	@Test
	public void testcollogSumExpCP() {
		runlogSumExpTest("cols", ExecType.CP);
	}

	@Test
	public void testcollogSumExpSP() {
		runlogSumExpTest("cols", ExecType.SPARK);
	}

	@Test
	public void testlogSumExpCP() {
		runlogSumExpTest("none", ExecType.CP);
	}

	@Test
	public void testlogSumExpSP() {
		runlogSumExpTest("none", ExecType.SPARK);
	}
	private void runlogSumExpTest(String axis, ExecType instType)
	{
		ExecMode platformOld = setExecMode(instType);
		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("A"), axis, output("B") };

			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + axis+ " " + expectedDir();

			//generate actual dataset
			double[][] A = getRandomMatrix(rows, 10, 10, 100, spDense, 7);
			writeInputMatrixWithMTD("A", A, true);

			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("B");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
