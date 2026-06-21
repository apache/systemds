/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.test.functions.builtin.part2;

import java.util.HashMap;

import org.junit.Test;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class BuiltinPowerTransformTest extends AutomatedTestBase {
	private static final String TEST_NAME = "powerTransformApply";
	private static final String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR =
		TEST_DIR + BuiltinPowerTransformTest.class.getSimpleName() + "/";

	private static final double EPS = 1e-9;

	@Override
	public void setUp() {
		// Register test configuration and declare the test output matrix Y
		addTestConfiguration(
			TEST_NAME,
			new TestConfiguration(
				TEST_CLASS_DIR,
				TEST_NAME,
				new String[] {"Y"}
			)
		);
	}

	@Test
	public void testPowerTransformApplyDenseCP() {
		// Only test single-node CP execution mode here
		runPowerTransformApplyTest(ExecType.CP);
	}

	private void runPowerTransformApplyTest(ExecType execType) {
		// Save the old execution mode and restore it after the test
		ExecMode oldExecMode = setExecMode(execType);

		try {
			// Load the configuration for this test
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String home = SCRIPT_DIR + TEST_DIR;

			// Set the DML test wrapper and R reference implementation
			fullDMLScriptName = home + TEST_NAME + ".dml";
			fullRScriptName = home + TEST_NAME + ".R";

			// DML arguments in order:
			// 1. Input matrix X
			// 2. Lambda row matrix L
			// 3. Output matrix Y
			programArgs = new String[] {
				"-exec", "singlenode",
				"-args",
				input("X"),
				input("L"),
				output("Y")
			};

			// R script receives the input directory and expected output directory
			rCmd = "Rscript " + fullRScriptName + " "
				+ inputDir() + " " + expectedDir();

			// Use the same input values in three columns to test lambda 0, 1, and 2
			double[][] X = {
				{-2, -2, -2},
				{-1, -1, -1},
				{ 0,  0,  0},
				{ 1,  1,  1},
				{ 2,  2,  2}
			};

			// First column lambda=0, second column lambda=1, third column lambda=2
			double[][] L = {
				{0, 1, 2}
			};

			// Write both input matrices to the test input directory and generate metadata
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("L", L, true);

			// DML test
			runTest(true, false, null, -1);

			// R reference implementation
			runRScript(true);

			// Read the results separately
			HashMap<CellIndex, Double> dmlResult =
				readDMLMatrixFromOutputDir("Y");

			HashMap<CellIndex, Double> rResult =
				readRMatrixFromExpectedDir("Y");

			// Compare the two result matrices within the given tolerance
			TestUtils.compareMatrices(
				dmlResult,
				rResult,
				EPS,
				"DML",
				"R"
			);
		}
		catch (Exception exception) {
			throw new RuntimeException(exception);
		}
		finally {
			// Restore the old execution mode whether the test succeeds or fails
			resetExecMode(oldExecMode);
		}
	}
}
