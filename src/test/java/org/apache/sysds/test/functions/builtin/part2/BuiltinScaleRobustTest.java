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

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.HashMap;

import org.junit.Test;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class BuiltinScaleRobustTest extends AutomatedTestBase {
	private final static String TEST_NAME = "scaleRobust";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinScaleRobustTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	private final static int rows = 70;
	private final static int cols = 50;


	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"B"}));
	}

	@Test
	public void testScaleRobustDenseCP() {
		runTest(false, ExecType.CP);
	}

	private void runTest(boolean sparse, ExecType et) {
		ExecMode old = setExecMode(et);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			double sparsity = sparse ? 0.1 : 0.9;
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			fullRScriptName = HOME + TEST_NAME + ".R";
			String fullPyScriptName = HOME + TEST_NAME + ".py";
			programArgs = new String[]{"-args", input("A"), output("B")};
			programArgs = new String[]{"-exec", "singlenode", "-args", input("A"), output("B")};
			rCmd = "Rscript " + fullRScriptName + " " + inputDir() + " " + expectedDir();
			String pyCmd = "python " + fullPyScriptName + " " + inputDir() + " " + expectedDir();

			double[][] A = getRandomMatrix(rows, cols, -10, 10, sparsity, 7);
			writeInputMatrixWithMTD("A", A, true);

			// Run DML
			runTest(true, false, null, -1); 

			// Run R
			runRScript(true);

			// Run Python 
			Process p = Runtime.getRuntime().exec(pyCmd);
			// Capture stdout
			BufferedReader stdInput = new BufferedReader(new InputStreamReader(p.getInputStream()));
			BufferedReader stdError = new BufferedReader(new InputStreamReader(p.getErrorStream()));

			String s;
			while ((s = stdInput.readLine()) != null) {
				System.out.println("[PYTHON OUT] " + s);
			}
			while ((s = stdError.readLine()) != null) {
				System.err.println("[PYTHON ERR] " + s);
			}

			int exitCode = p.waitFor();
			if(exitCode != 0) {
				throw new RuntimeException("Python script failed with exit code: " + exitCode);
			}

			// Read matrices and compare
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("B");
			HashMap<CellIndex, Double> pyfile = readRMatrixFromExpectedDir("B"); 

			TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");
			TestUtils.compareMatrices(dmlfile, pyfile, eps, "DML", "Python");
			

		} catch (Exception e) {
			throw new RuntimeException(e);
		} finally {
			rtplatform = old;
		}
	}

}