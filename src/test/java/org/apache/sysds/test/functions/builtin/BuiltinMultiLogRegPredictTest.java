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

import org.junit.Test;
import org.apache.sysds.common.Types;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import java.util.HashMap;

public class BuiltinMultiLogRegPredictTest extends AutomatedTestBase {
	private final static String TEST_NAME = "multiLogRegPredict";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinMultiLogRegPredictTest.class.getSimpleName() + "/";

	private final static double eps = 2;
	private final static int rows = 1000;
	private final static int cols = 150;
	private final static double spSparse = 0.3;
	private final static double spDense = 0.7;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
	}

	@Test
	public void testLmPredictMatrixDenseCP() {
		runPredictionTest(false, LopProperties.ExecType.CP);
	}

	@Test
	public void testLmPredictMatrixSparseCP() {
		runPredictionTest(true, LopProperties.ExecType.CP);
	}


	private void runPredictionTest(boolean sparse, LopProperties.ExecType instType)
	{
		Types.ExecMode platformOld = setExecMode(instType);

		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			double sparsity = sparse ? spSparse : spDense;

			String HOME = SCRIPT_DIR + TEST_DIR;


			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("A"), input("B"), input("C"), input("D"), output("O") };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " "  + expectedDir();


			//generate actual dataset
			double[][] A = getRandomMatrix(rows, cols, 1, 100, sparsity, 7);
			A = TestUtils.round(A);
			writeInputMatrixWithMTD("A", A, true);

			double[][] B = getRandomMatrix(rows, 1, 1, 5, 1.0, 3);
			B = TestUtils.round(B);
			writeInputMatrixWithMTD("B", B, true);

			double[][] C = getRandomMatrix(rows, cols, 1, 100, sparsity, 7);
			C = TestUtils.round(C);
			writeInputMatrixWithMTD("C", C, true);

			double[][] D = getRandomMatrix(rows, 1, 1, 5, 1.0, 3);
			D = TestUtils.round(D);
			writeInputMatrixWithMTD("D", D, true);


			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("O");
			HashMap<MatrixValue.CellIndex, Double> rfile  = readRMatrixFromFS("O");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
