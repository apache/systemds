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

package org.apache.sysds.test.functions.mlcontext;

import org.apache.log4j.Logger;
import org.apache.sysds.api.mlcontext.MLResults;
import org.apache.sysds.api.mlcontext.Script;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;

import static org.apache.sysds.api.mlcontext.ScriptFactory.dmlFromFile;

public class MLContextPNMFTest extends MLContextTestBase {
	protected static Logger log = Logger.getLogger(MLContextPNMFTest.class);

	protected final static String TEST_SCRIPT_PNMF = "scripts/staging/PNMF.dml";
	private final static double sparsity1 = 0.7; // dense
	private final static double sparsity2 = 0.1; // sparse

	private final static double eps = 1e-5;

	private final static int rows = 1468;
	private final static int cols = 1207;
	private final static int rank = 20;

	private final static double epsilon = 0.000000001;//1e-9
	private final static double maxiter = 10;

	@Test
	public void testPNMFSparse() {
		runPNMFTestMLC(true);
	}

	@Test
	public void testPNMFDense() {
		runPNMFTestMLC(false);
	}


	private void runPNMFTestMLC(boolean sparse) {

		//generate actual datasets
		double[][] X = getRandomMatrix(rows, cols, 0, 1, sparse?sparsity2:sparsity1, 234);
		double[][] W = getRandomMatrix(rows, rank, 0, 1e-14, 1, 71);
		double[][] H = getRandomMatrix(rank, cols, 0, 1e-14, 1, 72);
		writeInputMatrixWithMTD("X", X, true);
		writeInputMatrixWithMTD("W", W, true);
		writeInputMatrixWithMTD("H", H, true);


		fullRScriptName = "src/test/scripts/functions/codegenalg/Algorithm_PNMF.R";

		rCmd = getRCmd(inputDir(), String.valueOf(rank),
				String.valueOf(epsilon), String.valueOf(maxiter), expectedDir());
		runRScript(true);


		Script pnmf = dmlFromFile(TEST_SCRIPT_PNMF);
		pnmf.in("X", X).in("W", W).in("H", H).in("$4", rank)
				.in("$5", epsilon).in("$6", maxiter)
				.out("W").out("H");
		MLResults outres = ml.execute(pnmf);
		MatrixBlock dmlW = outres.getMatrix("W").toMatrixBlock();
		MatrixBlock dmlH = outres.getMatrix("H").toMatrixBlock();

		//compare matrices
		HashMap<MatrixValue.CellIndex, Double> rW = readRMatrixFromFS("W");
		HashMap<MatrixValue.CellIndex, Double> rH = readRMatrixFromFS("H");
		TestUtils.compareMatrices(rW, dmlW, eps);
		TestUtils.compareMatrices(rH, dmlH, eps);
	}
}
