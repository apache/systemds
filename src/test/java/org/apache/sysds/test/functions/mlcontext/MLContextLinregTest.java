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
import org.apache.sysds.api.mlcontext.Script;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;

public class MLContextLinregTest extends MLContextTestBase {
	protected static Logger log = Logger.getLogger(MLContextLinregTest.class);

	private final static double sparsity1 = 0.7; // dense
	private final static double sparsity2 = 0.1; // sparse

	public enum LinregType {
		CG, DS,
	}

	private final static double eps = 1e-3;

	private final static int rows = 2468;
	private final static int cols = 507;

	@Test
	public void testLinregCGSparse() {
		runLinregTestMLC(LinregType.CG, true);
	}

	@Test
	public void testLinregCGDense() {
		runLinregTestMLC(LinregType.CG, false);
	}

	@Test
	public void testLinregDSSparse() {
		runLinregTestMLC(LinregType.DS, true);
	}

	@Test
	public void testLinregDSDense() {
		runLinregTestMLC(LinregType.DS, false);
	}

	private void runLinregTestMLC(LinregType type, boolean sparse) {

		double[][] X = getRandomMatrix(rows, cols, 0, 1, sparse ? sparsity2 : sparsity1, 7);
		double[][] Y = getRandomMatrix(rows, 1, 0, 10, 1.0, 3);

		// Hack Alert
		// overwrite baseDirectory to the place where test data is stored.
		baseDirectory = "target/testTemp/functions/mlcontext/";

		fullRScriptName = "src/test/scripts/functions/codegenalg/Algorithm_LinregCG.R";

		writeInputMatrixWithMTD("X", X, true);
		writeInputMatrixWithMTD("y", Y, true);

		rCmd = getRCmd(inputDir(), "0", "0.000001", "0", "0.001", expectedDir());
		runRScript(true);

		MatrixBlock outmat = new MatrixBlock();

		switch (type) {
		case CG:
			Script lrcg = new Script(
				  "X = read($X);\n"
				+ "y = read($Y);\n"
				+ "beta_out = lmCG(X=X, y=y, intercept=$icpt, tol=$tol, maxIter=$maxi, reg=$reg);\n");
			lrcg.in("X", X).in("y", Y).in("$icpt", "0").in("$tol", "0.000001").in("$maxi", "0").in("$reg", "0.000001")
					.out("beta_out");
			outmat = ml.execute(lrcg).getMatrix("beta_out").toMatrixBlock();

			break;

		case DS:
			Script lrds = new Script(
				  "X = read($X);\n"
				+ "y = read($Y);\n"
				+ "beta_out = lmDS(X=X, y=y, intercept=$icpt, reg=$reg);\n");
			lrds.in("X", X).in("y", Y).in("$icpt", "0").in("$reg", "0.000001").out("beta_out");
			outmat = ml.execute(lrds).getMatrix("beta_out").toMatrixBlock();

			break;
		}

		//compare matrices
		HashMap<MatrixValue.CellIndex, Double> rfile = readRMatrixFromExpectedDir("w");
		TestUtils.compareMatrices(rfile, outmat, eps);
	}
}
