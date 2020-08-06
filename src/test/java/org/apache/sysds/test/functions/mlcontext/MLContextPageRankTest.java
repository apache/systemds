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

import static org.apache.sysds.api.mlcontext.ScriptFactory.dmlFromFile;

public class MLContextPageRankTest extends MLContextTestBase {
	protected static Logger log = Logger.getLogger(MLContextPageRankTest.class);

	protected final static String TEST_SCRIPT_PAGERANK = "scripts/staging/PageRank.dml";
	private final static double sparsity1 = 0.41; // dense
	private final static double sparsity2 = 0.05; // sparse

	private final static double eps = 0.1;

	private final static int rows = 1468;
	private final static int cols = 1468;
	private final static double alpha = 0.85;
	private final static double maxiter = 10;

	@Test
	public void testPageRankSparse() {
		runPageRankTestMLC(true);
	}

	@Test
	public void testPageRankDense() {
		runPageRankTestMLC(false);
	}


	private void runPageRankTestMLC(boolean sparse) {

		//generate actual datasets
		double[][] G = getRandomMatrix(rows, cols, 1, 1, sparse?sparsity2:sparsity1, 234);
		double[][] p = getRandomMatrix(cols, 1, 0, 1e-14, 1, 71);
		double[][] e = getRandomMatrix(rows, 1, 0, 1e-14, 1, 72);
		double[][] u = getRandomMatrix(1, cols, 0, 1e-14, 1, 73);
		writeInputMatrixWithMTD("G", G, true);
		writeInputMatrixWithMTD("p", p, true);
		writeInputMatrixWithMTD("e", e, true);
		writeInputMatrixWithMTD("u", u, true);


		fullRScriptName = "src/test/scripts/functions/codegenalg/Algorithm_PageRank.R";

		rCmd = getRCmd(inputDir(), String.valueOf(alpha),
				String.valueOf(maxiter), expectedDir());
		runRScript(true);

		MatrixBlock outmat = new MatrixBlock();

		Script pr = dmlFromFile(TEST_SCRIPT_PAGERANK);
		pr.in("G", G).in("p", p).in("e", e).in("u", u)
				.in("$5", alpha).in("$6", maxiter)
				.out("p");
		outmat = ml.execute(pr).getMatrix("p").toMatrixBlock();


		//compare matrices
		HashMap<MatrixValue.CellIndex, Double> rfile = readRMatrixFromFS("p");
		TestUtils.compareMatrices(rfile, outmat, eps);
	}
}
