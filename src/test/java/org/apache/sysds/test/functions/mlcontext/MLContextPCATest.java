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
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;

import static org.apache.sysds.api.mlcontext.ScriptFactory.dmlFromFile;

public class MLContextPCATest extends MLContextTestBase {
	protected static Logger log = Logger.getLogger(MLContextPCATest.class);

	protected final static String TEST_SCRIPT_PCA = "scripts/algorithms/PCA.dml";
	private final static double sparsity1 = 0.41; // dense
	private final static double sparsity2 = 0.05; // sparse

	private final static double eps = 0.1;

	private final static int rows = 1468;
	private final static int cols1 = 1007;
	private final static int cols2 = 387;

	@Test
	public void testPCASparse() {
		runPCATestMLC(ExecType.CP, true);
	}

	@Test
	public void testPCADense() {
		runPCATestMLC(ExecType.CP, false);
	}


	private void runPCATestMLC(ExecType instType, boolean sparse) {

		//generate actual datasets
		int cols = (instType== ExecType.SPARK) ? cols2 : cols1;
		double[][] A = getRandomMatrix(rows, cols, 0, 1, sparse?sparsity2:sparsity1, 714);
		writeInputMatrixWithMTD("A", A, true);


		fullRScriptName = "src/test/scripts/functions/codegenalg/Algorithm_PCA.R";

		rCmd = getRCmd(inputDir(), expectedDir());
		runRScript(true);


		Script pr = dmlFromFile(TEST_SCRIPT_PCA);
		pr.in("A", A)
				.out("eval_stdev_dominant")
				.out("eval_dominant")
				.out("evec_dominant");
		MLResults outres = ml.execute(pr);
		MatrixBlock dmlstd = outres.getMatrix("eval_stdev_dominant").toMatrixBlock();
		MatrixBlock dmleval = outres.getMatrix("eval_dominant").toMatrixBlock();
		MatrixBlock dmlevec = outres.getMatrix("evec_dominant").toMatrixBlock();


		//compare matrices
		HashMap<MatrixValue.CellIndex, Double> reval   = readRMatrixFromFS("dominant.eigen.values");
//		HashMap<MatrixValue.CellIndex, Double> revec = readRMatrixFromFS("dominant.eigen.vectors");
		HashMap<MatrixValue.CellIndex, Double> rstd   = readRMatrixFromFS("dominant.eigen.standard.deviations");
		TestUtils.compareMatrices(rstd, dmlstd, eps);
		TestUtils.compareMatrices(reval, dmleval, eps);
//		TestUtils.compareMatrices(revec, dmlevec, eps);
	}
}
