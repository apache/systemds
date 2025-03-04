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

package org.apache.sysds.test.functions.rewrite;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

import java.util.HashMap;

public class RewriteDetTest extends AutomatedTestBase 
{
	private static final String TEST_NAME_MIXED = "RewriteDetMixed";
	private static final String TEST_NAME_MULT = "RewriteDetMult";
	private static final String TEST_NAME_TRANSPOSE = "RewriteDetTranspose";
	private static final String TEST_NAME_SCALAR_MATRIX_MULT = "RewriteDetScalarMatrixMult";

	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteDetTest.class.getSimpleName() + "/";

	private final static int rows = 23;
	private final static double _sparsityDense = 0.7;
	private final static double _sparsitySparse = 0.2;
	private final static double eps = 1e-8;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME_MIXED, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_MIXED, new String[] { "d" }));
		// det(A%*%B) -> det(A)*det(B)
		addTestConfiguration(TEST_NAME_MULT, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_MULT, new String[] { "d" }));
		// det(t(A)) -> det(A)
		addTestConfiguration(TEST_NAME_TRANSPOSE, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_TRANSPOSE, new String[] { "d" }));
		// det(lambda*A) -> lambda^ncol*det(A)
		// This is faster, because lambda is a scalar, that can be multiplied
		// with in logarithmic time O(log(nrow(A))), whereas lambda needs to
		// be multiplied to every element in A, which is O(nrow(A)^2)).
		addTestConfiguration(TEST_NAME_SCALAR_MATRIX_MULT, new TestConfiguration(
			TEST_CLASS_DIR, TEST_NAME_SCALAR_MATRIX_MULT, new String[] { "d" }));
	}

	@Test
	public void testRewriteDetMixedSparseNoRewrite() {
		runRewriteDetTest(TEST_NAME_MIXED, false, true, ExecType.CP);
	}

	@Test
	public void testRewriteDetMixedSparseRewrite() {
		runRewriteDetTest(TEST_NAME_MIXED, true, true, ExecType.CP);
	}

	@Test
	public void testRewriteDetMixedDenseNoRewrite() {
		runRewriteDetTest(TEST_NAME_MIXED, false, false, ExecType.CP);
	}

	@Test
	public void testRewriteDetMixedDenseRewrite() {
		runRewriteDetTest(TEST_NAME_MIXED, true, false, ExecType.CP);
	}

	@Test
	public void testRewriteDetMultDenseRewrite() {
		runRewriteDetTest(TEST_NAME_MULT, true, false, ExecType.CP);
	}

	@Test
	public void testRewriteDetMultSparseRewrite() {
		runRewriteDetTest(TEST_NAME_MULT, true, true, ExecType.CP);
	}

	@Test
	public void testRewriteDetMultDenseNoRewrite() {
		runRewriteDetTest(TEST_NAME_MULT, false, false, ExecType.CP);
	}

	@Test
	public void testRewriteDetMultSparseNoRewrite() {
		runRewriteDetTest(TEST_NAME_MULT, false, true, ExecType.CP);
	}

	@Test
	public void testRewriteDetTransposeSparseRewrite() {
		runRewriteDetTest(TEST_NAME_TRANSPOSE, true, true, ExecType.CP);
	}

	@Test
	public void testRewriteDetTransposeDenseRewrite() {
		runRewriteDetTest(TEST_NAME_TRANSPOSE, true, false, ExecType.CP);
	}

	@Test
	public void testRewriteDetTransposeSparseNoRewrite() {
		runRewriteDetTest(TEST_NAME_TRANSPOSE, false, true, ExecType.CP);
	}

	@Test
	public void testRewriteDetTransposeDenseNoRewrite() {
		runRewriteDetTest(TEST_NAME_TRANSPOSE, false, false, ExecType.CP);
	}

	@Test
	public void testRewriteDetScalarMatrixMultDenseNoRewrite() {
		runRewriteDetTest(TEST_NAME_SCALAR_MATRIX_MULT, false, false, ExecType.CP);
	}

	@Test
	public void testRewriteDetScalarMatrixMultSparseNoRewrite() {
		runRewriteDetTest(TEST_NAME_SCALAR_MATRIX_MULT, false, true, ExecType.CP);
	}

	@Test
	public void testRewriteDetScalarMatrixMultDenseRewrite() {
		runRewriteDetTest(TEST_NAME_SCALAR_MATRIX_MULT, true, false, ExecType.CP);
	}

	@Test
	public void testRewriteDetScalarMatrixMultSparseRewrite() {
		runRewriteDetTest(TEST_NAME_SCALAR_MATRIX_MULT, true, true, ExecType.CP);
	}

	private void runRewriteDetTest(String testScriptName, boolean rewrites, boolean sparse, ExecType et) {
		// NOTE The sparsity of the matrix is considered, because rewrite
		// simplifications could be made if the matrix contains a lot of zeros.
		// Furthermore, some det-algorithms perform optimizations
		// (early termination, less recursions, ...) when the matrix is sparse.
		// Therefore dense and sparse matrices are part of the rewrite tests.

		ExecMode platformOld = setExecMode(et);
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;

		try {
			double sparsity = (sparse) ? _sparsitySparse : _sparsityDense;
			getAndLoadTestConfiguration(testScriptName);

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testScriptName + ".dml";

			boolean twoMatrixArg = testScriptName.equals(TEST_NAME_MULT);
			boolean oneMatrixOneScalarArg = testScriptName.equals(TEST_NAME_SCALAR_MATRIX_MULT);
			boolean twoMatrixOneScalarArg = testScriptName.equals(TEST_NAME_MIXED);
			if (twoMatrixArg) {
				programArgs = new String[]{"-stats", "-args", input("A"), input("B"), output("d")};
			}
			else if (oneMatrixOneScalarArg) {
				programArgs = new String[]{"-stats", "-args", input("A"), input("lambda"), output("d")};
			}
			else if (twoMatrixOneScalarArg) {
				programArgs = new String[]{"-stats", "-args", input("A"), input("B"), input("lambda"), output("d")};
			}
			else {
				programArgs = new String[]{"-stats", "-args", input("A"), output("d")};
			}

			fullRScriptName = HOME + testScriptName + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			double[][] A = getRandomMatrix(rows, rows, -1, 1, sparsity, 21332);
			writeInputMatrixWithMTD("A", A, true);
			if (twoMatrixArg) {
				double[][] B = getRandomMatrix(rows, rows, -1, 1, sparsity, 42422);
				writeInputMatrixWithMTD("B", B, true);
			}
			else if (twoMatrixOneScalarArg) {
				double[][] B = getRandomMatrix(rows, rows, -1, 1, sparsity, 4242);
				writeInputMatrixWithMTD("B", B, true);
				double[][] lambda = getRandomMatrix(1, 1, -1, 1, sparsity, 121);
				writeInputMatrixWithMTD("lambda", lambda, true);
			}
			else if (oneMatrixOneScalarArg) {
				double[][] lambda = getRandomMatrix(1, 1, -1, 1, sparsity, 121);
				writeInputMatrixWithMTD("lambda", lambda, true);
			}

			runTest(true, false, null, -1);
			runRScript(true);

			HashMap<CellIndex, Double> dmlfile = readDMLScalarFromOutputDir("d");
			HashMap<CellIndex, Double> rfile  = readRScalarFromExpectedDir("d");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

			if (rewrites) {
				Assert.assertTrue(
					(!testScriptName.equals(TEST_NAME_TRANSPOSE) || !heavyHittersContainsString(Opcodes.TRANSPOSE.toString()))
					&& (!testScriptName.equals(TEST_NAME_MULT) || Statistics.getCPHeavyHitterCount(Opcodes.DET.toString()) == 2)
					&& (!testScriptName.equals(TEST_NAME_SCALAR_MATRIX_MULT) || heavyHittersContainsString(Opcodes.POW.toString()))
					&& (!testScriptName.equals(TEST_NAME_MIXED) || (
						Statistics.getCPHeavyHitterCount(Opcodes.DET.toString()) == 2 
						&& !heavyHittersContainsString(Opcodes.TRANSPOSE.toString()) 
						&& heavyHittersContainsString(Opcodes.POW.toString()))));
			}
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
			resetExecMode(platformOld);
		}
	}
}
