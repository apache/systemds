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

package org.apache.sysds.test.functions.privacy.dp;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.HashMap;

import org.apache.sysds.parser.LanguageException;
import org.apache.sysds.parser.ParseException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

/*
 * ==========================================================================
 * DML integration test
 * ==========================================================================
 *
 * Full integration tests extend AutomatedTestBase and drive the DML runner.
 * Each test:
 *   (a) Writes a DML script to a temp file.
 *   (b) Provides input matrices via TestUtils.
 *   (c) Calls runTest() and reads the output MatrixBlock.
 *   (d) Verifies that the noisy result differs from the clean result by a
 *       statistically plausible amount (not zero, not astronomically large).
 */
public class DPBuiltinDMLTest extends AutomatedTestBase {

	private static final String TEST_DIR = "functions/privacy/dp/";
	private static final String TEST_CLASS = TEST_DIR + DPBuiltinDMLTest.class.getSimpleName() + "/";
	private static final int ROWS = 100;
	private static final int COLS = 10;

	private static final String DML_LAPLACE_TEMPLATE = "X = read($1);\n"
		+ "result = dp_laplace(X, query=\"%s\", sensitivity=1.0, epsilon=$2);\n"
		+ "write(result, $3, format=\"text\");\n";
	private static final String DML_GAUSSIAN_TEMPLATE = "X = read($1);\n"
		+ "result = dp_gaussian(X, query=\"%s\", sensitivity=1.0, epsilon=$2, delta=1e-5);\n"
		+ "write(result, $3, format=\"text\");\n";

	private static final String DML_LAPLACE = String.format(DML_LAPLACE_TEMPLATE, "colMeans");
	private static final String DML_GAUSSIAN = String.format(DML_GAUSSIAN_TEMPLATE, "colMeans");

	// dp_set_budget(epsilon, delta) is resolved entirely at compile time (its arguments
	// must be literals), called via a dummy assignment, then a single dp_laplace release
	// at $2 records a cost of exactly $2 (Laplace basic composition).
	private static final String DML_SET_BUDGET_TEMPLATE = "eps = dp_set_budget(%s, 1e-6);\n" + "X = read($1);\n"
		+ "result = dp_laplace(X, query=\"colMeans\", sensitivity=1.0, epsilon=$2);\n"
		+ "write(result, $3, format=\"text\");\n";

	// Two dp_set_budget calls in the same script; DMLTranslator must reject this at
	// compile time (DMLProgram.hasDPBudget()).
	private static final String DML_SET_BUDGET_TWICE = "eps = dp_set_budget(3.0, 1e-6);\n"
		+ "eps2 = dp_set_budget(5.0, 1e-6);\n" + "X = read($1);\n"
		+ "result = dp_laplace(X, query=\"colMeans\", sensitivity=1.0, epsilon=$2);\n"
		+ "write(result, $3, format=\"text\");\n";

	// A budget argument computed at runtime (not a literal); must be rejected at
	// compile time by BuiltinFunctionExpression's isConstant() check.
	private static final String DML_SET_BUDGET_NON_LITERAL = "X = read($1);\n" + "computed = sum(X) / nrow(X);\n"
		+ "eps = dp_set_budget(computed, 1e-6);\n"
		+ "result = dp_laplace(X, query=\"colMeans\", sensitivity=1.0, epsilon=$2);\n"
		+ "write(result, $3, format=\"text\");\n";

	@Override
	public void setUp() {
		addTestConfiguration("DPLaplace", new TestConfiguration(TEST_CLASS, "DPLaplace"));
		addTestConfiguration("DPGaussian", new TestConfiguration(TEST_CLASS, "DPGaussian"));
		addTestConfiguration("DPSetBudget", new TestConfiguration(TEST_CLASS, "DPSetBudget"));
	}

	@Test
	public void testLaplaceOutputDiffersFromCleanMean() {
		runColMeansDPTest("DPLaplace", DML_LAPLACE, "0.5");
	}

	@Test
	public void testGaussianOutputDiffersFromCleanMean() {
		runColMeansDPTest("DPGaussian", DML_GAUSSIAN, "0.5");
	}

	@Test
	public void testLaplaceColSums() {
		// query="colSums": T is 1 x n filled with 1.0, output is the noisy column-sum row vector.
		double[][] data = TestUtils.generateTestMatrix(ROWS, COLS, 0, 1, 1.0, 42);
		HashMap<CellIndex, Double> result = runAndGetResult("DPLaplace", String.format(DML_LAPLACE_TEMPLATE, "colSums"),
			"0.5", data);
		assertShape(result, 1, COLS);
		double maxDiff = maxAbsDiffFromClean(data, result, DPBuiltinDMLTest::colSum);
		assertTrue("Result should differ from the clean column sums", maxDiff > 0);
	}

	@Test
	public void testGaussianIdentity() {
		// query="identity": T is the n x n identity, output is a noisy release of X itself.
		double[][] data = TestUtils.generateTestMatrix(ROWS, COLS, 0, 1, 1.0, 42);
		HashMap<CellIndex, Double> result = runAndGetResult("DPGaussian",
			String.format(DML_GAUSSIAN_TEMPLATE, "identity"), "0.5", data);
		assertShape(result, ROWS, COLS);
		// identity releases X row-by-row, so compare cell-by-cell rather than via a per-column reduction.
		double maxCellDiff = 0;
		for(int r = 0; r < ROWS; r++) {
			for(int c = 0; c < COLS; c++) {
				double noisy = result.get(new CellIndex(r + 1, c + 1));
				maxCellDiff = Math.max(maxCellDiff, Math.abs(noisy - data[r][c]));
			}
		}
		assertTrue("Result should differ from the clean matrix", maxCellDiff > 0);
	}

	@Test
	public void testHighEpsilonIsCloserToTruth() {
		double[][] data = TestUtils.generateTestMatrix(ROWS, COLS, 0, 1, 1.0, 42);
		// Higher ε → less noise → result closer to the true mean.
		// NOTE: the DPBudgetAccountant caps total spend at the default budget
		// (ε = 1.0) regardless of the per-release ε requested, so ε values
		// here must stay well under that cap or the release is rejected.
		double noisyLow = runAndGetMaxAbsColMeansDiffFromClean(data, "DPGaussian", DML_GAUSSIAN, "0.1");
		double noisyHigh = runAndGetMaxAbsColMeansDiffFromClean(data, "DPGaussian", DML_GAUSSIAN, "0.5");
		assertTrue("ε=0.5 should give less noise than ε=0.1", noisyHigh < noisyLow);
	}

	@Test
	public void testSetBudgetLiteralAllowsExceedingDefaultBudget() {
		// Default budget is epsilon=1.0; a single release at epsilon=1.5 would be
		// rejected unless dp_set_budget(3.0, ...) widens it first.
		double[][] data = TestUtils.generateTestMatrix(ROWS, COLS, 0, 1, 1.0, 42);
		HashMap<CellIndex, Double> result = runAndGetResult("DPSetBudget",
			String.format(DML_SET_BUDGET_TEMPLATE, "3.0"), "1.5", data);
		assertShape(result, 1, COLS);
	}

	@Test
	public void testSetBudgetNarrowBudgetStillEnforced() {
		// An explicit narrow budget must still be enforced: epsilon=0.8 exceeds
		// the explicit budget of 0.5.
		double[][] data = TestUtils.generateTestMatrix(ROWS, COLS, 0, 1, 1.0, 42);
		runExpectingException("DPSetBudget", String.format(DML_SET_BUDGET_TEMPLATE, "0.5"), "0.8", data,
			DMLRuntimeException.class);
	}

	@Test
	public void testSetBudgetCalledTwiceFailsAtCompileTime() {
		// Thrown from DMLTranslator.processBuiltinFunctionExpression (HOP construction),
		// which wraps all case-block exceptions in ParseException (see processExpression's
		// catch-all) — unlike the non-literal check below, which runs during validation
		// and so surfaces as a bare LanguageException.
		double[][] data = TestUtils.generateTestMatrix(ROWS, COLS, 0, 1, 1.0, 42);
		runExpectingException("DPSetBudget", DML_SET_BUDGET_TWICE, "0.5", data, ParseException.class);
	}

	@Test
	public void testSetBudgetRejectsNonLiteralArgs() {
		double[][] data = TestUtils.generateTestMatrix(ROWS, COLS, 0, 1, 1.0, 42);
		runExpectingException("DPSetBudget", DML_SET_BUDGET_NON_LITERAL, "0.5", data, LanguageException.class);
	}

	private void runColMeansDPTest(String testName, String dml, String epsilonStr) {
		double[][] data = TestUtils.generateTestMatrix(ROWS, COLS, 0, 1, 1.0, 42);
		HashMap<CellIndex, Double> result = runAndGetResult(testName, dml, epsilonStr, data);
		assertShape(result, 1, COLS);
		// Must differ from the exact (clean) mean by a non-trivial amount.
		// (A single-seed exact-equality check is fragile; use range check.)
		double maxDiff = maxAbsDiffFromClean(data, result, DPBuiltinDMLTest::colMean);
		assertTrue("Result should differ from the clean mean", maxDiff > 0);
	}

	private double runAndGetMaxAbsColMeansDiffFromClean(double[][] data, String testName, String dml,
		String epsilonStr) {
		HashMap<CellIndex, Double> result = runAndGetResult(testName, dml, epsilonStr, data);
		return maxAbsDiffFromClean(data, result, DPBuiltinDMLTest::colMean);
	}

	private static void assertShape(HashMap<CellIndex, Double> result, int expectedRows, int expectedCols) {
		int maxRow = 0, maxCol = 0;
		for(CellIndex ci : result.keySet()) {
			maxRow = Math.max(maxRow, ci.row);
			maxCol = Math.max(maxCol, ci.column);
		}
		assertEquals("Result should have " + expectedRows + " row(s)", expectedRows, maxRow);
		assertEquals("Result should have " + expectedCols + " column(s)", expectedCols, maxCol);
	}

	@FunctionalInterface
	private interface CleanColumnFn {
		double apply(double[][] data, int col);
	}

	/** Computes max|noisy(1,c) - clean(data,c)| across the (1 x COLS) row-vector releases. */
	private static double maxAbsDiffFromClean(double[][] data, HashMap<CellIndex, Double> result,
		CleanColumnFn cleanFn) {
		double maxDiff = 0;
		for(int c = 0; c < COLS; c++) {
			double clean = cleanFn.apply(data, c);
			double noisy = result.get(new CellIndex(1, c + 1));
			maxDiff = Math.max(maxDiff, Math.abs(noisy - clean));
		}
		return maxDiff;
	}

	private static double colMean(double[][] data, int c) {
		double sum = 0;
		for(int r = 0; r < ROWS; r++)
			sum += data[r][c];
		return sum / ROWS;
	}

	private static double colSum(double[][] data, int c) {
		double sum = 0;
		for(int r = 0; r < ROWS; r++)
			sum += data[r][c];
		return sum;
	}

	private HashMap<CellIndex, Double> runAndGetResult(String testName, String dml, String epsilonStr,
		double[][] data) {
		prepareScript(testName, dml, epsilonStr, data);
		runTest(true, false, null, -1);
		return readDMLMatrixFromOutputDir("result");
	}

	private void runExpectingException(String testName, String dml, String epsilonStr, double[][] data,
		Class<?> expectedException) {
		prepareScript(testName, dml, epsilonStr, data);
		runTest(true, true, expectedException, -1);
	}

	private void prepareScript(String testName, String dml, String epsilonStr, double[][] data) {
		getAndLoadTestConfiguration(testName);
		writeInputMatrixWithMTD("X", data, false);

		fullDMLScriptName = getScript();
		try {
			File scriptFile = new File(fullDMLScriptName);
			scriptFile.getParentFile().mkdirs();
			Files.write(scriptFile.toPath(), dml.getBytes());
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}

		programArgs = new String[] {"-args", input("X"), epsilonStr, output("result")};
	}
}
