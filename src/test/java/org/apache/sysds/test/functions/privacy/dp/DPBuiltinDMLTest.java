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


import java.util.HashMap;

import org.apache.sysds.parser.LanguageException;
import org.apache.sysds.parser.ParseException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Assert;
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

	private double[][] data;

	@Override
	public void setUp() {
		addTestConfiguration("DPLaplace", new TestConfiguration(TEST_CLASS, "DPLaplaceTest"));
		addTestConfiguration("DPGaussian", new TestConfiguration(TEST_CLASS, "DPGaussianTest"));
		addTestConfiguration("DPSetBudget", new TestConfiguration(TEST_CLASS, "DPSetBudgetTest"));
		addTestConfiguration("DPSetBudgetTwice", new TestConfiguration(TEST_CLASS, "DPSetBudgetTwiceTest"));
		addTestConfiguration("DPSetBudgetNonLiteral", new TestConfiguration(TEST_CLASS, "DPSetBudgetNonLiteralTest"));
		data = this.getRandomMatrix(ROWS, COLS, 0, 1, 1.0, 42);
	}

	@Test
	public void testLaplaceOutputDiffersFromCleanMean() {
		runColMeansDPTest("DPLaplace", "0.5");
	}

	@Test
	public void testGaussianOutputDiffersFromCleanMean() {
		runColMeansDPTest("DPGaussian", "0.5");
	}

	@Test
	public void testLaplaceColSums() {
		// query="colSums": T is 1 x n filled with 1.0, output is the noisy column-sum row vector.
		HashMap<CellIndex, Double> result = runAndGetResult("DPLaplace", "colSums",
			"0.5", data);
		assertShape(result, 1, COLS);
		double maxDiff = maxAbsDiffFromClean(data, result, DPBuiltinDMLTest::colSum);
		Assert.assertTrue("Result should differ from the clean column sums", maxDiff > 0);
	}

	@Test
	public void testGaussianIdentity() {
		// query="identity": T is the n x n identity, output is a noisy release of X itself.
		HashMap<CellIndex, Double> result = runAndGetResult("DPGaussian", "identity", "0.5", data);
		assertShape(result, ROWS, COLS);
		// identity releases X row-by-row, so compare cell-by-cell rather than via a per-column reduction.
		double maxCellDiff = 0;
		for(int r = 0; r < ROWS; r++) {
			for(int c = 0; c < COLS; c++) {
				double noisy = result.get(new CellIndex(r + 1, c + 1));
				maxCellDiff = Math.max(maxCellDiff, Math.abs(noisy - data[r][c]));
			}
		}
		Assert.assertTrue("Result should differ from the clean matrix", maxCellDiff > 0);
	}

	@Test
	public void testHighEpsilonIsCloserToTruth() {
		// Higher epsilon => less noise => result closer to the true mean.
		// NOTE: the DPBudgetAccountant caps total spend at the default budget
		// (epsilon = 1.0) regardless of the per-release epsilon requested, so epsilon values
		// here must stay well under that cap or the release is rejected.
		double noisyLow = runAndGetMaxAbsColMeansDiffFromClean(data, "DPGaussian", "0.1");
		double noisyHigh = runAndGetMaxAbsColMeansDiffFromClean(data, "DPGaussian", "0.5");
		Assert.assertTrue("epsilon=0.5 should give less noise than epsilon=0.1", noisyHigh < noisyLow);
	}

	@Test
	public void testSetBudgetLiteralAllowsExceedingDefaultBudget() {
		// Default budget is epsilon=1.0; a single release at epsilon=1.5 would be
		// rejected unless dp_set_budget(3.0, ...) widens it first.
		HashMap<CellIndex, Double> result = runAndGetResult("DPSetBudget",
			"3.0", "1.5", data);
		assertShape(result, 1, COLS);
	}

	@Test
	public void testSetBudgetNarrowBudgetStillEnforced() {
		// An explicit narrow budget must still be enforced: epsilon=0.8 exceeds
		// the explicit budget of 0.5.
		runExpectingException("DPSetBudget", "0.5", "0.8", data,
			DMLRuntimeException.class);
	}

	@Test
	public void testSetBudgetCalledTwiceFailsAtCompileTime() {
		// Thrown from DMLTranslator.processBuiltinFunctionExpression (HOP construction),
		// which wraps all case-block exceptions in ParseException (see processExpression's
		// catch-all) - unlike the non-literal check below, which runs during validation
		// and so surfaces as a bare LanguageException.
		runExpectingException("DPSetBudgetTwice", "3.0", "0.5", data, ParseException.class);
	}

	@Test
	public void testSetBudgetRejectsNonLiteralArgs() {
		runExpectingException("DPSetBudgetNonLiteral", "unused", "0.5", data, LanguageException.class);
	}

	private void runColMeansDPTest(String testName, String epsilonStr) {
		HashMap<CellIndex, Double> result = runAndGetResult(testName, "colMeans", epsilonStr, data);
		assertShape(result, 1, COLS);
		// Must differ from the exact (clean) mean by a non-trivial amount.
		// (A single-seed exact-equality check is fragile; use range check.)
		double maxDiff = maxAbsDiffFromClean(data, result, DPBuiltinDMLTest::colMean);
		Assert.assertTrue("Result should differ from the clean mean", maxDiff > 0);
	}

	private double runAndGetMaxAbsColMeansDiffFromClean(double[][] data, String testName,
		String epsilonStr) {
		HashMap<CellIndex, Double> result = runAndGetResult(testName, "colMeans", epsilonStr, data);
		return maxAbsDiffFromClean(data, result, DPBuiltinDMLTest::colMean);
	}

	private static void assertShape(HashMap<CellIndex, Double> result, int expectedRows, int expectedCols) {
		int maxRow = 0, maxCol = 0;
		for(CellIndex ci : result.keySet()) {
			maxRow = Math.max(maxRow, ci.row);
			maxCol = Math.max(maxCol, ci.column);
		}
		Assert.assertEquals("Result should have " + expectedRows + " row(s)", expectedRows, maxRow);
		Assert.assertEquals("Result should have " + expectedCols + " column(s)", expectedCols, maxCol);
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

	private HashMap<CellIndex, Double> runAndGetResult(String testName, String query, String epsilonStr,
		double[][] data) {
		prepareScript(testName, query, epsilonStr, data);
		runTest(true, false, null, -1);
		return readDMLMatrixFromOutputDir("result");
	}

	private void runExpectingException(String testName, String query, String epsilonStr, double[][] data,
		Class<?> expectedException) {
		prepareScript(testName, query, epsilonStr, data);
		runTest(true, true, expectedException, -1);
	}

	private void prepareScript(String testName, String query, String epsilonStr, double[][] data) {
		getAndLoadTestConfiguration(testName);
		writeInputMatrixWithMTD("X", data, false);

		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + testName + "Test.dml";

		programArgs = new String[] {"-args", input("X"), query, epsilonStr, output("result")};
	}
}
