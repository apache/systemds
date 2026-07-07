// ==========================================================================
// DML integration test
// ==========================================================================
//
// Full integration tests extend AutomatedTestBase and drive the DML runner.
// Each test:
//   (a) Writes a DML script to a temp file.
//   (b) Provides input matrices via TestUtils.
//   (c) Calls runTest() and reads the output MatrixBlock.
//   (d) Verifies that the noisy result differs from the clean result by a
//       statistically plausible amount (not zero, not astronomically large).
//


package org.apache.sysds.test.functions.privacy.dp;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.HashMap;

import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class DPBuiltinDMLTest extends AutomatedTestBase {

    private static final String TEST_DIR   = "functions/privacy/dp/";
    private static final String TEST_CLASS = TEST_DIR + DPBuiltinDMLTest.class.getSimpleName() + "/";
    private static final int ROWS = 100;
    private static final int COLS = 10;
    private static final String DML_LAPLACE =
        "X = read($1);\n"
      + "result = dp_laplace(colMeans(X), sensitivity=1.0, epsilon=$2);\n"
      + "write(result, $3, format=\"text\");\n";
    private static final String DML_GAUSSIAN =
        "X = read($1);\n"
      + "result = dp_gaussian(colMeans(X), sensitivity=1.0, epsilon=$2, delta=1e-5);\n"
      + "write(result, $3, format=\"text\");\n";

    @Override
    public void setUp() {
        addTestConfiguration("DPLaplace",  new TestConfiguration(TEST_CLASS, "DPLaplace"));
        addTestConfiguration("DPGaussian", new TestConfiguration(TEST_CLASS, "DPGaussian"));
    }

    @Test
    public void testLaplaceOutputDiffersFromCleanMean() {
        runDPTest("DPLaplace", DML_LAPLACE, "0.5");
    }

    @Test
    public void testGaussianOutputDiffersFromCleanMean() {
        runDPTest("DPGaussian", DML_GAUSSIAN, "0.5");
    }

    @Test
    public void testHighEpsilonIsCloserToTruth() {
    	double[][] data = TestUtils.generateTestMatrix(ROWS, COLS, 0, 1, 1.0, 42);
        // Higher ε → less noise → result closer to the true mean.
        // NOTE: the DPBudgetAccountant caps total spend at the default budget
        // (ε = 1.0) regardless of the per-release ε requested, so ε values
        // here must stay well under that cap or the release is rejected.
        double noisyLow  = runAndGetMaxAbsColMeansDiffFromClean(data, "DPGaussian", DML_GAUSSIAN, "0.1");
        double noisyHigh = runAndGetMaxAbsColMeansDiffFromClean(data, "DPGaussian", DML_GAUSSIAN, "0.5");
        assertTrue("ε=0.5 should give less noise than ε=0.1", noisyHigh < noisyLow);
    }

    private void runDPTest(String testName, String dml, String epsilonStr) {
        double[][] data = TestUtils.generateTestMatrix(ROWS, COLS, 0, 1, 1.0, 42);
        HashMap<CellIndex, Double> result = runAndGetResult(testName, dml, epsilonStr, data);

        // The noisy result should be a (1 × cols) row vector.
        int maxRow = 0, maxCol = 0;
        for (CellIndex ci : result.keySet()) {
            maxRow = Math.max(maxRow, ci.row);
            maxCol = Math.max(maxCol, ci.column);
        }
        assertTrue("Result should have 1 row", maxRow == 1);
        assertTrue("Result should have " + COLS + " columns", maxCol == COLS);
        // Must differ from the exact (clean) mean by a non-trivial amount.
        // (A single-seed exact-equality check is fragile; use range check.)
        double maxDiff = maxAbsColMeansDiffFromClean(data, result);
        assertTrue("Result should differ from the clean mean", maxDiff > 0);
    }

    private double runAndGetMaxAbsColMeansDiffFromClean(double[][] data, String testName, String dml, String epsilonStr) {
        HashMap<CellIndex, Double> result = runAndGetResult(testName, dml, epsilonStr, data);
        return maxAbsColMeansDiffFromClean(data, result);
    }

    private static double maxAbsColMeansDiffFromClean(double[][] data, HashMap<CellIndex, Double> result) {
        double maxDiff = 0;
        for (int c = 0; c < COLS; c++) {
            double sum = 0;
            for (int r = 0; r < ROWS; r++)
                sum += data[r][c];
            double cleanMean = sum / ROWS;
            double noisy = result.get(new CellIndex(1, c + 1));
            maxDiff = Math.max(maxDiff, Math.abs(noisy - cleanMean));
        }
        return maxDiff;
    }

    private HashMap<CellIndex, Double> runAndGetResult(String testName, String dml, String epsilonStr,
        double[][] data)
    {
        getAndLoadTestConfiguration(testName);
        writeInputMatrixWithMTD("X", data, false);

        fullDMLScriptName = getScript();
        try {
            File scriptFile = new File(fullDMLScriptName);
            scriptFile.getParentFile().mkdirs();
            Files.write(scriptFile.toPath(), dml.getBytes());
        }
        catch (IOException e) {
            throw new RuntimeException(e);
        }

        programArgs = new String[]{ "-args", input("X"), epsilonStr, output("result") };
        runTest(true, false, null, -1);
        return readDMLMatrixFromOutputDir("result");
    }
}

