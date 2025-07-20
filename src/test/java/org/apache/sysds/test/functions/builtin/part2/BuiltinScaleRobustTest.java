package org.apache.sysds.test.functions.builtin.part2;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.IOException;
import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

public class BuiltinScaleRobustTest extends AutomatedTestBase {
	private final static String TEST_NAME = "scaleRobust";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinScaleRobustTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	private final static int cols = 500;

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
			programArgs = new String[]{"-exec", "singlenode", "-stats", "-args", input("A"), output("B")};
			rCmd = "Rscript " + fullRScriptName + " " + inputDir() + " " + expectedDir();
			String pyCmd = "python " + fullPyScriptName + " " + inputDir() + " " + expectedDir();

			int[] numRowsArr = new int[]{10000};
			int numReps = 10;

			for (int nRows : numRowsArr) {
				System.out.println("\n--- Running benchmark for matrix: " + nRows + " x " + cols + " ---");
				long totalTime = 0;

				for (int i = 0; i < numReps; i++) {
					System.out.println("Iteration " + (i + 1) + " of " + numReps);

					double[][] A = getRandomMatrix(nRows, cols, -10, 10, sparsity, 7);
					writeInputMatrixWithMTD("A", A, true);

					// Measure memory BEFORE DML execution
					Runtime runtime = Runtime.getRuntime();
					runtime.gc();
					long memBefore = runtime.totalMemory() - runtime.freeMemory();

					Statistics.reset();
					long startTime = System.nanoTime();
					runTest(true, false, null, -1); // DML run
					long endTime = System.nanoTime();
					long duration = endTime - startTime;
					totalTime += duration;

					long memAfter = runtime.totalMemory() - runtime.freeMemory();
					long memUsedBytes = memAfter - memBefore;
					double memUsedMB = memUsedBytes / (1024.0 * 1024.0);
					// Includes everything
					System.out.println("Iteration time (ms): " + (duration / 1e6));
					System.out.println("Memory used (MB): " + memUsedMB);

					// Run R and Python only for the first two iterations
					if (i == 0 || i == 1) {
						// Run R
						System.out.println("Running R script...");
						runRScript(true);

						// Run Python
						System.out.println("Running Python script...");
						Process p = Runtime.getRuntime().exec(pyCmd);
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
						if (exitCode != 0) {
							throw new RuntimeException("Python script failed with exit code: " + exitCode);
						}

						// Compare results
						HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
						HashMap<CellIndex, Double> rfile = readRMatrixFromExpectedDir("B");
						HashMap<CellIndex, Double> pyfile = readRMatrixFromExpectedDir("B");

						System.out.println("Comparing DML vs R...");
						TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");

						System.out.println("Comparing DML vs Python...");
						TestUtils.compareMatrices(dmlfile, pyfile, eps, "DML", "Python");
					}
				}

				double avgTimeMs = totalTime / 1e6 / numReps;
				System.out.println(">>> Average time for " + nRows + "x" + cols + " over " + numReps + " runs: " + avgTimeMs + " ms");
			}
		} catch (Exception e) {
			throw new RuntimeException(e);
		} finally {
			rtplatform = old;
		}
	}
}
