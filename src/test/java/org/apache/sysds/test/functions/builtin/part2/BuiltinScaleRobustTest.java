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

			int[] rowSizes = {10000, 20000, 50000/*, 100000 nur falls genug RAM*/};
			int[] colSizes = {100, 500, 1000};
			int numReps   = 5;

			for(int nRows : rowSizes) {
				for(int nCols : colSizes) {
					System.out.println("\n--- Benchmark for " + nRows + "×" + nCols + " ---");
					long totalTime = 0;

					for(int rep=1; rep<=numReps; rep++) {
						System.out.println("Iteration " + rep + " of " + numReps);

						// 1) Create input matrix
						double[][] A = getRandomMatrix(nRows, nCols, -10,10, sparsity, 7);
						writeInputMatrixWithMTD("A", A, true);

						Runtime.getRuntime().gc();
						long memBefore = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();

						// 2) Run DML script
						Statistics.reset();
						long t0 = System.nanoTime();
						runTest(true, false, null, -1);
						long t1 = System.nanoTime();
						long duration = t1 - t0;
						totalTime += duration;

						long memAfter = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
						System.out.println("  Time (ms): "   + (duration/1e6));
						System.out.println("  Mem used (MB): " + ((memAfter - memBefore)/(1024*1024.0)));

						// 3) Validate results
						if(rep == 1) {
							// Run R script
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

					double avgMs = totalTime/1e6/numReps;
					System.out.println(">>> Average for " + nRows + "×" + nCols + ": " + avgMs + " ms");
				}
			}
					
				

		} catch (Exception e) {
			throw new RuntimeException(e);
		} finally {
			rtplatform = old;
		}
	}
}
