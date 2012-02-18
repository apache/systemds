package dml.test.integration.applications;

import java.util.HashMap;

import org.junit.Test;

import dml.runtime.matrix.io.MatrixValue.CellIndex;
import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;
import dml.test.utils.TestUtils;

public class GNMFTest extends AutomatedTestBase {

	private final static String TEST_DIR = "applications/gnmf/";
	private final static String TEST_GNMF = "GNMF";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_GNMF, new TestConfiguration(TEST_DIR, TEST_GNMF, new String[] { "w", "h" }));
	}
	
	@Test
	public void testGNMFWithRDMLAndJava() {
		int m = 2000;
		int n = 1500;
		int k = 50;
		int maxiter = 3;
		
		TestConfiguration config = getTestConfiguration(TEST_GNMF);
		
		/* This is for running the junit test the old way, i.e., replace $$x$$ in DML script with its value */
		config.addVariable("m", m);
		config.addVariable("n", n);
		config.addVariable("k", k);
		config.addVariable("maxiter", maxiter);
		
		double Eps = Math.pow(10, -8);

		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String GNMF_HOME = SCRIPT_DIR + TEST_DIR;
		dmlArgs = new String[]{"-f", GNMF_HOME + TEST_GNMF + ".dml",
				               "-args", "\"" + GNMF_HOME + INPUT_DIR + "v" + "\"", 
				                        "\"" + GNMF_HOME + INPUT_DIR + "w" + "\"", 
				                        "\"" + GNMF_HOME + INPUT_DIR + "h" + "\"", 
				                        Integer.toString(m), Integer.toString(n), Integer.toString(k), Integer.toString(maxiter),
				                        "\"" + GNMF_HOME + OUTPUT_DIR + "w" + "\"", 
				                        "\"" + GNMF_HOME + OUTPUT_DIR + "h" + "\""};
		dmlArgsDebug = new String[]{"-f", GNMF_HOME + TEST_GNMF + ".dml", "-d",
	                                "-args", "\"" + GNMF_HOME + INPUT_DIR + "v" + "\"", 
	                                         "\"" + GNMF_HOME + INPUT_DIR + "w" + "\"", 
	                                         "\"" + GNMF_HOME + INPUT_DIR + "h" + "\"", 
	                                         Integer.toString(m), Integer.toString(n), Integer.toString(k), Integer.toString(maxiter),
	                                         "\"" + GNMF_HOME + OUTPUT_DIR + "w" + "\"", 
	                                         "\"" + GNMF_HOME + OUTPUT_DIR + "h" + "\""};
		rCmd = "Rscript" + " " + GNMF_HOME + TEST_GNMF + ".R" + " " + 
		       GNMF_HOME + INPUT_DIR + " " + Integer.toString(maxiter) + " " + GNMF_HOME + EXPECTED_DIR;
		
		loadTestConfiguration(config);

		double[][] v = getRandomMatrix(m, n, 1, 5, 0.2, -1);
		double[][] w = getRandomMatrix(m, k, 0, 1, 1, -1);
		double[][] h = getRandomMatrix(k, n, 0, 1, 1, -1);

		writeInputMatrix("v", v, true);
		writeInputMatrix("w", w, true);
		writeInputMatrix("h", h, true);

		for (int i = 0; i < maxiter; i++) {
			double[][] tW = TestUtils.performTranspose(w);
			double[][] tWV = TestUtils.performMatrixMultiplication(tW, v);
			double[][] tWW = TestUtils.performMatrixMultiplication(tW, w);
			double[][] tWWH = TestUtils.performMatrixMultiplication(tWW, h);
			for (int j = 0; j < k; j++) {
				for (int l = 0; l < n; l++) {
					h[j][l] = h[j][l] * (tWV[j][l] / (tWWH[j][l] + Eps));
				}
			}

			double[][] tH = TestUtils.performTranspose(h);
			double[][] vTH = TestUtils.performMatrixMultiplication(v, tH);
			double[][] hTH = TestUtils.performMatrixMultiplication(h, tH);
			double[][] wHTH = TestUtils.performMatrixMultiplication(w, hTH);
			for (int j = 0; j < m; j++) {
				for (int l = 0; l < k; l++) {
					w[j][l] = w[j][l] * (vTH[j][l] / (wHTH[j][l] + Eps));
				}
			}
		}

		boolean exceptionExpected = false;
		/*
		 * Expected number of jobs:
		 * Reblock - 1 job 
		 * While loop iteration - 10 jobs
		 * Final output write - 1 job
		 */
		int expectedNumberOfJobs = 12;
		
		/* GNMF must be run in the new way as GNMF.dml will be shipped */
		runTest(true, exceptionExpected, null, expectedNumberOfJobs); 
		
		runRScript(true);
		disableOutAndExpectedDeletion();

		HashMap<CellIndex, Double> hmWDML = readDMLMatrixFromHDFS("w");
		HashMap<CellIndex, Double> hmHDML = readDMLMatrixFromHDFS("h");
		HashMap<CellIndex, Double> hmWR = readRMatrixFromFS("w");
		HashMap<CellIndex, Double> hmHR = readRMatrixFromFS("h");
		HashMap<CellIndex, Double> hmWJava = TestUtils.convert2DDoubleArrayToHashMap(w);
		HashMap<CellIndex, Double> hmHJava = TestUtils.convert2DDoubleArrayToHashMap(h);

		TestUtils.compareMatrices(hmWDML, hmWR, 0.000001, "hmWDML", "hmWR");
		TestUtils.compareMatrices(hmWDML, hmWJava, 0.000001, "hmWDML", "hmWJava");
		TestUtils.compareMatrices(hmWR, hmWJava, 0.000001, "hmRDML", "hmWJava");
	}
}
