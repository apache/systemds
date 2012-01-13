package dml.test.integration.applications;

import java.util.HashMap;

import org.junit.Test;

import dml.runtime.matrix.io.MatrixValue.CellIndex;
import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;
import dml.test.utils.TestUtils;

public class GNMFTest extends AutomatedTestBase {

	private final static String TEST_DIR = "applications/gnmf/";
	private final static String TEST_GNMF = "GNMFTest";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_GNMF, new TestConfiguration(TEST_DIR, "GNMFTest", new String[] { "w", "h" }));
		addTestConfiguration("PartOfGNMF", new TestConfiguration(TEST_DIR, "PartOfGNMF", new String[] { "r", "h" }));
	}

	@Test
	public void testPartOfGNMF() {
		int m = 2000;
		int n = 1500;
		int k = 50;
		int maxiter = 3;

		TestConfiguration config = getTestConfiguration("PartOfGNMF");
		config.addVariable("m", m);
		config.addVariable("n", n);
		config.addVariable("k", k);
		config.addVariable("maxiter", maxiter);
		double Eps = Math.pow(10, -8);

		loadTestConfiguration(config);

		//double[][] v = { { 100 }, { 200 } };
		//double[][] w = { { 0 }, { 100 } };
		//double[][] h = { { 100 } };
		double[][] v = getRandomMatrix(m, n, 1, 5, 0.4, -1);
		double[][] w = getRandomMatrix(m, k, 0, 1, 1, -1);
		double[][] h = getRandomMatrix(k, n, 0, 1, 1, -1);
		double[][] r = null;

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
			double[][] hTH = TestUtils.performMatrixMultiplication(h, tH);
			r = hTH;
		}

		
		boolean exceptionExpected = false;
		/*
		 * Expected number of jobs:
		 * Reblock - 1 job 
		 * While loop iteration - 7 jobs
		 * Final output write - 1 job
		 */
		int expectedNumberOfJobs = 9;
		runTest(exceptionExpected, null, expectedNumberOfJobs);
		
		runRScript();

		HashMap<CellIndex, Double> hmRDML = readDMLMatrixFromHDFS("r");
		HashMap<CellIndex, Double> hmRJava = TestUtils.convert2DDoubleArrayToHashMap(r);
		HashMap<CellIndex, Double> hmRR = readRMatrixFromFS("r");

		TestUtils.compareMatrices(hmRDML, hmRJava, 0.000001, "hmRDML", "hmRJava");
		TestUtils.compareMatrices(hmRDML, hmRR, 0.000001, "hmRDML", "hmRR");
	}

	@Test
	public void testGNMFWithRDMLAndJava() {
		int m = 2000;
		int n = 1500;
		int k = 50;
		int maxiter = 3;

		TestConfiguration config = getTestConfiguration(TEST_GNMF);
		config.addVariable("m", m);
		config.addVariable("n", n);
		config.addVariable("k", k);
		config.addVariable("maxiter", maxiter);
		double Eps = Math.pow(10, -8);

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
		runTest(exceptionExpected, null, expectedNumberOfJobs);
		
		runRScript();
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
