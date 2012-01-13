package dml.test.integration.applications;

import java.util.HashMap;

import org.junit.Test;

import dml.runtime.matrix.io.MatrixValue.CellIndex;
import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;
import dml.test.utils.TestUtils;

public class HITSTest extends AutomatedTestBase {

	private final static String TEST_DIR = "applications/hits/";
	private final static String TEST_HITS = "HITSTest";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_HITS, new TestConfiguration(TEST_DIR, "HITSTest", new String[] { "hubs", "authorities" }));
	}

	@Test
	public void testHITSWithRDMLAndJava() {
		int rows = 10000;
		int cols = 10000;
		int maxiter = 2;
		double tol = 1e-6;
		
		TestConfiguration config = getTestConfiguration(TEST_HITS);
		config.addVariable("tol", tol);
		config.addVariable("maxiter", maxiter);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		double Eps = Math.pow(10, -8);

		loadTestConfiguration(config);

		
		boolean exceptionExpected = false;
		/*
		 * Expected number of jobs:
		 * Reblock - 1 job 
		 * While loop iteration - 9 jobs (Optimal = 8)
		 * Final output write - 1 job
		 */
		int expectedNumberOfJobs = 11;
		runTest(exceptionExpected, null, expectedNumberOfJobs);
		
		runRScript();
		disableOutAndExpectedDeletion();

		HashMap<CellIndex, Double> hubsDML = readDMLMatrixFromHDFS("hubs");
		HashMap<CellIndex, Double> authDML = readDMLMatrixFromHDFS("authorities");
		HashMap<CellIndex, Double> hubsR = readRMatrixFromFS("hubs");
		HashMap<CellIndex, Double> authR = readRMatrixFromFS("authorities");

		TestUtils.compareMatrices(hubsDML, hubsR, 0.001, "hubsDML", "hubsR");
		TestUtils.compareMatrices(authDML, authR, 0.001, "authDML", "authR");
		
	}
}
