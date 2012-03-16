package dml.test.integration.applications;

import java.util.HashMap;

import org.junit.Test;

import dml.runtime.matrix.io.MatrixValue.CellIndex;
import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;
import dml.test.utils.TestUtils;

public class HITSTest extends AutomatedTestBase {

	private final static String TEST_DIR = "applications/hits/";
	private final static String TEST_HITS = "HITS";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_HITS, new TestConfiguration(TEST_DIR, "HITS", new String[] { "hubs", "authorities" }));
	}

	@Test
	public void testHITSWithRDMLAndJava() {
		int rows = 1000;
		int cols = 1000;
		int maxiter = 2;

		
		TestConfiguration config = getTestConfiguration(TEST_HITS);
		
		/* This is for running the junit test the old way */
		config.addVariable("maxiter", maxiter);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		

		/* This is for running the junit test by constructing the arguments directly */
		String HITS_HOME = SCRIPT_DIR + TEST_DIR;
		dmlArgs = new String[]{"-f", HITS_HOME + TEST_HITS + ".dml",
				               "-args", "\"" + HITS_HOME + INPUT_DIR + "G" + "\"",
				                        Integer.toString(maxiter), Integer.toString(rows), Integer.toString(cols),
				                        Double.toString(Math.pow(10, -6)),
				                        "\"" + HITS_HOME + OUTPUT_DIR + "hubs" + "\"", 
				                        "\"" + HITS_HOME + OUTPUT_DIR + "authorities" + "\""};
		dmlArgsDebug = new String[]{"-f", HITS_HOME + TEST_HITS + ".dml", "-d",
							   "-args", "\"" + HITS_HOME + INPUT_DIR + "G" + "\"", 
                                        Integer.toString(maxiter), Integer.toString(rows), Integer.toString(cols),
                                        Double.toString(Math.pow(10, -6)),
                                        "\"" + HITS_HOME + OUTPUT_DIR + "hubs" + "\"", 
                                        "\"" + HITS_HOME + OUTPUT_DIR + "authorities" + "\""};
		rCmd = "Rscript" + " " + HITS_HOME + TEST_HITS + ".R" + " " + 
		       HITS_HOME + INPUT_DIR + " " + Integer.toString(maxiter) + " " + Double.toString(Math.pow(10, -6))+ " " + HITS_HOME + EXPECTED_DIR;
		
		loadTestConfiguration(config);

		double[][] G = getRandomMatrix(rows, cols, 0, 1, 1.0, -1);
		writeInputMatrix("G", G, true);
		
		boolean exceptionExpected = false;
		/*
		 * Expected number of jobs:
		 * Reblock - 1 job 
		 * While loop iteration - 9 jobs (Optimal = 8)
		 * Final output write - 1 job
		 */
		int expectedNumberOfJobs = 11;
		runTest(true, exceptionExpected, null, expectedNumberOfJobs);
		
		runRScript(true);
		disableOutAndExpectedDeletion();

		HashMap<CellIndex, Double> hubsDML = readDMLMatrixFromHDFS("hubs");
		HashMap<CellIndex, Double> authDML = readDMLMatrixFromHDFS("authorities");
		HashMap<CellIndex, Double> hubsR = readRMatrixFromFS("hubs");
		HashMap<CellIndex, Double> authR = readRMatrixFromFS("authorities");

		TestUtils.compareMatrices(hubsDML, hubsR, 0.001, "hubsDML", "hubsR");
		TestUtils.compareMatrices(authDML, authR, 0.001, "authDML", "authR");
		
	}
}
