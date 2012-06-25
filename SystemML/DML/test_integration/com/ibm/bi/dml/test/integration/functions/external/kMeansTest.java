package com.ibm.bi.dml.test.integration.functions.external;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class kMeansTest extends AutomatedTestBase {
	
	private final static String TEST_KMEANS = "kMeans"; // The version w/o init kCenters
	
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/external/";
		availableTestConfigurations.put(TEST_KMEANS, new TestConfiguration(TEST_KMEANS, new String[] { "kCenters"}));
	}

	@Test
	public void testkMeansTest() {
		
		int rows = 100;
		int cols = 10;
		int centers = 5;

		TestConfiguration config = availableTestConfigurations.get(TEST_KMEANS);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String KMEANS_HOME = baseDirectory;
		dmlArgs = new String[]{"-f", KMEANS_HOME + TEST_KMEANS + ".dml",
				               "-args",  KMEANS_HOME + INPUT_DIR + "M" , 
				                        Integer.toString(rows), Integer.toString(cols), 
				                         KMEANS_HOME + OUTPUT_DIR + "kcenters" };
		dmlArgsDebug = new String[]{"-f", KMEANS_HOME + TEST_KMEANS + ".dml", "-d", 
	                                "-args",  KMEANS_HOME + INPUT_DIR + "M" , 
	                                         Integer.toString(rows), Integer.toString(cols), 
	                                          KMEANS_HOME + OUTPUT_DIR + "kcenters" };
		
		double[][] M = getRandomMatrix(rows, cols, -1, 1, 0.05, 10);
		
		writeInputMatrix("M", M);
		
		HashMap<CellIndex, Double> expected_kmeans = TestUtils.readDMLMatrixFromHDFS(baseDirectory + "kMeans/kMeansWrapperOutput");
		
		
		double [][] expected_means_arr = TestUtils.convertHashMapToDoubleArray(expected_kmeans);
		
		writeExpectedMatrix("kCenters", expected_means_arr);
		
				
		loadTestConfiguration(config);

		runTest(true, false, null, -1);

		compareResults(0.0001);
	}
}
