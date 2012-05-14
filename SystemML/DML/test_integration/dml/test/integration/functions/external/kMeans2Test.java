package dml.test.integration.functions.external;

import java.util.HashMap;

import org.junit.Test;

import dml.runtime.matrix.io.MatrixValue.CellIndex;
import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;
import dml.test.utils.TestUtils;

/**
 * 
 * @author Amol Ghoting
 */
public class kMeans2Test extends AutomatedTestBase {

	private final static String TEST_KMEANS = "kMeans2";  // The version w/ init kCenters

	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/external/";
		availableTestConfigurations.put(TEST_KMEANS, new TestConfiguration(TEST_KMEANS, new String[] { "kcenters", "kcentersWithInit"}));
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
				 KMEANS_HOME + INPUT_DIR + "initialCenters" , 
				 KMEANS_HOME + OUTPUT_DIR + "kcenters" ,
				 KMEANS_HOME + OUTPUT_DIR + "kcentersWithInit" };
		dmlArgsDebug = new String[]{"-f", KMEANS_HOME + TEST_KMEANS + ".dml", "-d", 
				"-args",  KMEANS_HOME + INPUT_DIR + "M" , 
				Integer.toString(rows), Integer.toString(cols), 
				 KMEANS_HOME + INPUT_DIR + "initialCenters" , 
				 KMEANS_HOME + OUTPUT_DIR + "kcenters" ,
				 KMEANS_HOME + OUTPUT_DIR + "kcentersWithInit" };

		double[][] M = getRandomMatrix(rows, cols, -1, 1, 0.05, 10);
		double[][] initCenters = getRandomMatrix(centers, cols, -1, 1, 0.9, 20);

		writeInputMatrix("M", M);
		writeInputMatrix("initialCenters",initCenters);


		HashMap<CellIndex, Double> expected_kCenters = TestUtils.readDMLMatrixFromHDFS(baseDirectory + "kMeans2/kMeansWrapperOutput1");
		HashMap<CellIndex, Double> expected_kCentersWithInit = TestUtils.readDMLMatrixFromHDFS(baseDirectory + "kMeans2/kMeansWrapperOutput2");



		double [][] kcenters_arr = TestUtils.convertHashMapToDoubleArray(expected_kCenters);
		double [][] kcenters_init_arr = TestUtils.convertHashMapToDoubleArray(expected_kCentersWithInit);

		writeExpectedMatrix("kcenters", kcenters_arr);
		writeExpectedMatrix("kcentersWithInit", kcenters_init_arr);


		loadTestConfiguration(config);

		runTest(true, false, null, -1);
		

		compareResults(0.0001);
	}
}
