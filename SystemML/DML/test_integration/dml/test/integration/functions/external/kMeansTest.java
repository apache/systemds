package dml.test.integration.functions.external;

import org.junit.Test;

import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;

/**
 * 
 * @author Amol Ghoting
 */
public class kMeansTest extends AutomatedTestBase {
	
	private final static String TEST_KMEANS = "kMeans";
	
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/external/";
		availableTestConfigurations.put(TEST_KMEANS, new TestConfiguration(TEST_KMEANS, new String[] { "kCenters", "kCentersWithInit" }));
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
				               "-args", "\"" + KMEANS_HOME + INPUT_DIR + "M" + "\"", 
				                        Integer.toString(rows), Integer.toString(cols), 
				                        "\"" + KMEANS_HOME + INPUT_DIR + "initCenters" + "\"", 
				                        "\"" + KMEANS_HOME + OUTPUT_DIR + "kcenters" + "\"",
				                        "\"" + KMEANS_HOME + OUTPUT_DIR + "kcentersWithInit" + "\""};
		dmlArgsDebug = new String[]{"-f", KMEANS_HOME + TEST_KMEANS + ".dml", "-d", 
	                                "-args", "\"" + KMEANS_HOME + INPUT_DIR + "M" + "\"", 
	                                         Integer.toString(rows), Integer.toString(cols), 
	                                         "\"" + KMEANS_HOME + INPUT_DIR + "initCenters" + "\"", 
	                                         "\"" + KMEANS_HOME + OUTPUT_DIR + "kcenters" + "\"",
	                                         "\"" + KMEANS_HOME + OUTPUT_DIR + "kcentersWithInit" + "\""};
		
		double[][] M = getRandomMatrix(rows, cols, -1, 1, 0.05, -1);
		double[][] initCenters = getRandomMatrix(centers, cols, -1, 1, 0.05, -1);
		
		writeInputMatrix("M", M);
		writeInputMatrix("initCenters",initCenters);
		
		loadTestConfiguration(config);

		runTest(true, false, null, -1);

		checkForResultExistence();
	}
}
