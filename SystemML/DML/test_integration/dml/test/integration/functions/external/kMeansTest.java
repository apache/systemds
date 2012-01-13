package dml.test.integration.functions.external;

import org.junit.Test;
import org.netlib.lapack.LAPACK;

import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;
import dml.test.utils.TestUtils;

/**
 * 
 * @author Amol Ghoting
 */
public class kMeansTest extends AutomatedTestBase {

	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/external/";
		availableTestConfigurations.put("kMeansTest", new TestConfiguration("kMeansTest", new String[] { "kCenters", "kCentersWithInit" }));
	}

	@Test
	public void testkMeansTest() {
		
		int rows = 100;
		int cols = 10;
		int centers = 5;

		TestConfiguration config = availableTestConfigurations.get("kMeansTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);

		double[][] M = getRandomMatrix(rows, cols, -1, 1, 0.05, -1);
		double[][] initCenters = getRandomMatrix(centers, cols, -1, 1, 0.05, -1);
		
		
		writeInputMatrix("M", M);
		writeInputMatrix("initCenters",initCenters);
		
		loadTestConfiguration(config);

		runTest();

		
		checkForResultExistence();
	}
}
