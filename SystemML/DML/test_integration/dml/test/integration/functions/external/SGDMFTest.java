package dml.test.integration.functions.external;

import org.junit.Test;
import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;

/**
 * JUnit test for SGD based MF
 * @author Amol Ghoting
 */

public class SGDMFTest extends AutomatedTestBase {

	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/external/";
		availableTestConfigurations.put("SGDMFTest", new TestConfiguration("SGDMFTest", new String[] { "W", "tH" }));
	}

	@Test
	public void testSGDMFTest() {
		
		int rows = 100;
		int cols = 50;
		
		TestConfiguration config = availableTestConfigurations.get("SGDMFTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);

		double[][] V = getRandomMatrix(rows, cols, 1, 5, 0.05, -1);
		
		writeInputMatrix("V", V);
		
		loadTestConfiguration(config);

		runTest();

		checkForResultExistence();
	}
}
