package dml.test.integration.functions.external;

import org.junit.Test;
import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;

/**
 * 
 * @author Amol Ghoting
 */
public class OutlierTest extends AutomatedTestBase {

	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/external/";
		availableTestConfigurations.put("OutlierTest", new TestConfiguration("OutlierTest", new String[] { "o" }));
	}

	@Test
	public void testOutlierTest() {
		
		int rows = 100;
		int cols = 10;

		TestConfiguration config = availableTestConfigurations.get("OutlierTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);

		double[][] M = getRandomMatrix(rows, cols, -1, 1, 0.05, -1);
			
		writeInputMatrix("M", M);
		
		loadTestConfiguration(config);

		runTest();

		
		checkForResultExistence();
	}
}
