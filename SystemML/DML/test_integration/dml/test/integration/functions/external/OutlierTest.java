package dml.test.integration.functions.external;

import org.junit.Test;
import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;

/**
 * 
 * @author Amol Ghoting
 */
public class OutlierTest extends AutomatedTestBase {

	private final static String TEST_OUTLIER = "Outlier"; 
	
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/external/";
		availableTestConfigurations.put("Outlier", new TestConfiguration(baseDirectory, "Outlier", new String[] { "o" }));
	}

	@Test
	public void testOutlierTest() {
		
		int rows = 100;
		int cols = 10;

		TestConfiguration config = availableTestConfigurations.get("Outlier");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);

		/* This is for running the junit test by constructing the arguments directly */
		String OUTLIER_HOME = baseDirectory;
		dmlArgs = new String[]{"-f", OUTLIER_HOME + TEST_OUTLIER + ".dml",
				               "-args", "\"" + OUTLIER_HOME + INPUT_DIR + "M" + "\"", 
				                        Integer.toString(rows), Integer.toString(cols), 
				                        "\"" + OUTLIER_HOME + OUTPUT_DIR + "o" + "\""};
		dmlArgsDebug = new String[]{"-f", OUTLIER_HOME + TEST_OUTLIER + ".dml", "-d", 
	                                "-args", "\"" + OUTLIER_HOME + INPUT_DIR + "M" + "\"", 
	                                         Integer.toString(rows), Integer.toString(cols), 
	                                         "\"" + OUTLIER_HOME + OUTPUT_DIR + "o" + "\""};
		
		double[][] M = getRandomMatrix(rows, cols, -1, 1, 0.05, -1);
			
		writeInputMatrix("M", M);
		
		loadTestConfiguration(config);

		// there is no expected number of M/R job calculated, set to default for now
		runTest(true, false, null, -1);

		
		checkForResultExistence();
	}
}
