package dml.test.integration.functions.data;

import org.junit.Test;

import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;

/**
 * <p>
 * <b>Positive tests:</b>
 * </p>
 * <ul>
 * <li>random matrix generation (rows, cols, min, max)</li>
 * <li>random scalar generation (min, max)</li>
 * </ul>
 * <p>
 * <b>Negative tests:</b>
 * </p>
 * 
 * @author Hamborg
 */
public class RandTest extends AutomatedTestBase {

	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/data/";

		// positive tests
		availableTestConfigurations.put("MatrixTest", new TestConfiguration("RandTest", new String[] { "rand" }));
		availableTestConfigurations.put("ScalarTest", new TestConfiguration("RandScalarTest", new String[] { "rand" }));

		// negative tests
	}

	@Test
	public void testMatrix() {
		int rows = 10;
		int cols = 10;
		double min = -1;
		double max = 1;

		TestConfiguration config = availableTestConfigurations.get("MatrixTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("min", min);
		config.addVariable("max", max);

		loadTestConfiguration("MatrixTest");

		runTest();

		checkResults(rows, cols, min, max);
	}

	@Test
	public void testScalar() {
		int min = -1;
		int max = 1;

		TestConfiguration config = availableTestConfigurations.get("ScalarTest");
		config.addVariable("min", min);
		config.addVariable("max", max);

		loadTestConfiguration("ScalarTest");

		createHelperMatrix();

		runTest();

		checkResults(1, 1, min, max);
	}

}
