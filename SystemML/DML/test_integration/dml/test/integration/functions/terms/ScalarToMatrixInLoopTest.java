package dml.test.integration.functions.terms;

import org.junit.Test;

import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;
import dml.test.utils.TestUtils;

public class ScalarToMatrixInLoopTest extends AutomatedTestBase {
	@SuppressWarnings("deprecation")
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/terms/";

		availableTestConfigurations.put("ScalarToMatrixInLoop", new TestConfiguration("TestScalarToMatrixInLoop", new String[] {}));
	}

	@Test
	public void testScalarToMatrixInLoop() {
		int rows = 5, cols = 5;

		TestConfiguration config = getTestConfiguration("ScalarToMatrixInLoop");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);

		loadTestConfiguration(config);

		runTest();
	}
}
