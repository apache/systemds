package dml.test.integration.functions.external;

import org.junit.Test;
import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;

/**
 * 
 * @author Felix Hamborg, Amol Ghoting
 */
public class EigenTest extends AutomatedTestBase {

	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/external/";
		availableTestConfigurations.put("EigenTest", new TestConfiguration("EigenTest", new String[] { "val", "vec" }));
	}

	@Test
	public void testEigen() {
		
		int rows = 3;
		int cols = rows;

		TestConfiguration config = availableTestConfigurations.get("EigenTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);

		double[][] A = {{0, 1, -1},{1, 1, 0},{-1, 0, 1}};
		
		
		writeInputMatrix("A", A);
		
		loadTestConfiguration(config);

		runTest();
		
		double [][] val = {{-1, 0, 0}, {0, 1, 0}, {0, 0, 2}};
		double [][] vec = {
				{0.81649, 0, -0.57735},
				{-0.4082, 0.7071, -0.57735},
				{0.4082, 0.7071, 0.57735}
				};

		writeExpectedMatrix("val", val);
		writeExpectedMatrix("vec", vec);
		compareResults(0.001);
		
	}
}
