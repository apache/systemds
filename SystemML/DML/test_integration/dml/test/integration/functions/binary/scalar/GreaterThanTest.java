package dml.test.integration.functions.binary.scalar;

import org.junit.Test;

import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;


public class GreaterThanTest extends AutomatedTestBase {
	
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/binary/scalar/";
		availableTestConfigurations.put("GreaterThanTest", new TestConfiguration("GreaterThanTest",
				new String[] { "left_1", "left_2", "left_3", "right_1", "right_2", "right_3" }));
	}
	
	@Test
	public void testGreater() {
		loadTestConfiguration("GreaterThanTest");
		
		createHelperMatrix();
		writeExpectedHelperMatrix("left_1", 1);
		writeExpectedHelperMatrix("left_2", 1);
		writeExpectedHelperMatrix("left_3", 2);
		writeExpectedHelperMatrix("right_1", 2);
		writeExpectedHelperMatrix("right_2", 1);
		writeExpectedHelperMatrix("right_3", 1);
		
		runTest();
		
		compareResults();
	}

}
