package dml.test.integration.functions.binary.scalar;

import org.junit.Test;

import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;


public class LessThanOrEqualTest extends AutomatedTestBase {
	
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/binary/scalar/";
		availableTestConfigurations.put("LessThanOrEqualTest", new TestConfiguration("LessThanOrEqualTest",
				new String[] { "left_1", "left_2", "left_3", "right_1", "right_2", "right_3" }));
	}
	
	@Test
	public void testLessThanOrEqual() {
		loadTestConfiguration("LessThanOrEqualTest");
		
		createHelperMatrix();
		writeExpectedHelperMatrix("left_1", 2);
		writeExpectedHelperMatrix("left_2", 2);
		writeExpectedHelperMatrix("left_3", 1);
		writeExpectedHelperMatrix("right_1", 1);
		writeExpectedHelperMatrix("right_2", 2);
		writeExpectedHelperMatrix("right_3", 2);
		
		runTest();
		
		compareResults();
	}

}
