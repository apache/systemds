package dml.test.integration.functions.binary.scalar;

import org.junit.Test;

import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;


public class OrTest extends AutomatedTestBase {
	
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/binary/scalar/";
		availableTestConfigurations.put("OrTest", new TestConfiguration("OrTest",
				new String[] { "left_1", "left_2", "left_3", "left_4", "right_1", "right_2", "right_3", "right_4" }));
	}
	
	@Test
	public void testOr() {
		loadTestConfiguration("OrTest");
		
		createHelperMatrix();
		writeExpectedHelperMatrix("left_1", 2);
		writeExpectedHelperMatrix("left_2", 2);
		writeExpectedHelperMatrix("left_3", 2);
		writeExpectedHelperMatrix("left_4", 1);
		writeExpectedHelperMatrix("right_1", 2);
		writeExpectedHelperMatrix("right_2", 2);
		writeExpectedHelperMatrix("right_3", 2);
		writeExpectedHelperMatrix("right_4", 1);
		
		runTest();
		
		compareResults();
	}

}
