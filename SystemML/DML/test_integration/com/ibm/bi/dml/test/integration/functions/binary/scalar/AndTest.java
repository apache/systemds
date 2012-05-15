package com.ibm.bi.dml.test.integration.functions.binary.scalar;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;



public class AndTest extends AutomatedTestBase {

	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/binary/scalar/";
		availableTestConfigurations.put("AndTest", new TestConfiguration("AndTest",
				new String[] { "left_1", "left_2", "left_3", "left_4", "right_1", "right_2", "right_3", "right_4" }));
	}
	
	@Test
	public void testAnd() {
		loadTestConfiguration("AndTest");
		
		createHelperMatrix();
		writeExpectedHelperMatrix("left_1", 2);
		writeExpectedHelperMatrix("left_2", 1);
		writeExpectedHelperMatrix("left_3", 1);
		writeExpectedHelperMatrix("left_4", 1);
		writeExpectedHelperMatrix("right_1", 2);
		writeExpectedHelperMatrix("right_2", 1);
		writeExpectedHelperMatrix("right_3", 1);
		writeExpectedHelperMatrix("right_4", 1);
		
		runTest();
		
		compareResults();
	}

}
