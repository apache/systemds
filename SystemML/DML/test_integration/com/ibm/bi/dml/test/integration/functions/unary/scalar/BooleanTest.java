package com.ibm.bi.dml.test.integration.functions.unary.scalar;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;



public class BooleanTest extends AutomatedTestBase {

	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/unary/scalar/";
		
		// positive tests
		availableTestConfigurations.put("WhileTest", new TestConfiguration("BooleanWhileTest",
				new String[] { "true", "false" }));
		
		// negative tests
	}
	
	@Test
	public void testWhile() {
		loadTestConfiguration("WhileTest");
		
		createHelperMatrix();
		
		writeExpectedHelperMatrix("true", 2);
		writeExpectedHelperMatrix("false", 1);
		
		runTest();
		
		compareResults();
	}

}
