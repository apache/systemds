package com.ibm.bi.dml.test.integration.functions.unary.scalar;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;


/**
 * Tests the print function
 * 
 * @author Felix Hamborg
 */
public class PrintTest extends AutomatedTestBase {

	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/unary/scalar/";

		availableTestConfigurations.put("PrintTest", new TestConfiguration("PrintTest", new String[] {}));
		availableTestConfigurations.put("PrintTest2", new TestConfiguration("PrintTest2", new String[] {}));
	}

	@Test
	public void testInt() {
		int value = 0;

		TestConfiguration config = availableTestConfigurations.get("PrintTest");
		config.addVariable("value", value);

		loadTestConfiguration(config);

		setExpectedStdOut("X= " + value);
		runTest();
	}
	
	@Test
	public void testDouble() {
		double value = 1337.3;

		TestConfiguration config = availableTestConfigurations.get("PrintTest");
		config.addVariable("value", value);

		loadTestConfiguration(config);
		
		setExpectedStdOut("X= " + value);
		runTest();
	}
	
	@Test
	public void testBoolean() {
		boolean value = true;

		TestConfiguration config = availableTestConfigurations.get("PrintTest");
		config.addVariable("value", value);

		loadTestConfiguration(config);

		setExpectedStdOut("X= " + value);
		runTest();
	}
	
	@Test
	public void testString() {
		String value = "\"Hello World!\"";

		TestConfiguration config = availableTestConfigurations.get("PrintTest");
		config.addVariable("value", value);

		loadTestConfiguration(config);

		setExpectedStdOut("X= " + value.substring(1, value.length()-1));
		runTest();
	}
	
	@Test
	public void testStringWithoutMsg() {
		String value = "\"Hello World!\"";

		TestConfiguration config = availableTestConfigurations.get("PrintTest2");
		config.addVariable("value", value);

		loadTestConfiguration(config);

		setExpectedStdOut(value.substring(1, value.length()-1));
		runTest();
	}
}
