/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.unary.scalar;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;


/**
 * Tests the print function
 */
public class PrintTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/unary/scalar/";

		availableTestConfigurations.put("PrintTest", new TestConfiguration("PrintTest", new String[] {}));
		availableTestConfigurations.put("PrintTest2", new TestConfiguration("PrintTest2", new String[] {}));
		availableTestConfigurations.put("PrintTest3", new TestConfiguration("PrintTest3", new String[] {}));
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

	@Test
	public void testPrint() {
		TestConfiguration config = availableTestConfigurations.get("PrintTest3");

		loadTestConfiguration(config);

		String value = "fooboo, 0.0";
		setExpectedStdOut(value.substring(1, value.length()-1));
		runTest();
	}
}
