/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.unary.scalar;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;



/**
 * <p><b>Positive tests:</b></p>
 * <ul>
 * 	<li>sqrt (int, double)</li>
 * 	<li>random int</li>
 * 	<li>random double</li>
 * </ul>
 * <p><b>Negative tests:</b></p>
 * <ul>
 * 	<li>negative int</li>
 * 	<li>negative double</li>
 * 	<li>random int</li>
 * 	<li>random double</li>
 * </ul>
 * 
 * 
 */
public class SqrtTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final String TEST_DIR = "functions/unary/scalar/";
	
	@Override
	public void setUp() {
		
		// positive tests
		addTestConfiguration("PositiveTest",
				new TestConfiguration(TEST_DIR, "SqrtTest", new String[] { "int", "double" }));
		
		// random tests
		addTestConfiguration("RandomIntTest",
				new TestConfiguration(TEST_DIR, "SqrtSingleTest", new String[] { "computed" }));
		addTestConfiguration("RandomDoubleTest",
				new TestConfiguration(TEST_DIR, "SqrtSingleTest", new String[] { "computed" }));
		
		// negative tests
		addTestConfiguration("NegativeIntTest",
				new TestConfiguration(TEST_DIR, "SqrtSingleTest", new String[] { "computed" }));
		addTestConfiguration("NegativeDoubleTest",
				new TestConfiguration(TEST_DIR, "SqrtSingleTest", new String[] { "computed" }));
	}
	
	@Test
	public void testPositive() {
		int intValue = 5;
		double doubleValue = 5.0;
		
		TestConfiguration config = availableTestConfigurations.get("PositiveTest");
		config.addVariable("int", intValue);
		config.addVariable("double", doubleValue);
		
		loadTestConfiguration(config);
		
		double computedIntValue = Math.sqrt(intValue);
		double computedDoubleValue = Math.sqrt(doubleValue);
		
		createHelperMatrix();
		writeExpectedHelperMatrix("int", computedIntValue);
		writeExpectedHelperMatrix("double", computedDoubleValue);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testRandomInt() {
		int intValue = TestUtils.getRandomInt();
		
		TestConfiguration config = availableTestConfigurations.get("RandomIntTest");
		config.addVariable("value", intValue);
		
		loadTestConfiguration(config);
		
		createHelperMatrix();
		
		double computedIntValue = Math.sqrt(intValue);
		writeExpectedHelperMatrix("computed", computedIntValue);
		runTest();
		compareResults();
	}
	
	@Test
	public void testRandomDouble() {
		double doubleValue = TestUtils.getRandomDouble();
		
		TestConfiguration config = availableTestConfigurations.get("RandomDoubleTest");
		config.addVariable("value", doubleValue);
		
		loadTestConfiguration(config);
		
		createHelperMatrix();
		
		double computedDoubleValue = Math.sqrt(doubleValue);
		writeExpectedHelperMatrix("computed", computedDoubleValue);
		runTest();
		compareResults();
	}
	
	@Test
	public void testNegativeInt() {
		int intValue = -5;
		
		TestConfiguration config = availableTestConfigurations.get("NegativeIntTest");
		config.addVariable("value", intValue);
		
		loadTestConfiguration(config);
		
		createHelperMatrix();
		
		runTest(false);
	}
	
	@Test
	public void testNegativeDouble() {
		double doubleValue = -5.0;
		
		TestConfiguration config = availableTestConfigurations.get("NegativeDoubleTest");
		config.addVariable("value", doubleValue);
		
		loadTestConfiguration(config);
		
		createHelperMatrix();
		
		runTest(false);
	}
}
