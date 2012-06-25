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
public class SqrtTest extends AutomatedTestBase {

	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/unary/scalar/";
		
		// positive tests
		availableTestConfigurations.put("PositiveTest",
				new TestConfiguration("SqrtTest", new String[] { "int", "double" }));
		
		// random tests
		availableTestConfigurations.put("RandomIntTest",
				new TestConfiguration("SqrtSingleTest", new String[] { "computed" }));
		availableTestConfigurations.put("RandomDoubleTest",
				new TestConfiguration("SqrtSingleTest", new String[] { "computed" }));
		
		// negative tests
		availableTestConfigurations.put("NegativeIntTest",
				new TestConfiguration("SqrtSingleTest", new String[] { "computed" }));
		availableTestConfigurations.put("NegativeDoubleTest",
				new TestConfiguration("SqrtSingleTest", new String[] { "computed" }));
	}
	
	@Test
	public void testPositive() {
		int intValue = 5;
		double doubleValue = 5.0;
		
		TestConfiguration config = availableTestConfigurations.get("PositiveTest");
		config.addVariable("int", intValue);
		config.addVariable("double", doubleValue);
		
		loadTestConfiguration("PositiveTest");
		
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
		
		loadTestConfiguration("RandomIntTest");
		
		createHelperMatrix();
		
		if(intValue < 0) {
			runTest(true);
		} else {
			double computedIntValue = Math.sqrt(intValue);
			
			writeExpectedHelperMatrix("computed", computedIntValue);
			
			runTest();
			
			compareResults();
		}
	}
	
	@Test
	public void testRandomDouble() {
		double doubleValue = TestUtils.getRandomDouble();
		
		TestConfiguration config = availableTestConfigurations.get("RandomDoubleTest");
		config.addVariable("value", doubleValue);
		
		loadTestConfiguration("RandomDoubleTest");
		
		createHelperMatrix();
		
		if(doubleValue < 0) {
			runTest(true);
		} else {
			double computedDoubleValue = Math.sqrt(doubleValue);
			
			writeExpectedHelperMatrix("computed", computedDoubleValue);
			
			runTest();
			
			compareResults();
		}
	}
	
	@Test
	public void testNegativeInt() {
		int intValue = -5;
		
		TestConfiguration config = availableTestConfigurations.get("NegativeIntTest");
		config.addVariable("value", intValue);
		
		loadTestConfiguration("NegativeIntTest");
		
		createHelperMatrix();
		
		runTest(true);
	}
	
	@Test
	public void testNegativeDouble() {
		double doubleValue = -5.0;
		
		TestConfiguration config = availableTestConfigurations.get("NegativeDoubleTest");
		config.addVariable("value", doubleValue);
		
		loadTestConfiguration("NegativeDoubleTest");
		
		createHelperMatrix();
		
		runTest(true);
	}
	
}
