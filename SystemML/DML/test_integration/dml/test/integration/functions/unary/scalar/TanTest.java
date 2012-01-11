package dml.test.integration.functions.unary.scalar;

import org.junit.Test;

import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;
import dml.test.utils.TestUtils;


/**
 * <p><b>Positive tests:</b></p>
 * <ul>
 * 	<li>positive scalar (int, double)</li>
 * 	<li>negative scalar (int, double)</li>
 * 	<li>random scalar (int, double)</li>
 * </ul>
 * <p><b>Negative tests:</b></p>
 * 
 * @author schnetter
 */
public class TanTest extends AutomatedTestBase {
	
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/unary/scalar/";
		
		// positive tests
		availableTestConfigurations.put("PositiveTest",
				new TestConfiguration("TanTest", new String[] { "int", "double" }));
		availableTestConfigurations.put("NegativeTest",
				new TestConfiguration("TanTest", new String[] { "int", "double" }));
		availableTestConfigurations.put("RandomTest",
				new TestConfiguration("TanTest", new String[] { "int", "double" }));
		
		// negative tests
	}
	
	@Test
	public void testPositive() {
		int intValue = 5;
		double doubleValue = 5.0;
		
		TestConfiguration config = availableTestConfigurations.get("PositiveTest");
		config.addVariable("int", intValue);
		config.addVariable("double", doubleValue);
		
		loadTestConfiguration("PositiveTest");
		
		double computedIntValue = Math.tan(intValue);
		double computedDoubleValue = Math.tan(doubleValue);
		
		createHelperMatrix();
		writeExpectedHelperMatrix("int", computedIntValue);
		writeExpectedHelperMatrix("double", computedDoubleValue);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testNegative() {
		int intValue = -5;
		double doubleValue = -5.0;
		
		TestConfiguration config = availableTestConfigurations.get("NegativeTest");
		config.addVariable("int", intValue);
		config.addVariable("double", doubleValue);
		
		loadTestConfiguration("NegativeTest");
		
		double computedIntValue = Math.tan(intValue);
		double computedDoubleValue = Math.tan(doubleValue);
		
		createHelperMatrix();
		writeExpectedHelperMatrix("int", computedIntValue);
		writeExpectedHelperMatrix("double", computedDoubleValue);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testRandom() {
		int intValue = TestUtils.getRandomInt();
		double doubleValue = TestUtils.getRandomDouble();
		
		TestConfiguration config = availableTestConfigurations.get("RandomTest");
		config.addVariable("int", intValue);
		config.addVariable("double", doubleValue);
		
		loadTestConfiguration("RandomTest");
		
		double computedIntValue = Math.tan(intValue);
		double computedDoubleValue = Math.tan(doubleValue);
		
		createHelperMatrix();
		writeExpectedHelperMatrix("int", computedIntValue);
		writeExpectedHelperMatrix("double", computedDoubleValue);
		
		runTest();
		
		compareResults();
	}

}
