package dml.test.integration.functions.unary.scalar;

import org.junit.Test;

import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;


/**
 * <p><b>Positive tests:</b></p>
 * <ul>
 * 	<li>negation (int, negative int, double, negative double)</li>
 * </ul>
 * <p><b>Negative tests:</b></p>
 * 
 * @author schnetter
 */
public class NegationTest extends AutomatedTestBase {

	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/unary/scalar/";
		
		// positive tests
		availableTestConfigurations.put("NegationTest", new TestConfiguration("NegationTest",
				new String[] { }));
		
		// negative tests
	}
	
	@Test
	public void testNegation() {
		int intValue = 2;
		int negativeIntValue = -2;
		double doubleValue = 2;
		double negativeDoubleValue = -2;
		
		TestConfiguration config = availableTestConfigurations.get("NegationTest");
		config.addVariable("int", intValue);
		config.addVariable("negativeint", negativeIntValue);
		config.addVariable("double", doubleValue);
		config.addVariable("negativedouble", negativeDoubleValue);
		
		loadTestConfiguration("NegationTest");
		
		createHelperMatrix();
		writeExpectedHelperMatrix("int", -intValue);
		writeExpectedHelperMatrix("negative_int", -negativeIntValue);
		writeExpectedHelperMatrix("double", -doubleValue);
		writeExpectedHelperMatrix("negative_double", -negativeDoubleValue);
		
		runTest();
		
		compareResults();
	}
	
}
