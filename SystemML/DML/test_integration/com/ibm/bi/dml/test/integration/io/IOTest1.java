package com.ibm.bi.dml.test.integration.io;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;


/**
 * <p>
 * <b>Positive tests:</b>
 * </p>
 * <ul>
 * <li>text format</li>
 * <li>binary format</li>
 * </ul>
 * <p>
 * <b>Negative tests:</b>
 * </p>
 * <ul>
 * <li>wrong row dimension (format=text)</li>
 * <li>wrong column dimension (format=text)</li>
 * <li>wrong row and column dimensions (format=text)</li>
 * <li>wrong format (format=text)</li>
 * <li>wrong row dimension (format=binary)</li>
 * <li>wrong column dimension (format=binary)</li>
 * <li>wrong row and column dimensions (format=binary)</li>
 * <li>wrong format (format=binary)</li>
 * </ul>
 * 
 * 
 */
public class IOTest1 extends AutomatedTestBase {

	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/io/";

		// positive tests
		availableTestConfigurations.put("SimpleTest", new TestConfiguration("IOTest1", new String[] { "a" }));
		

		// negative tests
		
	}

	@Test
	public void testSimple() {
		int rows = 10;
		int cols = 10;

		TestConfiguration config = availableTestConfigurations.get("SimpleTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "text");
	
		loadTestConfiguration("SimpleTest");

		double[][] a = getRandomMatrix(rows, cols, -1, 1, 0.5, -1);
		writeInputMatrix("a", a);
		writeExpectedMatrix("a", a);

		runTest();

		compareResults();
	}

}
