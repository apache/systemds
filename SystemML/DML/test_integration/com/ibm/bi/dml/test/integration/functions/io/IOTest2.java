package com.ibm.bi.dml.test.integration.functions.io;

import org.junit.Test;

import com.ibm.bi.dml.parser.ParseException;
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
 * </ul>
 * 
 * 
 */
public class IOTest2 extends AutomatedTestBase {

	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/io/";

		// positive tests
		
		

		// negative tests
		availableTestConfigurations.put("SimpleTest", new TestConfiguration("functions/io/", "IOTest2", new String[] { "a" }));
	}

	@Test
	public void testSimple() {
		int rows = 10;
		int cols = 10;

		TestConfiguration config = availableTestConfigurations.get("SimpleTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "text");
	
		loadTestConfiguration(config);

		double[][] a = getRandomMatrix(rows, cols, -1, 1, 0.5, -1);
		writeInputMatrix("a", a);
		writeExpectedMatrix("a", a);

		runTest(true, ParseException.class);

	}

}
