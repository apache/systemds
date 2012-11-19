package com.ibm.bi.dml.test.integration.functions.unary.matrix;

import org.junit.Test;

import com.ibm.bi.dml.parser.ParseException;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;



/**
 * <p><b>Positive tests:</b></p>
 * <ul>
 * 	<li>matrix to vector & matrix 2 vector</li>
 * </ul>
 * <p><b>Negative tests:</b></p>
 * <ul>
 * 	<li>wrong dimensions</li>
 * </ul>
 * 
 * 
 */
public class DiagTest extends AutomatedTestBase {

	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/unary/matrix/";
		
		// positive tests
		availableTestConfigurations.put("DiagTest", new TestConfiguration("DiagTest", new String[] { "b", "d" }));
		
		// negative tests
		availableTestConfigurations.put("WrongDimensionsTest", new TestConfiguration("DiagSingleTest",
				new String[] { "b" }));
	}
	
	@Test
	public void testDiag() {
		int rowsCols = 10;
		
		TestConfiguration config = availableTestConfigurations.get("DiagTest");
		config.addVariable("rows", rowsCols);
		config.addVariable("cols", rowsCols);
		
		loadTestConfiguration("DiagTest");
		
		double[][] a = getRandomMatrix(rowsCols, rowsCols, -1, 1, 0.5, -1);
		writeInputMatrix("a", a);
		
		double[][] b = new double[rowsCols][1];
		for(int i = 0; i < rowsCols; i++) {
			b[i][0] = a[i][i];
		}
		writeExpectedMatrix("b", b);
		
		double[][] c = getRandomMatrix(rowsCols, 1, -1, 1, 0.5, -1);
		writeInputMatrix("c", c);
		
		double[][] d = new double[rowsCols][rowsCols];
		for(int i = 0; i < rowsCols; i++) {
			d[i][i] = c[i][0];
		}
		writeExpectedMatrix("d", d);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testWrongDimensions() {
		int rows = 10;
		int cols = 9;
		
		TestConfiguration config = availableTestConfigurations.get("WrongDimensionsTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		loadTestConfiguration("WrongDimensionsTest");
		
		createRandomMatrix("a", rows, cols, -1, 1, 0.5, -1);
		
		runTest(true, ParseException.class);
	}

}
