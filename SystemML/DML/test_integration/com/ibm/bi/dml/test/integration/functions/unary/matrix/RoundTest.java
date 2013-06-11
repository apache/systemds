package com.ibm.bi.dml.test.integration.functions.unary.matrix;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;



public class RoundTest extends AutomatedTestBase {

	private final static String TEST_NAME = "RoundTest";
	private final static String TEST_DIR = "functions/unary/matrix/";

	private final static int rows = 100;
	private final static int cols = 100;    
	private final static double sparsity = 0.75;
	
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/unary/matrix/";
		availableTestConfigurations.put("RoundTest", new TestConfiguration("RoundTest", new String[] { "matrix" }));
	}
	
	@Test
	public void testRound() {
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", HOME + INPUT_DIR + "matrix" , 
						                Integer.toString(rows),
						                Integer.toString(cols),
				                        HOME + OUTPUT_DIR + "matrix" };

		loadTestConfiguration("RoundTest");
		
		long seed = System.nanoTime();
        double[][] matrix = getRandomMatrix(rows, cols, 0, 1, sparsity, seed);
		
		double[][] roundMatrix = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				roundMatrix[i][j] = Math.round(matrix[i][j]);
			}
		}

		writeInputMatrix("matrix", matrix, false);
		writeExpectedMatrix("matrix", roundMatrix);
		
		runTest(true, false, null, -1);
		
		compareResults();
	}
	
	
}
