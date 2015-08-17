/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.unary.matrix;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;



public class ACosTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final String TEST_DIR = "functions/unary/matrix/";
	private static final double EPS = 1e-14;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration("PositiveTest",
				new TestConfiguration(TEST_DIR, "ACosTest", new String[] { "vector", "matrix" }));
		addTestConfiguration("NegativeTest",
				new TestConfiguration(TEST_DIR, "ACosTest", new String[] { "vector", "matrix" }));
		addTestConfiguration("RandomTest",
				new TestConfiguration(TEST_DIR, "ACosTest", new String[] { "vector", "matrix" }));
	}
	
	@Test
	public void testPositive() {
		int rows = 10;
		int cols = 10;
		
		TestConfiguration config = availableTestConfigurations.get("PositiveTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		loadTestConfiguration(config);
		
		double[][] vector = getRandomMatrix(rows, 1, 0, 1, 1, -1);
		double[][] acosVector = new double[rows][1];
		for(int i = 0; i < rows; i++) {
			acosVector[i][0] = Math.acos(vector[i][0]);
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector", acosVector);
		
		double[][] matrix = getRandomMatrix(rows, cols, 0, 1, 1, -1);
		double[][] acosMatrix = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				acosMatrix[i][j] = Math.acos(matrix[i][j]);
			}
		}
		writeInputMatrix("matrix", matrix);
		writeExpectedMatrix("matrix", acosMatrix);
		
		runTest();
		
		compareResults(EPS);
	}
	
	@Test
	public void testNegative() {
		int rows = 10;
		int cols = 10;
		
		TestConfiguration config = availableTestConfigurations.get("NegativeTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		loadTestConfiguration(config);
		
		double[][] vector = getRandomMatrix(rows, 1, -1, 0, 1, -1);
		double[][] acosVector = new double[rows][1];
		for(int i = 0; i < rows; i++) {
			acosVector[i][0] = Math.acos(vector[i][0]);
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector", acosVector);
		
		double[][] matrix = getRandomMatrix(rows, cols, -1, 0, 1, -1);
		double[][] acosMatrix = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				acosMatrix[i][j] = Math.acos(matrix[i][j]);
			}
		}
		writeInputMatrix("matrix", matrix);
		writeExpectedMatrix("matrix", acosMatrix);
		
		runTest();
		
		compareResults(EPS);
	}
	
	@Test
	public void testRandom() {
		int rows = 10;
		int cols = 10;
		
		TestConfiguration config = availableTestConfigurations.get("RandomTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		loadTestConfiguration(config);
		
		double[][] vector = getRandomMatrix(rows, 1, -1, 1, 1, -1);
		double[][] acosVector = new double[rows][1];
		for(int i = 0; i < rows; i++) {
			acosVector[i][0] = Math.acos(vector[i][0]);
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector", acosVector);
		
		double[][] matrix = getRandomMatrix(rows, cols, -1, 1, 1, -1);
		double[][] acosMatrix = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				acosMatrix[i][j] = Math.acos(matrix[i][j]);
			}
		}
		writeInputMatrix("matrix", matrix);
		writeExpectedMatrix("matrix", acosMatrix);
		
		runTest();
		
		compareResults(EPS);
	}
	
}
