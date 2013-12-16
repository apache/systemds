/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.unary.matrix;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;

public class QRFactorizeTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "qr";
	private final static String TEST_DIR = "functions/unary/matrix/";

	private final static int rows = 1500;
	private final static int cols = 500;
	private final static double sparsity = 0.5;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME1, 
				new TestConfiguration(TEST_DIR, TEST_NAME1, 
				new String[] { "I", "D" })   ); 

		// *** GENERATE data ONCE, and use it FOR ALL tests involving w/ different platforms
		// Therefore, data generation is done in setUp() method.
		
		TestConfiguration config = getTestConfiguration(TEST_NAME1);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
		programArgs = new String[]{"-args", HOME + INPUT_DIR + "A" , 
											HOME + OUTPUT_DIR + "I", 
											HOME + OUTPUT_DIR + "D" };

		loadTestConfiguration(config);
		
		double[][] A = getRandomMatrix(rows, cols, 0, 1, sparsity, 10);
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, -1, -1, -1);
		writeInputMatrixWithMTD("A", A, true, mc);
		
		// 1st Expected matrix = identity matrix
		double[][] I = new double[rows][rows];
		for(int i=0; i<rows; i++)
			I[i][i] = 1.0;
		writeExpectedMatrix("I", I);
		
		// 1st Expected matrix = 1x1 zero matrix 
		double[][] D  = new double[1][1];
		D[0][0] = 0.0;
		writeExpectedMatrix("D", D);
		
	
	}
	
	@Test
	public void testQRFactorizeDenseCP() 
	{
		runTestQRFactorize( RUNTIME_PLATFORM.SINGLE_NODE );
	}
	
	@Test
	public void testQRFactorizeDenseMR() 
	{
		runTestQRFactorize( RUNTIME_PLATFORM.HADOOP );
	}
	
	@Test
	public void testQRFactorizeDenseHybrid() 
	{
		runTestQRFactorize( RUNTIME_PLATFORM.HYBRID );
	}
	
	private void runTestQRFactorize( RUNTIME_PLATFORM rt)
	{		

		RUNTIME_PLATFORM rtold = rtplatform;
		rtplatform = rt;
		
		boolean exceptionExpected = false;
		runTest(true, exceptionExpected, null, -1);
		compareResults(1e-8);
		
		rtplatform = rtold;
	}
	

	
}