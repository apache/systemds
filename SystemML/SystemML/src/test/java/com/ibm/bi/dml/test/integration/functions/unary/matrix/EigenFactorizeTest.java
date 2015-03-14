/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.unary.matrix;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;

public class EigenFactorizeTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "eigen";
	private final static String TEST_DIR = "functions/unary/matrix/";

	private final static int rows1 = 500;
	private final static int rows2 = 1000;
	private final static double sparsity = 0.9;
	private final static int numEigenValuesToEvaluate = 15;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME1, 
				new TestConfiguration(TEST_DIR, TEST_NAME1, 
				new String[] { "D" })   ); 

		// *** GENERATE data ONCE, and use it FOR ALL tests involving w/ different platforms
		// Therefore, data generation is done in setUp() method.
		
	}
	
	@Test
	public void testEigenFactorizeDenseCP() 
	{
		runTestEigenFactorize( rows1, RUNTIME_PLATFORM.SINGLE_NODE );
	}
	
	@Test
	public void testEigenFactorizeDenseMR() 
	{
		runTestEigenFactorize( rows1, RUNTIME_PLATFORM.HADOOP );
	}
	
	@Test
	public void testEigenFactorizeDenseHybrid() 
	{
		runTestEigenFactorize( rows1, RUNTIME_PLATFORM.HYBRID );
	}
	
	@Test
	public void testLargeEigenFactorizeDenseCP() 
	{
		runTestEigenFactorize( rows2, RUNTIME_PLATFORM.SINGLE_NODE );
	}
	
	@Test
	public void testLargeEigenFactorizeDenseMR() 
	{
		runTestEigenFactorize( rows2, RUNTIME_PLATFORM.HADOOP );
	}
	
	@Test
	public void testLargeEigenFactorizeDenseHybrid() 
	{
		runTestEigenFactorize( rows2, RUNTIME_PLATFORM.HYBRID );
	}
	
	private void runTestEigenFactorize( int rows, RUNTIME_PLATFORM rt)
	{		
		RUNTIME_PLATFORM rtold = rtplatform;
		rtplatform = rt;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME1);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "A" ,
												Integer.toString(numEigenValuesToEvaluate),
												HOME + OUTPUT_DIR + "D" };
	
			loadTestConfiguration(config);
			
			double[][] A = getRandomMatrix(rows, rows, 0, 1, sparsity, 10);
			MatrixCharacteristics mc = new MatrixCharacteristics(rows, rows, -1, -1, -1);
			writeInputMatrixWithMTD("A", A, false, mc);
			
			// Expected matrix = 1x1 zero matrix 
			double[][] D  = new double[numEigenValuesToEvaluate][1];
			//D[0][0] = 0.0;
			for(int i=0; i < numEigenValuesToEvaluate; i++)
				D[i][0] = 0.0;
			writeExpectedMatrix("D", D);		
			
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1);
			compareResults(1e-8);
		}
		finally
		{
			rtplatform = rtold;
		}
	}
	
}