/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.unary.matrix;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;

public class MatrixInverseTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "inverse";
	private final static String TEST_DIR = "functions/unary/matrix/";

	private final static int rows = 1001;
	private final static int cols = 1001;
	private final static double sparsity = 0.7;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "AI" })   ); 

		// *** GENERATE data ONCE, and use it FOR ALL tests involving w/ different platforms
		// Therefore, data generation is done in setUp() method.
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", HOME + INPUT_DIR + "A" , 
											HOME + OUTPUT_DIR + config.getOutputFiles()[0] };

		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + "A.mtx" + " " + 
			       HOME + EXPECTED_DIR + config.getOutputFiles()[0];

		loadTestConfiguration(config);
		
		double[][] A = getRandomMatrix(rows, cols, 0, 1, sparsity, 10);
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, -1, -1, -1);
		writeInputMatrixWithMTD("A", A, true, mc);
	}
	
	@Test
	public void testInverseCP() 
	{
		runTestMatrixInverse( RUNTIME_PLATFORM.SINGLE_NODE );
	}
	
	@Test
	public void testInverseSP() 
	{
		if(rtplatform == RUNTIME_PLATFORM.SPARK)
			runTestMatrixInverse( RUNTIME_PLATFORM.SPARK );
	}
	
	@Test
	public void testInverseMR() 
	{
		runTestMatrixInverse( RUNTIME_PLATFORM.HADOOP );
	}
	
	@Test
	public void testInverseHybrid() 
	{
		runTestMatrixInverse( RUNTIME_PLATFORM.HYBRID );
	}
	
	private void runTestMatrixInverse( RUNTIME_PLATFORM rt )
	{		

		RUNTIME_PLATFORM rtold = rtplatform;
		rtplatform = rt;
		
		boolean exceptionExpected = false;
		runTest(true, exceptionExpected, null, -1);
		runRScript(true);
	
		compareResultsWithR(1e-5);
		
		rtplatform = rtold;
	}
	

	
}
