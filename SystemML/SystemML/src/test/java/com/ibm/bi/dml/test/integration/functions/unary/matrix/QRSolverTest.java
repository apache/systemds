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

public class QRSolverTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "QRsolve";
	private final static String TEST_DIR = "functions/unary/matrix/";

	private final static int rows = 1500;
	private final static int cols = 50;
	private final static double sparsity = 0.7;
	
	/** Main method for running single tests from Eclipse. */
	public static void main(String[] args) {
		long startMsec = System.currentTimeMillis();
		QRSolverTest t = new QRSolverTest();
		
		t.setUpBase();
		t.setUp();
		t.testQRSolveMR();
		
		t.tearDown();
		
		long elapsedMsec = System.currentTimeMillis() - startMsec;
		System.err.printf("Finished in %1.3f sec.\n", elapsedMsec / 1000.0);
	}
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "x" })   ); 

		// *** GENERATE data ONCE, and use it FOR ALL tests involving w/ different platforms
		// Therefore, data generation is done in setUp() method.
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", HOME + INPUT_DIR + "A" , 
											HOME + INPUT_DIR + "y", 
											HOME + OUTPUT_DIR + config.getOutputFiles()[0] };

		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + "A.mtx" + " " + 
			       HOME + INPUT_DIR + "y.mtx" + " " + 
			       HOME + EXPECTED_DIR + config.getOutputFiles()[0];

		loadTestConfiguration(config);
		
		double[][] A = getRandomMatrix(rows, cols, 0, 1, sparsity, 10);
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, -1, -1, -1);
		writeInputMatrixWithMTD("A", A, true, mc);
		
		double[][] y = getRandomMatrix(rows, 1, 0, 1, 1.0, 11);
		mc.set(rows, 1, -1, -1);
		writeInputMatrixWithMTD("y", y, true, mc);
	}
	
	@Test
	public void testQRSolveCP() 
	{
		runTestQRSolve( RUNTIME_PLATFORM.SINGLE_NODE );
	}
	
	@Test
	public void testQRSolveSP() 
	{
		if(rtplatform == RUNTIME_PLATFORM.SPARK)
			runTestQRSolve( RUNTIME_PLATFORM.SPARK );
	}
	
	@Test
	public void testQRSolveMR() 
	{
		runTestQRSolve( RUNTIME_PLATFORM.HADOOP );
	}
	
	@Test
	public void testQRSolveHybrid() 
	{
		runTestQRSolve( RUNTIME_PLATFORM.HYBRID );
	}
	
	private void runTestQRSolve( RUNTIME_PLATFORM rt)
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