/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.test.integration.functions.unary.matrix;

import org.junit.Test;

import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;

public class QRSolverTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "QRsolve";
	private final static String TEST_DIR = "functions/unary/matrix/";
	private static final String TEST_CLASS_DIR = TEST_DIR + QRSolverTest.class.getSimpleName() + "/";

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
		addTestConfiguration(TEST_NAME, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "x" }) ); 

		// *** GENERATE data ONCE, and use it FOR ALL tests involving w/ different platforms
		// Therefore, data generation is done in setUp() method.
		
		TestConfiguration config = getAndLoadTestConfiguration(TEST_NAME);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", input("A"), input("y"), 
			output(config.getOutputFiles()[0]) };

		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + input("A.mtx") + " " + 
			input("y.mtx") + " " + expected(config.getOutputFiles()[0]);
		
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