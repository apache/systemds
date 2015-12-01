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

public class MatrixInverseTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "Inverse";
	private final static String TEST_DIR = "functions/unary/matrix/";
	private static final String TEST_CLASS_DIR = TEST_DIR + MatrixInverseTest.class.getSimpleName() + "/";

	private final static int rows = 1001;
	private final static int cols = 1001;
	private final static double sparsity = 0.7;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "AI" }) );

		// *** GENERATE data ONCE, and use it FOR ALL tests involving w/ different platforms
		// Therefore, data generation is done in setUp() method.
		
		TestConfiguration config = getAndLoadTestConfiguration(TEST_NAME);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", input("A"), output(config.getOutputFiles()[0]) };

		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + input("A.mtx") + " " + 
		    expected(config.getOutputFiles()[0]);
		
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
