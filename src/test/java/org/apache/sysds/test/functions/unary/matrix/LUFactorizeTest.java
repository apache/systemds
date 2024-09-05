/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.test.functions.unary.matrix;

import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

public class LUFactorizeTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "lu";
	private final static String TEST_DIR = "functions/unary/matrix/";
	private static final String TEST_CLASS_DIR = TEST_DIR + LUFactorizeTest.class.getSimpleName() + "/";

	private final static int rows1 = 500;
	private final static int rows2 = 2500;
	private final static double sparsity = 0.9;
	
	@Override
	public void setUp() {
		addTestConfiguration( TEST_NAME1, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "D" })   ); 
	}
	
	@Test
	public void testLUFactorizeDenseCP() {
		runTestLUFactorize( rows1, ExecMode.SINGLE_NODE );
	}
	
	@Test
	public void testLUFactorizeDenseSP() {
		runTestLUFactorize( rows1, ExecMode.SPARK );
	}
	
	@Test
	public void testLUFactorizeDenseHybrid() {
		runTestLUFactorize( rows1, ExecMode.HYBRID );
	}
	
	@Test
	public void testLargeLUFactorizeDenseCP() {
		runTestLUFactorize( rows2, ExecMode.SINGLE_NODE );
	}
	
	@Test
	public void testLargeLUFactorizeDenseSP() {
		runTestLUFactorize( rows2, ExecMode.SPARK );
	}
	
	@Test
	public void testLargeLUFactorizeDenseHybrid() {
		runTestLUFactorize( rows2, ExecMode.HYBRID );
	}

	@Test
	public void testLUFactorizeSingularDenseCP() {
		runTestLUFactorize( 2, ExecMode.SINGLE_NODE, true);
	}

	private void runTestLUFactorize( int rows, ExecMode rt){
		runTestLUFactorize(rows, rt, false);
	}
	
	private void runTestLUFactorize( int rows, ExecMode rt, boolean exceptionExpected)
	{		
		ExecMode rtold = rtplatform;
		rtplatform = rt;
		
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		try
		{
			getAndLoadTestConfiguration(TEST_NAME1);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[]{"-args", input("A"), output("D") };

			double[][] A = getRandomMatrix(rows, rows, 0, 1, sparsity, 10);
			// create singular matrix
			if(exceptionExpected)
				A = new double[][]{{1, 0}, {1, 0}};

			MatrixCharacteristics mc = new MatrixCharacteristics(rows, rows, -1, -1);
			writeInputMatrixWithMTD("A", A, false, mc);
			
			// Expected matrix = 1x1 zero matrix 
			double[][] D  = new double[1][1];
			D[0][0] = 0.0;
			writeExpectedMatrix("D", D);

			runTest(true, exceptionExpected, null, -1);
			if(!exceptionExpected)
				compareResults(1e-8);
		}
		finally {
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			rtplatform = rtold;
		}
	}
	
}