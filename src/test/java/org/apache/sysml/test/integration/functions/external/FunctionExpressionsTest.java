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

package org.apache.sysml.test.integration.functions.external;

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.utils.Statistics;

public class FunctionExpressionsTest extends AutomatedTestBase 
{

	
	private final static String TEST_NAME1 = "FunctionExpressions1";
	private final static String TEST_NAME2 = "FunctionExpressions2";
	private final static String TEST_DIR = "functions/external/";
	private final static double eps = 1e-10;
	
	private final static int rows = 12;
	private final static int cols = 11;    
	private final static double sparsity = 0.7;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME1, 
				new TestConfiguration(TEST_DIR, TEST_NAME1, 
				new String[] { "Y" })   ); 
		addTestConfiguration(
				TEST_NAME2, 
				new TestConfiguration(TEST_DIR, TEST_NAME2, 
				new String[] { "Y" })   ); 
	}

	
	@Test
	public void testDMLFunction() 
	{
		runFunctionExpressionsTest( TEST_NAME1 );
	}
	
	@Test
	public void testExternalFunction() 
	{
		runFunctionExpressionsTest( TEST_NAME2 );
	}

	private void runFunctionExpressionsTest( String TEST_NAME )
	{		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", HOME + INPUT_DIR + "X", 
							                Integer.toString(rows),
							                Integer.toString(cols),
				                            HOME + OUTPUT_DIR + "Y" };
		loadTestConfiguration(config);

		try 
		{
			long seed = System.nanoTime();
	        double[][] X = getRandomMatrix(rows, cols, 0, 1, sparsity, seed);
			writeInputMatrix("X", X, false);

			runTest(true, false, null, -1);

			double[][] Y = MapReduceTool.readMatrixFromHDFS(HOME + OUTPUT_DIR + "Y", InputInfo.TextCellInputInfo, rows, cols, 1000,1000);
		
			double sx = sum(X,rows,cols);
			double sy = sum(Y,rows,cols);
			Assert.assertEquals(sx, sy, eps);
			
			Assert.assertEquals("Unexpected number of executed MR jobs.", 
					             0, Statistics.getNoOfExecutedMRJobs());
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
		}
	}
	
	private static double sum( double[][] X, int rows, int cols )
	{
		double sum = 0;
		for( int i=0; i<rows; i++ )
			for( int j=0; j<cols; j++ )
				sum += X[i][j];
		return sum;
	}
}