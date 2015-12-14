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

import org.junit.Test;

import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class DynReadWriteTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME = "DynReadWrite";
	private final static String TEST_DIR = "functions/external/";
	private final static String TEST_CLASS_DIR = TEST_DIR + DynReadWriteTest.class.getSimpleName() + "/";
	
	private final static double eps = 1e-10;
	
	private final static int rows = 1200;
	private final static int cols = 1100;    
	private final static double sparsity = 0.7;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME,
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "Rout" }) );
	}

	
	@Test
	public void testTextCell() 
	{
		runDynReadWriteTest("textcell");
	}

	@Test
	public void testBinaryCell() 
	{
		runDynReadWriteTest("binarycell");
	}
	
	@Test
	public void testBinaryBlock() 
	{
		runDynReadWriteTest("binaryblock");
	}
		
	
	private void runDynReadWriteTest( String format )
	{		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		loadTestConfiguration(config);
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", input("X"),
			Integer.toString(rows), Integer.toString(cols), format, output("Y") };

		try 
		{
			long seed = System.nanoTime();
	        double[][] X = getRandomMatrix(rows, cols, 0, 1, sparsity, seed);
			writeInputMatrix("X", X, false);

			runTest(true, false, null, -1);

			double[][] Y = MapReduceTool.readMatrixFromHDFS(output("Y"), InputInfo.stringToInputInfo(format), rows, cols, 1000, 1000);
		
			TestUtils.compareMatrices(X, Y, rows, cols, eps);
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
		}
	}
}