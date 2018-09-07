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

package org.apache.sysml.test.integration.functions.parfor;

import java.util.HashMap;

import org.junit.Test;

import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class ParForFunctionSerializationTest extends AutomatedTestBase 
{

	private final static String TEST_NAME1 = "parfor_funct";
	private final static String TEST_NAME2 = "parfor_extfunct";
	private final static String TEST_DIR = "functions/parfor/";
	private final static String TEST_CLASS_DIR = TEST_DIR + ParForFunctionSerializationTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rows = 20;
	private final static int cols = 10;    
	private final static double sparsity = 1.0;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "Rout" }) );
		addTestConfiguration(TEST_NAME2, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "Rout" }) );
	}

	@Test
	public void testParForFunctSerialization() 
	{
		runFunctionTest(1);
	}
	
	@Test
	public void testParForExtFunctSerialization() 
	{
		runFunctionTest(2);
	}

	
	private void runFunctionTest( int testNum )
	{
		String TEST_NAME = null;
		switch( testNum )
		{
			case 1: TEST_NAME = TEST_NAME1; break;
			case 2: TEST_NAME = TEST_NAME2; break;
		}
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		loadTestConfiguration(config);
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", input("V"), 
			Integer.toString(rows), Integer.toString(cols), output("R") };
		
		fullRScriptName = HOME + TEST_NAME1 + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();

		long seed = System.nanoTime();
        double[][] V = getRandomMatrix(rows, cols, 0, 1, sparsity, seed);
		writeInputMatrix("V", V, true);

		boolean exceptionExpected = false;
		runTest(true, exceptionExpected, null, -1);
		runRScript(true);
		
		//compare matrices
		HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
		HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("Rout");
		TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");			

	}
	
}