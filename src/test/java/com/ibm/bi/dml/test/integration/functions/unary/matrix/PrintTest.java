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

package com.ibm.bi.dml.test.integration.functions.unary.matrix;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;

public class PrintTest extends AutomatedTestBase 
{

	
	private final static String TEST_NAME = "PrintTest";
	private final static String TEST_DIR = "functions/unary/matrix/";

	//note: (1) even number of rows/cols required, (2) same dims because controlled via exec platform
	private final static int rows = 10;
	private final static int cols = 10;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "V" })   ); 
	}
	
	@Test
	public void testPrintMatrixScalarNames() 
	{
		runPrintTest();
	}
	
	private void runPrintTest()
	{	
		//register test configuration
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", HOME + INPUT_DIR + "X" , 
				                            String.valueOf(rows),
											String.valueOf(cols) };
		
		loadTestConfiguration(config);
		
		double[][] X = getRandomMatrix(rows, cols, 0, 1, 1.0, 7);
        writeInputMatrix("X", X, true);
        runTest(true, false, null, -1);
	}	
}