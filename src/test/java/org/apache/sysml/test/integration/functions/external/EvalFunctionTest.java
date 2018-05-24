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

public class EvalFunctionTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "SecondOrderExternal";
	private final static String TEST_NAME2 = "SecondOrderBuiltin";
	
	private final static String TEST_DIR = "functions/external/";
	private final static String TEST_CLASS_DIR = TEST_DIR + EvalFunctionTest.class.getSimpleName() + "/";
	
	private final static double eps = 1e-7;
	private final static int rows = 120;
	private final static int cols = 110;
	private final static double sparsity = 0.7;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "Y" }) );
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "Y" }) );
	}

	@Test
	public void runEvalFunctionExternalTest() {
		runEvalFunctionTest(TEST_NAME1);
	}
	
	@Test
	public void runEvalFunctionBuiltinTest() {
		runEvalFunctionTest(TEST_NAME2);
	}
	
	private void runEvalFunctionTest(String testname) {
		TestConfiguration config = getTestConfiguration(testname);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		loadTestConfiguration(config);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + testname + ".dml";
		programArgs = new String[]{"-args", input("X"), output("Y") };
		
		try {
			double[][] X = getRandomMatrix(rows, cols, 0, 1, sparsity, -7);
			writeInputMatrixWithMTD("X", X, false);
			runTest(true, false, null, -1);
			double[][] Y = MapReduceTool.readMatrixFromHDFS(output("Y"),
				InputInfo.stringToInputInfo("textcell"), rows, cols, 1000, 1000);
			for( int i=0; i<rows; i++ )
				for( int j=0; j<cols; j++ )
					if( Math.abs(X[i][j]-Y[i][j]+7) > eps )
						Assert.fail("Wrong results: "+X[i][j]+" vs "+Y[i][j]);
		} 
		catch (Exception e)  {
			e.printStackTrace();
		}
	}
}
