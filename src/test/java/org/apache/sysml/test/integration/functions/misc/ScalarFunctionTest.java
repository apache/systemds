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

package org.apache.sysml.test.integration.functions.misc;

import java.util.HashMap;

import org.junit.Test;

import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

/**
 * 
 */
public class ScalarFunctionTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "ScalarFunctionTest1";
	private final static String TEST_NAME2 = "ScalarFunctionTest2";
	
	private final static String TEST_DIR = "functions/misc/";
	private final static double eps = 1e-8;
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] { "R" })); 
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] { "R" })); 
	}
	
	
	@Test
	public void testScalarFunctionLiteral() {
		runScalarFunctionTest(TEST_NAME1);
	}
	
	@Test
	public void testScalarFunctionVariable() {
		runScalarFunctionTest(TEST_NAME2);
	}
	
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runScalarFunctionTest( String testname )
	{
		String TEST_NAME = testname;
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		
		String HOME = SCRIPT_DIR + TEST_DIR;			
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", HOME + OUTPUT_DIR + "R"};
		fullRScriptName = HOME + TEST_NAME +".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + HOME + EXPECTED_DIR;
		
		loadTestConfiguration(config);

		//run Tests
		runTest(true, false, null, -1); 
		runRScript(true); 
		
		//compare matrices 
		HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
		HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
		TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			
		//check meta data
		checkDMLMetaDataFile("R", new MatrixCharacteristics(1,1,1,1));
	}

}