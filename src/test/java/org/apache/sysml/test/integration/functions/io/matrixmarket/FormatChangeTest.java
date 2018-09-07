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

package org.apache.sysml.test.integration.functions.io.matrixmarket;

import org.junit.Test;

import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class FormatChangeTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME = "mm_test";
	private final static String TEST_DIR = "functions/io/matrixmarket/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FormatChangeTest.class.getSimpleName() + "/";
	
	private final static int rows = 2000;
	private final static int cols = 500;
	private final static double sparsity = 0.01;
	private static String format1, format2;
	private final static double eps = 1e-9;

	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "Rout" }) );  
	}
	
	@Test
	public void testFormatChange() {
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format1", cols);
		config.addVariable("format2", cols);
		
		loadTestConfiguration(config);
		
		int scriptNum = 1;
		
		//generate actual dataset
		double[][] D = getRandomMatrix(rows, cols, 0, 1, sparsity, 7777); 
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, -1, -1);
		writeInputMatrixWithMTD("D", D, true, mc);

		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME +scriptNum + ".dml";
		String[] oldProgramArgs = programArgs = new String[]{"-args", 
			input("D"), format1, input("D.binary"), format2 };
		
		String txtFile = input("D");
		String binFile = input("D.binary");
		String mmFile  = output("D.mm");
		
		// text to binary format
		programArgs[2] = "text";
		programArgs[3] = binFile;
		programArgs[4] = "binary";
		runTest(true, false, null, -1);

		// Test TextCell -> MM conversion
		programArgs[2] = "text";
		programArgs[3] = mmFile;
		programArgs[4] = "mm";
		runTest(true, false, null, -1);
		
		verifyDMLandMMFiles(rows, cols, sparsity, txtFile, "text", mmFile);

		// Test BinaryBlock -> MM conversion
		programArgs = oldProgramArgs;
		programArgs[1] = binFile;
		programArgs[2] = "binary";
		programArgs[3] = mmFile;
		programArgs[4] = "mm";
		runTest(true, false, null, -1);
		
		verifyDMLandMMFiles(rows, cols, sparsity, binFile, "binary", mmFile);

		// Test MM -> TextCell conversion
		programArgs = oldProgramArgs;
		programArgs[1] = mmFile;
		programArgs[2] = "mm";
		programArgs[3] = txtFile;
		programArgs[4] = "text";
		runTest(true, false, null, -1);
		
		verifyDMLandMMFiles(rows, cols, sparsity, txtFile, "text", mmFile);

		// Test MM -> BinaryBlock conversion
		programArgs = oldProgramArgs;
		programArgs[1] = mmFile;
		programArgs[2] = "mm";
		programArgs[3] = binFile;
		programArgs[4] = "binary";
		runTest(true, false, null, -1);
		
		verifyDMLandMMFiles(rows, cols, sparsity, binFile, "binary", mmFile);
		
		//fullRScriptName = HOME + TEST_NAME + ".R";
		//rCmd = "Rscript" + " " + fullRScriptName + " " + 
		//        inputDir() + " " + Integer.toString((int)maxVal) + " " + expectedDir();
	}
	
	private void verifyDMLandMMFiles(int rows, int cols, double sparsity, String dmlFile, String dmlFormat, String mmFile) {
		String HOME = SCRIPT_DIR + TEST_DIR;
		
		// backup old DML and R script files
		String oldDMLScript = fullDMLScriptName;
		String oldRScript = fullRScriptName;
		
		String dmlOutput = output("dml.scalar");
		String rOutput = output("R.scalar");
		
		fullDMLScriptName = HOME + "mm_verify.dml";
		programArgs = new String[]{"-args", dmlFile,
			Integer.toString(rows), Integer.toString(cols), dmlFormat, dmlOutput };
		
		fullRScriptName = HOME + "mm_verify.R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + mmFile + " " + rOutput;
		
		// Run the verify test
		runTest(true, false, null, -1);
		runRScript(true);
		
		double dmlScalar = TestUtils.readDMLScalar(dmlOutput); 
		double rScalar = TestUtils.readRScalar(rOutput); 
		
		TestUtils.compareScalars(dmlScalar, rScalar, eps);
		
		// restore old DML and R script files
		fullDMLScriptName = oldDMLScript;
		fullRScriptName = oldRScript;
		
	}
	
}