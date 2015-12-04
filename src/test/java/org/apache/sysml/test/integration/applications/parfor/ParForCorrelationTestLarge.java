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

package org.apache.sysml.test.integration.applications.parfor;

import java.util.HashMap;

import org.junit.Test;

import org.apache.sysml.hops.Hop;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PExecMode;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

/**
 * Intension is to test file-based result merge with regard to its integration
 * with the different execution modes. Hence we need at least a dataset of size
 * CPThreshold^2
 * 
 * 
 */
public class ParForCorrelationTestLarge extends AutomatedTestBase 
{

	
	private final static String TEST_NAME = "parfor_corr_large";
	private final static String TEST_DIR = "applications/parfor/";
	private final static double eps = 1e-10;
	
	private final static int rows = (int)Hop.CPThreshold+1;  // # of rows in each vector (for MR instructions)
	private final static int cols = (int)Hop.CPThreshold+1;      // # of columns in each vector  
	
	private final static double minVal=0;    // minimum value in each vector 
	private final static double maxVal=1000; // maximum value in each vector 

	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "Rout" })   ); //TODO this specification is not intuitive
	}

	
	@Test
	public void testParForCorrleationLargeLocalLocal() 
	{
		runParForCorrelationTest(PExecMode.LOCAL, PExecMode.LOCAL);
	}

	/*
	@Test
	public void testParForCorrleationLargeLocalRemote() 
	{
		runParForCorrelationTest(PExecMode.LOCAL, PExecMode.REMOTE_MR);
	}
	
	@Test
	public void testParForCorrleationRemoteLocalCP() 
	{
		runParForCorrelationTest(PExecMode.REMOTE_MR, PExecMode.LOCAL);
	}
	*/
	
	@Test
	public void testParForCorrleationLargeDefault() 
	{
		runParForCorrelationTest(null, null);
	}
	
	/**
	 * 
	 * @param outer execution mode of outer parfor loop
	 * @param inner execution mode of inner parfor loop
	 * @param instType execution mode of instructions
	 */
	private void runParForCorrelationTest( PExecMode outer, PExecMode inner )
	{
		//script
		int scriptNum = -1;
		if( inner == PExecMode.REMOTE_MR )      scriptNum=2;
		else if( outer == PExecMode.REMOTE_MR ) scriptNum=3;
		else if( outer == PExecMode.LOCAL )     scriptNum=1;
		else                                    scriptNum=4; //optimized
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME +scriptNum + ".dml";
		programArgs = new String[]{"-args", HOME + INPUT_DIR + "V" , 
				                        Integer.toString(rows),
				                        Integer.toString(cols),
				                        HOME + OUTPUT_DIR + "PearsonR" };
		
		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
		
		loadTestConfiguration(config);

		long seed = System.nanoTime();
        double[][] V = getRandomMatrix(rows, cols, minVal, maxVal, 1.0, seed);
		writeInputMatrix("V", V, true);

		boolean exceptionExpected = false;
		runTest(true, exceptionExpected, null, -1);
		runRScript(true);
		
		//compare matrices
		HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("PearsonR");
		HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("Rout");
		TestUtils.compareMatrices(dmlfile, rfile, eps, "PearsonR-DML", "PearsonR-R");
		
	}
}