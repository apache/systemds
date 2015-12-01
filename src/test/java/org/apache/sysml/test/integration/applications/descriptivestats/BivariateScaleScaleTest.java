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

package com.ibm.bi.dml.test.integration.applications.descriptivestats;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;


public class BivariateScaleScaleTest extends AutomatedTestBase 
{

	
	private final static String TEST_DIR = "applications/descriptivestats/";
	private final static String TEST_SCALE_SCALE = "ScaleScale";
	private final static String TEST_SCALE_SCALE_WEIGHTS = "ScaleScalePearsonRWithWeightsTest";

	private final static double eps = 1e-10;
	
	private final static int rows = 100000;      // # of rows in each vector
	private final static double minVal=0;       // minimum value in each vector 
	private final static double maxVal=10000;    // maximum value in each vector 
	private int maxW = 1000;    // maximum weight
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_SCALE_SCALE, new TestConfiguration(TEST_DIR,
				TEST_SCALE_SCALE, new String[] { "PearsonR" + ".scalar" }));
		addTestConfiguration(TEST_SCALE_SCALE_WEIGHTS, new TestConfiguration(
				TEST_DIR, "ScaleScalePearsonRWithWeightsTest",
				new String[] { "PearsonR" + ".scalar" }));
	}
	
	@Test
	public void testPearsonR() {

		TestConfiguration config = getTestConfiguration(TEST_SCALE_SCALE);
		
		config.addVariable("rows", rows);
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String SS_HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = SS_HOME + TEST_SCALE_SCALE + ".dml";
		programArgs = new String[]{"-args",  SS_HOME + INPUT_DIR + "X" , 
				                        Integer.toString(rows),
				                         SS_HOME + INPUT_DIR + "Y" , 
				                         SS_HOME + OUTPUT_DIR + "PearsonR" };
		fullRScriptName = SS_HOME + TEST_SCALE_SCALE + ".R";
		rCmd = "Rscript" + " " + SS_HOME + TEST_SCALE_SCALE + ".R" + " " + 
		       SS_HOME + INPUT_DIR + " " + SS_HOME + EXPECTED_DIR;
		
		loadTestConfiguration(config);

		long seed = System.currentTimeMillis();
		//System.out.println("Seed = " + seed);
        double[][] X = getRandomMatrix(rows, 1, minVal, maxVal, 0.1, seed);
        double[][] Y = getRandomMatrix(rows, 1, minVal, maxVal, 0.1, seed+1);

		writeInputMatrix("X", X, true);
		writeInputMatrix("Y", Y, true);

		boolean exceptionExpected = false;
		/*
		 * Expected number of jobs:
		 */
		// int expectedNumberOfJobs = 5; // This will cause failure
		runTest(true, exceptionExpected, null, -1);
		runRScript(true);
		
		for(String file: config.getOutputFiles())
		{
			/* NOte that some files do not contain matrix, but just a single scalar value inside */
			HashMap<CellIndex, Double> dmlfile;
			HashMap<CellIndex, Double> rfile;
			if (file.endsWith(".scalar")) {
				file = file.replace(".scalar", "");
				dmlfile = readDMLScalarFromHDFS(file);
				rfile = readRScalarFromFS(file);
			}
			else {
				dmlfile = readDMLMatrixFromHDFS(file);
				rfile = readRMatrixFromFS(file);
			}
			TestUtils.compareMatrices(dmlfile, rfile, eps, file+"-DML", file+"-R");
		}
	}
	
	private void round(double[][] weight) {
		for(int i=0; i<weight.length; i++)
			weight[i][0]=Math.floor(weight[i][0]);
	}

	@Test
	public void testPearsonRWithWeights() {

		TestConfiguration config = getTestConfiguration(TEST_SCALE_SCALE_WEIGHTS);
		
		config.addVariable("rows", rows);

		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String SS_HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = SS_HOME + TEST_SCALE_SCALE_WEIGHTS + ".dml";
		programArgs = new String[]{"-args",  SS_HOME + INPUT_DIR + "X" , 
				                        Integer.toString(rows),
				                         SS_HOME + INPUT_DIR + "Y" , 
				                         SS_HOME + INPUT_DIR + "WM" , 
				                         SS_HOME + OUTPUT_DIR + "PearsonR" };
		fullRScriptName = SS_HOME + TEST_SCALE_SCALE_WEIGHTS + ".R";
		rCmd = "Rscript" + " " + SS_HOME + TEST_SCALE_SCALE_WEIGHTS + ".R" + " " + 
		       SS_HOME + INPUT_DIR + " " + SS_HOME + EXPECTED_DIR;

		loadTestConfiguration(config);

		//long seed = System.currentTimeMillis();
		//System.out.println("Seed = " + seed);
        double[][] X = getRandomMatrix(rows, 1, minVal, maxVal, 0.1, System.currentTimeMillis());
        double[][] Y = getRandomMatrix(rows, 1, minVal, maxVal, 0.1, System.currentTimeMillis());
        double[][] WM = getRandomMatrix(rows, 1, 1, maxW, 1, System.currentTimeMillis());
        round(WM);
        
		writeInputMatrix("X", X, true);
		writeInputMatrix("Y", Y, true);
		writeInputMatrix("WM", WM, true);
        createHelperMatrix();
		
		boolean exceptionExpected = false;
		/*
		 * Expected number of jobs:
		 * Mean etc - 2 jobs (reblock & gmr)
		 * Cov etc - 2 jobs
		 * Final output write - 1 job
		 */
		//int expectedNumberOfJobs = 6;
		runTest(true, exceptionExpected, null, -1);
		runRScript(true);
		
		for(String file: config.getOutputFiles())
		{
			/* NOte that some files do not contain matrix, but just a single scalar value inside */
			HashMap<CellIndex, Double> dmlfile;
			HashMap<CellIndex, Double> rfile;
			if (file.endsWith(".scalar")) {
				file = file.replace(".scalar", "");
				dmlfile = readDMLScalarFromHDFS(file);
				rfile = readRScalarFromFS(file);
			}
			else {
				dmlfile = readDMLMatrixFromHDFS(file);
				rfile = readRMatrixFromFS(file);
			}
			TestUtils.compareMatrices(dmlfile, rfile, eps, file+"-DML", file+"-R");
		}

	}
	

}
