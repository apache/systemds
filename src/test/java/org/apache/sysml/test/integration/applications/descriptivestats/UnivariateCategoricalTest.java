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

package org.apache.sysml.test.integration.applications.descriptivestats;

import java.util.HashMap;

import org.junit.Test;

import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;


/** Tests of univariate statistics functions over categorical data. */
public class UnivariateCategoricalTest extends UnivariateStatsBase
{
	
	@Test
	public void testCategoricalWithR() {
	
        TestConfiguration config = getTestConfiguration("Categorical");
        config.addVariable("rows1", rows1);
        
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String C_HOME = SCRIPT_DIR + TEST_DIR;	
		fullDMLScriptName = C_HOME + "Categorical" + ".dml";
		programArgs = new String[]{"-args",  C_HOME + INPUT_DIR + "vector" , 
	                        Integer.toString(rows1),
	                         C_HOME + OUTPUT_DIR + "Nc" , 
	                         C_HOME + OUTPUT_DIR + "R" , 
	                         C_HOME + OUTPUT_DIR + "Pc" ,
	                         C_HOME + OUTPUT_DIR + "C" ,
	                         C_HOME + OUTPUT_DIR + "Mode" };
		fullRScriptName = C_HOME + "Categorical" + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       C_HOME + INPUT_DIR + " " + C_HOME + EXPECTED_DIR;

		loadTestConfiguration(config);

        double[][] vector = getRandomMatrix(rows1, 1, 1, 10, 1, System.currentTimeMillis());
        OrderStatisticsTest.round(vector);
        
        writeInputMatrix("vector", vector, true);

		/*
		 * Expected number of jobs:
		 * Reblock - 1 job 
		 * While loop iteration - 10 jobs
		 * Final output write - 1 job
		 */
        boolean exceptionExpected = false;
		int expectedNumberOfJobs = 12;
		runTest(true, exceptionExpected, null, expectedNumberOfJobs);
		
		runRScript(true);
		//disableOutAndExpectedDeletion();
	
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
			TestUtils.compareMatrices(dmlfile, rfile, epsilon, file+"-DML", file+"-R");
		}
	}
	
	@Test
	public void testWeightedCategoricalWithR() {
	
        TestConfiguration config = getTestConfiguration("WeightedCategoricalTest");
        config.addVariable("rows1", rows1);

		// This is for running the junit test the new way, i.e., construct the arguments directly
		String C_HOME = SCRIPT_DIR + TEST_DIR;	
		fullDMLScriptName = C_HOME + "WeightedCategoricalTest" + ".dml";
		programArgs = new String[]{"-args",  C_HOME + INPUT_DIR + "vector" , 
	                        Integer.toString(rows1),
	                         C_HOME + INPUT_DIR + "weight" , 
	                         C_HOME + OUTPUT_DIR + "Nc" , 
	                         C_HOME + OUTPUT_DIR + "R" , 
	                         C_HOME + OUTPUT_DIR + "Pc" ,
	                         C_HOME + OUTPUT_DIR + "C" ,
	                         C_HOME + OUTPUT_DIR + "Mode" };
		fullRScriptName = C_HOME + "WeightedCategoricalTest" + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       C_HOME + INPUT_DIR + " " + C_HOME + EXPECTED_DIR;
		
		loadTestConfiguration(config);

		createHelperMatrix();
        double[][] vector = getRandomMatrix(rows1, 1, 1, 10, 1, System.currentTimeMillis());
        OrderStatisticsTest.round(vector);
        double[][] weight = getRandomMatrix(rows1, 1, 1, 10, 1, System.currentTimeMillis());
        OrderStatisticsTest.round(weight);

        writeInputMatrix("vector", vector, true);
        writeInputMatrix("weight", weight, true);
  
        boolean exceptionExpected = false;
		runTest(true, exceptionExpected, null, -1);
		runRScript(true);
		//disableOutAndExpectedDeletion();
	
		for(String file: config.getOutputFiles())
		{
			// NOte that some files do not contain matrix, but just a single scalar value inside
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
			TestUtils.compareMatrices(dmlfile, rfile, epsilon, file+"-DML", file+"-R");
		}
	}
}
