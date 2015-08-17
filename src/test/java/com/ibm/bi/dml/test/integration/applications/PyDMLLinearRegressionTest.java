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

package com.ibm.bi.dml.test.integration.applications;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

@RunWith(value = Parameterized.class)
public class PyDMLLinearRegressionTest extends AutomatedTestBase
{
	
	// Reuse directory and test name
    private final static String TEST_DIR = "applications/linear_regression/";
    private final static String TEST_LINEAR_REGRESSION = "LinearRegression";

    private int numRecords, numFeatures;
    private double sparsity;
    
	public PyDMLLinearRegressionTest(int rows, int cols, double sp) {
		numRecords = rows;
		numFeatures = cols;
		sparsity = sp;
	}
	
	@Parameters
	 public static Collection<Object[]> data() {
	   Object[][] data = new Object[][] { 
			   //sparse tests (sparsity=0.01)
			   {100, 50, 0.01}, {1000, 500, 0.01}, {10000, 750, 0.01}, {100000, 1000, 0.01},
			   //dense tests (sparsity=0.7)
			   {100, 50, 0.7}, {1000, 500, 0.7}, {10000, 750, 0.7} };
	   
	   return Arrays.asList(data);
	 }
	 
    @Override
    public void setUp()
    {
        addTestConfiguration(TEST_LINEAR_REGRESSION, new TestConfiguration(TEST_DIR, TEST_LINEAR_REGRESSION,
                new String[] { "w" }));
    }
    
    @Test
    public void testLinearRegression()
    {
    	int rows = numRecords;
        int cols = numFeatures;

        TestConfiguration config = getTestConfiguration(TEST_LINEAR_REGRESSION);
        config.addVariable("rows", rows);
        config.addVariable("cols", cols);
        config.addVariable("eps", Math.pow(10, -8));
        
        /* This is for running the junit test the new way, i.e., construct the arguments directly */
		String LR_HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = LR_HOME + TEST_LINEAR_REGRESSION + ".pydml";
		programArgs = new String[]{"-python", "-stats","-args", LR_HOME + INPUT_DIR + "v", 
				                        Integer.toString(rows), Integer.toString(cols),
				                        LR_HOME + INPUT_DIR + "y", 
				                        Double.toString(Math.pow(10,-8)), 
				                        LR_HOME + OUTPUT_DIR + "w"};
		
		fullRScriptName = LR_HOME + TEST_LINEAR_REGRESSION + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       LR_HOME + INPUT_DIR + " " + Double.toString(Math.pow(10, -8)) + " " + LR_HOME + EXPECTED_DIR;
      
        loadTestConfiguration(config);

        double[][] v = getRandomMatrix(rows, cols, 0, 1, sparsity, -1);
        double[][] y = getRandomMatrix(rows, 1, 1, 10, 1, -1);
        writeInputMatrix("v", v, true);
        writeInputMatrix("y", y, true);
        
		boolean exceptionExpected = false;
		/*
		 * Expected number of jobs:
		 * Rand - 1 job 
		 * Computation before while loop - 4 jobs
		 * While loop iteration - 10 jobs
		 * Final output write - 1 job
		 */
		int expectedNumberOfJobs = 16;
		runTest(true, exceptionExpected, null, expectedNumberOfJobs);
        
		runRScript(true);
        
        HashMap<CellIndex, Double> wR = readRMatrixFromFS("w");
        HashMap<CellIndex, Double> wDML= readDMLMatrixFromHDFS("w");
        TestUtils.compareMatrices(wR, wDML, Math.pow(10, -10), "wR", "wDML");
    }
}
