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

import java.io.IOException;
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
public class LinearLogRegTest extends AutomatedTestBase
{
	
    private final static String TEST_DIR = "applications/linearLogReg/";
    private final static String TEST_LINEAR_LOG_REG = "LinearLogReg";

    private int numRecords, numFeatures, numTestRecords;
    private double sparsity;
    
	public LinearLogRegTest(int numRecords, int numFeatures, int numTestRecords, double sparsity) {
		this.numRecords = numRecords;
		this.numFeatures = numFeatures;
		this.numTestRecords = numTestRecords;
		this.sparsity = sparsity;
	}
	
	@Parameters
	 public static Collection<Object[]> data() {
	   Object[][] data = new Object[][] { 
			 //sparse tests (sparsity=0.01)
			 {100, 50, 25, 0.01}, {1000, 500, 200, 0.01}, {10000, 750, 1500, 0.01}, {100000, 1000, 1500, 0.01},
			 //dense tests (sparsity=0.7)
			 {100, 50, 25, 0.7}, {1000, 500, 200, 0.7}, {10000, 750, 1500, 0.7}};
	   return Arrays.asList(data);
	 }
 
    @Override
    public void setUp()
    {
    	setUpBase();
    	addTestConfiguration(TEST_LINEAR_LOG_REG, new TestConfiguration(TEST_DIR, TEST_LINEAR_LOG_REG,
                new String[] { "w" }));
    }
    
    @Test
    public void testLinearLogReg() throws ClassNotFoundException, IOException
    {
    	int rows = numRecords;			// # of rows in the training data 
        int cols = numFeatures;
        int rows_test = numTestRecords; // # of rows in the test data 
        int cols_test = cols; 			// # of rows in the test data 

        TestConfiguration config = getTestConfiguration(TEST_LINEAR_LOG_REG);
        config.addVariable("rows", rows);
        config.addVariable("cols", cols);
        config.addVariable("rows_test", rows_test);
        config.addVariable("cols_test", cols_test);
        
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String LLR_HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = LLR_HOME + TEST_LINEAR_LOG_REG + ".dml";
		programArgs = new String[]{"-stats","-args", LLR_HOME + INPUT_DIR + "X" , 
				                        Integer.toString(rows), Integer.toString(cols),
				                         LLR_HOME + INPUT_DIR + "Xt" , 
				                        Integer.toString(rows_test), Integer.toString(cols_test),
				                         LLR_HOME + INPUT_DIR + "y" ,
				                         LLR_HOME + INPUT_DIR + "yt" ,
				                         LLR_HOME + OUTPUT_DIR + "w" };
		
		fullRScriptName = LLR_HOME + TEST_LINEAR_LOG_REG + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       LLR_HOME + INPUT_DIR + " " + LLR_HOME + EXPECTED_DIR;
      
        loadTestConfiguration(config);

        // prepare training data set
        double[][] X = getRandomMatrix(rows, cols, 1, 10, sparsity, 100);
        double[][] y = getRandomMatrix(rows, 1, 0.01, 1, 1, 100);
        writeInputMatrix("X", X, true);
        writeInputMatrix("y", y, true);
        
        // prepare test data set
        double[][] Xt = getRandomMatrix(rows_test, cols_test, 1, 10, sparsity, 100);
        double[][] yt = getRandomMatrix(rows_test, 1, 0.01, 1, 1, 100);
        writeInputMatrix("Xt", Xt, true);
        writeInputMatrix("yt", yt, true);
        
        
		boolean exceptionExpected = false;
		int expectedNumberOfJobs = 31;
		runTest(true, exceptionExpected, null, expectedNumberOfJobs);
        
		runRScript(true);
        
        HashMap<CellIndex, Double> wR = readRMatrixFromFS("w");
        HashMap<CellIndex, Double> wDML= readDMLMatrixFromHDFS("w");
        TestUtils.compareMatrices(wR, wDML, Math.pow(10, -14), "wR", "wDML");
    }
}
