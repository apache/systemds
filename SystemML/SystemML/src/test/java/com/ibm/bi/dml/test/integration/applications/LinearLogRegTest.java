/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
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
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
