/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
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
public class LinearRegressionTest extends AutomatedTestBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
    private final static String TEST_DIR = "applications/linear_regression/";
    private final static String TEST_LINEAR_REGRESSION = "LinearRegression";

    private int numRecords, numFeatures;
    private double sparsity;
    
	public LinearRegressionTest(int rows, int cols, double sp) {
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
		fullDMLScriptName = LR_HOME + TEST_LINEAR_REGRESSION + ".dml";
		programArgs = new String[]{"-stats","-args", LR_HOME + INPUT_DIR + "v", 
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
