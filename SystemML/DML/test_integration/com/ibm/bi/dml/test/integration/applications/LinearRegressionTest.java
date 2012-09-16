package com.ibm.bi.dml.test.integration.applications;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

@RunWith(value = Parameterized.class)
public class LinearRegressionTest extends AutomatedTestBase
{

    private final static String TEST_DIR = "applications/linear_regression/";
    private final static String TEST_LINEAR_REGRESSION = "LinearRegression";

    private int numRecords, numFeatures;
    
	public LinearRegressionTest(int rows, int cols) {
		numRecords = rows;
		numFeatures = cols;
	}
	
	@Parameters
	 public static Collection<Object[]> data() {
	   Object[][] data = new Object[][] { {100, 50}, {1000, 500}, {10000, 750}};
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
		programArgs = new String[]{"-args", LR_HOME + INPUT_DIR + "v", 
				                        Integer.toString(rows), Integer.toString(cols),
				                        LR_HOME + INPUT_DIR + "y", 
				                        Double.toString(Math.pow(10,-8)), 
				                        LR_HOME + OUTPUT_DIR + "w"};
		
		fullRScriptName = LR_HOME + TEST_LINEAR_REGRESSION + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       LR_HOME + INPUT_DIR + " " + Double.toString(Math.pow(10, -8)) + " " + LR_HOME + EXPECTED_DIR;
      
        loadTestConfiguration(TEST_LINEAR_REGRESSION);

        double[][] v = getRandomMatrix(rows, cols, 0, 1, 0.01, -1);
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
        
        HashMap<CellIndex, Double> wR = this.readRMatrixFromFS("w");
        HashMap<CellIndex, Double> wDML= this.readDMLMatrixFromHDFS("w");
        TestUtils.compareMatrices(wR, wDML, Math.pow(10, -10), "wR", "wDML");
    }
}
