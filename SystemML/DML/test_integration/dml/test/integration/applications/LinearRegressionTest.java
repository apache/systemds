package dml.test.integration.applications;

import java.util.HashMap;

import org.junit.Test;

import dml.runtime.matrix.io.MatrixValue.CellIndex;
import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;
import dml.test.utils.TestUtils;

public class LinearRegressionTest extends AutomatedTestBase
{

    private final static String TEST_DIR = "applications/linear_regression/";
    private final static String TEST_LINEAR_REGRESSION = "LinearRegression";


    @Override
    public void setUp()
    {
        addTestConfiguration(TEST_LINEAR_REGRESSION, new TestConfiguration(TEST_DIR, TEST_LINEAR_REGRESSION,
                new String[] { "w" }));
    }
    
    @Test
    public void testLinearRegression()
    {
    	int rows = 50;
        int cols = 30;

        TestConfiguration config = getTestConfiguration(TEST_LINEAR_REGRESSION);
        config.addVariable("rows", rows);
        config.addVariable("cols", cols);
        config.addVariable("eps", Math.pow(10, -8));
        
        /* This is for running the junit test the new way, i.e., construct the arguments directly */
		String LR_HOME = SCRIPT_DIR + TEST_DIR;
		dmlArgs = new String[]{"-f", LR_HOME + TEST_LINEAR_REGRESSION + ".dml",
				               "-args", LR_HOME + INPUT_DIR + "v", 
				                        Integer.toString(rows), Integer.toString(cols),
				                        LR_HOME + INPUT_DIR + "y", 
				                        Double.toString(Math.pow(10,-8)), 
				                        LR_HOME + OUTPUT_DIR + "w"};
		dmlArgsDebug = new String[]{"-f", LR_HOME + TEST_LINEAR_REGRESSION + ".dml", "-d",
	                                "-args", LR_HOME + INPUT_DIR + "v", 
	                                         Integer.toString(rows), Integer.toString(cols),
	                                         LR_HOME + INPUT_DIR + "y", 
	                                         Double.toString(Math.pow(10,-8)), 
	                                         LR_HOME + OUTPUT_DIR + "w" };
		
		rCmd = "Rscript" + " " + LR_HOME + TEST_LINEAR_REGRESSION + ".R" + " " + 
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
        TestUtils.compareMatrices(wR, wDML, Math.pow(10, -14), "wR", "wDML");
    }
}
