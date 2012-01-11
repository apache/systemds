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
    private final static String TEST_LINEAR_REGRESSION = "LinearRegressionTest";


    @Override
    public void setUp()
    {
        addTestConfiguration(TEST_LINEAR_REGRESSION, new TestConfiguration(TEST_DIR, "LinearRegressionTest",
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
		runTest(exceptionExpected, null, expectedNumberOfJobs);
        
		runRScript();
        
        HashMap<CellIndex, Double> wR = this.readRMatrixFromFS("w");
        HashMap<CellIndex, Double> wDML= this.readDMLMatrixFromHDFS("w");
        TestUtils.compareMatrices(wR, wDML, Math.pow(10, -14), "wR", "wDML");
    }
}
