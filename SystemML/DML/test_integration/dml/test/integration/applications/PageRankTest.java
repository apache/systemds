package dml.test.integration.applications;

import org.junit.Test;

import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;
import dml.test.utils.TestUtils;

public class PageRankTest extends AutomatedTestBase
{

    private final static String TEST_DIR = "applications/page_rank/";
    private final static String TEST_PAGE_RANK = "PageRankTest";


    @Override
    public void setUp()
    {
        addTestConfiguration(TEST_PAGE_RANK, new TestConfiguration(TEST_DIR, "PageRankTest", new String[] { "p" }));
    }

    @Test
    public void testPageRank()
    {
    	int rows = 100;
    	int cols = 100;
    	
        loadTestConfiguration(TEST_PAGE_RANK);

        double[][] g = getRandomMatrix(rows, cols, 1, 1, 0.000374962, -1);
        double[][] p = getRandomMatrix(rows, 1, 1, 1, 1, -1);
        double[][] e = getRandomMatrix(rows, 1, 1, 1, 1, -1);
        double[][] u = getRandomMatrix(1, cols, 1, 1, 1, -1);
        writeInputMatrix("g", g);
        writeInputMatrix("p", p);
        writeInputMatrix("e", e);
        writeInputMatrix("u", u);
        
        double alpha = 0.85;
        for(int i = 0; i < 3; i++) {
        	double[][] gp = TestUtils.performMatrixMultiplication(g, p);
        	double[][] eu = TestUtils.performMatrixMultiplication(e, u);
        	double[][] eup = TestUtils.performMatrixMultiplication(eu, p);
        	for(int j = 0; j < rows; j++) {
        		p[j][0] = alpha * gp[j][0] + (1 - alpha) * eup[j][0];
        	}
        }
        
        writeExpectedMatrix("p", p);

		boolean exceptionExpected = false;
		/*
		 * Expected number of jobs:
		 * Reblock - 1 job 
		 * While loop iteration - 6 jobs (can be reduced if MMCJ job can produce multiple outputs)
		 * Final output write - 1 job
		 */
		int expectedNumberOfJobs = 8;
		runTest(exceptionExpected, null, expectedNumberOfJobs);
        
        compareResults();
    }

}
