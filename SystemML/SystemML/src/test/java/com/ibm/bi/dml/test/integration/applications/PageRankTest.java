/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.applications;

import java.util.Arrays;
import java.util.Collection;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;


@RunWith(value = Parameterized.class)
public class PageRankTest extends AutomatedTestBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
    private final static String TEST_DIR = "applications/page_rank/";
    private final static String TEST_PAGE_RANK = "PageRank";

    private int numRows, numCols;

	public PageRankTest(int rows, int cols) {
		this.numRows = rows;
		this.numCols = cols;
	}

	@Parameters
	 public static Collection<Object[]> data() {
	   Object[][] data = new Object[][] { { 50, 50 }, { 1500, 1500 }, { 7500, 7500 }};
	   return Arrays.asList(data);
	 }
   
    @Override
    public void setUp()
    {
        addTestConfiguration(TEST_PAGE_RANK, new TestConfiguration(TEST_DIR, "PageRank", new String[] { "p" }));
    }

    @Test
    public void testPageRank()
    {
    	int rows = numRows;
    	int cols = numCols;
    	int maxiter = 2;
    	double alpha = 0.85;
    	
    	/* This is for running the junit test by constructing the arguments directly */
		String PAGE_RANK_HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = PAGE_RANK_HOME + TEST_PAGE_RANK + ".dml";
		programArgs = new String[]{"-args",  PAGE_RANK_HOME + INPUT_DIR + "g" , 
				                         PAGE_RANK_HOME + INPUT_DIR + "p" , 
				                         PAGE_RANK_HOME + INPUT_DIR + "e" ,
				                         PAGE_RANK_HOME + INPUT_DIR + "u" ,
				                        Integer.toString(rows), Integer.toString(cols),
				                        Double.toString(alpha), Integer.toString(maxiter),
				                         PAGE_RANK_HOME + OUTPUT_DIR + "p" };
		
        TestConfiguration config = getTestConfiguration(TEST_PAGE_RANK);
        loadTestConfiguration(config);
        
        double[][] g = getRandomMatrix(rows, cols, 1, 1, 0.000374962, -1);
        double[][] p = getRandomMatrix(rows, 1, 1, 1, 1, -1);
        double[][] e = getRandomMatrix(rows, 1, 1, 1, 1, -1);
        double[][] u = getRandomMatrix(1, cols, 1, 1, 1, -1);
        writeInputMatrix("g", g);
        writeInputMatrix("p", p);
        writeInputMatrix("e", e);
        writeInputMatrix("u", u);
        
        for(int i = 0; i < maxiter; i++) {
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
		runTest(true, exceptionExpected, null, expectedNumberOfJobs);
        
        compareResults(0.0000001);
    }

}
