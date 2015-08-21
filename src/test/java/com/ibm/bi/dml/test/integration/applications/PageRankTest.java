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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;


@RunWith(value = Parameterized.class)
public class PageRankTest extends AutomatedTestBase {
	
    private final static String TEST_DIR = "applications/page_rank/";
    private final static String TEST_PAGE_RANK = "PageRank";
	private final static String PAGE_RANK_HOME = SCRIPT_DIR + TEST_DIR;

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
	public void testPageRankDml() {
		System.out.println("------------ BEGIN " + TEST_PAGE_RANK + " DML TEST {" + numRows + ", " + numCols + "} ------------");
		testPageRank(ScriptType.DML);
	}

	@Test
	public void testPageRankPyDml() {
		System.out.println("------------ BEGIN " + TEST_PAGE_RANK + " PYDML TEST {" + numRows + ", " + numCols + "} ------------");
		testPageRank(ScriptType.PYDML);
	}
    
    public void testPageRank(ScriptType scriptType)
    {
    	this.scriptType = scriptType;
    	
    	int rows = numRows;
    	int cols = numCols;
    	int maxiter = 2;
    	double alpha = 0.85;
    	
    	/* This is for running the junit test by constructing the arguments directly */
		List<String> proArgs = new ArrayList<String>();
		proArgs.add("-args");
		proArgs.add(PAGE_RANK_HOME + INPUT_DIR + "g");
		proArgs.add(PAGE_RANK_HOME + INPUT_DIR + "p");
		proArgs.add(PAGE_RANK_HOME + INPUT_DIR + "e");
		proArgs.add(PAGE_RANK_HOME + INPUT_DIR + "u");
		proArgs.add(Integer.toString(rows));
		proArgs.add(Integer.toString(cols));
		proArgs.add(Double.toString(alpha));
		proArgs.add(Integer.toString(maxiter));
		proArgs.add(PAGE_RANK_HOME + OUTPUT_DIR + "p");
		
		switch (scriptType) {
		case DML:
			fullDMLScriptName = PAGE_RANK_HOME + TEST_PAGE_RANK + ".dml";
			break;
		case PYDML:
			fullPYDMLScriptName = PAGE_RANK_HOME + TEST_PAGE_RANK + ".pydml";
			proArgs.add(0, "-python");
			break;
		}
		programArgs = proArgs.toArray(new String[proArgs.size()]);
		System.out.println("arguments from test case: " + Arrays.toString(programArgs));
		
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

		/*
		 * Expected number of jobs:
		 * Reblock - 1 job 
		 * While loop iteration - 6 jobs (can be reduced if MMCJ job can produce multiple outputs)
		 * Final output write - 1 job
		 */
		int expectedNumberOfJobs = 8;
		runTest(true, EXCEPTION_NOT_EXPECTED, null, expectedNumberOfJobs);
        
        compareResults(0.0000001);
    }

}
