package com.ibm.bi.dml.test.integration.functions.indexing;

import java.util.HashMap;
import java.util.Random;

import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;



public class RightIndexingTest  extends AutomatedTestBase{

	private final static String TEST_DIR = "functions/indexing/";

	private final static double epsilon=0.0000000001;
	private final static int rows = 2279;
	private final static int cols = 1050;
	private final static int min=0;
	private final static int max=100;
	
	@Override
	public void setUp() {
		addTestConfiguration("RightIndexingTest", new TestConfiguration(TEST_DIR, "RightIndexingTest", 
				new String[] {"B", "C", "D"}));
	}
	@Test
	public void testRightIndexing() {
	    TestConfiguration config = getTestConfiguration("RightIndexingTest");
	    
	    
        config.addVariable("rows", rows);
        config.addVariable("cols", cols);
        
        long rowstart=216, rowend=429, colstart=967, colend=1009;
        Random rand=new Random(System.currentTimeMillis());
        rowstart=(long)(rand.nextDouble()*((double)rows))+1;
        rowend=(long)(rand.nextDouble()*((double)(rows-rowstart+1)))+rowstart;
        colstart=(long)(rand.nextDouble()*((double)cols))+1;
        colend=(long)(rand.nextDouble()*((double)(cols-colstart+1)))+colstart;
        config.addVariable("rowstart", rowstart);
        config.addVariable("rowend", rowend);
        config.addVariable("colstart", colstart);
        config.addVariable("colend", colend);
        
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String C_HOME = SCRIPT_DIR + TEST_DIR;	
		dmlArgs = new String[]{"-f", C_HOME + "RightIndexingTest" + ".dml",
	               "-args",  C_HOME + INPUT_DIR + "A" , 
	               			Long.toString(rows), Long.toString(cols),
	                        Long.toString(rowstart), Long.toString(rowend),
	                        Long.toString(colstart), Long.toString(colend),
	                         C_HOME + OUTPUT_DIR + "B" , 
	                         C_HOME + OUTPUT_DIR + "C" , 
	                         C_HOME + OUTPUT_DIR + "D" };
		dmlArgsDebug = new String[]{"-f", C_HOME + "RightIndexingTest" + ".dml", "-d",
				 "-args",  C_HOME + INPUT_DIR + "A" , 
		        			Long.toString(rows), Long.toString(cols),
			                 Long.toString(rowstart), Long.toString(rowend),
			                 Long.toString(colstart), Long.toString(colend),
			                  C_HOME + OUTPUT_DIR + "B" , 
			                  C_HOME + OUTPUT_DIR + "C" , 
			                  C_HOME + OUTPUT_DIR + "D" };
		rCmd = "Rscript" + " " + C_HOME + "RightIndexingTest" + ".R" + " " + 
		       C_HOME + INPUT_DIR + " "+rowstart+" "+rowend+" "+colstart+" "+colend+" " + C_HOME + EXPECTED_DIR;

		loadTestConfiguration(config);
		double sparsity=rand.nextDouble();
        double[][] A = getRandomMatrix(rows, cols, min, max, sparsity, System.currentTimeMillis());
        writeInputMatrix("A", A, true);

		/*
		 * Expected number of jobs:
		 * Reblock - 1 job 
		 * While loop iteration - 10 jobs
		 * Final output write - 1 job
		 */
        //boolean exceptionExpected = false;
		//int expectedNumberOfJobs = 12;
		//runTest(exceptionExpected, null, expectedNumberOfJobs);
        boolean exceptionExpected = false;
		int expectedNumberOfJobs = -1;
		runTest(true, exceptionExpected, null, expectedNumberOfJobs);
		
		runRScript(true);
		//disableOutAndExpectedDeletion();
	
		for(String file: config.getOutputFiles())
		{
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS(file);
			HashMap<CellIndex, Double> rfile = readRMatrixFromFS(file);
		//	System.out.println(file+"-DML: "+dmlfile);
		//	System.out.println(file+"-R: "+rfile);
			TestUtils.compareMatrices(dmlfile, rfile, epsilon, file+"-DML", file+"-R");
		}
	}
}
