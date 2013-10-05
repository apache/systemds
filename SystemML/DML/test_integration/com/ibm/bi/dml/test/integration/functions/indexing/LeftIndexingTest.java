/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.indexing;

import java.util.HashMap;
import java.util.Random;

import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class LeftIndexingTest extends AutomatedTestBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_DIR = "functions/indexing/";

	private final static double epsilon=0.0000000001;
	private final static int rows = 2279;
	private final static int cols = 1050;
	private final static int min=0;
	private final static int max=100;
	
	@Override
	public void setUp() {
		addTestConfiguration("LeftIndexingTest", new TestConfiguration(TEST_DIR, "LeftIndexingTest", 
				new String[] {"AB", "AC", "AD"}));
	}
	@Test
	public void testLeftIndexing() {
	    TestConfiguration config = getTestConfiguration("LeftIndexingTest");
	    
	    
        config.addVariable("rows", rows);
        config.addVariable("cols", cols);
        
        long rowstart=816, rowend=1229, colstart=967, colend=1009;
      //  long rowstart=2, rowend=4, colstart=9, colend=10;
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
		String LI_HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = LI_HOME + "LeftIndexingTest" + ".dml";
		programArgs = new String[]{"-args",  LI_HOME + INPUT_DIR + "A" , 
	               			Long.toString(rows), Long.toString(cols),
	                        Long.toString(rowstart), Long.toString(rowend),
	                        Long.toString(colstart), Long.toString(colend),
	                        LI_HOME + OUTPUT_DIR + "AB" , 
	                         LI_HOME + OUTPUT_DIR + "AC" , 
	                         LI_HOME + OUTPUT_DIR + "AD",
	                         LI_HOME + INPUT_DIR + "B" , 
	                         LI_HOME + INPUT_DIR + "C" , 
	                         LI_HOME + INPUT_DIR + "D",
	                         Long.toString(rowend-rowstart+1), 
	                         Long.toString(colend-colstart+1),
		                     Long.toString(cols-colstart+1)};
		fullRScriptName = LI_HOME + "LeftIndexingTest" + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       LI_HOME + INPUT_DIR + " "+rowstart+" "+rowend+" "+colstart+" "+colend+" " + LI_HOME + EXPECTED_DIR;

		loadTestConfiguration(config);
		double sparsity=rand.nextDouble();
        double[][] A = getRandomMatrix(rows, cols, min, max, sparsity, System.currentTimeMillis());
        writeInputMatrix("A", A, true);
        
        sparsity=rand.nextDouble();
        double[][] B = getRandomMatrix((int)(rowend-rowstart+1), (int)(colend-colstart+1), min, max, sparsity, System.currentTimeMillis());
        writeInputMatrix("B", B, true);
        
        sparsity=rand.nextDouble();
        double[][] C = getRandomMatrix((int)(rowend), (int)(cols-colstart+1), min, max, sparsity, System.currentTimeMillis());
        writeInputMatrix("C", C, true);
        
        sparsity=rand.nextDouble();
        double[][] D = getRandomMatrix(rows, (int)(colend-colstart+1), min, max, sparsity, System.currentTimeMillis());
        writeInputMatrix("D", D, true);

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

