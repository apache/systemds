/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.append;

import java.util.HashMap;
import java.util.Random;

import junit.framework.Assert;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;
import com.ibm.bi.dml.utils.Statistics;

public class AppendMatrixTest extends AutomatedTestBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "AppendMatrixTest";
	private final static String TEST_DIR = "functions/append/";

	private final static double epsilon=0.0000000001;
	private final static int min=1;
	private final static int max=100;
	
	private final static int rows = 1692;
	//usecase a: inblock single
	private final static int cols1a = 375;
	private final static int cols2a = 92;
	//usecase b: inblock multiple
	private final static int cols1b = 1059;
	private final static int cols2b = 1010;
	//usecase c: outblock blocksize 
	private final static int cols1c = 2*DMLTranslator.DMLBlockSize;
	private final static int cols2c = 1010;
	//usecase d: outblock blocksize 
	private final static int cols1d = 1460;
	private final static int cols2d = 1920;
		
	
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] {"C"}));
	}

	@Test
	public void testAppendInBlock1CP() {
		commonAppendTest(RUNTIME_PLATFORM.SINGLE_NODE, rows, cols1a, cols2a);
	}
	
	//NOTE: different dimension use cases only relvant for MR
	/*
	@Test
	public void testAppendInBlock2CP() {
		commonAppendTest(RUNTIME_PLATFORM.SINGLE_NODE, rows, cols1b, cols2b);
	}
	
	@Test
	public void testAppendOutBlock1CP() {
		commonAppendTest(RUNTIME_PLATFORM.SINGLE_NODE, rows, cols1c, cols2c);
	}	

	@Test
	public void testAppendOutBlock2CP() {
		commonAppendTest(RUNTIME_PLATFORM.SINGLE_NODE, rows, cols1d, cols2d);
	}*/
	
	@Test
	public void testAppendInBlock1MR() {
		commonAppendTest(RUNTIME_PLATFORM.HADOOP, rows, cols1a, cols2a);
	}   
	
	@Test
	public void testAppendInBlock2MR() {
		commonAppendTest(RUNTIME_PLATFORM.HADOOP, rows, cols1b, cols2b);
	}   
	
	@Test
	public void testAppendOutBlock1MR() {
		commonAppendTest(RUNTIME_PLATFORM.HADOOP, rows, cols1c, cols2c);
	}
	
	@Test
	public void testAppendOutBlock2MR() {
		commonAppendTest(RUNTIME_PLATFORM.HADOOP, rows, cols1d, cols2d);
	}
	
	public void commonAppendTest(RUNTIME_PLATFORM platform, int rows, int cols1, int cols2)
	{
		TestConfiguration config = getTestConfiguration(TEST_NAME);
	    
		RUNTIME_PLATFORM prevPlfm=rtplatform;
		
	    rtplatform = platform;

        config.addVariable("rows", rows);
        config.addVariable("cols", cols1);
          
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String RI_HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = RI_HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args",  RI_HOME + INPUT_DIR + "A" , 
				                             Long.toString(rows), 
				                             Long.toString(cols1),
							                 RI_HOME + INPUT_DIR + "B" ,
							                 Long.toString(cols2),
	                                         RI_HOME + OUTPUT_DIR + "C" };
		fullRScriptName = RI_HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       RI_HOME + INPUT_DIR + " "+ RI_HOME + EXPECTED_DIR;

		Random rand=new Random(System.currentTimeMillis());
		loadTestConfiguration(config);
		double sparsity=rand.nextDouble();
		double[][] A = getRandomMatrix(rows, cols1, min, max, sparsity, System.currentTimeMillis());
        writeInputMatrix("A", A, true);
        sparsity=rand.nextDouble();
        double[][] B= getRandomMatrix(rows, cols2, min, max, sparsity, System.currentTimeMillis());
        writeInputMatrix("B", B, true);
        
        boolean exceptionExpected = false;
        int expectedCompiledMRJobs = (rtplatform==RUNTIME_PLATFORM.HADOOP)? 2 : 1;
		int expectedExecutedMRJobs = (rtplatform==RUNTIME_PLATFORM.HADOOP)? 2 : 0;
		runTest(true, exceptionExpected, null, expectedCompiledMRJobs);
		runRScript(true);
		Assert.assertEquals("Wrong number of executed MR jobs.",
				            expectedExecutedMRJobs, Statistics.getNoOfExecutedMRJobs());

		for(String file: config.getOutputFiles())
		{
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS(file);
			HashMap<CellIndex, Double> rfile = readRMatrixFromFS(file);
			TestUtils.compareMatrices(dmlfile, rfile, epsilon, file+"-DML", file+"-R");
		}
		rtplatform = prevPlfm;
	}
   
}
