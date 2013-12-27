/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.external;

import java.util.HashMap;

import junit.framework.Assert;

import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;
import com.ibm.bi.dml.utils.Statistics;

public class OrderTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "Order";
	private final static String TEST_DIR = "functions/external/";
	
	private final static int rows = 1200;
	private final static int cols = 1100; 
	private final static int sc = 1; 

	private final static double sparsity = 0.7;
	private final static double eps = 1e-10;
	
	@Override
	public void setUp() {
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "B.mtx" })   ); 
	}

	@Test
	public void testOrderAsc()
	{
		runtestOrder( true ); 
	}
	
	@Test
	public void testOrderDesc()
	{
		runtestOrder( false ); 
	}
	
	private void runtestOrder( boolean asc ) 
	{	
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		int sortcol = sc * (asc ? 1 : -1);
		int namesuffix = (asc ? 1 : 2);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + namesuffix + ".dml";
		programArgs = new String[]{"-args", HOME + INPUT_DIR + "A" , 
				                        Integer.toString(rows),
				                        Integer.toString(cols),
				                        Integer.toString(sc),
				                        HOME + OUTPUT_DIR + "B" };
		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       HOME + INPUT_DIR + " " + sortcol + " " + HOME + EXPECTED_DIR;
		
		loadTestConfiguration(config);

		try 
		{
			long seed = System.nanoTime();
	        double[][] A = getRandomMatrix(rows, cols, 0, 1, sparsity, seed);
			writeInputMatrix("A", A, true);
			
			runTest(true, false, null, -1);
			runRScript(true);
			
			//check number of compiled and executed scripts (assumes IPA and recompile)
			Assert.assertEquals("Unexpected number of compiled MR jobs.", 1, Statistics.getNoOfCompiledMRJobs()); //reblock
			Assert.assertEquals("Unexpected number of executed MR jobs.", 0, Statistics.getNoOfExecutedMRJobs());
			
			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("B");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("B.mtx");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
		}
	}
}
