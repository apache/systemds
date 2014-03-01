/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.external;

import java.io.IOException;

import junit.framework.Assert;

import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.utils.Statistics;

public class FunctionExpressionsTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "FunctionExpressions1";
	private final static String TEST_NAME2 = "FunctionExpressions2";
	private final static String TEST_DIR = "functions/external/";
	private final static double eps = 1e-10;
	
	private final static int rows = 12;
	private final static int cols = 11;    
	private final static double sparsity = 0.7;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME1, 
				new TestConfiguration(TEST_DIR, TEST_NAME1, 
				new String[] { "Y" })   ); 
		addTestConfiguration(
				TEST_NAME2, 
				new TestConfiguration(TEST_DIR, TEST_NAME2, 
				new String[] { "Y" })   ); 
	}

	
	@Test
	public void testDMLFunction() 
	{
		runFunctionExpressionsTest( TEST_NAME1 );
	}
	
	@Test
	public void testExternalFunction() 
	{
		runFunctionExpressionsTest( TEST_NAME2 );
	}

	private void runFunctionExpressionsTest( String TEST_NAME )
	{		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", HOME + INPUT_DIR + "X", 
							                Integer.toString(rows),
							                Integer.toString(cols),
				                            HOME + OUTPUT_DIR + "Y" };
		loadTestConfiguration(config);

		try 
		{
			long seed = System.nanoTime();
	        double[][] X = getRandomMatrix(rows, cols, 0, 1, sparsity, seed);
			writeInputMatrix("X", X, false);

			runTest(true, false, null, -1);

			double[][] Y = MapReduceTool.readMatrixFromHDFS(HOME + OUTPUT_DIR + "Y", InputInfo.TextCellInputInfo, rows, cols, 1000,1000);
		
			double sx = sum(X,rows,cols);
			double sy = sum(Y,rows,cols);
			Assert.assertEquals(sx, sy, eps);
			
			Assert.assertEquals("Unexpected number of executed MR jobs.", 
					             0, Statistics.getNoOfExecutedMRJobs());
		} 
		catch (IOException e) 
		{
			e.printStackTrace();
		}
	}
	
	private static double sum( double[][] X, int rows, int cols )
	{
		double sum = 0;
		for( int i=0; i<rows; i++ )
			for( int j=0; j<cols; j++ )
				sum += X[i][j];
		return sum;
	}
}