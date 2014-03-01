/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.external;

import java.io.IOException;

import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class DynReadWriteTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "DynReadWrite";
	private final static String TEST_DIR = "functions/external/";
	private final static double eps = 1e-10;
	
	private final static int rows = 1200;
	private final static int cols = 1100;    
	private final static double sparsity = 0.7;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "Rout" })   ); 
	}

	
	@Test
	public void testTextCell() 
	{
		runDynReadWriteTest("textcell");
	}

	@Test
	public void testBinaryCell() 
	{
		runDynReadWriteTest("binarycell");
	}
	
	@Test
	public void testBinaryBlock() 
	{
		runDynReadWriteTest("binaryblock");
	}
		
	
	private void runDynReadWriteTest( String format )
	{		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", HOME + INPUT_DIR + "X" , 
				                        Integer.toString(rows),
				                        Integer.toString(cols),
				                        format,
				                        HOME + OUTPUT_DIR + "Y" };
		loadTestConfiguration(config);

		try 
		{
			long seed = System.nanoTime();
	        double[][] X = getRandomMatrix(rows, cols, 0, 1, sparsity, seed);
			writeInputMatrix("X", X, false);

			runTest(true, false, null, -1);

			double[][] Y = MapReduceTool.readMatrixFromHDFS(HOME + OUTPUT_DIR + "Y", InputInfo.stringToInputInfo(format), rows, cols, 1000,1000);
		
			TestUtils.compareMatrices(X, Y, rows, cols, eps);
		} 
		catch (IOException e) 
		{
			e.printStackTrace();
		}
	}
}