/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.recompile;

import java.util.HashMap;

import junit.framework.Assert;

import org.junit.Test;

import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;
import com.ibm.bi.dml.utils.Statistics;

public class RandJobRecompileTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "grpagg_rand_recompile";
	private final static String TEST_DIR = "functions/recompile/";
	
	private final static int rows = 27;
	private final static int cols = 2; 
	
	private final static int min = 1;
	private final static int max = 4;
	
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "Z" })   );
	}

	
	
	@Test
	public void testRandRecompileNoEstSizeEval() 
	{
		runRandJobRecompileTest(false);
	}
	
	@Test
	public void testRandRecompilEstSizeEval() 
	{
		runRandJobRecompileTest(true);
	}

	/**
	 * 
	 * @param estSizeEval
	 */
	private void runRandJobRecompileTest( boolean estSizeEval )
	{	
		boolean oldFlagSizeEval = OptimizerUtils.ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args",HOME + INPUT_DIR + "X",
					                           Integer.toString(rows),
					                           HOME + OUTPUT_DIR + "Z" };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;			
			loadTestConfiguration(config);

			OptimizerUtils.ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION = estSizeEval;
			
			double[][] V = round( getRandomMatrix(rows, cols, min, max, 1.0d, 7) );
			writeInputMatrix("X", V, true);
			
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("Z");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("Z");
			TestUtils.compareMatrices(dmlfile, rfile, 0, "Stat-DML", "Stat-R");
			
			//check expected number of compiled and executed MR jobs
			int expectedNumCompiled = (estSizeEval?1:2); //rand, write
			int expectedNumExecuted = 0;			
			
			checkNumCompiledMRJobs(expectedNumCompiled); 
			checkNumExecutedMRJobs(expectedNumExecuted); 
		}
		finally
		{
			OptimizerUtils.ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION = oldFlagSizeEval;
		}
	}
	
	/**
	 * 
	 * @param data
	 * @return
	 */
	private double[][] round( double[][] data )
	{
		for( int i=0; i<data.length; i++ )
			for( int j=0; j<data[i].length; j++ )
				data[i][j] = Math.round(data[i][j]);
		return data;
	}
}