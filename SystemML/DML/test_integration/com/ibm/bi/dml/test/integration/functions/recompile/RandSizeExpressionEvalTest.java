/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.recompile;

import java.util.HashMap;

import junit.framework.Assert;

import org.junit.Test;

import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.utils.Statistics;

public class RandSizeExpressionEvalTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "rand_size_expr_eval";
	private final static String TEST_DIR = "functions/recompile/";
	
	private final static int rows = 14;
	private final static int cols = 14;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration( TEST_NAME, new TestConfiguration(TEST_DIR, TEST_NAME, new String[]{} ));
	}

	@Test
	public void testComplexRandWithoutEvalExpression() 
	{
		runRandTest(TEST_NAME, false);
	}
	
	@Test
	public void testComplexRandWithEvalExpression() 
	{
		runRandTest(TEST_NAME, true);
	}
	
	private void runRandTest( String testName, boolean evalExpr )
	{	
		boolean oldFlag = OptimizerUtils.ALLOW_SIZE_EXPRESSION_EVALUATION;
		
		try
		{
			TestConfiguration config = getTestConfiguration(testName);
			config.addVariable("rows", rows);
			config.addVariable("cols", cols);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testName + ".dml";
			programArgs = new String[]{"-args", Integer.toString(rows), Integer.toString(cols), HOME + OUTPUT_DIR + "R" };
			
			loadTestConfiguration(config);
	
			OptimizerUtils.ALLOW_SIZE_EXPRESSION_EVALUATION = evalExpr;
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			
			//check correct propagated size via final results
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			Assert.assertEquals("Unexpected results.", rows*cols*3.0, dmlfile.get(new CellIndex(1,1)).doubleValue());
			
			//check expected number of compiled and executed MR jobs
			if( evalExpr )
			{
				Assert.assertEquals("Unexpected number of executed MR jobs.", 
						  0, Statistics.getNoOfExecutedMRJobs());			
			}
			else
			{
				Assert.assertEquals("Unexpected number of executed MR jobs.", 
						            2, Statistics.getNoOfExecutedMRJobs()); //Rand, GMR (sum)
			}		
		}
		finally
		{
			OptimizerUtils.ALLOW_SIZE_EXPRESSION_EVALUATION = oldFlag;
		}
	}
	
}