/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.recompile;

import org.junit.Assert;
import org.junit.Test;

import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.utils.Statistics;

public class RandRecompileTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "rand_recompile"; //scalar values, IPA irrelevant
	private final static String TEST_NAME2 = "rand_recompile2"; //nrow
	private final static String TEST_NAME3 = "rand_recompile3"; //ncol
	private final static String TEST_DIR = "functions/recompile/";
	
	private final static int rows = 200;
	private final static int cols = 200;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME1, 
				new TestConfiguration(TEST_DIR, TEST_NAME1, new String[]{} ));
		addTestConfiguration(
				TEST_NAME2, 
				new TestConfiguration(TEST_DIR, TEST_NAME2, new String[]{} ));
		addTestConfiguration(
				TEST_NAME3, 
				new TestConfiguration(TEST_DIR, TEST_NAME3, new String[]{} ));
	}

	@Test
	public void testRandScalarWithoutRecompile() 
	{
		runRandTest(TEST_NAME1, false, false);
	}
	
	@Test
	public void testRandScalarWithRecompile() 
	{
		runRandTest(TEST_NAME1, true, false);
	}

	@Test
	public void testRandNRowWithoutRecompileWithoutIPA() 
	{
		runRandTest(TEST_NAME2, false, false);
	}
	
	@Test
	public void testRandNRowWithRecompileWithoutIPA() 
	{
		runRandTest(TEST_NAME2, true, false);
	}
	
	@Test
	public void testRandNColWithoutRecompileWithoutIPA() 
	{
		runRandTest(TEST_NAME3, false, false);
	}
	
	@Test
	public void testRandNColWithRecompileWithoutIPA() 
	{
		runRandTest(TEST_NAME3, true, false);
	}

	@Test
	public void testRandNRowWithoutRecompileWithIPA() 
	{
		runRandTest(TEST_NAME2, false, true);
	}
	
	@Test
	public void testRandNRowWithRecompileWithIPA() 
	{
		runRandTest(TEST_NAME2, true, true);
	}
	
	@Test
	public void testRandNColWithoutRecompileWithIPA() 
	{
		runRandTest(TEST_NAME3, false, true);
	}
	
	@Test
	public void testRandNColWithRecompileWithIPA() 
	{
		runRandTest(TEST_NAME3, true, true);
	}
	 
	
	private void runRandTest( String testName, boolean recompile, boolean IPA )
	{	
		boolean oldFlagRecompile = OptimizerUtils.ALLOW_DYN_RECOMPILATION;
		boolean oldFlagIPA = OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS;
		
		boolean oldFlagRand1 = OptimizerUtils.ALLOW_RAND_JOB_RECOMPILE;
		boolean oldFlagRand2 = OptimizerUtils.ALLOW_BRANCH_REMOVAL;
		boolean oldFlagRand3 = OptimizerUtils.ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION;
		
		
		try
		{
			TestConfiguration config = getTestConfiguration(testName);
			config.addVariable("rows", rows);
			config.addVariable("cols", cols);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testName + ".dml";
			programArgs = new String[]{"-args", Integer.toString(rows), Integer.toString(cols) };
			
			loadTestConfiguration(config);
	
			OptimizerUtils.ALLOW_DYN_RECOMPILATION = recompile;
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = IPA;
			
			//disable rand specific recompile
			OptimizerUtils.ALLOW_RAND_JOB_RECOMPILE = false;
			OptimizerUtils.ALLOW_BRANCH_REMOVAL = false;
			OptimizerUtils.ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION = false;
			
			
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			
			//CHECK compiled MR jobs
			int expectNumCompiled = -1;
			if( IPA ) expectNumCompiled = 0; 
			else      expectNumCompiled = 2;//rand, GMR
			Assert.assertEquals("Unexpected number of compiled MR jobs.", 
					            expectNumCompiled, Statistics.getNoOfCompiledMRJobs());
		
			//CHECK executed MR jobs
			int expectNumExecuted = -1;
			if( recompile ) expectNumExecuted = 0;
			else if( IPA )  expectNumExecuted = 0; 
			else            expectNumExecuted = 2; //rand, GMR
			Assert.assertEquals("Unexpected number of executed MR jobs.", 
		                        expectNumExecuted, Statistics.getNoOfExecutedMRJobs());		
		}
		finally
		{
			OptimizerUtils.ALLOW_DYN_RECOMPILATION = oldFlagRecompile;
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = oldFlagIPA;
			
			OptimizerUtils.ALLOW_RAND_JOB_RECOMPILE = oldFlagRand1;
			OptimizerUtils.ALLOW_BRANCH_REMOVAL = oldFlagRand2;
			OptimizerUtils.ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION = oldFlagRand3;
		}
	}
	
}