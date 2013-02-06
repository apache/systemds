package com.ibm.bi.dml.test.integration.functions.recompile;

import junit.framework.Assert;

import org.junit.Test;

import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.utils.Statistics;

public class RandRecompileTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "rand_recompile";
	private final static String TEST_DIR = "functions/recompile/";
	
	private final static int rows = 200;   
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME1, 
				new TestConfiguration(TEST_DIR, TEST_NAME1, 
				new String[] { "Rout" })   );
	}

	@Test
	public void testRandWithoutRecompile() 
	{
		runRandTest(false);
	}
	
	@Test
	public void testRandWithRecompile() 
	{
		runRandTest(true);
	}


	
	private void runRandTest( boolean recompile )
	{	
		boolean oldFlag = OptimizerUtils.ALLOW_DYN_RECOMPILATION;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME1);
			config.addVariable("rows", rows);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[]{"-args", Integer.toString(rows) };
			
			loadTestConfiguration(config);
	
			OptimizerUtils.ALLOW_DYN_RECOMPILATION = recompile;
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			
			//check expected number of compiled and executed MR jobs
			if( recompile )
			{
				Assert.assertEquals("Unexpected number of executed MR jobs.", 
						  0, Statistics.getNoOfExecutedMRJobs());			
			}
			else
			{
				Assert.assertEquals("Unexpected number of executed MR jobs.", 
						            2, Statistics.getNoOfExecutedMRJobs()); //rand, GMR
			}		
		}
		finally
		{
			OptimizerUtils.ALLOW_DYN_RECOMPILATION = oldFlag;
		}
	}
	
}