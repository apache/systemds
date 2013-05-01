package com.ibm.bi.dml.test.integration.functions.recompile;

import junit.framework.Assert;

import org.junit.Test;

import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.utils.Statistics;

public class RandRecompileTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "rand_recompile"; //scalar values
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
		runRandTest(TEST_NAME1, false);
	}
	
	@Test
	public void testRandScalarWithRecompile() 
	{
		runRandTest(TEST_NAME1, true);
	}

	@Test
	public void testRandNRowWithoutRecompile() 
	{
		runRandTest(TEST_NAME2, false);
	}
	
	@Test
	public void testRandNRowWithRecompile() 
	{
		runRandTest(TEST_NAME2, true);
	}
	
	@Test
	public void testRandNColWithoutRecompile() 
	{
		runRandTest(TEST_NAME3, false);
	}
	
	@Test
	public void testRandNColWithRecompile() 
	{
		runRandTest(TEST_NAME3, true);
	}


	
	private void runRandTest( String testName, boolean recompile )
	{	
		boolean oldFlag = OptimizerUtils.ALLOW_DYN_RECOMPILATION;
		
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