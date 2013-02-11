package com.ibm.bi.dml.test.integration.functions.recompile;

import java.util.HashMap;

import junit.framework.Assert;

import org.junit.Test;

import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.utils.Statistics;

public class PredicateRecompileTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "while_recompile";
	private final static String TEST_NAME2 = "if_recompile";
	private final static String TEST_NAME3 = "for_recompile";
	private final static String TEST_NAME4 = "parfor_recompile";
	private final static String TEST_DIR = "functions/recompile/";
	
	private final static int rows = 10;
	private final static int cols = 15;    
	private final static int val = 7;    
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME1, 
				new TestConfiguration(TEST_DIR, TEST_NAME1, 
				new String[] { "Rout" })   );
		addTestConfiguration(
				TEST_NAME2, 
				new TestConfiguration(TEST_DIR, TEST_NAME2, 
				new String[] { "Rout" })   );
		addTestConfiguration(
				TEST_NAME3, 
				new TestConfiguration(TEST_DIR, TEST_NAME3, 
				new String[] { "Rout" })   );
		addTestConfiguration(
				TEST_NAME4, 
				new TestConfiguration(TEST_DIR, TEST_NAME4, 
				new String[] { "Rout" })   );
	}

	@Test
	public void testWhileRecompile() 
	{
		runRecompileTest(TEST_NAME1, true);
	}
	
	@Test
	public void testWhileNoRecompile() 
	{
		runRecompileTest(TEST_NAME1, false);
	}
	
	@Test
	public void testIfRecompile() 
	{
		runRecompileTest(TEST_NAME2, true);
	}
	
	@Test
	public void testIfNoRecompile() 
	{
		runRecompileTest(TEST_NAME2, false);
	}
	
	@Test
	public void testForRecompile() 
	{
		runRecompileTest(TEST_NAME3, true);
	}
	
	@Test
	public void testForNoRecompile() 
	{
		runRecompileTest(TEST_NAME3, false);
	}
	
	@Test
	public void testParForRecompile() 
	{
		runRecompileTest(TEST_NAME4, true);
	}
	
	@Test
	public void testParForNoRecompile() 
	{
		runRecompileTest(TEST_NAME4, false);
	}

	
	private void runRecompileTest( String testname, boolean recompile )
	{	
		boolean oldFlag = OptimizerUtils.ALLOW_DYN_RECOMPILATION;
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-args",Integer.toString(rows),
					                           Integer.toString(cols),
					                           Integer.toString(val),
					                           HOME + OUTPUT_DIR + "R" };
			fullRScriptName = HOME + testname + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;			
			loadTestConfiguration(config);

			OptimizerUtils.ALLOW_DYN_RECOMPILATION = recompile;
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			
			//check expected number of compiled and executed MR jobs
			if( recompile )
			{
				Assert.assertEquals("Unexpected number of executed MR jobs.", 
						  1, Statistics.getNoOfExecutedMRJobs()); //rand	
			}
			else
			{
				if( testname.equals(TEST_NAME1) )
					Assert.assertEquals("Unexpected number of executed MR jobs.", 
				            4, Statistics.getNoOfExecutedMRJobs()); //rand, 2xgmr while pred, 1x gmr while body				
				else //if( testname.equals(TEST_NAME2) )
					Assert.assertEquals("Unexpected number of executed MR jobs.", 
				            3, Statistics.getNoOfExecutedMRJobs()); //rand, 1xgmr if pred, 1x gmr if body
			}
			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			Assert.assertEquals((double)val, dmlfile.get(new CellIndex(1,1)));
		}
		finally
		{
			OptimizerUtils.ALLOW_DYN_RECOMPILATION = oldFlag;
		}
	}
	
}