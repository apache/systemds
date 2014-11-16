/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.misc;

import junit.framework.Assert;

import org.junit.Test;

import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.utils.Statistics;

public class FunctionInliningTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_DIR = "functions/misc/";
	private final static String TEST_NAME1 = "function_chain_inlining";
	private final static String TEST_NAME2 = "function_chain_non_inlining";
	private final static String TEST_NAME3 = "function_recursive_inlining";
	
	private final static long rows = 3400;
	private final static long cols = 2700;
	private final static double val = 1.0;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] { "Rout" })   );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] { "Rout" })   );
		addTestConfiguration( TEST_NAME3, new TestConfiguration(TEST_DIR, TEST_NAME3, new String[] { "Rout" })   );
		
	}

	@Test
	public void testChainInliningIPA() 
	{
		runInliningTest(TEST_NAME1, true);
	}
	
	@Test
	public void testChainNoInliningIPA() 
	{
		runInliningTest(TEST_NAME2, true);
	}
	
	@Test
	public void testRecursiveInliningIPA() 
	{
		runInliningTest(TEST_NAME3, true);
	}
	
	@Test
	public void testChainInliningNoIPA() 
	{
		runInliningTest(TEST_NAME1, false);
	}
	
	@Test
	public void testChainNoInliningNoIPA() 
	{
		runInliningTest(TEST_NAME2, false);
	}
	
	@Test
	public void testRecursiveInliningNoIPA() 
	{
		runInliningTest(TEST_NAME3, false);
	}

	/**
	 * 
	 * @param testname
	 * @param IPA
	 */
	private void runInliningTest( String testname, boolean IPA )
	{	
		boolean oldIPA = OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS;
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{/*"-explain",*/"-args",String.valueOf(rows),
					                           String.valueOf(cols),
					                           String.valueOf(val),
					                           HOME + OUTPUT_DIR + "Rout" };
			loadTestConfiguration(config);

			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = IPA;
			
			//run testcase
			runTest(true, false, null, -1); 
			
			//compare output
			double ret = MapReduceTool.readDoubleFromHDFSFile(HOME + OUTPUT_DIR + "Rout");
			Assert.assertEquals(rows*cols*val*6, ret);
			
			
			//compiled MR jobs
			int expectNumCompiled = IPA ? 0 : (testname.equals(TEST_NAME1)?2: //2GMR in foo1 and foo2 (not removed w/o IPA)
				                               (testname.equals(TEST_NAME2)?4: //3GMR in foo1 and foo2, 1GMR for subsequent sum  
				                            	5 )); //5GMR in foo1-foo5 (not removed w/o IPA)			
			Assert.assertEquals("Unexpected number of compiled MR jobs.", 
					            expectNumCompiled, Statistics.getNoOfCompiledMRJobs());
		
			//check executed MR jobs
			int expectNumExecuted = 0; //executed jobs should always be 0 due to dynamic recompilation
			Assert.assertEquals("Unexpected number of executed MR jobs.", expectNumExecuted, Statistics.getNoOfExecutedMRJobs());
		}
		catch(Exception ex)
		{
			Assert.fail("Failed to run test: "+ex.getMessage());
		}
		finally
		{
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = oldIPA;
		}
	}
	
}