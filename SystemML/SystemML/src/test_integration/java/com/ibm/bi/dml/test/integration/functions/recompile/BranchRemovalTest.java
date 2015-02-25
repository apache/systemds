/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
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

public class BranchRemovalTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "if_branch_removal";
	private final static String TEST_DIR = "functions/recompile/";
	
	private final static int rows = 10;
	private final static int cols = 15;    
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "X" })   );
	}

	
	
	@Test
	public void testTrueConditionNoBranchRemovalNoIPA() 
	{
		runBranchRemovalTest(true, false, false);
	}
	
	@Test
	public void testFalseConditionNoBranchRemovalNoIPA() 
	{
		runBranchRemovalTest(false, false, false);
	}
	
	@Test
	public void testTrueConditionBranchRemovalNoIPA() 
	{
		runBranchRemovalTest(true, true, false);
	}
	
	@Test
	public void testFalseConditionBranchRemovalNoIPA() 
	{
		runBranchRemovalTest(false, true, false);
	}
	
	@Test
	public void testTrueConditionNoBranchRemovalIPA() 
	{
		runBranchRemovalTest(true, false, true);
	}
	
	@Test
	public void testFalseConditionNoBranchRemovalIPA() 
	{
		runBranchRemovalTest(false, false, true);
	}
	
	@Test
	public void testTrueConditionBranchRemovalIPA() 
	{
		runBranchRemovalTest(true, true, true);
	}
	
	@Test
	public void testFalseConditionBranchRemovalIPA() 
	{
		runBranchRemovalTest(false, true, true);
	}

	/**
	 * 
	 * @param condition
	 * @param branchRemoval
	 * @param IPA
	 */
	private void runBranchRemovalTest( boolean condition, boolean branchRemoval, boolean IPA )
	{	
		boolean oldFlagBranchRemoval = OptimizerUtils.ALLOW_BRANCH_REMOVAL;
		boolean oldFlagIPA = OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS;
		
		//boolean oldFlagRand1 = OptimizerUtils.ALLOW_RAND_JOB_RECOMPILE;
		//boolean oldFlagRand3 = OptimizerUtils.ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION;
		
		
		int val = (condition?1:0);
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args",HOME + INPUT_DIR + "X",
					                           Integer.toString(rows),
					                           Integer.toString(cols),
					                           Integer.toString(val),
					                           HOME + OUTPUT_DIR + "X" };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + val + " " + HOME + EXPECTED_DIR;			
			loadTestConfiguration(config);

			OptimizerUtils.ALLOW_BRANCH_REMOVAL = branchRemoval;
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = IPA;
			
			//disable rand specific recompile
			//OptimizerUtils.ALLOW_RAND_JOB_RECOMPILE = false;
			//OptimizerUtils.ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION = false;

			
			double[][] V = getRandomMatrix(rows, cols, -1, 1, 1.0d, 7);
			writeInputMatrix("X", V, true);
			
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("X");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("X");
			TestUtils.compareMatrices(dmlfile, rfile, 0, "Stat-DML", "Stat-R");
			
			//check expected number of compiled and executed MR jobs
			int expectedNumCompiled = 5; //reblock, 3xGMR (append), write
			int expectedNumExecuted = 0;			
			if( branchRemoval && IPA )
				expectedNumCompiled = 1; //reblock
			else if( branchRemoval ){
				if( condition ) expectedNumCompiled = 4; //reblock, 2xGMR (append), write
				else            expectedNumCompiled = 3; //reblock, GMR (append), write
			}
				
			
			Assert.assertEquals("Unexpected number of compiled MR jobs.", expectedNumCompiled, Statistics.getNoOfCompiledMRJobs()); 
			Assert.assertEquals("Unexpected number of executed MR jobs.", expectedNumExecuted, Statistics.getNoOfExecutedMRJobs()); 
		}
		finally
		{
			OptimizerUtils.ALLOW_BRANCH_REMOVAL = oldFlagBranchRemoval;
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = oldFlagIPA;
			

			//OptimizerUtils.ALLOW_RAND_JOB_RECOMPILE = oldFlagRand1;
			//OptimizerUtils.ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION = oldFlagRand3;
		}
	}
	
}