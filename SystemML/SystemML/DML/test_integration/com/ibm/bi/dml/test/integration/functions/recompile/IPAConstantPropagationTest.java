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
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;
import com.ibm.bi.dml.utils.Statistics;

public class IPAConstantPropagationTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "constant_propagation_if";
	private final static String TEST_NAME2 = "constant_propagation_while";
	private final static String TEST_DIR = "functions/recompile/";
	
	private final static int rows = 10;
	private final static int cols = 15;    
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] { "X" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] { "X" }) );
	}

	
	
	@Test
	public void testConstantPropagationNoUpdateNoBranchRemovalNoIPA() 
	{
		runIPAConstantPropagationTest(false, false, false);
	}
	
	@Test
	public void testConstantPropagationNoUpdateNoBranchRemovalIPA() 
	{
		runIPAConstantPropagationTest(false, false, true);
	}
	
	@Test
	public void testConstantPropagationNoUpdateBranchRemovalNoIPA() 
	{
		runIPAConstantPropagationTest(false, true, false);
	}
	
	@Test
	public void testConstantPropagationNoUpdateBranchRemovalIPA() 
	{
		runIPAConstantPropagationTest(false, true, true);
	}
	
	@Test
	public void testConstantPropagationUpdateNoBranchRemovalNoIPA() 
	{
		runIPAConstantPropagationTest(true, false, false);
	}
	
	@Test
	public void testConstantPropagationUpdateNoBranchRemovalIPA() 
	{
		runIPAConstantPropagationTest(true, false, true);
	}
	
	@Test
	public void testConstantPropagationUpdateBranchRemovalNoIPA() 
	{
		runIPAConstantPropagationTest(true, true, false);
	}
	
	@Test
	public void testConstantPropagationUpdateBranchRemovalIPA() 
	{
		runIPAConstantPropagationTest(true, true, true);
	}

	/**
	 * 
	 * @param condition
	 * @param branchRemoval
	 * @param IPA
	 */
	private void runIPAConstantPropagationTest( boolean update, boolean branchRemoval, boolean IPA )
	{	
		boolean oldFlagBranchRemoval = OptimizerUtils.ALLOW_BRANCH_REMOVAL;
		boolean oldFlagIPA = OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS;
		
		String TEST_NAME = update ? TEST_NAME2 : TEST_NAME1;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-args", Integer.toString(rows),
					                            Integer.toString(cols),
					                            HOME + OUTPUT_DIR + "X" };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
				   Integer.toString(rows) + " " + Integer.toString(cols) + " " +
			       HOME + EXPECTED_DIR;			
			loadTestConfiguration(config);

			OptimizerUtils.ALLOW_BRANCH_REMOVAL = branchRemoval;
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = IPA;
			
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("X");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("X");
			TestUtils.compareMatrices(dmlfile, rfile, 0, "Stat-DML", "Stat-R");
			
			//check expected number of compiled and executed MR jobs
			int expectedNumCompiled = ( branchRemoval && IPA && !update ) ? 0 : 1; //rand
			int expectedNumExecuted = 0;			
			
			Assert.assertEquals("Unexpected number of compiled MR jobs.", expectedNumCompiled, Statistics.getNoOfCompiledMRJobs()); 
			Assert.assertEquals("Unexpected number of executed MR jobs.", expectedNumExecuted, Statistics.getNoOfExecutedMRJobs()); 
		}
		finally
		{
			OptimizerUtils.ALLOW_BRANCH_REMOVAL = oldFlagBranchRemoval;
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = oldFlagIPA;
		}
	}
	
}
