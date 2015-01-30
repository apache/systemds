/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.recompile;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

/**
 * The main purpose of this test is to ensure that encountered and fixed
 * issues, related to remove empty rewrites (or issues, which showed up due
 * to those rewrites) will never happen again.
 * 
 */
public class RemoveEmptyPotpourriTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "remove_empty_potpourri1";
	private final static String TEST_NAME2 = "remove_empty_potpourri2";
	private final static String TEST_NAME3 = "remove_empty_potpourri3";
	
	private final static String TEST_DIR = "functions/recompile/";
	private final static double eps = 1e-10;
	
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] { "R" }));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] { "R" }));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_DIR, TEST_NAME3, new String[] { "R" }));
	}
	
	@Test
	public void testRemoveEmptySequenceReshapeNoRewrite() 
	{
		runRemoveEmptyTest(TEST_NAME1, false);
	}
	
	@Test
	public void testRemoveEmptySequenceReshapeRewrite() 
	{
		runRemoveEmptyTest(TEST_NAME1, true);
	}
	
	@Test
	public void testRemoveEmptySumColSumNoRewrite() 
	{
		runRemoveEmptyTest(TEST_NAME2, false);
	}
	
	@Test
	public void testRemoveEmptySumColSumRewrite() 
	{
		runRemoveEmptyTest(TEST_NAME2, true);
	}
	
	@Test
	public void testRemoveEmptyComplexDagSplitNoRewrite() 
	{
		runRemoveEmptyTest(TEST_NAME3, false);
	}
	
	
	@Test
	public void testRemoveEmptyComplexDagSplitRewrite() 
	{
		runRemoveEmptyTest(TEST_NAME3, true);
	}

	/**
	 * 
	 * @param type
	 * @param empty
	 */
	private void runRemoveEmptyTest( String TEST_NAME, boolean rewrite )
	{	
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		
		try
		{
			// This is for running the junit test the new way, i.e., construct the arguments directly
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			//note: stats required for runtime check of rewrite
			programArgs = new String[]{"-args", HOME + OUTPUT_DIR + "R" };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			runTest(true, false, null, -1); 
			runRScript(true);
					
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");	
		}
		finally
		{
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}
	}
	

}