/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.parfor;

import java.util.HashMap;

import junit.framework.Assert;

import org.junit.Test;

import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class ParForReplaceThreadIDRecompileTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_DIR = "functions/parfor/";
	private final static String TEST_NAME1 = "parfor_threadid_recompile1"; //for
	private final static String TEST_NAME2 = "parfor_threadid_recompile2"; //parfor 
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] { "B" }));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] { "B" }));
	}

	@Test
	public void testThreadIDReplaceForRecompile() 
	{
		runThreadIDReplaceTest(TEST_NAME1, true);
	}
	
	@Test
	public void testThreadIDReplaceParForRecompile() 
	{
		runThreadIDReplaceTest(TEST_NAME2, true);
	}
	
	
	/**
	 * 
	 * @param TEST_NAME
	 * @param recompile
	 */
	private void runThreadIDReplaceTest( String TEST_NAME, boolean recompile )
	{
		boolean flag = OptimizerUtils.ALLOW_DYN_RECOMPILATION;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			OptimizerUtils.ALLOW_DYN_RECOMPILATION = recompile;
			
			// This is for running the junit test the new way, i.e., construct the arguments directly 
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "A" , 
												HOME + OUTPUT_DIR + "B" };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	        double[][] A = new double[][]{{2.0},{3.0}};
			writeInputMatrixWithMTD("A", A, false);
	
			runTest(true, false, null, -1);
			
			//compare matrices
			HashMap<CellIndex, Double> dmlout = readDMLMatrixFromHDFS("B");
			Assert.assertTrue( dmlout.size()>=1 );
		}
		finally{
			OptimizerUtils.ALLOW_DYN_RECOMPILATION = flag;
		}
	}
	
}