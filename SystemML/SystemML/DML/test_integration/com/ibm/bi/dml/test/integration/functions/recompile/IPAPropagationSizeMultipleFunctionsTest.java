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

public class IPAPropagationSizeMultipleFunctionsTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "multiple_function_calls1";
	private final static String TEST_NAME2 = "multiple_function_calls2";
	private final static String TEST_NAME3 = "multiple_function_calls3";
	private final static String TEST_NAME4 = "multiple_function_calls4";
	private final static String TEST_NAME5 = "multiple_function_calls5";
	
	private final static String TEST_DIR = "functions/recompile/";
	
	private final static int rows = 10;
	private final static int cols = 15;    
	private final static double sparsity = 0.7;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME3, new TestConfiguration(TEST_DIR, TEST_NAME3, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME4, new TestConfiguration(TEST_DIR, TEST_NAME4, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME5, new TestConfiguration(TEST_DIR, TEST_NAME5, new String[] { "R" }) );
	}

	
	
	@Test
	public void testFunctionSizePropagationSameInput() 
	{
		runIPASizePropagationMultipleFunctionsTest(TEST_NAME1, false);
	}
	
	@Test
	public void testFunctionSizePropagationEqualDimsUnknownNnzRight() 
	{
		runIPASizePropagationMultipleFunctionsTest(TEST_NAME2, false);
	}
	
	@Test
	public void testFunctionSizePropagationEqualDimsUnknownNnzLeft() 
	{
		runIPASizePropagationMultipleFunctionsTest(TEST_NAME3, false);
	}
	
	@Test
	public void testFunctionSizePropagationEqualDimsUnknownNnz() 
	{
		runIPASizePropagationMultipleFunctionsTest(TEST_NAME4, false);
	}
	
	@Test
	public void testFunctionSizePropagationDifferentDims() 
	{
		runIPASizePropagationMultipleFunctionsTest(TEST_NAME5, false);
	}
	
	@Test
	public void testFunctionSizePropagationSameInputIPA() 
	{
		runIPASizePropagationMultipleFunctionsTest(TEST_NAME1, true);
	}
	
	@Test
	public void testFunctionSizePropagationEqualDimsUnknownNnzRightIPA() 
	{
		runIPASizePropagationMultipleFunctionsTest(TEST_NAME2, true);
	}
	
	@Test
	public void testFunctionSizePropagationEqualDimsUnknownNnzLeftIPA() 
	{
		runIPASizePropagationMultipleFunctionsTest(TEST_NAME3, true);
	}
	
	@Test
	public void testFunctionSizePropagationEqualDimsUnknownNnzIPA() 
	{
		runIPASizePropagationMultipleFunctionsTest(TEST_NAME4, true);
	}
	
	@Test
	public void testFunctionSizePropagationDifferentDimsIPA() 
	{
		runIPASizePropagationMultipleFunctionsTest(TEST_NAME5, true);
	}
	
	

	/**
	 * 
	 * @param condition
	 * @param branchRemoval
	 * @param IPA
	 */
	private void runIPASizePropagationMultipleFunctionsTest( String TEST_NAME, boolean IPA )
	{	
		boolean oldFlagIPA = OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-args", 
												 HOME + INPUT_DIR + "V",
					                             HOME + OUTPUT_DIR + "R" };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
				   HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;			
			
			loadTestConfiguration(config);

			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = IPA;

			//generate input data
			double[][] V = getRandomMatrix(rows, cols, 0, 1, sparsity, 7);
			writeInputMatrixWithMTD("V", V, true);
	
			//run tests
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, 0, "Stat-DML", "Stat-R");
			
			//check expected number of compiled and executed MR jobs
			int expectedNumCompiled = (IPA) ? (TEST_NAME.equals(TEST_NAME5)?3:1) : 4; //reblock, 2xGMR foo, GMR 
			int expectedNumExecuted = 0;			
			
			Assert.assertEquals("Unexpected number of compiled MR jobs.", expectedNumCompiled, Statistics.getNoOfCompiledMRJobs()); 
			Assert.assertEquals("Unexpected number of executed MR jobs.", expectedNumExecuted, Statistics.getNoOfExecutedMRJobs()); 
		}
		finally
		{
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = oldFlagIPA;
		}
	}
	
}
