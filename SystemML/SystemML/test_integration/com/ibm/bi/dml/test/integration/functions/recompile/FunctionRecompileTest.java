/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
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

public class FunctionRecompileTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "funct_recompile";
	private final static String TEST_DIR = "functions/recompile/";
	private final static double eps = 1e-10;
	
	private final static int rows = 20;
	private final static int cols = 10;    
	private final static double sparsity = 1.0;
	
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(
				TEST_NAME1, 
				new TestConfiguration(TEST_DIR, TEST_NAME1, 
				new String[] { "Rout" })   );
	}

	@Test
	public void testFunctionWithoutRecompileWithoutIPA() 
	{
		runFunctionTest(false, false);
	}
	
	@Test
	public void testFunctionWithoutRecompileWithIPA() 
	{
		runFunctionTest(false, true);
	}
	

	@Test
	public void testFunctionWithRecompileWithoutIPA() 
	{
		runFunctionTest(true, false);
	}
	
	@Test
	public void testFunctionWithRecompileWithIPA() 
	{
		runFunctionTest(true, true);
	}
	

	private void runFunctionTest( boolean recompile, boolean IPA )
	{	
		boolean oldFlagRecompile = OptimizerUtils.ALLOW_DYN_RECOMPILATION;
		boolean oldFlagIPA = OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME1);
			config.addVariable("rows", rows);
			config.addVariable("cols", cols);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "V" , 
					                        Integer.toString(rows),
					                        Integer.toString(cols),
					                        HOME + OUTPUT_DIR + "R" };
			fullRScriptName = HOME + TEST_NAME1 + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			long seed = System.nanoTime();
	        double[][] V = getRandomMatrix(rows, cols, 0, 1, sparsity, seed);
			writeInputMatrix("V", V, true);
	
			OptimizerUtils.ALLOW_DYN_RECOMPILATION = recompile;
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = IPA;
			
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			runRScript(true);
			
			//note: change from previous version due to fix in op selection (unknown size XtX and mapmult)
			//CHECK compiled MR jobs
			int expectNumCompiled = -1;
			if( IPA ) expectNumCompiled = 1; //reblock (with recompile right indexing); before: 3 reblock, GMR,GMR 
			else      expectNumCompiled = 5;//reblock, GMR,GMR,GMR,GMR (last two should piggybacked)
			Assert.assertEquals("Unexpected number of compiled MR jobs.", 
					            expectNumCompiled, Statistics.getNoOfCompiledMRJobs());
		
			//CHECK executed MR jobs
			int expectNumExecuted = -1;
			if( recompile ) expectNumExecuted = 0;
			else if( IPA )  expectNumExecuted = 1; //reblock (with recompile right indexing); before: 21 reblock, 10*(GMR,GMR)
			else            expectNumExecuted = 41; //reblock, 10*(GMR,GMR,GMR, GMR) (last two should piggybacked)
			Assert.assertEquals("Unexpected number of executed MR jobs.", 
		                        expectNumExecuted, Statistics.getNoOfExecutedMRJobs());
			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("Rout");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");			
		}
		finally
		{
			OptimizerUtils.ALLOW_DYN_RECOMPILATION = oldFlagRecompile;
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = oldFlagIPA;
		}
	}
	
}