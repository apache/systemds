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

public class RemoveEmptyRecompileTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "remove_empty_recompile";
	
	private final static String TEST_DIR = "functions/recompile/";
	private final static double eps = 1e-10;
	
	private final static int rows = 20;
	private final static int cols = 20;    
	private final static double sparsity = 1.0;
	
	private enum OpType{
		SUM, //aggregate unary
		ROUND, //unary
		TRANSPOSE, //reorg
		MULT, //binary
		PLUS, //binary
		MM, //aggregate binary
		//RIX, //right indexing
		//LIX, //left indexing
	}
	
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_DIR, TEST_NAME, new String[] { "R" }));
	}

	
	@Test
	public void testRemoveEmptySumNonEmpty() 
	{
		runRemoveEmptyTest(OpType.SUM, false);
	}
	
	@Test
	public void testRemoveEmptyRoundNonEmpty() 
	{
		runRemoveEmptyTest(OpType.ROUND, false);
	}
	
	@Test
	public void testRemoveEmptyTransposeNonEmpty() 
	{
		runRemoveEmptyTest(OpType.TRANSPOSE, false);
	}
	
	@Test
	public void testRemoveEmptyMultNonEmpty() 
	{
		runRemoveEmptyTest(OpType.MULT, false);
	}
	
	@Test
	public void testRemoveEmptyPlusNonEmpty() 
	{
		runRemoveEmptyTest(OpType.PLUS, false);
	}
	
	@Test
	public void testRemoveEmptyMatMultNonEmpty() 
	{
		runRemoveEmptyTest(OpType.MM, false);
	}

	@Test
	public void testRemoveEmptySumEmpty() 
	{
		runRemoveEmptyTest(OpType.SUM, true);
	}
	
	@Test
	public void testRemoveEmptyRoundEmpty() 
	{
		runRemoveEmptyTest(OpType.ROUND, true);
	}
	
	@Test
	public void testRemoveEmptyTransposeEmpty() 
	{
		runRemoveEmptyTest(OpType.TRANSPOSE, true);
	}
	
	@Test
	public void testRemoveEmptyMultEmpty() 
	{
		runRemoveEmptyTest(OpType.MULT, true);
	}
	
	@Test
	public void testRemoveEmptyPlusEmpty() 
	{
		runRemoveEmptyTest(OpType.PLUS, true);
	}
	
	@Test
	public void testRemoveEmptyMatMultEmpty() 
	{
		runRemoveEmptyTest(OpType.MM, true);
	}
	

	/**
	 * 
	 * @param type
	 * @param empty
	 */
	private void runRemoveEmptyTest( OpType type, boolean empty )
	{	
		boolean oldFlagIPA = OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS;
		
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			config.addVariable("rows", rows);
			config.addVariable("cols", cols);
			
			//IPA always disabled to force recompile
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = false;
			
			// This is for running the junit test the new way, i.e., construct the arguments directly
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			//note: stats required for runtime check of rewrite
			programArgs = new String[]{"-stats","-args", HOME + INPUT_DIR + "X" , 
					                        Integer.toString(type.ordinal()),
					                        HOME + OUTPUT_DIR + "R" };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + Integer.toString(type.ordinal()) + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			long seed = System.nanoTime();
	        double[][] X = getRandomMatrix(rows, cols, 0, empty?0:1, sparsity, seed);
			writeInputMatrixWithMTD("X", X, true);
	
			runTest(true, false, null, -1); 
			runRScript(true);
			
			//CHECK compiled MR jobs
			int expectNumCompiled = 9; //reblock, 5xGMR, MMCJ+GMR, write
			Assert.assertEquals("Unexpected number of compiled MR jobs.", 
					            expectNumCompiled, Statistics.getNoOfCompiledMRJobs());
			//CHECK executed MR jobs
			int expectNumExecuted = 0;
			Assert.assertEquals("Unexpected number of executed MR jobs.", 
		                        expectNumExecuted, Statistics.getNoOfExecutedMRJobs());
			
			//CHECK rewrite application
			String opcode = getOpcode(type);
			Assert.assertEquals(empty, !Statistics.getCPHeavyHitterOpCodes().contains(opcode));
			
			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");		
			
		}
		finally
		{
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = oldFlagIPA;
		}
	}
	

	private static String getOpcode( OpType type )
	{
		switch(type){
			case SUM: 		return "uak+";
			case ROUND: 	return "round";
			case TRANSPOSE: return "r'";
			case MULT: 		return "*";
			case PLUS: 		return "+";
			case MM: 		return "ba+*";			
		}
		
		return null;
	}
}