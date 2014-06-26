/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.recompile;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;
import com.ibm.bi.dml.utils.Statistics;

/**
 * INTERESTING NOTE: see MINUS_RIGHT; if '(X+1)-X' instead of '(X+2)-X'
 * R's writeMM returns (and hence the test fails)
 *   - MatrixMarket matrix coordinate pattern symmetric
 * instead of 
 *   - MatrixMarket matrix coordinate integer symmetric 
 * 
 */
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
		MULT_LEFT, //binary, left empty
		MULT_RIGHT, //binary, right empty
		PLUS_LEFT, //binary, left empty
		PLUS_RIGHT, //binary, right empty
		MINUS_LEFT, //binary, left empty
		MINUS_RIGHT, //binary, right empty
		MM_LEFT, //aggregate binary, left empty
		MM_RIGHT, //aggregate binary, right empty
		RIX, //right indexing
		LIX, //left indexing
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
	public void testRemoveEmptyMultLeftNonEmpty() 
	{
		runRemoveEmptyTest(OpType.MULT_LEFT, false);
	}
	
	@Test
	public void testRemoveEmptyMultRightNonEmpty() 
	{
		runRemoveEmptyTest(OpType.MULT_RIGHT, false);
	}
	
	@Test
	public void testRemoveEmptyPlusLeftNonEmpty() 
	{
		runRemoveEmptyTest(OpType.PLUS_LEFT, false);
	}
	
	@Test
	public void testRemoveEmptyPlusRightNonEmpty() 
	{
		runRemoveEmptyTest(OpType.PLUS_RIGHT, false);
	}
	
	@Test
	public void testRemoveEmptyMinusLeftNonEmpty() 
	{
		runRemoveEmptyTest(OpType.MINUS_LEFT, false);
	}
	
	@Test
	public void testRemoveEmptyMinusRightNonEmpty() 
	{
		runRemoveEmptyTest(OpType.MINUS_RIGHT, false);
	}
	
	@Test
	public void testRemoveEmptyMatMultLeftNonEmpty() 
	{
		runRemoveEmptyTest(OpType.MM_LEFT, false);
	}
	
	@Test
	public void testRemoveEmptyMatMultRightNonEmpty() 
	{
		runRemoveEmptyTest(OpType.MM_RIGHT, false);
	}
	
	@Test
	public void testRemoveEmptyRIXNonEmpty() 
	{
		runRemoveEmptyTest(OpType.RIX, false);
	}
	
	@Test
	public void testRemoveEmptyLIXNonEmpty() 
	{
		runRemoveEmptyTest(OpType.LIX, false);
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
	public void testRemoveEmptyMultLeftEmpty() 
	{
		runRemoveEmptyTest(OpType.MULT_LEFT, true);
	}
	
	@Test
	public void testRemoveEmptyMultRightEmpty() 
	{
		runRemoveEmptyTest(OpType.MULT_RIGHT, true);
	}
	
	@Test
	public void testRemoveEmptyPlusLeftEmpty() 
	{
		runRemoveEmptyTest(OpType.PLUS_LEFT, true);
	}
	
	@Test
	public void testRemoveEmptyPlusRightEmpty() 
	{
		runRemoveEmptyTest(OpType.PLUS_RIGHT, true);
	}
	
	@Test
	public void testRemoveEmptyMinusLeftEmpty() 
	{
		runRemoveEmptyTest(OpType.MINUS_LEFT, true);
	}
	
	@Test
	public void testRemoveEmptyMinusRightEmpty() 
	{
		runRemoveEmptyTest(OpType.MINUS_RIGHT, true);
	}
	
	@Test
	public void testRemoveEmptyMatMultLeftEmpty() 
	{
		runRemoveEmptyTest(OpType.MM_LEFT, true);
	}
	
	@Test
	public void testRemoveEmptyMatMultRightEmpty() 
	{
		runRemoveEmptyTest(OpType.MM_RIGHT, true);
	}
	
	@Test
	public void testRemoveEmptyRIXEmpty() 
	{
		runRemoveEmptyTest(OpType.RIX, true);
	}
	
	@Test
	public void testRemoveEmptyLIXEmpty() 
	{
		runRemoveEmptyTest(OpType.LIX, true);
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
			programArgs = new String[]{"-explain","-stats","-args", HOME + INPUT_DIR + "X" , 
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
			int expectNumCompiled = 18; //reblock, 10xGMR, 2x(MMCJ+GMR), 2xGMR(LIX), write
			Assert.assertEquals("Unexpected number of compiled MR jobs.", 
					            expectNumCompiled, Statistics.getNoOfCompiledMRJobs());
			//CHECK executed MR jobs
			int expectNumExecuted = 0;
			Assert.assertEquals("Unexpected number of executed MR jobs.", 
		                        expectNumExecuted, Statistics.getNoOfExecutedMRJobs());
			
			//CHECK rewrite application 
			//(for minus_left we replace X-Y with 0-Y and hence still execute -)
			if( type != OpType.MINUS_LEFT ){
				String opcode = getOpcode(type);
				Assert.assertEquals(empty, !Statistics.getCPHeavyHitterOpCodes().contains(opcode));
			}
			
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
			case SUM: 		  return "uak+";
			case ROUND: 	  return "round";
			case TRANSPOSE:   return "r'";
			case MULT_LEFT:
			case MULT_RIGHT:  return "*";
			case PLUS_LEFT:
			case PLUS_RIGHT:  return "+";
			case MINUS_LEFT:
			case MINUS_RIGHT: return "-";
			case MM_LEFT:
			case MM_RIGHT: 	  return "ba+*";		
			case RIX:		  return "rangeReIndex";
			case LIX:		  return "leftIndex";
		}
		
		return null;
	}
}