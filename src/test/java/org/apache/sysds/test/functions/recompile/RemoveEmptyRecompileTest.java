/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.test.functions.recompile;

import java.util.HashMap;

import org.apache.sysds.common.Opcodes;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

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
	private final static String TEST_NAME = "remove_empty_recompile";
	
	private final static String TEST_DIR = "functions/recompile/";
	private final static String TEST_CLASS_DIR = TEST_DIR + RemoveEmptyRecompileTest.class.getSimpleName() + "/";
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
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "R" }));
	}

	
	@Test
	public void testRemoveEmptySumNonEmpty() {
		runRemoveEmptyTest(OpType.SUM, false);
	}
	
	@Test
	public void testRemoveEmptyRoundNonEmpty() {
		runRemoveEmptyTest(OpType.ROUND, false);
	}
	
	@Test
	public void testRemoveEmptyTransposeNonEmpty() {
		runRemoveEmptyTest(OpType.TRANSPOSE, false);
	}
	
	@Test
	public void testRemoveEmptyMultLeftNonEmpty() {
		runRemoveEmptyTest(OpType.MULT_LEFT, false);
	}
	
	@Test
	public void testRemoveEmptyMultRightNonEmpty() {
		runRemoveEmptyTest(OpType.MULT_RIGHT, false);
	}
	
	@Test
	public void testRemoveEmptyPlusLeftNonEmpty() {
		runRemoveEmptyTest(OpType.PLUS_LEFT, false);
	}
	
	@Test
	public void testRemoveEmptyPlusRightNonEmpty() {
		runRemoveEmptyTest(OpType.PLUS_RIGHT, false);
	}
	
	@Test
	public void testRemoveEmptyMinusLeftNonEmpty() {
		runRemoveEmptyTest(OpType.MINUS_LEFT, false);
	}
	
	@Test
	public void testRemoveEmptyMinusRightNonEmpty() {
		runRemoveEmptyTest(OpType.MINUS_RIGHT, false);
	}
	
	@Test
	public void testRemoveEmptyMatMultLeftNonEmpty() {
		runRemoveEmptyTest(OpType.MM_LEFT, false);
	}
	
	@Test
	public void testRemoveEmptyMatMultRightNonEmpty() {
		runRemoveEmptyTest(OpType.MM_RIGHT, false);
	}
	
	@Test
	public void testRemoveEmptyRIXNonEmpty() {
		runRemoveEmptyTest(OpType.RIX, false);
	}
	
	@Test
	public void testRemoveEmptyLIXNonEmpty() {
		runRemoveEmptyTest(OpType.LIX, false);
	}

	@Test
	public void testRemoveEmptySumEmpty() {
		runRemoveEmptyTest(OpType.SUM, true);
	}
	
	@Test
	public void testRemoveEmptyRoundEmpty() {
		runRemoveEmptyTest(OpType.ROUND, true);
	}
	
	@Test
	public void testRemoveEmptyTransposeEmpty() {
		runRemoveEmptyTest(OpType.TRANSPOSE, true);
	}
	
	@Test
	public void testRemoveEmptyMultLeftEmpty() {
		runRemoveEmptyTest(OpType.MULT_LEFT, true);
	}
	
	@Test
	public void testRemoveEmptyMultRightEmpty() {
		runRemoveEmptyTest(OpType.MULT_RIGHT, true);
	}
	
	@Test
	public void testRemoveEmptyPlusLeftEmpty() {
		runRemoveEmptyTest(OpType.PLUS_LEFT, true);
	}
	
	@Test
	public void testRemoveEmptyPlusRightEmpty() {
		runRemoveEmptyTest(OpType.PLUS_RIGHT, true);
	}
	
	@Test
	public void testRemoveEmptyMinusLeftEmpty() {
		runRemoveEmptyTest(OpType.MINUS_LEFT, true);
	}
	
	@Test
	public void testRemoveEmptyMinusRightEmpty() {
		runRemoveEmptyTest(OpType.MINUS_RIGHT, true);
	}
	
	@Test
	public void testRemoveEmptyMatMultLeftEmpty() {
		runRemoveEmptyTest(OpType.MM_LEFT, true);
	}
	
	@Test
	public void testRemoveEmptyMatMultRightEmpty() {
		runRemoveEmptyTest(OpType.MM_RIGHT, true);
	}
	
	@Test
	public void testRemoveEmptyRIXEmpty() {
		runRemoveEmptyTest(OpType.RIX, true);
	}
	
	@Test
	public void testRemoveEmptyLIXEmpty() {
		runRemoveEmptyTest(OpType.LIX, true);
	}

	private void runRemoveEmptyTest( OpType type, boolean empty )
	{
		boolean oldFlagIPA = OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			config.addVariable("rows", rows);
			config.addVariable("cols", cols);
			loadTestConfiguration(config);
			
			//IPA always disabled to force recompile
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = false;
			
			// This is for running the junit test the new way, i.e., construct the arguments directly
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			//note: stats required for runtime check of rewrite
			programArgs = new String[]{"-stats","-args",
				input("X"), Integer.toString(type.ordinal()), output("R") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
				inputDir() + " " + Integer.toString(type.ordinal()) + " " + expectedDir();
	
			long seed = System.nanoTime();
			double[][] X = getRandomMatrix(rows, cols, 0, empty?0:1, sparsity, seed);
			writeInputMatrixWithMTD("X", X, true);
	
			runTest(true, false, null, -1); 
			runRScript(true);
			
			//CHECK compiled Spark jobs
			int expectNumCompiled = 24; //reblock, 4x1, 9x2, write
			Assert.assertEquals("Unexpected number of compiled Spark jobs.", 
				expectNumCompiled, Statistics.getNoOfCompiledSPInst());
			//CHECK executed Spark jobs
			int expectNumExecuted = 0;
			Assert.assertEquals("Unexpected number of executed Spark jobs.", 
				expectNumExecuted, Statistics.getNoOfExecutedSPInst());
			
			//CHECK rewrite application 
			//(for minus_left we replace X-Y with 0-Y and hence still execute -)
			if( type != OpType.MINUS_LEFT ){
				String opcode = getOpcode(type);
				//sum subject to literal replacement (no-op for empty) which happens before the rewrites
				boolean lempty = (type == OpType.SUM && empty) ? false : empty;
				Assert.assertEquals(lempty, !Statistics.getCPHeavyHitterOpCodes().contains(opcode));
			}
			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");	
		}
		finally {
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = oldFlagIPA;
		}
	}
	
	private static String getOpcode( OpType type ) {
		switch(type){
			//for sum, literal replacement of unary aggregates applies
			case SUM:         return "rlit";//return "uak+";
			case ROUND:       return Opcodes.ROUND.toString();
			case TRANSPOSE:   return Opcodes.TRANSPOSE.toString();
			case MULT_LEFT:
			case MULT_RIGHT:  return Opcodes.MULT.toString();
			case PLUS_LEFT:
			case PLUS_RIGHT:  return Opcodes.PLUS.toString();
			case MINUS_LEFT:
			case MINUS_RIGHT: return Opcodes.MINUS.toString();
			case MM_LEFT:
			case MM_RIGHT:    return Opcodes.MMULT.toString();
			case RIX:         return Opcodes.RIGHT_INDEX.toString();
			case LIX:         return Opcodes.LEFT_INDEX.toString();
		}
		return null;
	}
}