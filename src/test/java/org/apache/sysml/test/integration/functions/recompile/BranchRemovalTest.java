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

package org.apache.sysml.test.integration.functions.recompile;

import java.util.HashMap;

import org.junit.Test;

import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class BranchRemovalTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME = "if_branch_removal";
	private final static String TEST_DIR = "functions/recompile/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BranchRemovalTest.class.getSimpleName() + "/";
	
	private final static int rows = 10;
	private final static int cols = 15;    
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "X" }) );
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
			loadTestConfiguration(config);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("X"),
				Integer.toString(rows), Integer.toString(cols),
				Integer.toString(val), output("X") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
				inputDir() + " " + val + " " + expectedDir();

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
			int expectedNumCompiled = 4; //reblock, 3xGMR (append), write
			int expectedNumExecuted = 0;			
			if( branchRemoval && IPA )
				expectedNumCompiled = 1; //reblock
			else if( branchRemoval ){
				expectedNumCompiled = 3; //reblock, GMR (append), write
			}
			
			checkNumCompiledMRJobs(expectedNumCompiled); 
			checkNumExecutedMRJobs(expectedNumExecuted); 
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