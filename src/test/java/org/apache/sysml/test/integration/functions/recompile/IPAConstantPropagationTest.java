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

public class IPAConstantPropagationTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME1 = "constant_propagation_if";
	private final static String TEST_NAME2 = "constant_propagation_while";
	private final static String TEST_DIR = "functions/recompile/";
	private final static String TEST_CLASS_DIR = TEST_DIR + IPAConstantPropagationTest.class.getSimpleName() + "/";
	
	private final static int rows = 10;
	private final static int cols = 15;    
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "X" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "X" }) );
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
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-args", 
				Integer.toString(rows), Integer.toString(cols), output("X") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
				Integer.toString(rows) + " " + Integer.toString(cols) + " " + expectedDir();

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
			
			checkNumCompiledMRJobs(expectedNumCompiled); 
			checkNumExecutedMRJobs(expectedNumExecuted); 
		}
		finally
		{
			OptimizerUtils.ALLOW_BRANCH_REMOVAL = oldFlagBranchRemoval;
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = oldFlagIPA;
		}
	}
	
}
