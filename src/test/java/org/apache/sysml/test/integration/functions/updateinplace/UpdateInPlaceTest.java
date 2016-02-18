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

package org.apache.sysml.test.integration.functions.updateinplace;

import java.util.Arrays;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.controlprogram.parfor.opt.OptimizerRuleBased;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class UpdateInPlaceTest extends AutomatedTestBase 
{
	
	private final static String TEST_DIR = "functions/updateinplace/";
	private final static String TEST_NAME = "updateinplace";
	private final static String TEST_CLASS_DIR = TEST_DIR + UpdateInPlaceTest.class.getSimpleName() + "/";
	
	/* Test cases to test following scenarios
	 * 
	 * Test scenarios										Test case
	 * ------------------------------------------------------------------------------------
	 * 
	 * Positive case::
	 * ===============
	 * 
	 * Candidate UIP applicable								testUIP
	 * 
	 * Interleave Operalap::
	 * =====================
	 * 
	 * Various loop types::
	 * --------------------
	 * 
	 * Overlap for Consumer	within while loop				testUIPNAConsUsed
	 * Overlap for Consumer	outside loop					testUIPNAConsUsedOutsideDAG
	 * Overlap for Consumer	within loop(not used)			testUIPNAConsUsed
	 * Overlap for Consumer	within for loop					testUIPNAConsUsedForLoop
	 * Overlap for Consumer	within inner parfor loop		testUIPNAParFor
	 * Overlap for Consumer	inside loop						testUIPNAConsUsedInsideDAG
	 * 
	 * Complex Statement:: 
	 * -------------------
	 * 
	 * Overlap for Consumer	within complex statement		testUIPNAComplexConsUsed
	 * 		(Consumer in complex statement)
	 * Overlap for Consumer	within complex statement		testUIPNAComplexCandUsed
	 * 		(Candidate in complex statement)
	 * 
	 * Else and Predicate case::
	 * -------------------------
	 * 
	 * Overlap for Consumer	within else clause				testUIPNAConsUsedElse
	 * Overlap with consumer in predicate					testUIPNACandInPredicate
	 * 
	 * Multiple LIX for same object with interleave::
	 * ----------------------------------------------
	 * 
	 * Overlap for Consumer	with multiple lix				testUIPNAMultiLIX
	 * 
	 * 
	 * Function Calls::
	 * ================ 
	 * 
	 * Overlap for candidate used in function call 			testUIPNACandInFuncCall
	 * Overlap for consumer used in function call 			testUIPNAConsInFuncCall
	 * Function call without consumer/candidate 			testUIPFuncCall
	 * 
	 */
	
	
	
	//Note: In order to run these tests against ParFor loop, parfor's DEBUG flag needs to be set in the script.
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		OptimizerUtils.ALLOW_DYN_RECOMPILATION = true;
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, null));
	}

	//public void testUIPOverlapStatement(1)
	@Test
	public void testUIP() 
	{
		List<String> listUIPRes = Arrays.asList("A");

		runUpdateInPlaceTest(TEST_NAME, 1, listUIPRes);
	}
	
	@Test
	public void testUIPNAConsUsed() 
	{
		List<String> listUIPRes = Arrays.asList();

		runUpdateInPlaceTest(TEST_NAME, 2, listUIPRes);
	}
	
	@Test
	public void testUIPNAConsUsedOutsideDAG()
	{
		List<String> listUIPRes = Arrays.asList();

		runUpdateInPlaceTest(TEST_NAME, 3, listUIPRes);
	}
	
	@Test
	public void testUIPConsNotUsed() 
	{
		List<String> listUIPRes = Arrays.asList("A");

		runUpdateInPlaceTest(TEST_NAME, 4, listUIPRes);
	}
	
	@Test
	public void testUIPNAConsUsedForLoop() 
	{
		List<String> listUIPRes = Arrays.asList();

		runUpdateInPlaceTest(TEST_NAME, 5, listUIPRes);
	}
	
	@Test
	public void testUIPNAComplexConsUsed() 
	{
		List<String> listUIPRes = Arrays.asList();

		runUpdateInPlaceTest(TEST_NAME, 6, listUIPRes);
	}
	
	@Test
	public void testUIPNAComplexCandUsed() 
	{
		List<String> listUIPRes = Arrays.asList();

		runUpdateInPlaceTest(TEST_NAME, 7, listUIPRes);
	}
	
	@Test
	public void testUIPNAConsUsedElse() 
	{
		List<String> listUIPRes = Arrays.asList();

		runUpdateInPlaceTest(TEST_NAME, 8, listUIPRes);
	}
	
	@Test
	public void testUIPNACandInPredicate()
	{
		List<String> listUIPRes = Arrays.asList();

		runUpdateInPlaceTest(TEST_NAME, 9, listUIPRes);
	}
	
	@Test
	public void testUIPNAMultiLIX() 
	{
		List<String> listUIPRes = Arrays.asList();

		runUpdateInPlaceTest(TEST_NAME, 10, listUIPRes);
	}
	
	@Test
	public void testUIPNAParFor() 
	{
		List<String> listUIPRes = Arrays.asList();

		runUpdateInPlaceTest(TEST_NAME, 11, listUIPRes);
	}
	
	@Test
	public void testUIPNACandInFuncCall() 
	{
		List<String> listUIPRes = Arrays.asList();

		runUpdateInPlaceTest(TEST_NAME, 12, listUIPRes);
	}
	
	@Test
	public void testUIPNAConsInFuncCall() 
	{
		List<String> listUIPRes = Arrays.asList();

		runUpdateInPlaceTest(TEST_NAME, 13, listUIPRes);
	}
	
	@Test
	public void testUIPFuncCall() 
	{
		List<String> listUIPRes = Arrays.asList("A");

		runUpdateInPlaceTest(TEST_NAME, 14, listUIPRes);
	}
		
	@Test
	public void testUIPNAConsUsedInsideDAG()
	{
		List<String> listUIPRes = Arrays.asList();

		runUpdateInPlaceTest(TEST_NAME, 15, listUIPRes);
	}
	

	/**
	 * 
	 * @param TEST_NAME
	 * @param iTestNumber
	 * @param listUIPRes
	 */
	private void runUpdateInPlaceTest( String TEST_NAME, int iTestNumber, List<String> listUIPExp )
	{
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			
			// This is for running the junit test the new way, i.e., construct the arguments directly 
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + iTestNumber + ".dml";
			programArgs = new String[]{"-stats"}; //new String[]{"-args", input("A"), output("B") };
			
			runTest(true, false, null, -1);

			if(OptimizerRuleBased.APPLY_REWRITE_UPDATE_INPLACE_INTERMEDIATE)
			{
				List<String> listUIPRes = OptimizerRuleBased.getUIPList();
				int iUIPResCount = 0;
				
				// If UpdateInPlace list specified in the argument, verify the list.
				if (listUIPExp != null)
				{
					if(listUIPRes != null)
					{
						for (String strUIPMatName: listUIPExp)
							Assert.assertTrue("Expected UpdateInPlace matrix " + strUIPMatName 
									+ " does not exist in the result UpdateInPlace matrix list.", 
									listUIPRes.contains(strUIPMatName));
						
						iUIPResCount = listUIPRes.size();
					}
	
					Assert.assertTrue("Expected # of UpdateInPlace matrix object/s " + listUIPExp.size() + 
						" does not match with the # of matrix objects " + iUIPResCount + " from optimization result.", 
						(iUIPResCount == listUIPExp.size()));
				}
				else
				{
					Assert.assertTrue("Expected # of UpdateInPlace matrix object/s " + "0" + 
							" does not match with the # of matrix objects " + "0" + " from optimization result.", 
							(listUIPRes == null || listUIPRes.size() == 0));
				}
			}
		}
		finally{
		}
	}
	
}
