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

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysml.conf.CompilerConfig;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.utils.Statistics;

public class PredicateRecompileTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME1 = "while_recompile";
	private final static String TEST_NAME2 = "if_recompile";
	private final static String TEST_NAME3 = "for_recompile";
	private final static String TEST_NAME4 = "parfor_recompile";
	private final static String TEST_DIR = "functions/recompile/";
	private final static String TEST_CLASS_DIR = TEST_DIR + PredicateRecompileTest.class.getSimpleName() + "/";
	
	private final static int rows = 10;
	private final static int cols = 15;    
	private final static int val = 7;    
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "Rout" }) );
		addTestConfiguration(TEST_NAME2, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "Rout" }) );
		addTestConfiguration(TEST_NAME3, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "Rout" }) );
		addTestConfiguration(TEST_NAME4, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] { "Rout" }) );
	}

	@Test
	public void testWhileRecompile() 
	{
		runRecompileTest(TEST_NAME1, true, false, false, false);
	}
	
	@Test
	public void testWhileNoRecompile() 
	{
		runRecompileTest(TEST_NAME1, false, false, false, false);
	}
	
	@Test
	public void testIfRecompile() 
	{
		runRecompileTest(TEST_NAME2, true, false, false, false);
	}
	
	@Test
	public void testIfNoRecompile() 
	{
		runRecompileTest(TEST_NAME2, false, false, false, false);
	}
	
	@Test
	public void testForRecompile() 
	{
		runRecompileTest(TEST_NAME3, true, false, false, false);
	}
	
	@Test
	public void testForNoRecompile() 
	{
		runRecompileTest(TEST_NAME3, false, false, false, false);
	}
	
	@Test
	public void testParForRecompile() 
	{
		runRecompileTest(TEST_NAME4, true, false, false, false);
	}
	
	@Test
	public void testParForNoRecompile() 
	{
		runRecompileTest(TEST_NAME4, false, false, false, false);
	}

	@Test
	public void testWhileRecompileExprEval() 
	{
		runRecompileTest(TEST_NAME1, true, true, false, false);
	}
	
	@Test
	public void testWhileNoRecompileExprEval() 
	{
		runRecompileTest(TEST_NAME1, false, true, false, false);
	}
	
	@Test
	public void testIfRecompileExprEval() 
	{
		runRecompileTest(TEST_NAME2, true, true, false, false);
	}
	
	@Test
	public void testIfNoRecompileExprEval() 
	{
		runRecompileTest(TEST_NAME2, false, true, false, false);
	}
	
	@Test
	public void testForRecompileExprEval() 
	{
		runRecompileTest(TEST_NAME3, true, true, false, false);
	}
	
	@Test
	public void testForNoRecompileExprEval() 
	{
		runRecompileTest(TEST_NAME3, false, true, false, false);
	}
	
	@Test
	public void testParForRecompileExprEval() 
	{
		runRecompileTest(TEST_NAME4, true, true, false, false);
	}
	
	@Test
	public void testParForNoRecompileExprEval() 
	{
		runRecompileTest(TEST_NAME4, false, true, false, false);
	}

	@Test
	public void testWhileRecompileConstFold() 
	{
		runRecompileTest(TEST_NAME1, true, false, true, false);
	}
	
	@Test
	public void testWhileNoRecompileConstFold() 
	{
		runRecompileTest(TEST_NAME1, false, false, true, false);
	}
	
	@Test
	public void testIfRecompileConstFold() 
	{
		runRecompileTest(TEST_NAME2, true, false, true, false);
	}
	
	@Test
	public void testIfNoRecompileConstFold() 
	{
		runRecompileTest(TEST_NAME2, false, false, true, false);
	}
	
	@Test
	public void testForRecompileConstFold() 
	{
		runRecompileTest(TEST_NAME3, true, false, true, false);
	}
	
	@Test
	public void testForNoRecompileConstFold() 
	{
		runRecompileTest(TEST_NAME3, false, false, true, false);
	}
	
	@Test
	public void testParForRecompileConstFold() 
	{
		runRecompileTest(TEST_NAME4, true, false, true, false);
	}
	
	@Test
	public void testParForNoRecompileConstFold() 
	{
		runRecompileTest(TEST_NAME4, false, false, true, false);
	}

	@Test
	public void testWhileNoRecompileIPA() 
	{
		runRecompileTest(TEST_NAME1, false, false, false, true);
	}
	
	@Test
	public void testIfNoRecompileIPA() 
	{
		runRecompileTest(TEST_NAME2, false, false, false, true);
	}

	@Test
	public void testForNoRecompileIPA() 
	{
		runRecompileTest(TEST_NAME3, false, false, false, true);
	}
	
	@Test
	public void testParForNoRecompileIPA() 
	{
		runRecompileTest(TEST_NAME4, false, false, false, true);
	}
	
	@Test
	public void testWhileNoRecompileExprEvalIPA() 
	{
		runRecompileTest(TEST_NAME1, false, true, false, true);
	}

	@Test
	public void testIfNoRecompileExprEvalIPA() 
	{
		runRecompileTest(TEST_NAME2, false, true, false, true);
	}
	
	@Test
	public void testForNoRecompileExprEvalIPA() 
	{
		runRecompileTest(TEST_NAME3, false, true, false, true);
	}
	
	@Test
	public void testParForNoRecompileExprEvalIPA() 
	{
		runRecompileTest(TEST_NAME4, false, true, false, true);
	}

	@Test
	public void testWhileNoRecompileConstFoldIPA() 
	{
		runRecompileTest(TEST_NAME1, false, false, true, true);
	}

	@Test
	public void testIfNoRecompileConstFoldIPA() 
	{
		runRecompileTest(TEST_NAME2, false, false, true, true);
	}

	
	@Test
	public void testForNoRecompileConstFoldIPA() 
	{
		runRecompileTest(TEST_NAME3, false, false, true, true);
	}
	
	@Test
	public void testParForNoRecompileConstFoldIPA() 
	{
		runRecompileTest(TEST_NAME4, false, false, true, true);
	}
	
	
	private void runRecompileTest( String testname, boolean recompile, boolean evalExpr, boolean constFold, boolean IPA )
	{	
		boolean oldFlagRecompile = CompilerConfig.FLAG_DYN_RECOMPILE;
		boolean oldFlagEval = OptimizerUtils.ALLOW_SIZE_EXPRESSION_EVALUATION;
		boolean oldFlagFold = OptimizerUtils.ALLOW_CONSTANT_FOLDING;
		boolean oldFlagIPA = OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS;

		boolean oldFlagRand1 = OptimizerUtils.ALLOW_RAND_JOB_RECOMPILE;
		boolean oldFlagRand2 = OptimizerUtils.ALLOW_BRANCH_REMOVAL;
		boolean oldFlagRand3 = OptimizerUtils.ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION;
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-args",
				Integer.toString(rows),
				Integer.toString(cols),
				Integer.toString(val),
				output("R") };

			CompilerConfig.FLAG_DYN_RECOMPILE = recompile;
			OptimizerUtils.ALLOW_SIZE_EXPRESSION_EVALUATION = evalExpr;
			OptimizerUtils.ALLOW_CONSTANT_FOLDING = constFold;
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = IPA;
			
			//disable rand specific recompile
			OptimizerUtils.ALLOW_RAND_JOB_RECOMPILE = false;
			OptimizerUtils.ALLOW_BRANCH_REMOVAL = false;
			OptimizerUtils.ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION = false;
			
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			
			//check expected number of compiled and executed MR jobs
			if( recompile )
			{
				Assert.assertEquals("Unexpected number of executed MR jobs.", 
						  1 - ((evalExpr || constFold)?1:0), Statistics.getNoOfExecutedMRJobs()); //rand	
			}
			else
			{
				if( IPA )
				{
					//old expected numbers before IPA
					if( testname.equals(TEST_NAME1) )
						Assert.assertEquals("Unexpected number of executed MR jobs.", 
					            4 - ((evalExpr||constFold)?4:0), Statistics.getNoOfExecutedMRJobs()); //rand, 2xgmr while pred, 1x gmr while body				
					else //if( testname.equals(TEST_NAME2) )
						Assert.assertEquals("Unexpected number of executed MR jobs.", 
					            3 - ((evalExpr||constFold)?3:0), Statistics.getNoOfExecutedMRJobs()); //rand, 1xgmr if pred, 1x gmr if body	
				}
				else
				{
					//old expected numbers before IPA
					if( testname.equals(TEST_NAME1) )
						Assert.assertEquals("Unexpected number of executed MR jobs.", 
					            4 - ((evalExpr)?1:0), Statistics.getNoOfExecutedMRJobs()); //rand, 2xgmr while pred, 1x gmr while body				
					else //if( testname.equals(TEST_NAME2) )
						Assert.assertEquals("Unexpected number of executed MR jobs.", 
					            3 - ((evalExpr)?1:0), Statistics.getNoOfExecutedMRJobs()); //rand, 1xgmr if pred, 1x gmr if body
				}
			}
			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			Assert.assertEquals(Double.valueOf((double)val), dmlfile.get(new CellIndex(1,1)));
		}
		finally
		{
			CompilerConfig.FLAG_DYN_RECOMPILE = oldFlagRecompile;
			OptimizerUtils.ALLOW_SIZE_EXPRESSION_EVALUATION = oldFlagEval;
			OptimizerUtils.ALLOW_CONSTANT_FOLDING = oldFlagFold;
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = oldFlagIPA;
			
			OptimizerUtils.ALLOW_RAND_JOB_RECOMPILE = oldFlagRand1;
			OptimizerUtils.ALLOW_BRANCH_REMOVAL = oldFlagRand2;
			OptimizerUtils.ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION = oldFlagRand3;
		}
	}
	
}