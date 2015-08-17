/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.test.integration.functions.recompile;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.utils.Statistics;

public class RandSizeExpressionEvalTest extends AutomatedTestBase 
{

	
	private final static String TEST_NAME = "rand_size_expr_eval";
	private final static String TEST_DIR = "functions/recompile/";
	
	private final static int rows = 14;
	private final static int cols = 14;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration( TEST_NAME, new TestConfiguration(TEST_DIR, TEST_NAME, new String[]{} ));
	}

	@Test
	public void testComplexRand() 
	{
		runRandTest(TEST_NAME, false, false);
	}
	
	@Test
	public void testComplexRandExprEval() 
	{
		runRandTest(TEST_NAME, true, false);
	}
	
	@Test
	public void testComplexRandConstFold() 
	{
		runRandTest(TEST_NAME, false, true);
	}
	
	private void runRandTest( String testName, boolean evalExpr, boolean constFold )
	{	
		boolean oldFlagEval = OptimizerUtils.ALLOW_SIZE_EXPRESSION_EVALUATION;
		boolean oldFlagFold = OptimizerUtils.ALLOW_CONSTANT_FOLDING;
		
		boolean oldFlagRand1 = OptimizerUtils.ALLOW_RAND_JOB_RECOMPILE;
		boolean oldFlagRand2 = OptimizerUtils.ALLOW_BRANCH_REMOVAL;
		boolean oldFlagRand3 = OptimizerUtils.ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION;
		
		
		try
		{
			TestConfiguration config = getTestConfiguration(testName);
			config.addVariable("rows", rows);
			config.addVariable("cols", cols);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testName + ".dml";
			programArgs = new String[]{"-args", Integer.toString(rows), Integer.toString(cols), HOME + OUTPUT_DIR + "R" };
			
			loadTestConfiguration(config);
	
			OptimizerUtils.ALLOW_SIZE_EXPRESSION_EVALUATION = evalExpr;
			OptimizerUtils.ALLOW_CONSTANT_FOLDING = constFold;
			
			//disable rand specific recompile
			OptimizerUtils.ALLOW_RAND_JOB_RECOMPILE = false;
			OptimizerUtils.ALLOW_BRANCH_REMOVAL = false;
			OptimizerUtils.ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION = false;
			
			
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			
			//check correct propagated size via final results
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			Assert.assertEquals("Unexpected results.", Double.valueOf(rows*cols*3.0), dmlfile.get(new CellIndex(1,1)));
			
			//check expected number of compiled and executed MR jobs
			if( evalExpr || constFold )
			{
				Assert.assertEquals("Unexpected number of executed MR jobs.", 
						  0, Statistics.getNoOfExecutedMRJobs());			
			}
			else
			{
				Assert.assertEquals("Unexpected number of executed MR jobs.", 
						            2, Statistics.getNoOfExecutedMRJobs()); //Rand, GMR (sum)
			}		
		}
		finally
		{
			OptimizerUtils.ALLOW_SIZE_EXPRESSION_EVALUATION = oldFlagEval;
			OptimizerUtils.ALLOW_CONSTANT_FOLDING = oldFlagFold;
			
			OptimizerUtils.ALLOW_RAND_JOB_RECOMPILE = oldFlagRand1;
			OptimizerUtils.ALLOW_BRANCH_REMOVAL = oldFlagRand2;
			OptimizerUtils.ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION = oldFlagRand3;
		}
	}
	
}