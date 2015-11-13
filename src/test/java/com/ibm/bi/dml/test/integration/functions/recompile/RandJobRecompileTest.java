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

import org.junit.Test;

import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class RandJobRecompileTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME = "grpagg_rand_recompile";
	private final static String TEST_DIR = "functions/recompile/";
	private final static String TEST_CLASS_DIR = TEST_DIR + RandJobRecompileTest.class.getSimpleName() + "/";
	
	private final static int rows = 27;
	private final static int cols = 2; 
	
	private final static int min = 1;
	private final static int max = 4;
	
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "Z" }) );
	}
	
	
	@Test
	public void testRandRecompileNoEstSizeEval() 
	{
		runRandJobRecompileTest(false);
	}
	
	@Test
	public void testRandRecompilEstSizeEval() 
	{
		runRandJobRecompileTest(true);
	}

	/**
	 * 
	 * @param estSizeEval
	 */
	private void runRandJobRecompileTest( boolean estSizeEval )
	{	
		boolean oldFlagSizeEval = OptimizerUtils.ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("X"), Integer.toString(rows), output("Z") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();

			OptimizerUtils.ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION = estSizeEval;
			
			double[][] V = round( getRandomMatrix(rows, cols, min, max, 1.0d, 7) );
			writeInputMatrix("X", V, true);
			
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("Z");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("Z");
			TestUtils.compareMatrices(dmlfile, rfile, 0, "Stat-DML", "Stat-R");
			
			//check expected number of compiled and executed MR jobs
			int expectedNumCompiled = (estSizeEval?1:2); //rand, write
			int expectedNumExecuted = 0;			
			
			checkNumCompiledMRJobs(expectedNumCompiled); 
			checkNumExecutedMRJobs(expectedNumExecuted); 
		}
		finally
		{
			OptimizerUtils.ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION = oldFlagSizeEval;
		}
	}
	
	/**
	 * 
	 * @param data
	 * @return
	 */
	private double[][] round( double[][] data )
	{
		for( int i=0; i<data.length; i++ )
			for( int j=0; j<data[i].length; j++ )
				data[i][j] = Math.round(data[i][j]);
		return data;
	}
}