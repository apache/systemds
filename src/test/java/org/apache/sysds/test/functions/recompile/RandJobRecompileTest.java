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

import org.junit.Test;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

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
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "Z" }) );
	}
	
	@Test
	public void testRandRecompileNoEstSizeEval() {
		runRandJobRecompileTest(false);
	}
	
	@Test
	public void testRandRecompileEstSizeEval() {
		runRandJobRecompileTest(true);
	}
	
	private void runRandJobRecompileTest( boolean estSizeEval )
	{	
		boolean oldFlagSizeEval = OptimizerUtils.ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION;
		boolean oldFlagSplit = OptimizerUtils.ALLOW_SPLIT_HOP_DAGS;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-args", input("X"), Integer.toString(rows), output("Z") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();

			OptimizerUtils.ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION = estSizeEval;
			OptimizerUtils.ALLOW_SPLIT_HOP_DAGS = false; //test size eval in single program block
			
			double[][] V = TestUtils.round( getRandomMatrix(rows, cols, min, max, 1.0d, 7) );
			writeInputMatrix("X", V, true);
			
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("Z");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("Z");
			TestUtils.compareMatrices(dmlfile, rfile, 0, "Stat-DML", "Stat-R");
			
			//check expected number of compiled and executed Spark jobs
			int expectedNumCompiled = (estSizeEval?1:3); //rbl, rand, write
			int expectedNumExecuted = (estSizeEval?0:1);
			
			checkNumCompiledSparkInst(expectedNumCompiled);
			checkNumExecutedSparkInst(expectedNumExecuted);
		}
		finally {
			OptimizerUtils.ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION = oldFlagSizeEval;
			OptimizerUtils.ALLOW_SPLIT_HOP_DAGS = oldFlagSplit;
		}
	}
}