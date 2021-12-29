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

package org.apache.sysds.test.functions.builtin.part2;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

public class BuiltinNormalizeTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "normalize";
	private final static String TEST_NAME2 = "normalizeAll";
	private final static String TEST_NAME3 = "normalizeListEval";
	private final static String TEST_NAME4 = "normalizeListEvalAll";
	
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinNormalizeTest.class.getSimpleName() + "/";
	
	private final static double eps = 1e-10;
	private final static int rows = 70;
	private final static int cols = 50;
	private final static double spSparse = 0.1;
	private final static double spDense = 0.9;
	
	@Override
	public void setUp() {
		//only needed for directory here
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"})); 
	}

	@Test
	public void testNormalizeMatrixDenseCP() {
		runNormalizeTest(TEST_NAME, false, ExecType.CP);
	}
	
	@Test
	public void testNormalizeMatrixSparseCP() {
		runNormalizeTest(TEST_NAME, true, ExecType.CP);
	}
	
	@Test
	public void testNormalizeMatrixDenseSP() {
		runNormalizeTest(TEST_NAME, false, ExecType.SPARK);
	}
	
	@Test
	public void testNormalizeMatrixSparseSP() {
		runNormalizeTest(TEST_NAME, true, ExecType.SPARK);
	}
	
	@Test
	public void testNormalize2MatrixDenseCP() {
		runNormalizeTest(TEST_NAME2, false, ExecType.CP);
	}
	
	@Test
	public void testNormalize2MatrixSparseCP() {
		runNormalizeTest(TEST_NAME2, true, ExecType.CP);
	}
	
	@Test
	public void testNormalize2MatrixDenseSP() {
		runNormalizeTest(TEST_NAME2, false, ExecType.SPARK);
	}
	
	@Test
	public void testNormalize2MatrixSparseSP() {
		runNormalizeTest(TEST_NAME2, true, ExecType.SPARK);
	}
	
	@Test
	public void testNormalizeListEvalMatrixDenseCP() {
		runNormalizeTest(TEST_NAME3, false, ExecType.CP);
	}
	
	@Test
	public void testNormalizeListEvalMatrixSparseCP() {
		runNormalizeTest(TEST_NAME3, true, ExecType.CP);
	}
	
	@Test
	public void testNormalizeListEvalMatrixDenseSP() {
		runNormalizeTest(TEST_NAME3, false, ExecType.SPARK);
	}
	
	@Test
	public void testNormalizeListEvalMatrixSparseSP() {
		runNormalizeTest(TEST_NAME3, true, ExecType.SPARK);
	}
	
	@Test
	public void testNormalizeListEval2MatrixDenseCP() {
		runNormalizeTest(TEST_NAME4, false, ExecType.CP);
	}
	
	@Test
	public void testNormalizeListEval2MatrixSparseCP() {
		runNormalizeTest(TEST_NAME4, true, ExecType.CP);
	}
	
	@Test
	public void testNormalizeListEval2MatrixDenseSP() {
		runNormalizeTest(TEST_NAME4, false, ExecType.SPARK);
	}
	
	@Test
	public void testNormalizeListEval2MatrixSparseSP() {
		runNormalizeTest(TEST_NAME4, true, ExecType.SPARK);
	}
	
	private void runNormalizeTest(String testname, boolean sparse, ExecType instType)
	{
		ExecMode platformOld = setExecMode(instType);
		
		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			double sparsity = sparse ? spSparse : spDense;
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-explain","-args", input("A"), output("B") };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			//generate actual dataset 
			double[][] A = getRandomMatrix(rows, cols, -10, 10, sparsity, 7);
			writeInputMatrixWithMTD("A", A, true);
			
			runTest(true, false, null, -1);
			runRScript(true); 
		
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("B");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		
			//check number of compiler Spark instructions
			if( instType == ExecType.CP ) {
				int expected = testname.equals(TEST_NAME4) ? 2 : 1;
				Assert.assertEquals(expected, Statistics.getNoOfCompiledSPInst()); //reblock, [write]
				Assert.assertEquals(0, Statistics.getNoOfExecutedSPInst());
			}
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
