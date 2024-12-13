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

package org.apache.sysds.test.functions.rewrite;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

public class RewriteIfElseTest extends AutomatedTestBase
{
	private static final String TEST_NAME1 = "RewriteIfElseScalar";
	private static final String TEST_NAME2 = "RewriteIfElseMatrix";
	
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteIfElseTest.class.getSimpleName() + "/";
	
	private static final int rows = 10;
	private static final int cols = 10;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }) );
	}

	@Test
	public void testIfElseScalarTrueNoRewritesCP() {
		testRewriteIfElse(TEST_NAME1, true, false, ExecType.CP);
	}
	
	@Test
	public void testIfElseScalarFalseNoRewritesCP() {
		testRewriteIfElse(TEST_NAME1, false, false, ExecType.CP);
	}
	
	@Test
	public void testIfElseScalarTrueRewritesCP() {
		testRewriteIfElse(TEST_NAME1, true, true, ExecType.CP);
	}
	
	@Test
	public void testIfElseScalarFalseRewritesCP() {
		testRewriteIfElse(TEST_NAME1, false, true, ExecType.CP);
	}
	
	@Test
	public void testIfElseMatrixTrueNoRewritesCP() {
		testRewriteIfElse(TEST_NAME2, true, false, ExecType.CP);
	}
	
	@Test
	public void testIfElseMatrixFalseNoRewritesCP() {
		testRewriteIfElse(TEST_NAME2, false, false, ExecType.CP);
	}
	
	@Test
	public void testIfElseMatrixTrueRewritesCP() {
		testRewriteIfElse(TEST_NAME2, true, true, ExecType.CP);
	}
	
	@Test
	public void testIfElseMatrixFalseRewritesCP() {
		testRewriteIfElse(TEST_NAME2, false, true, ExecType.CP);
	}

	@Test
	public void testIfElseScalarTrueNoRewritesSP() {
		testRewriteIfElse(TEST_NAME1, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testIfElseScalarFalseNoRewritesSP() {
		testRewriteIfElse(TEST_NAME1, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testIfElseScalarTrueRewritesSP() {
		testRewriteIfElse(TEST_NAME1, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testIfElseScalarFalseRewritesSP() {
		testRewriteIfElse(TEST_NAME1, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testIfElseMatrixTrueNoRewritesSP() {
		testRewriteIfElse(TEST_NAME2, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testIfElseMatrixFalseNoRewritesSP() {
		testRewriteIfElse(TEST_NAME2, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testIfElseMatrixTrueRewritesSP() {
		testRewriteIfElse(TEST_NAME2, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testIfElseMatrixFalseRewritesSP() {
		testRewriteIfElse(TEST_NAME2, false, true, ExecType.SPARK);
	}

	private void testRewriteIfElse(String testname, boolean pred, boolean rewrites, ExecType et)
	{
		ExecMode platformOld = setExecMode(et);
		boolean rewritesOld = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[] { "-explain", "hops", "-stats", "-args",
				testname.equals(TEST_NAME1) ? String.valueOf(pred).toUpperCase() : input("X"), output("R") };
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(testname.equals(TEST_NAME1) ? String.valueOf(pred).toUpperCase() : inputDir(), expectedDir());
			
			if( testname.equals(TEST_NAME2) ) {
				double val = pred ? 1 : 0;
				double[][] X = getRandomMatrix(rows, cols, val, val, 1.0, 7);
				writeInputMatrixWithMTD("X", X, true,
					new MatrixCharacteristics(10,10,1000,pred?100:0));
			}
			
			//execute tests
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("R");
			Assert.assertTrue(TestUtils.compareMatrices(dmlfile, rfile, Math.pow(10,-10), "Stat-DML", "Stat-R"));
			
			//check for presence of power operator, if we did a rewrite
			if( rewrites ) {
				String opcode = et==ExecType.SPARK ? Instruction.SP_INST_PREFIX + "rand" : "rand";
				Assert.assertTrue(heavyHittersContainsString(opcode) && Statistics.getCPHeavyHitterCount(opcode)==1);
			}
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewritesOld;
			resetExecMode(platformOld);
		}
	}
}
