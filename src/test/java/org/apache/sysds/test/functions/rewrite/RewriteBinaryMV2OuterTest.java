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

import org.apache.sysds.common.Opcodes;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import java.util.HashMap;

public class RewriteBinaryMV2OuterTest extends AutomatedTestBase 
{
	private static final String TEST_NAME = "RewriteBinaryMV2Outer";
	
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteBinaryMV2OuterTest.class.getSimpleName() + "/";
	
	private double eps = Math.pow(10, -10);
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "R" }) );
	}
	
	@Test
	public void testRewriteBinaryMV2OuterEquals() {
		testRewriteBinaryMV2Outer(Opcodes.EQUAL.toString(), false);
	}
	
	@Test
	public void testRewriteBinaryMV2OuterNotEquals() {
		testRewriteBinaryMV2Outer(Opcodes.NOTEQUAL.toString(), false);
	}
	
	@Test
	public void testRewriteBinaryMV2OuterMinus() {
		testRewriteBinaryMV2Outer(Opcodes.MINUS.toString(), false);
	}
	
	@Test
	public void testRewriteBinaryMV2OuterPlus() {
		testRewriteBinaryMV2Outer(Opcodes.PLUS.toString(), false);
	}
	
	@Test
	public void testRewriteBinaryMV2OuterEqualsRewrites() {
		testRewriteBinaryMV2Outer(Opcodes.EQUAL.toString(), true);
	}
	
	@Test
	public void testRewriteBinaryMV2OuterNotEqualsRewrites() {
		testRewriteBinaryMV2Outer(Opcodes.NOTEQUAL.toString(), true);
	}
	
	@Test
	public void testRewriteBinaryMV2OuterMinusRewrites() {
		testRewriteBinaryMV2Outer(Opcodes.MINUS.toString(), true);
	}
	
	@Test
	public void testRewriteBinaryMV2OuterPlusRewrites() {
		testRewriteBinaryMV2Outer(Opcodes.PLUS.toString(), true);
	}
	
	private void testRewriteBinaryMV2Outer(String opcode, boolean rewrites)
	{	
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{ "-stats","-args", 
				input("A"), input("B"), opcode, output("R") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = getRCmd(inputDir(), opcode, expectedDir());			

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			
			//generate actual dataset 
			double[][] A = getRandomMatrix(128, 1, -5, 5, 0.9, 123); 
			double[][] B = getRandomMatrix(1, 256, -5, 5, 0.9, 456); 
			writeInputMatrixWithMTD("A", A, true);
			writeInputMatrixWithMTD("B", B, true);
			
			//run test
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			
			//check for applied rewrites
			if( rewrites )
				Assert.assertTrue(!heavyHittersContainsSubString(Opcodes.MMULT.toString()));
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}	
	}	
}
