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

public class RewriteFusedRandTest extends AutomatedTestBase 
{
	private static final String TEST_NAME1 = "RewriteFusedRandLit";
	private static final String TEST_NAME2 = "RewriteFusedRandVar1";
	private static final String TEST_NAME3 = "RewriteFusedRandVar2";
	private static final String TEST_NAME4 = "RewriteFusedRandVar3";
	
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteFusedRandTest.class.getSimpleName() + "/";
	
	private static final int rows = 1932;
	private static final int cols = 14;
	private static final int seed = 7;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] { "R" }) );
	}

	@Test
	public void testRewriteFusedRandUniformNoRewrite() {
		testRewriteFusedRand( TEST_NAME1, "uniform", false );
	}
	
	@Test
	public void testRewriteFusedRandNormalNoRewrite() {
		testRewriteFusedRand( TEST_NAME1, "normal", false );
	}
	
	@Test
	public void testRewriteFusedRandPoissonNoRewrite() {
		testRewriteFusedRand( TEST_NAME1, "poisson", false );
	}
	
	@Test
	public void testRewriteFusedRandUniform() {
		testRewriteFusedRand( TEST_NAME1, "uniform", true );
	}
	
	@Test
	public void testRewriteFusedRandNormal() {
		testRewriteFusedRand( TEST_NAME1, "normal", true );
	}
	
	@Test
	public void testRewriteFusedRandPoisson() {
		testRewriteFusedRand( TEST_NAME1, "poisson", true );
	}
	
	@Test
	public void testRewriteFusedZerosPlusVarUniform() {
		testRewriteFusedRand( TEST_NAME2, "uniform", true );
	}
	
	@Test
	public void testRewriteFusedOnesMultVarUniform() {
		testRewriteFusedRand( TEST_NAME3, "uniform", true );
	}
	
	@Test
	public void testRewriteFusedOnesMult2VarUniform() {
		testRewriteFusedRand( TEST_NAME4, "uniform", true );
	}
	
	@Test
	public void testRewriteFusedOnesMult2VarNormal() {
		testRewriteFusedRand( TEST_NAME4, "normal", true );
	}
	
	private void testRewriteFusedRand( String testname, String pdf, boolean rewrites )
	{	
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		
		try {
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{ "-stats", "-args", String.valueOf(rows), 
					String.valueOf(cols), pdf, String.valueOf(seed), output("R") };
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			//run performance tests
			runTest(true, false, null, -1); 
			
			//compare matrices 
			Double ret = readDMLMatrixFromOutputDir("R").get(new CellIndex(1,1));
			if( testname.equals(TEST_NAME1) )
				Assert.assertEquals("Wrong result", Double.valueOf(rows), ret);
			else if( testname.equals(TEST_NAME2) )
				Assert.assertEquals("Wrong result", Double.valueOf(Math.pow(rows*cols, 2)), ret);
			else if( testname.equals(TEST_NAME3) )
				Assert.assertEquals("Wrong result", Double.valueOf(Math.pow(rows*cols, 2)), ret);
			
			//check for applied rewrites
			if( rewrites ) {
				boolean expected = testname.equals(TEST_NAME2) || pdf.equals("uniform");
				Assert.assertTrue(expected == (!heavyHittersContainsString(Opcodes.PLUS.toString())
					&& !heavyHittersContainsString("*")));
			}
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}
	}
}
