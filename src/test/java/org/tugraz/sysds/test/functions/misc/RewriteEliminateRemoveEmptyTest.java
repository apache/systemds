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

package org.tugraz.sysds.test.functions.misc;

import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

public class RewriteEliminateRemoveEmptyTest extends AutomatedTestBase 
{
	private static final String TEST_NAME1 = "RewriteEliminateRmEmpty1";
	private static final String TEST_NAME2 = "RewriteEliminateRmEmpty2";
	
	private static final String TEST_DIR = "functions/misc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteEliminateRemoveEmptyTest.class.getSimpleName() + "/";
	
	private static final int rows = 1092;
	private static final double sparsity = 0.4;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "B" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "B" }) );
	}
	
	@Test
	public void testEliminateRmEmpty1() {
		testRewriteEliminateRmEmpty(TEST_NAME1, false);
	}
	
	@Test
	public void testEliminateRmEmpty2() {
		testRewriteEliminateRmEmpty(TEST_NAME2, false);
	}
	
	@Test
	public void testEliminateRmEmpty1Rewrites() {
		testRewriteEliminateRmEmpty(TEST_NAME1, true);
	}
	
	@Test
	public void testEliminateRmEmpty2Rewrites() {
		testRewriteEliminateRmEmpty(TEST_NAME2, true);
	}

	private void testRewriteEliminateRmEmpty(String test, boolean rewrites)
	{	
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		
		try
		{
			TestConfiguration config = getTestConfiguration(test);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + test + ".dml";
			programArgs = new String[]{ "-explain", "-stats",
				"-args", input("A"), output("B") };

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
				
			//generate actual dataset 
			double[][] A = getRandomMatrix(rows, 1, -10, 10, sparsity, 7);
			writeInputMatrixWithMTD("A", A, true);
			long nnz = TestUtils.computeNNZ(A);
			
			//run test
			runTest(true, false, null, -1); 
			
			//compare scalars 
			double ret1 = readDMLMatrixFromHDFS("B").get(new CellIndex(1,1));
			TestUtils.compareScalars(ret1, nnz, 1e-10);
			
			//check for applied rewrites
			if( rewrites )
				Assert.assertFalse(heavyHittersContainsSubString("rmempty"));
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}
	}
}
