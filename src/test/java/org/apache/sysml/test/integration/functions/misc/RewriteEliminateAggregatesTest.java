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

package org.apache.sysml.test.integration.functions.misc;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class RewriteEliminateAggregatesTest extends AutomatedTestBase 
{
	private static final String TEST_NAME = "RewriteEliminateAggregate";
	private static final String TEST_DIR = "functions/misc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteEliminateAggregatesTest.class.getSimpleName() + "/";
	
	private double tol = Math.pow(10, -10);
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "R" }) );
	}
	
	@Test
	public void testEliminateSumSumNoRewrite() {
		testRewriteEliminateAggregate(1, false);
	}
	
	@Test
	public void testEliminateMinMinNoRewrite() {
		testRewriteEliminateAggregate(2, false);
	}
	
	@Test
	public void testEliminateMaxMaxNoRewrite() {
		testRewriteEliminateAggregate(3, false);
	}
	
	@Test
	public void testEliminateSumSqSumNoRewrite() {
		testRewriteEliminateAggregate(4, false);
	}
	
	@Test
	public void testEliminateMinSumNoRewrite() {
		testRewriteEliminateAggregate(5, false);
	}
	
	@Test
	public void testEliminateSumSumRewrite() {
		testRewriteEliminateAggregate(1, true);
	}
	
	@Test
	public void testEliminateMinMinRewrite() {
		testRewriteEliminateAggregate(2, true);
	}
	
	@Test
	public void testEliminateMaxMaxRewrite() {
		testRewriteEliminateAggregate(3, true);
	}
	
	@Test
	public void testEliminateSumSqSumRewrite() {
		testRewriteEliminateAggregate(4, true);
	}
	
	@Test
	public void testEliminateMinSumRewrite() {
		testRewriteEliminateAggregate(5, true);
	}
	
	private void testRewriteEliminateAggregate(int type, boolean rewrites)
	{	
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{ "-stats","-args", 
				input("A"), String.valueOf(type), output("Scalar") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = getRCmd(inputDir(), String.valueOf(type), expectedDir());			

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			
			//generate actual dataset 
			double[][] A = getRandomMatrix(123, 12, -5, 5, 0.9, 7); 
			writeInputMatrixWithMTD("A", A, true);
			
			//run test
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare scalars 
			double ret1 = readDMLScalarFromHDFS("Scalar").get(new CellIndex(1,1));
			double ret2 = readRScalarFromFS("Scalar").get(new CellIndex(1,1));
			TestUtils.compareScalars(ret1, ret2, tol);
			
			//check for applied rewrites
			if( rewrites ) {
				Assert.assertEquals(type==5, 
					heavyHittersContainsSubString("uar", "uac"));
			} 
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}	
	}	
}
