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

package org.apache.sysds.test.functions.aggregate;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

public class ContainsTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "Contains";

	private final static String TEST_DIR = "functions/aggregate/";
	private static final String TEST_CLASS_DIR = TEST_DIR + AggregateInfTest.class.getSimpleName() + "/";
	
	private final static int rows = 1205;
	private final static int cols = 1179;
	private final static double sparsity1 = 0.1;
	private final static double sparsity2 = 0.7;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"B"})); 
	}

	
	@Test
	public void testNaNTrueDenseCP() {
		runContainsTest(Double.NaN, true, false, ExecType.CP);
	}
	
	@Test
	public void testNaNFalseDenseCP() {
		runContainsTest(Double.NaN, false, false, ExecType.CP);
	}
	
	@Test
	public void testNaNTrueSparseCP() {
		runContainsTest(Double.NaN, true, true, ExecType.CP);
	}
	
	@Test
	public void testNaNFalseSpaseCP() {
		runContainsTest(Double.NaN, false, true, ExecType.CP);
	}
	
	@Test
	public void testInfTrueDenseCP() {
		runContainsTest(Double.POSITIVE_INFINITY, true, false, ExecType.CP);
	}
	
	@Test
	public void testInfFalseDenseCP() {
		runContainsTest(Double.POSITIVE_INFINITY, false, false, ExecType.CP);
	}
	
	@Test
	public void testInfTrueSparseCP() {
		runContainsTest(Double.POSITIVE_INFINITY, true, true, ExecType.CP);
	}
	
	@Test
	public void testInfFalseSpaseCP() {
		runContainsTest(Double.POSITIVE_INFINITY, false, true, ExecType.CP);
	}

	@Test
	public void testNaNTrueDenseSpark() {
		runContainsTest(Double.NaN, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testNaNFalseDenseSpark() {
		runContainsTest(Double.NaN, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testNaNTrueSparseSpark() {
		runContainsTest(Double.NaN, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testNaNFalseSpaseSpark() {
		runContainsTest(Double.NaN, false, true, ExecType.SPARK);
	}
	
	private void runContainsTest( double check, boolean expected, boolean sparse, ExecType instType)
	{
		ExecMode oldMode = setExecMode(instType);
	
		try
		{
			double sparsity = (sparse) ? sparsity1 : sparsity2;
			getAndLoadTestConfiguration(TEST_NAME);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args",
				input("A"), String.valueOf(check), output("B") };
			
			//generate actual dataset 
			double[][] A = getRandomMatrix(rows, cols, -0.05, 1, sparsity, 7); 
			A[7][7] = expected ? check : 7;
			writeInputMatrixWithMTD("A", A, false);
	
			//run test
			runTest(true, false, null, -1); 
			boolean ret = TestUtils.readDMLBoolean(output("B"));
			Assert.assertEquals(expected, ret);
			if( instType == ExecType.CP ) {
				Assert.assertEquals(Statistics.getNoOfCompiledSPInst(), 1); //reblock
				Assert.assertEquals(Statistics.getNoOfExecutedSPInst(), 0);
			}
		}
		finally {
			resetExecMode(oldMode);
		}
	}
}
