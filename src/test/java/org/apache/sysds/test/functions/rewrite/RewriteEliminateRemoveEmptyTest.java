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

public class RewriteEliminateRemoveEmptyTest extends AutomatedTestBase 
{
	private static final String TEST_NAME1 = "RewriteEliminateRmEmpty1";
	private static final String TEST_NAME2 = "RewriteEliminateRmEmpty2";
	private static final String TEST_NAME3 = "RewriteEliminateRmEmptySum";
	private static final String TEST_NAME4 = "RewriteEliminateRmEmptySumSelect";
	private static final String TEST_NAME5 = "RewriteEliminateRmEmptyRowSum";
	private static final String TEST_NAME6 = "RewriteEliminateRmEmptyColSum";
	
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteEliminateRemoveEmptyTest.class.getSimpleName() + "/";
	
	private static final int rows = 1092;
	private static final int cols = 5;
	private static final double sparsity = 0.4;
	private static double[][] ADefault;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "B" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "B" }) );
		addTestConfiguration( TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "s" }) );
		addTestConfiguration( TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] { "s" }) );
		addTestConfiguration( TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5, new String[] { "s" }) );
		addTestConfiguration( TEST_NAME6, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME6, new String[] { "s" }) );
		ADefault = getRandomMatrix(rows, 1, -10, 10, sparsity, 7);
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

	@Test
	public void testEliminateRmEmptySumRow() {
		double [][] A = {{1,1},{1,1},{0,0}};
		double [][] sum = {{4}};
		testRewriteEliminateRmEmpty(TEST_NAME3, true, A, sum, false);
	}

	@Test
	public void testEliminateRmEmptySumRow2() {
		double [][] A = {{1,0},{1,0},{1,0}};
		double [][] sum = {{3}};
		testRewriteEliminateRmEmpty(TEST_NAME3, true, A, sum, false);
	}

	@Test
	public void testEliminateRmEmptyRowSumRow1() {
		double [][] A = {{1,1},{1,1},{0,0}};
		double [][] ARowSum = {{2},{2},{0}};
		testRewriteEliminateRmEmpty(TEST_NAME5, true, A, ARowSum, false);
	}

	@Test
	public void testEliminateRmEmptySumRowLarge() {
		double[][] sum = {{sum(ADefault)}};
		testRewriteEliminateRmEmpty(TEST_NAME3, true, ADefault, sum, false);
	}

	@Test
	public void testEliminateRmEmptyRowSumRowLarge() {
		double [][] A = getRandomMatrix(rows, cols, -10, 10, sparsity, 7);
		double [][] ARowSum = rowSum(A);
		testRewriteEliminateRmEmpty(TEST_NAME5, true, A, ARowSum, false);
	}

	@Test
	public void testEliminateRmEmptySumRowSelect() {
		double [][] A = {{1,1},{1,1},{0,0}};
		double [][] sum = {{sum(A)}};
		testRewriteEliminateRmEmpty(TEST_NAME4, true, A, sum, true);
	}

	@Test
	public void testEliminateRmEmptyColSum() {
		double [][] A = {{1,0,1},{1,0,1},{1,0,1}};
		double [][] AColSum = {{3,0,3}};
		testRewriteEliminateRmEmpty(TEST_NAME6, true, A, AColSum, false);
	}

	@Test
	public void testEliminateRmEmptyColSumLarge() {
		double [][] A = getRandomMatrix(rows, cols, -10, 10, sparsity, 7);
		double [][] AColSum = colSum(A);
		testRewriteEliminateRmEmpty(TEST_NAME6, true, A, AColSum, false);
	}

	private static double sum(double[][] A) {
		double sum = 0;
		for (double[] na : A) {
			for (double n : na) {
				sum += n;
			}
		}
		return sum;
	}

	private static double[][] rowSum(double[][] A) {
		double[][] matrixRowSum = new double[rows][1];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				matrixRowSum[i][0] += A[i][j];
			}
		}
		return matrixRowSum;
	}

	private static double[][] colSum(double[][] A) {
		double[][] matrixColSum = new double[1][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				matrixColSum[0][j] += A[i][j];
			}
		}
		return matrixColSum;
	}

	private void compareNNZ(double[][] A) {
		//give input
		writeInputMatrixWithMTD("A", A, true);
		long nnz = TestUtils.computeNNZ(A);
		
		//run test
		runTest(true, false, null, -1); 
		
		//compare NNZ
		double ret1 = readDMLMatrixFromOutputDir("B").get(new CellIndex(1,1));
		TestUtils.compareScalars(ret1, nnz, 1e-10); 
	}

	private void compareScalar(double[][] A, double[][] sum) {
		//give input
		writeInputMatrixWithMTD("A", A, true);
		
		//run test
		runTest(true, false, null, -1);
		
		//compare scalar
		double s = readDMLScalarFromOutputDir("s").get(new CellIndex(1, 1));
		TestUtils.compareScalars(s,sum[0][0],1e-10);
	}

	private void compareMatrix(double[][] A, double[][] sum) {
		//give input and expectation
		writeInputMatrixWithMTD("A", A, true);
		writeExpectedMatrix("s", sum);
		
		//run test
		runTest(true, false, null, -1);
		
		//compare matrices
		compareResults(1e-10);
	}

	private void testRewriteEliminateRmEmpty (String test, boolean rewrites) {
		testRewriteEliminateRmEmpty(test, rewrites, ADefault, null, false);
	}

	private void testRewriteEliminateRmEmpty(String test, boolean rewrites, double[][] A, double[][] sum, boolean select) {
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;

		try
		{
			TestConfiguration config = getTestConfiguration(test);
			loadTestConfiguration(config);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + test + ".dml";
			programArgs = new String[]{ "-explain", "-stats",
				"-args", input("A"), output(config.getOutputFiles()[0]) };
		
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			switch(config.getTestScript()) {
				case TEST_NAME1:
				case TEST_NAME2:
					compareNNZ(A);
					break;
				case TEST_NAME3: 
				case TEST_NAME4:
					compareScalar(A, sum);
					break;
				case TEST_NAME5:
				case TEST_NAME6:
					compareMatrix(A, sum);
					break;
				default:
					throw new AssertionError("No test case specified!");
			}

			if( rewrites && !select ) {
				boolean noRmempty = heavyHittersContainsSubString(Opcodes.RMEMPTY.toString());
				Assert.assertFalse(noRmempty);
			}
		} 
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}
	}
}
