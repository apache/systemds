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

package org.tugraz.sysds.test.functions.aggregate;

import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.util.DataConverter;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;

import java.util.Arrays;

public class AggregateNaNTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "NaNTest";
	private final static String TEST_DIR = "functions/aggregate/";
	private static final String TEST_CLASS_DIR = TEST_DIR + AggregateNaNTest.class.getSimpleName() + "/";
	private final static int rows = 120;
	private final static int cols = 117;
	private final static double sparsity1 = 0.1;
	private final static double sparsity2 = 0.7;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"B"})); 
	}

	
	@Test
	public void testSumDenseNaN() {
		runNaNAggregateTest(0, false);
	}
	
	@Test
	public void testSumSparseNaN() {
		runNaNAggregateTest(0, true);
	}
	
	@Test
	public void testSumSqDenseNaN() {
		runNaNAggregateTest(1, false);
	}
	
	@Test
	public void testSumSqSparseNaN() {
		runNaNAggregateTest(1, true);
	}
	
	@Test
	public void testMinDenseNaN() {
		runNaNAggregateTest(2, false);
	}
	
	@Test
	public void testMinSparseNaN() {
		runNaNAggregateTest(2, true);
	}
	
	@Test
	public void testMaxDenseNaN() {
		runNaNAggregateTest(3, false);
	}
	
	@Test
	public void testMaxSparseNaN() {
		runNaNAggregateTest(3, true);
	}
	
	@Test
	public void testRowIndexMaxDenseNaN() {
		runNaNRowIndexMxxTest("uarimax", false);
	}
	
	@Test
	public void testRowIndexMaxSparseNaN() {
		runNaNRowIndexMxxTest("uarimax", true);
	}
	
	@Test
	public void testRowIndexMinDenseNaN() {
		runNaNRowIndexMxxTest("uarimin", false);
	}
	
	@Test
	public void testRowIndexMinSparseNaN() {
		runNaNRowIndexMxxTest("uarimin", true);
	}
	
	private void runNaNAggregateTest(int type, boolean sparse) {
		//generate input
		double sparsity = sparse ? sparsity1 : sparsity2;
		double[][] A = getRandomMatrix(rows, cols, -0.05, 1, sparsity, 7); 
		A[7][7] = Double.NaN;
		MatrixBlock mb = DataConverter.convertToMatrixBlock(A);
		
		double ret = -1;
		switch(type) {
			case 0: ret = mb.sum();
			case 1: ret = mb.sumSq();
			case 2: ret = mb.min();
			case 3: ret = mb.max();
		}
		
		Assert.assertTrue(Double.isNaN(ret));
	}
	
	private void runNaNRowIndexMxxTest(String type, boolean sparse) {
		//generate input
		double sparsity = sparse ? sparsity1 : sparsity2;
		double[][] A = getRandomMatrix(rows, cols, -0.05, 1, sparsity, 7);
		Arrays.fill(A[7], Double.NaN);
		MatrixBlock mb = DataConverter.convertToMatrixBlock(A);
		
		double ret = mb.aggregateUnaryOperations(
			InstructionUtils.parseBasicAggregateUnaryOperator(type),
			new MatrixBlock(), -1, new MatrixIndexes(1, 1), true).getValue(7, 0);

		Assert.assertTrue(ret == 1);
	}
}
