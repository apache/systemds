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

package org.apache.sysds.test.component.matrix.libMatrixReorg;

import java.util.Arrays;
import java.util.Collection;

import org.apache.sysds.runtime.functionobjects.IndexFunction;
import org.apache.sysds.runtime.functionobjects.RollIndex;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(Parameterized.class)
public class SparseMatrixRollOperationCorrectnessTest {

	private final double[][] input;
	private final double[][] expected;
	private final int shift;

	public SparseMatrixRollOperationCorrectnessTest(double[][] input, double[][] expected, int shift) {
		this.input = input;
		this.expected = expected;
		this.shift = shift;
	}

	@Parameterized.Parameters(name = "Shift={2}, Size={0}x{1} (Sparse)")
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			{
				new double[][] {{1, 0, 0}, {0, 2, 0}, {0, 0, 3}},
				new double[][] {{0, 0, 3}, {1, 0, 0}, {0, 2, 0}},
				1
			},
			{
				new double[][] {{1, 0, 0}, {0, 2, 0}, {0, 0, 3}},
				new double[][] {{0, 2, 0}, {0, 0, 3}, {1, 0, 0}},
				-1
			},
			{
				new double[][] {{0}, {10}, {0}, {20}, {0}},
				new double[][] {{20}, {0}, {0}, {10}, {0}},
				2
			},
			{
				new double[][] {{1, 2}, {0, 0}, {3, 4}, {0, 0}},
				new double[][] {{0, 0}, {1, 2}, {0, 0}, {3, 4}},
				1
			},
			{
				new double[][] {{0, 0, 0}, {0, 0, 0}, {0, 5, 0}, {0, 0, 0}},
				new double[][] {{0, 5, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
				2
			},
			{
				new double[][] {{1, 0}, {0, 2}, {3, 0}},
				new double[][] {{3, 0}, {1, 0}, {0, 2}},
				4
			},
			{
				new double[][] {{0, 1}, {0, 0}, {2, 0}},
				new double[][] {{0, 0}, {2, 0}, {0, 1}},
				-1
			},
			{
				new double[][] {{0, 0}, {0, 0}},
				new double[][] {{0, 0}, {0, 0}},
				1
			},
			{
				new double[][] {{1, 0, 1}, {0, 1, 0}, {1, 0, 1}},
				new double[][] {{1, 0, 1}, {1, 0, 1}, {0, 1, 0}},
				1
			},
			{
				new double[][] {{0, 5}, {0, 0}, {2, 0}},
				new double[][] {{0, 5}, {0, 0}, {2, 0}},
				0
			},
			{
				new double[][] {{0, 5}, {0, 0}, {2, 0}},
				new double[][] {{0, 5}, {0, 0}, {2, 0}},
				3
			},
			{
				new double[][] {{0, 5}, {0, 0}, {2, 0}},
				new double[][] {{0, 5}, {0, 0}, {2, 0}},
				-3
			},
			{
				new double[][] {{0, 0, 1, 0}, {0, 2, 0, 0}},
				new double[][] {{0, 2, 0, 0}, {0, 0, 1, 0}},
				1
			},
			{
				new double[][] {{0, 0, 1, 0}, {0, 2, 0, 0}},
				new double[][] {{0, 2, 0, 0}, {0, 0, 1, 0}},
				-1
			},
			{
				new double[][] {{1, 1}, {0, 0}, {2, 2}, {0, 0}},
				new double[][] {{0, 0}, {1, 1}, {0, 0}, {2, 2}},
				1
			},
			{
				new double[][] {{0, 0}, {0, 0}, {1, 2}, {3, 4}},
				new double[][] {{1, 2}, {3, 4}, {0, 0}, {0, 0}},
				2
			},
			{
				new double[][] {{1, 0}, {0, 0}, {0, 2}},
				new double[][] {{0, 2}, {1, 0}, {0, 0}},
				10
			},
			{
				new double[][] {{1, 0}, {0, 0}, {0, 2}},
				new double[][] {{0, 0}, {0, 2}, {1, 0}},
				-10
			},
			{
				new double[][] {{5, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}},
				new double[][] {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {5, 0, 0, 0}},
				3
			},
			{
				new double[][] {{5, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}},
				new double[][] {{5, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}},
				4
			},
			{
				new double[][] {{0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}},
				new double[][] {{0, 3}, {0, 4}, {0, 5}, {0, 1}, {0, 2}},
				3
			},
			{
				new double[][] {{-1, 0}, {0, 0}, {0, 5}},
				new double[][] {{0, 5}, {-1, 0}, {0, 0}},
				1
			}
		});
	}

	@Test
	public void testRollOperationProducesExpectedOutputSparse() {
		MatrixBlock inBlock = new MatrixBlock(input.length, input[0].length, false);
		inBlock.init(input, input.length, input[0].length);

		inBlock.denseToSparse(true);

		Assert.assertTrue("Input block must be in sparse format", inBlock.isInSparseFormat());

		IndexFunction op = new RollIndex(shift);
		ReorgOperator reorgOperator = new ReorgOperator(op);
		MatrixBlock matrixBlock = new MatrixBlock();

		MatrixBlock outBlock = inBlock.reorgOperations(reorgOperator, matrixBlock, 0, 0, 0);

		MatrixBlock expectedBlock = new MatrixBlock(expected.length, expected[0].length, false);
		expectedBlock.init(expected, expected.length, expected[0].length);

		TestUtils.compareMatrices(outBlock, expectedBlock, 1e-12,
			"Sparse Roll operation does not match expected output");
	}
}
