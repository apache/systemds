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
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(Parameterized.class)
public class DenseMatrixRollOperationCorrectnessTest {

	private final double[][] input;
	private final double[][] expected;
	private final int shift;

	public DenseMatrixRollOperationCorrectnessTest(double[][] input, double[][] expected, int shift) {
		this.input = input;
		this.expected = expected;
		this.shift = shift;
	}

	@Parameterized.Parameters(name = "Shift={2}, Size={0}x{1}")
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			{
				new double[][] {{1, 2, 3, 4, 5}},
				new double[][] {{1, 2, 3, 4, 5}},
				0
			},
			{
				new double[][] {{1, 2, 3, 4, 5}},
				new double[][] {{1, 2, 3, 4, 5}},
				1
			},
			{
				new double[][] {{1, 2, 3, 4, 5}},
				new double[][] {{1, 2, 3, 4, 5}},
				-3
			},
			{
				new double[][] {{1, 2, 3, 4, 5}},
				new double[][] {{1, 2, 3, 4, 5}},
				999
			},
			{
				new double[][] {{1}, {2}, {3}, {4}, {5}},
				new double[][] {{4}, {5}, {1}, {2}, {3}},
				2
			},
			{
				new double[][] {{1}, {2}, {3}, {4}, {5}},
				new double[][] {{2}, {3}, {4}, {5}, {1}},
				-1
			},
			{
				new double[][] {{1}, {2}, {3}, {4}, {5}},
				new double[][] {{1}, {2}, {3}, {4}, {5}},
				5
			},
			{
				new double[][] {{1, 2, 3}, {4, 5, 6}},
				new double[][] {{4, 5, 6}, {1, 2, 3}},
				1
			},
			{
				new double[][] {{1, 2, 3}, {4, 5, 6}},
				new double[][] {{4, 5, 6}, {1, 2, 3}},
				7
			},
			{
				new double[][] {{1, 2, 3}, {4, 5, 6}},
				new double[][] {{1, 2, 3}, {4, 5, 6}},
				2
			},
			{
				new double[][] {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
				new double[][] {{7, 8, 9}, {1, 2, 3}, {4, 5, 6}},
				1
			},
			{
				new double[][] {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
				new double[][] {{4, 5, 6}, {7, 8, 9}, {1, 2, 3}},
				-1
			},
			{
				new double[][] {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}},
				new double[][] {{3, 2, 1}, {9, 8, 7}, {6, 5, 4}},
				1
			},
			{
				new double[][] {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
				new double[][] {{9, 10, 11, 12}, {1, 2, 3, 4}, {5, 6, 7, 8}},
				1
			},
			{
				new double[][] {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
				new double[][] {{5, 6, 7, 8}, {9, 10, 11, 12}, {1, 2, 3, 4}},
				-1
			},
			{
				new double[][] {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}, {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}},
				new double[][] {{21, 22, 23, 24, 25}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}, {16, 17, 18, 19, 20}},
				1
			},
			{
				new double[][] {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}, {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}},
				new double[][] {{11, 12, 13, 14, 15}, {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
				-2
			},
			{
				new double[][] {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}, {13, 14, 15}, {16, 17, 18}, {19, 20, 21},
					{22, 23, 24}, {25, 26, 27}, {28, 29, 30}},
				new double[][] {{22, 23, 24}, {25, 26, 27}, {28, 29, 30}, {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12},
					{13, 14, 15}, {16, 17, 18}, {19, 20, 21}},
				3
			},
			{
				new double[][] {{1, 2}, {3, 4}, {5, 6}, {7, 8}},
				new double[][] {{5, 6}, {7, 8}, {1, 2}, {3, 4}},
				1002
			},
			{
				new double[][] {{1}, {2}, {3}, {4}, {5}},
				new double[][] {{3}, {4}, {5}, {1}, {2}},
				-12
			},
			{
				new double[][] {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
				new double[][] {{4, 5, 6}, {7, 8, 9}, {1, 2, 3}},
				-10
			},
			{
				new double[][] {{1, 2}, {3, 4}, {5, 6}, {7, 8}},
				new double[][] {{1, 2}, {3, 4}, {5, 6}, {7, 8}},
				-4
			},
			{
				new double[][] {{1, 2}, {3, 4}, {5, 6}, {7, 8}},
				new double[][] {{3, 4}, {5, 6}, {7, 8}, {1, 2}},
				-5
			}
		});
	}

	@Test
	public void testRollOperationProducesExpectedOutput() {
		MatrixBlock inBlock = new MatrixBlock(input.length, input[0].length, false);
		inBlock.init(input, input.length, input[0].length);

		IndexFunction op = new RollIndex(shift);
		MatrixBlock outBlock = inBlock.reorgOperations(new ReorgOperator(op), new MatrixBlock(), 0, 0, 5);

		MatrixBlock expectedBlock = new MatrixBlock(expected.length, expected[0].length, false);
		expectedBlock.init(expected, expected.length, expected[0].length);

		TestUtils.compareMatrices(outBlock, expectedBlock, 1e-12, "Dense Roll operation does not match expected output");
	}
}
