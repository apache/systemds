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

package org.apache.sysds.runtime.matrix.data;

import static org.junit.Assert.assertEquals;

import java.util.HashSet;

import org.apache.sysds.common.Types;
import org.junit.Test;

/**
 * Tests LibMatrixSketch unique paths with k=1 and k>1.
 */
public class LibMatrixSketchUniqueParallelTest {
	@Test
	public void testRowColUniqueMatchesBaseline() {
		MatrixBlock input = new MatrixBlock(20000, 2, false).allocateBlock();
		for( int i = 0; i < input.getNumRows(); i++ ) {
			input.set(i, 0, i % 7);
			input.set(i, 1, (i + 3) % 11);
		}
		input.recomputeNonZeros();

		MatrixBlock baseline = LibMatrixSketch.getUniqueValues(input, Types.Direction.RowCol);
		MatrixBlock parallel = LibMatrixSketch.getUniqueValues(input, Types.Direction.RowCol, 4);

		assertDimensions(parallel, 11, 1, "RowCol parallel dimensions");
		assertSameScalarSet(baseline, parallel, "RowCol baseline vs parallel");
	}

	@Test
	public void testRowUniqueMatchesBaselineAndExpectedValues() {
		MatrixBlock input = new MatrixBlock(12000, 4, false).allocateBlock();
		for( int i = 0; i < input.getNumRows(); i++ ) {
			int pattern = i % 4;
			input.set(i, 0, pattern);
			input.set(i, 1, pattern);
			input.set(i, 2, pattern + 10);
			input.set(i, 3, pattern + 20);
		}
		input.recomputeNonZeros();

		MatrixBlock baseline = LibMatrixSketch.getUniqueValues(input, Types.Direction.Row);
		MatrixBlock parallel = LibMatrixSketch.getUniqueValues(input, Types.Direction.Row, 4);

		assertDimensions(parallel, input.getNumRows(), 3, "Row parallel dimensions");
		assertBlockEquals(baseline, parallel, "Row baseline vs parallel");
		assertSameRowSet(parallel, 5, new double[] {1, 11, 21}, "Row expected values");
	}

	@Test
	public void testColumnUniqueMatchesBaselineAndExpectedValues() {
		MatrixBlock input = new MatrixBlock(4, 5000, false).allocateBlock();
		for( int j = 0; j < input.getNumColumns(); j++ ) {
			int pattern = j % 4;
			input.set(0, j, pattern);
			input.set(1, j, pattern);
			input.set(2, j, pattern + 10);
			input.set(3, j, pattern + 20);
		}
		input.recomputeNonZeros();

		MatrixBlock baseline = LibMatrixSketch.getUniqueValues(input, Types.Direction.Col);
		MatrixBlock parallel = LibMatrixSketch.getUniqueValues(input, Types.Direction.Col, 4);

		assertDimensions(parallel, 3, input.getNumColumns(), "Col parallel dimensions");
		assertBlockEquals(baseline, parallel, "Col baseline vs parallel");
		assertSameColumnSet(parallel, 5, new double[] {1, 11, 21}, "Col expected values");
	}

	@Test
	public void testLargerInputsMatchBaseline() {
		testRowColLargeInput();
		testRowLargeInput();
		testColumnLargeInput();
	}

	private static void testRowColLargeInput() {
		MatrixBlock input = new MatrixBlock(1200000, 1, false).allocateBlock();
		for( int i = 0; i < input.getNumRows(); i++ )
			input.set(i, 0, i % 7);
		input.recomputeNonZeros();

		MatrixBlock baseline = LibMatrixSketch.getUniqueValues(input, Types.Direction.RowCol);
		MatrixBlock parallel = LibMatrixSketch.getUniqueValues(input, Types.Direction.RowCol, 4);

		assertDimensions(parallel, 7, 1, "RowCol large input dimensions");
		assertSameScalarSet(baseline, parallel, "RowCol large input baseline vs parallel");
	}

	private static void testRowLargeInput() {
		MatrixBlock input = new MatrixBlock(80000, 16, false).allocateBlock();
		for( int i = 0; i < input.getNumRows(); i++ )
			for( int j = 0; j < input.getNumColumns(); j++ )
				input.set(i, j, j % 4 + 1);
		input.recomputeNonZeros();

		MatrixBlock baseline = LibMatrixSketch.getUniqueValues(input, Types.Direction.Row);
		MatrixBlock parallel = LibMatrixSketch.getUniqueValues(input, Types.Direction.Row, 4);

		assertDimensions(parallel, input.getNumRows(), 4, "Row large input dimensions");
		assertBlockEquals(baseline, parallel, "Row large input baseline vs parallel");
		assertSameRowSet(parallel, 0, new double[] {1, 2, 3, 4}, "Row large input expected values");
	}

	private static void testColumnLargeInput() {
		MatrixBlock input = new MatrixBlock(16, 80000, false).allocateBlock();
		for( int j = 0; j < input.getNumColumns(); j++ )
			for( int i = 0; i < input.getNumRows(); i++ )
				input.set(i, j, i % 4 + 1);
		input.recomputeNonZeros();

		MatrixBlock baseline = LibMatrixSketch.getUniqueValues(input, Types.Direction.Col);
		MatrixBlock parallel = LibMatrixSketch.getUniqueValues(input, Types.Direction.Col, 4);

		assertDimensions(parallel, 4, input.getNumColumns(), "Col large input dimensions");
		assertBlockEquals(baseline, parallel, "Col large input baseline vs parallel");
		assertSameColumnSet(parallel, 0, new double[] {1, 2, 3, 4}, "Col large input expected values");
	}

	/**
	 * Compares two MatrixBlocks cell by cell with exact equality.
	 */
	private static void assertBlockEquals(MatrixBlock expected, MatrixBlock actual, String message) {
		assertDimensions(actual, expected.getNumRows(), expected.getNumColumns(), message);
		for( int i = 0; i < expected.getNumRows(); i++ )
			for( int j = 0; j < expected.getNumColumns(); j++ )
				assertEquals(message + " mismatch at (" + i + ", " + j + ")", expected.get(i, j),
					actual.get(i, j), 0);
	}

	/**
	 * Compares RowCol output as a set.
	 */
	private static void assertSameScalarSet(MatrixBlock expected, MatrixBlock actual, String message) {
		assertDimensions(actual, expected.getNumRows(), expected.getNumColumns(), message);
		HashSet<Double> expectedValues = collectScalars(expected);
		HashSet<Double> actualValues = collectScalars(actual);
		assertEquals(message + " mismatch", expectedValues, actualValues);
	}

	/**
	 * Collects all values from a one-column MatrixBlock into a set.
	 */
	private static HashSet<Double> collectScalars(MatrixBlock block) {
		HashSet<Double> ret = new HashSet<>();
		for( int i = 0; i < block.getNumRows(); i++ )
			ret.add(block.get(i, 0));
		return ret;
	}

	/**
	 * Compares one output row as a set of scalar values.
	 */
	private static void assertSameRowSet(MatrixBlock block, int row, double[] expectedValues, String message) {
		HashSet<Double> expected = collectExpected(expectedValues);
		HashSet<Double> actual = new HashSet<>();
		for( int j = 0; j < block.getNumColumns(); j++ )
			actual.add(block.get(row, j));
		assertEquals(message + " mismatch", expected, actual);
	}

	/**
	 * Compares one output column as a set of scalar values.
	 */
	private static void assertSameColumnSet(MatrixBlock block, int col, double[] expectedValues, String message) {
		HashSet<Double> expected = collectExpected(expectedValues);
		HashSet<Double> actual = new HashSet<>();
		for( int i = 0; i < block.getNumRows(); i++ )
			actual.add(block.get(i, col));
		assertEquals(message + " mismatch", expected, actual);
	}

	/**
	 * Collects expected scalar values into a set.
	 */
	private static HashSet<Double> collectExpected(double[] values) {
		HashSet<Double> ret = new HashSet<>();
		for( double value : values )
			ret.add(value);
		return ret;
	}

	/**
	 * Checks MatrixBlock dimensions and reports a readable failure.
	 */
	private static void assertDimensions(MatrixBlock block, int rows, int cols, String message) {
		assertEquals(message + " row dimension", rows, block.getNumRows());
		assertEquals(message + " column dimension", cols, block.getNumColumns());
	}
}
