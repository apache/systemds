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

import java.util.HashSet;

import org.apache.sysds.common.Types;

/**
 * Small standalone parallel unique test for LibMatrixSketch with k=1 and k>1.
 *
 * This class intentionally avoids JUnit so it can be run directly from an IDE
 * or from a full SystemDS checkout with the normal project classpath.
 */
public class LibMatrixSketchUniqueParallelTest {
	public static void main(String[] args) {
		testRowColUniqueMatchesBaseline();
		testRowUniqueMatchesBaselineAndExpectedRows();
		testColumnUniqueMatchesBaselineAndExpectedColumns();
		System.out.println("LibMatrixSketch unique parallel tests passed.");
	}

	/**
	 * Checks RowCol unique on a large single-column vector so the k=4 path is used.
	 */
	private static void testRowColUniqueMatchesBaseline() {
		MatrixBlock input = new MatrixBlock(20000, 1, false).allocateBlock();
		for( int i = 0; i < input.getNumRows(); i++ )
			input.set(i, 0, i % 7);
		input.recomputeNonZeros();

		MatrixBlock baseline = LibMatrixSketch.getUniqueValues(input, Types.Direction.RowCol);
		MatrixBlock parallel = LibMatrixSketch.getUniqueValues(input, Types.Direction.RowCol, 4);

		assertDimensions(parallel, 7, 1, "RowCol parallel dimensions");
		assertSameScalarSet(baseline, parallel, "RowCol baseline vs parallel");
	}

	/**
	 * Checks Row unique with repeated rows. The expected output is ordered by first
	 * occurrence, which also verifies the ordered global merge across row partitions.
	 */
	private static void testRowUniqueMatchesBaselineAndExpectedRows() {
		MatrixBlock input = new MatrixBlock(12000, 2, false).allocateBlock();
		for( int i = 0; i < input.getNumRows(); i++ ) {
			int pattern = i % 4;
			input.set(i, 0, pattern);
			input.set(i, 1, pattern * 10);
		}
		input.recomputeNonZeros();

		MatrixBlock baseline = LibMatrixSketch.getUniqueValues(input, Types.Direction.Row);
		MatrixBlock parallel = LibMatrixSketch.getUniqueValues(input, Types.Direction.Row, 4);
		MatrixBlock expected = matrix(new double[][] {
			{0, 0},
			{1, 10},
			{2, 20},
			{3, 30}
		});

		assertBlockEquals(expected, baseline, "Row expected vs baseline");
		assertBlockEquals(expected, parallel, "Row expected vs parallel");
	}

	/**
	 * Checks Col unique with repeated columns. The expected output shape is
	 * original_num_rows x number_of_unique_columns.
	 */
	private static void testColumnUniqueMatchesBaselineAndExpectedColumns() {
		MatrixBlock input = new MatrixBlock(4, 5000, false).allocateBlock();
		double[][] uniqueColumns = new double[][] {
			{1, 2, 3, 4},
			{5, 6, 7, 8},
			{9, 10, 11, 12}
		};

		for( int j = 0; j < input.getNumColumns(); j++ ) {
			double[] column = uniqueColumns[j % uniqueColumns.length];
			for( int i = 0; i < input.getNumRows(); i++ )
				input.set(i, j, column[i]);
		}
		input.recomputeNonZeros();

		MatrixBlock baseline = LibMatrixSketch.getUniqueValues(input, Types.Direction.Col);
		MatrixBlock parallel = LibMatrixSketch.getUniqueValues(input, Types.Direction.Col, 4);
		MatrixBlock expected = matrix(new double[][] {
			{1, 5, 9},
			{2, 6, 10},
			{3, 7, 11},
			{4, 8, 12}
		});

		assertBlockEquals(expected, baseline, "Col expected vs baseline");
		assertBlockEquals(expected, parallel, "Col expected vs parallel");
	}

	/**
	 * Builds a dense MatrixBlock from a plain two-dimensional Java array.
	 */
	private static MatrixBlock matrix(double[][] values) {
		MatrixBlock ret = new MatrixBlock(values.length, values[0].length, false).allocateBlock();
		for( int i = 0; i < values.length; i++ )
			for( int j = 0; j < values[i].length; j++ )
				ret.set(i, j, values[i][j]);
		ret.recomputeNonZeros();
		return ret;
	}

	/**
	 * Compares two MatrixBlocks cell by cell with exact equality.
	 */
	private static void assertBlockEquals(MatrixBlock expected, MatrixBlock actual, String message) {
		assertDimensions(actual, expected.getNumRows(), expected.getNumColumns(), message);
		for( int i = 0; i < expected.getNumRows(); i++ )
			for( int j = 0; j < expected.getNumColumns(); j++ )
				if( expected.get(i, j) != actual.get(i, j) )
					throw new AssertionError(message + " mismatch at (" + i + ", " + j + "): expected "
						+ expected.get(i, j) + " but found " + actual.get(i, j));
	}

	/**
	 * Compares RowCol output as a set because the scalar unique path is hash-set based.
	 */
	private static void assertSameScalarSet(MatrixBlock expected, MatrixBlock actual, String message) {
		assertDimensions(actual, expected.getNumRows(), expected.getNumColumns(), message);
		HashSet<Double> expectedValues = collectScalars(expected);
		HashSet<Double> actualValues = collectScalars(actual);
		if( !expectedValues.equals(actualValues) )
			throw new AssertionError(message + " mismatch: expected " + expectedValues + " but found " + actualValues);
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
	 * Checks MatrixBlock dimensions and reports a readable failure.
	 */
	private static void assertDimensions(MatrixBlock block, int rows, int cols, String message) {
		if( block.getNumRows() != rows || block.getNumColumns() != cols )
			throw new AssertionError(message + " dimensions mismatch: expected " + rows + "x" + cols
				+ " but found " + block.getNumRows() + "x" + block.getNumColumns());
	}
}
