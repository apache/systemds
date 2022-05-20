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

package org.apache.sysds.test.component.compress.functional;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.stream.DoubleStream;

import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.functional.LinearRegression;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.component.compress.colgroup.ColGroupLinearFunctionalBase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
public class LinearRegressionTests {
	protected final double[][] data;
	protected final int[] colIndexes;
	protected final boolean isTransposed;
	protected final double[] expectedCoefficients;
	protected final Exception expectedException;

	protected final double EQUALITY_TOLERANCE = 1e-4;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		try {
			addCases(tests);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed constructing tests");
		}

		return tests;
	}

	public LinearRegressionTests(double[][] data, int[] colIndexes, boolean isTransposed, double[] expectedCoefficients,
		Exception expectedException) {
		this.data = data;
		this.colIndexes = colIndexes;
		this.isTransposed = isTransposed;
		this.expectedCoefficients = expectedCoefficients;
		this.expectedException = expectedException;
	}

	protected static void addCases(ArrayList<Object[]> tests) {
		double[][] data = new double[][] {{1, 1, -3, 4, 5}, {2, 2, 3, 4, 5}, {3, 3, 3, 4, 5}};
		int[] colIndexes = new int[] {0, 1, 3, 4};
		double[] trueCoefficients = new double[] {0, 0, 4, 5, 1, 1, 0, 0};
		tests.add(new Object[] {data, colIndexes, false, trueCoefficients, null});

		data = new double[][] {{1}, {2}, {3}};
		colIndexes = new int[] {0};
		trueCoefficients = new double[] {0, 1};
		tests.add(new Object[] {data, colIndexes, false, trueCoefficients, null});

		// expect exception if passing columns with single data points each
		tests.add(new Object[] {new double[][] {{1, 2, 3}}, Util.genColsIndices(1), false, null,
			new DMLCompressionException("At least 2 data points are required to fit a linear function.")});

		// expect exception if passing no colIndexes
		tests.add(new Object[] {new double[][] {{1, 2, 3}, {2, 3, 4}}, Util.genColsIndices(0), false, null,
			new DMLCompressionException("At least 1 column must be specified for compression.")});

		// random matrix
		int rows = 100;
		int cols = 200;
		// TODO: move generateRandomInterceptsSlopes in an appropriate Util class
		// TODO: move generateTestMatrixLinearColumns in an appropriate Util class
		double[][] randomCoefficients = ColGroupLinearFunctionalBase.generateRandomInterceptsSlopes(cols, -1000, 1000,
			-20, 20, 42);
		double[][] testData = ColGroupLinearFunctionalBase.generateTestMatrixLinearColumns(rows, cols,
			randomCoefficients[0], randomCoefficients[1]);
		tests.add(new Object[] {testData, Util.genColsIndices(cols), false,
			DoubleStream.concat(Arrays.stream(randomCoefficients[0]), Arrays.stream(randomCoefficients[1])).toArray(),
			null});
	}

	@Test
	public void testLinearRegression() {
		MatrixBlock mbt = DataConverter.convertToMatrixBlock(data);
		try {
			double[] coefficients = LinearRegression.regressMatrixBlock(mbt, colIndexes, isTransposed);
			assertArrayEquals(expectedCoefficients, coefficients, EQUALITY_TOLERANCE);
		}
		catch(Exception e) {
			assertEquals(expectedException.getClass(), e.getClass());
			assertEquals(expectedException.getMessage(), e.getMessage());
		}
	}

	@Test
	public void testLineratRegressionEquivalentTransposed() {
		MatrixBlock mbt = DataConverter.convertToMatrixBlock(data);
		MatrixBlock mbtt = LibMatrixReorg.transpose(mbt);
		try {
			double[] coefficients = LinearRegression.regressMatrixBlock(mbtt, colIndexes, !isTransposed);
			assertArrayEquals(expectedCoefficients, coefficients, EQUALITY_TOLERANCE);
		}
		catch(Exception e) {
			assertEquals(expectedException.getClass(), e.getClass());
			assertEquals(expectedException.getMessage(), e.getMessage());
		}
	}

}
