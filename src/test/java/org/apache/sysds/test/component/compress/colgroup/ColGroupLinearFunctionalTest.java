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

package org.apache.sysds.test.component.compress.colgroup;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimatorExact;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.functionobjects.*;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.util.DataConverter;
import org.junit.Assert;
import org.junit.Test;

import java.util.EnumSet;
import java.util.Random;

public class ColGroupLinearFunctionalTest {

	protected static final Log LOG = LogFactory.getLog(ColGroupLinearFunctionalTest.class.getName());
	private final static Random random = new Random();

	public double[][] generatePointsOnLine(double intercept, double slope, int length) {
		double[] result = new double[length];
		for(int i = 0; i < length; i++) {
			result[i] = intercept + slope * i;
		}

		return new double[][] {result};
	}

	@Test
	public void testColSumsSq() {
		boolean isTransposed = false;
		double[][] data = new double[][] {{1, 2, 3, 4, 5}, {-4, 2, 8, 53, -100}};

		double[] colSumsExpected = new double[data[0].length];
		for(int j = 0; j < data[0].length; j++) {
			double colSum = 0;
			for(int i = 0; i < data.length; i++) {
				colSum += Math.pow(data[i][j], 2);
			}
			colSumsExpected[j] = colSum;
		}

		double[] colSums = new double[data[0].length];
		AggregateOperator aop = new AggregateOperator(0, KahanPlusSq.getKahanPlusSqFnObject());
		AggregateUnaryOperator auop = new AggregateUnaryOperator(aop, ReduceRow.getReduceRowFnObject());
		unaryAggregate(data, isTransposed, auop, colSums);

		Assert.assertArrayEquals(colSums, colSumsExpected, 0.001);
	}

	@Test
	public void testProduct() {
		boolean isTransposed = false;
		double[][] data = new double[][] {{1, 2, 3, 4, 5}, {-4, 2, 8, 5.3, -100}};

		double productExpected = 1;
		for(int j = 0; j < data[0].length; j++) {
			for(int i = 0; i < data.length; i++) {
				productExpected *= data[i][j];
			}
		}

		double[] product = new double[data[0].length];
		AggregateOperator aop = new AggregateOperator(0, Multiply.getMultiplyFnObject());
		AggregateUnaryOperator auop = new AggregateUnaryOperator(aop, ReduceAll.getReduceAllFnObject());
		unaryAggregate(data, isTransposed, auop, product);

		Assert.assertEquals(product[0], productExpected, 0.001);
	}

	@Test
	public void testColProduct() {
		boolean isTransposed = false;
		double[][] data = new double[][] {{1, 2, 3, 4, 5}, {-4, 2, 8, 5.3, -100}};

		double[] productExpected = new double[data[0].length];
		for(int j = 0; j < data[0].length; j++) {
			double colProduct = 1;
			for(int i = 0; i < data.length; i++) {
				colProduct *= data[i][j];
			}
			productExpected[j] = colProduct;
		}

		double[] product = new double[data[0].length];
		AggregateOperator aop = new AggregateOperator(0, Multiply.getMultiplyFnObject());
		AggregateUnaryOperator auop = new AggregateUnaryOperator(aop, ReduceRow.getReduceRowFnObject());
		unaryAggregate(data, isTransposed, auop, product);

		Assert.assertArrayEquals(productExpected, product, 0.001);
	}

	@Test
	public void testSumSq() {
		boolean isTransposed = false;
		double[][] data = new double[][] {{1, 2, 3, 4, 5}, {-4, 2, 8, 53, -100}};

		double sumSqExpected = 0;
		for(int j = 0; j < data[0].length; j++) {
			for(int i = 0; i < data.length; i++) {
				sumSqExpected += Math.pow(data[i][j], 2);
			}
		}

		double[] sumSq = new double[data[0].length];
		AggregateOperator aop = new AggregateOperator(0, KahanPlusSq.getKahanPlusSqFnObject());
		AggregateUnaryOperator auop = new AggregateUnaryOperator(aop, ReduceAll.getReduceAllFnObject());
		unaryAggregate(data, isTransposed, auop, sumSq);

		Assert.assertEquals(sumSq[0], sumSqExpected, 0.001);
	}

	@Test
	public void testSum() {
		boolean isTransposed = false;
		double[][] data = new double[][] {{1, 2, 3, 4, 5}, {-4, 2, 8, 53, -100}};

		double sumExpected = 0;
		for(int j = 0; j < data[0].length; j++) {
			for(int i = 0; i < data.length; i++) {
				sumExpected += data[i][j];
			}
		}

		double[] sumSq = new double[data[0].length];
		AggregateOperator aop = new AggregateOperator(0, Plus.getPlusFnObject());
		AggregateUnaryOperator auop = new AggregateUnaryOperator(aop, ReduceAll.getReduceAllFnObject());
		unaryAggregate(data, isTransposed, auop, sumSq);

		Assert.assertEquals(sumSq[0], sumExpected, 0.001);
	}

	@Test
	public void testRowSums() {
		boolean isTransposed = false;
		double[][] data = new double[][] {{1, 2, 3, 4, 5}, {-4, 2, 8, 53, -100}};

		double[] rowSumExpected = new double[data.length];
		for(int i = 0; i < data.length; i++) {
			double rowSum = 0;
			for(int j = 0; j < data[0].length; j++) {
				rowSum += data[i][j];
			}
			rowSumExpected[i] = rowSum;
		}

		double[] rowSums = new double[data.length];
		AggregateOperator aop = new AggregateOperator(0, Plus.getPlusFnObject());
		AggregateUnaryOperator auop = new AggregateUnaryOperator(aop, ReduceCol.getReduceColFnObject());
		unaryAggregate(data, isTransposed, auop, rowSums);

		Assert.assertArrayEquals(rowSums, rowSumExpected, 0.001);
	}

	public void unaryAggregate(double[][] data, boolean isTransposed, AggregateUnaryOperator auop, double[] res) {
		MatrixBlock mbt = DataConverter.convertToMatrixBlock(data);

		final int numCols = isTransposed ? mbt.getNumRows() : mbt.getNumColumns();
		final int numRows = isTransposed ? mbt.getNumColumns() : mbt.getNumRows();
		int[] colIndexes = new int[numCols];
		for(int x = 0; x < numCols; x++)
			colIndexes[x] = x;

		AColGroup cg = createCompressedColGroup(mbt, colIndexes, isTransposed);
		cg.unaryAggregateOperations(auop, res, numRows, 0, numRows);
	}

	@Test
	public void testColSums() {
		boolean isTransposed = false;
		double[][] data = new double[][] {{1, 2, 3, 4, 5}, {-4, 2, 8, 53, -100}};

		double[] colSumsExpected = new double[data[0].length];
		for(int j = 0; j < data[0].length; j++) {
			double colSum = 0;
			for(int i = 0; i < data.length; i++) {
				colSum += data[i][j];
			}
			colSumsExpected[j] = colSum;
		}

		double[] colSums = new double[data[0].length];
		AggregateOperator aop = new AggregateOperator(0, Plus.getPlusFnObject());
		AggregateUnaryOperator auop = new AggregateUnaryOperator(aop, ReduceRow.getReduceRowFnObject());
		unaryAggregate(data, isTransposed, auop, colSums);

		Assert.assertArrayEquals(colSums, colSumsExpected, 0.001);
	}

	public AColGroup createCompressedColGroup(MatrixBlock mbt, int[] colIndexes, boolean isTransposed) {
		CompressionSettings cs = new CompressionSettingsBuilder().setSamplingRatio(1.0)
			.setValidCompressions(EnumSet.of(AColGroup.CompressionType.LinearFunctional)).create();
		cs.transposed = isTransposed;

		final CompressedSizeInfoColGroup cgi = new CompressedSizeEstimatorExact(mbt, cs)
			.getColGroupInfo(colIndexes);
		CompressedSizeInfo csi = new CompressedSizeInfo(cgi);
		AColGroup cg = ColGroupFactory.compressColGroups(mbt, csi, cs, 1).get(0);

		Assert.assertSame(cg.getCompType(), AColGroup.CompressionType.LinearFunctional);
		return cg;
	}

	@Test
	public void testRandomColumnCompression() {
		double intercept = random.nextInt(1000) - 500 + random.nextDouble();
		double slope = random.nextInt(1000) - 500 + random.nextDouble();

		double[][] column = generatePointsOnLine(intercept, slope, 5000);

		testDecompressToDenseBlock(column, true);
	}

	@Test
	public void testDecompressToDenseBlockSingleColumn() {
		testDecompressToDenseBlock(new double[][] {{1, 2, 3, 4, 5}}, true);
	}

	@Test
	public void testDecompressToDenseBlockSingleColumnTransposed() {
		testDecompressToDenseBlock(new double[][] {{1}, {2}, {3}, {4}, {5}}, false);
	}

	@Test
	public void testDecompressToDenseBlockTwoColumns() {
		testDecompressToDenseBlock(new double[][] {{1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}}, false);
	}

	@Test
	public void testDecompressToDenseBlockTwoColumnsUnequalSlopeTransposed() {
		testDecompressToDenseBlock(new double[][] {{1, 2, 3, 4, 5}, {-1, -2, -3, -4, -5}}, true);
	}

	@Test
	public void testDecompressToDenseBlockTwoColumnsTransposed() {
		testDecompressToDenseBlock(new double[][] {{1, 2, 3, 4, 5}, {1, 1, 1, 1, 1}}, true);
	}

	@Test
	public void testDecompressToDenseBlockTwoColumnsNonLinearTransposed() {
		testDecompressToDenseBlock(new double[][] {{1, 2, 3, 4, 5, 0}, {1, 1, 1, 1, 1, 2}}, true);
	}

	public void testDecompressToSparseBlock(double[][] data, boolean isTransposed) {
		MatrixBlock mbt = DataConverter.convertToMatrixBlock(data);

	}

	public void testDecompressToDenseBlock(double[][] data, boolean isTransposed) {
		MatrixBlock mbt = DataConverter.convertToMatrixBlock(data);

		final int numCols = isTransposed ? mbt.getNumRows() : mbt.getNumColumns();
		final int numRows = isTransposed ? mbt.getNumColumns() : mbt.getNumRows();
		int[] colIndexes = new int[numCols];
		for(int x = 0; x < numCols; x++)
			colIndexes[x] = x;

		AColGroup cg = createCompressedColGroup(mbt, colIndexes, isTransposed);

		try {
			// Decompress to dense block
			MatrixBlock ret = new MatrixBlock(numRows, numCols, false);
			ret.allocateDenseBlock();
			cg.decompressToDenseBlock(ret.getDenseBlock(), 0, numRows);

			MatrixBlock expected = DataConverter.convertToMatrixBlock(data);
			if(isTransposed)
				LibMatrixReorg.transposeInPlace(expected, 1);
			Assert.assertArrayEquals(expected.getDenseBlockValues(), ret.getDenseBlockValues(), 0.01);

		}
		catch(Exception e) {
			e.printStackTrace();
			throw new DMLRuntimeException("Failed construction : " + this.getClass().getSimpleName());
		}
	}

}
