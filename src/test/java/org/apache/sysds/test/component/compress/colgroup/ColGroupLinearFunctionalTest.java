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
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimatorExact;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.functionobjects.*;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
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
	public void testTsmm() {
		boolean isTransposed = true;
		// only column 0 and 2 will be compressed using LF
		double[][] data = new double[][] {{1, 2, 3, 4, 5}, {0, -1, 5, 12, 33}, {-4, -2, 0, 2, 4}};
		MatrixBlock mbt = DataConverter.convertToMatrixBlock(data);

		final int numCols = isTransposed ? mbt.getNumRows() : mbt.getNumColumns();
		final int numRows = isTransposed ? mbt.getNumColumns() : mbt.getNumRows();
		int[] colIndexes = new int[]{0, 2};

		AColGroup cgCompressed = createCompressedColGroup(mbt, colIndexes, isTransposed);
		AColGroup cgUncompressed = createUncompressedColGroup(mbt, colIndexes, isTransposed);

		final MatrixBlock resultUncompressed = new MatrixBlock(numCols, numCols, false);
		resultUncompressed.allocateDenseBlock();
		cgUncompressed.tsmm(resultUncompressed, numRows);

		final MatrixBlock resultCompressed = new MatrixBlock(numCols, numCols, false);
		resultCompressed.allocateDenseBlock();
		cgCompressed.tsmm(resultCompressed, numRows);

		Assert.assertArrayEquals(resultUncompressed.getDenseBlockValues(), resultCompressed.getDenseBlockValues(), 0.001);
	}

	@Test
	public void testRightMultByMatrix() {
		boolean transposedRight = false;
		boolean transposedLeft = true;
		double[][] dataLeft = new double[][] {{8, 4, 0, -4, -8}, {-1, 0, 1, 2, 3}};
		double[][] dataRight = new double[][] {{8, 3, 7, 12}, {-1, 8, 4, -2}};

		MatrixBlock mbtLeft = DataConverter.convertToMatrixBlock(dataLeft);
		MatrixBlock mbtRight = DataConverter.convertToMatrixBlock(dataRight);

		final int numColsRight = transposedRight ? mbtRight.getNumRows() : mbtRight.getNumColumns();
		final int numRowsRight = transposedRight ? mbtRight.getNumColumns() : mbtRight.getNumRows();
		int[] colIndexesRight = new int[numColsRight];
		for(int x = 0; x < numColsRight; x++)
			colIndexesRight[x] = x;

		final int numColsLeft = transposedLeft ? mbtLeft.getNumRows() : mbtLeft.getNumColumns();
		final int numRowsLeft = transposedLeft ? mbtLeft.getNumColumns() : mbtLeft.getNumRows();
		int[] colIndexesLeft = new int[numColsLeft];
		for(int x = 0; x < numColsLeft; x++)
			colIndexesLeft[x] = x;

		AColGroup cgCompressedLeft = createCompressedColGroup(mbtLeft, colIndexesLeft, transposedLeft);

		final MatrixBlock resultExpected = new MatrixBlock(numRowsLeft, numColsRight, false);

		ColGroupUncompressed colGroupResult = (ColGroupUncompressed) cgCompressedLeft.rightMultByMatrix(mbtRight);
		final MatrixBlock result = colGroupResult.getData();

		if(transposedLeft) {
			mbtLeft = LibMatrixReorg.transpose(mbtLeft, InfrastructureAnalyzer.getLocalParallelism());
		}

		if(transposedRight) {
			mbtRight = LibMatrixReorg.transpose(mbtRight, InfrastructureAnalyzer.getLocalParallelism());
		}

		LibMatrixMult.matrixMult(mbtLeft, mbtRight, resultExpected);

		Assert.assertArrayEquals(resultExpected.getDenseBlockValues(), result.getDenseBlockValues(), 0.001);
	}

	@Test
	public void testLeftMultByAColGroupLFCompressed() {
		leftMultByAColGroup(true);
	}

	@Test
	public void testLeftMultByAColGroupUncompressed() {
		leftMultByAColGroup(false);
	}

	public void leftMultByAColGroup(boolean compressedLeft) {
		boolean transposedRight = true;
		boolean transposedLeft = true;
		double[][] dataRight = new double[][] {{1, 2, 3, 4, 5}, {-4, -2, 0, 2, 4}};
		int[] colIndexesRight = new int[]{1};
		double[][] dataLeft;
		int[] colIndexesLeft;
		if(compressedLeft) {
			dataLeft = new double[][] {{8, 4, 0, -4, -8}, {-1, 0, 1, 2, 3}};
			colIndexesLeft = new int[]{0};
		} else {
			dataLeft = new double[][] {{8, 3, 7, 12, -3}, {-1, 8, 4, -2, -2}, {3, 4, 2, 0, -1}};
			colIndexesLeft = new int[]{0, 2};
		}

		MatrixBlock mbtLeft = DataConverter.convertToMatrixBlock(dataLeft);
		MatrixBlock mbtRight = DataConverter.convertToMatrixBlock(dataRight);

		final int numColsRight = transposedRight ? mbtRight.getNumRows() : mbtRight.getNumColumns();
		final int numRowsRight = transposedRight ? mbtRight.getNumColumns() : mbtRight.getNumRows();

		final int numColsLeft = transposedLeft ? mbtLeft.getNumRows() : mbtLeft.getNumColumns();
		final int numRowsLeft = transposedLeft ? mbtLeft.getNumColumns() : mbtLeft.getNumRows();

		AColGroup cgCompressedRight = createCompressedColGroup(mbtRight, colIndexesRight, transposedRight);
		AColGroup cgUncompressedRight = createUncompressedColGroup(mbtRight, colIndexesRight, transposedRight);
		AColGroup cgCompressedLeft = createCompressedColGroup(mbtLeft, colIndexesLeft, transposedLeft);;
		AColGroup cgUncompressedLeft = createUncompressedColGroup(mbtLeft, colIndexesLeft, transposedLeft);

		final MatrixBlock result = new MatrixBlock(numColsLeft, numColsRight, false);
		final MatrixBlock resultExpected = new MatrixBlock(numColsLeft, numColsRight, false);
		result.allocateDenseBlock();
		resultExpected.allocateDenseBlock();

		cgUncompressedRight.leftMultByAColGroup(cgUncompressedLeft, resultExpected);

		if(compressedLeft) {
			cgCompressedRight.leftMultByAColGroup(cgCompressedLeft, result);
		} else {
			cgCompressedRight.leftMultByAColGroup(cgUncompressedLeft, result);
		}

		int[][] colIndexesArray = new int[][]{colIndexesLeft, colIndexesRight};
		MatrixBlock[] mbts = new MatrixBlock[]{mbtLeft, mbtRight};
		boolean[] transposedArray = new boolean[]{transposedLeft, transposedRight};

		for(int idx = 0; idx < mbts.length; idx++) {
			MatrixBlock mbt = mbts[idx];
			int[] colIndexes = colIndexesArray[idx];
			boolean transposed = transposedArray[idx];

			zeroColumsNotInColIndexes(mbt, colIndexes, transposed);
		}

		// since left matrix is transposed in leftMultByAColGroup, we simulate this behaviour here
		if(!transposedLeft) {
			mbtLeft = LibMatrixReorg.transpose(mbtLeft, InfrastructureAnalyzer.getLocalParallelism());
		}

		if(transposedRight) {
			mbtRight = LibMatrixReorg.transpose(mbtRight, InfrastructureAnalyzer.getLocalParallelism());
		}

		LibMatrixMult.matrixMult(mbtLeft, mbtRight, resultExpected);

		Assert.assertArrayEquals(resultExpected.getDenseBlockValues(), result.getDenseBlockValues(), 0.001);
	}

	public void zeroColumsNotInColIndexes(MatrixBlock mbt, int[] colIndexes, boolean transposed) {
		for(int r = 0; r < mbt.getNumRows(); r++) {
			for(int c = 0; c < mbt.getNumColumns(); c++) {
				boolean setZero = true;
				for(int colIndex : colIndexes) {
					if((!transposed && colIndex == c) || (transposed && colIndex == r)) {
						setZero = false;
						break;
					}
				}

				if(setZero) {
					mbt.setValue(r, c, 0);
				}
			}
		}
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

		Assert.assertArrayEquals(colSumsExpected, colSums, 0.001);
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

		Assert.assertEquals(productExpected, product[0], 0.001);
	}

	@Test
	public void testMax() {
		boolean isTransposed = false;
		double[][] data = new double[][] {{1, 2, 3, 4, 5}, {-4, 2, 8, 5.3, -100}};

		double maxExpected = Double.NEGATIVE_INFINITY;
		for(int j = 0; j < data[0].length; j++) {
			for(int i = 0; i < data.length; i++) {
				if(data[i][j] > maxExpected) {
					maxExpected = data[i][j];
				}
			}
		}

		MatrixBlock mbt = DataConverter.convertToMatrixBlock(data);

		final int numCols = isTransposed ? mbt.getNumRows() : mbt.getNumColumns();
		final int numRows = isTransposed ? mbt.getNumColumns() : mbt.getNumRows();
		int[] colIndexes = new int[numCols];
		for(int x = 0; x < numCols; x++)
			colIndexes[x] = x;

		AColGroup cg = createCompressedColGroup(mbt, colIndexes, isTransposed);

		Assert.assertEquals(maxExpected, cg.getMax(), 0.001);
	}

	@Test
	public void testMin() {
		boolean isTransposed = false;
		double[][] data = new double[][] {{1, 2, 3, 4, 5}, {-4, 2, 8, 5.3, -100}};

		double minExpected = Double.POSITIVE_INFINITY;
		for(int j = 0; j < data[0].length; j++) {
			for(int i = 0; i < data.length; i++) {
				if(data[i][j] < minExpected) {
					minExpected = data[i][j];
				}
			}
		}

		MatrixBlock mbt = DataConverter.convertToMatrixBlock(data);

		final int numCols = isTransposed ? mbt.getNumRows() : mbt.getNumColumns();
		final int numRows = isTransposed ? mbt.getNumColumns() : mbt.getNumRows();
		int[] colIndexes = new int[numCols];
		for(int x = 0; x < numCols; x++)
			colIndexes[x] = x;

		AColGroup cg = createCompressedColGroup(mbt, colIndexes, isTransposed);

		Assert.assertEquals(minExpected, cg.getMin(), 0.001);
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

		Assert.assertEquals(sumSqExpected, sumSq[0], 0.001);
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

		Assert.assertEquals(sumExpected, sumSq[0], 0.001);
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

		Assert.assertArrayEquals(rowSumExpected, rowSums, 0.001);
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

		Assert.assertArrayEquals(colSumsExpected, colSums, 0.001);
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

	public AColGroup createUncompressedColGroup(MatrixBlock mbt, int[] colIndexes, boolean isTransposed) {
		CompressionSettings cs = new CompressionSettingsBuilder().setSamplingRatio(1.0)
			.setValidCompressions(EnumSet.of(AColGroup.CompressionType.UNCOMPRESSED)).create();
		cs.transposed = isTransposed;

		final CompressedSizeInfoColGroup cgi = new CompressedSizeEstimatorExact(mbt, cs)
			.getColGroupInfo(colIndexes);
		CompressedSizeInfo csi = new CompressedSizeInfo(cgi);
		AColGroup cg = ColGroupFactory.compressColGroups(mbt, csi, cs, 1).get(0);

		Assert.assertSame(cg.getCompType(), AColGroup.CompressionType.UNCOMPRESSED);
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

	@Test(expected = AssertionError.class)
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
