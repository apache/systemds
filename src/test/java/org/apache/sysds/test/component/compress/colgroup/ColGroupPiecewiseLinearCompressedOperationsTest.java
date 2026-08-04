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

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.colgroup.ColGroupPiecewiseLinearCompressed;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.DenseBlockFP64;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Multiply2;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.Power2;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests for ColGroupPiecewiseLinearCompressed operations containing: scalarOperation, binaryRowOps, computeSum,
 * containsValue, getIdx, getExactSizeOnDisk.
 */
public class ColGroupPiecewiseLinearCompressedOperationsTest extends AutomatedTestBase {

	private static final long SEED = 42L;
	private static final int NROWS = 50;
	private static final int NCOLS = 3;
	private static final double TARGET_LOSS = 50;
	private static final double DELTA = 1e-9;

	private ColGroupPiecewiseLinearCompressed piecewiseLinearColGroup;
	private MatrixBlock originalMB;
	private MatrixBlock decompressedMB;
	private IColIndex colIndexes;
	private int numRows;
	private int numCols;

	@Before
	public void setUp() {
		numRows = NROWS;
		numCols = NCOLS;

		/// generate random matrix
		double[][] data = super.getRandomMatrix(numRows, numCols, -30, 30, 1.0, SEED);
		originalMB = DataConverter.convertToMatrixBlock(data);
		originalMB.allocateDenseBlock();

		colIndexes = ColIndexFactory.create(buildColArray(numCols));

		CompressionSettings cs = new CompressionSettingsBuilder().create();
		cs.setPiecewiseTargetLoss(TARGET_LOSS);

		/// create ColGroupPiecewiseLinearCompressed instance
		AColGroup result = ColGroupFactory.compressPiecewiseLinearFunctional(colIndexes, originalMB, cs);
		assertTrue(result instanceof ColGroupPiecewiseLinearCompressed);
		piecewiseLinearColGroup = (ColGroupPiecewiseLinearCompressed) result;

		/// decompress again
		decompressedMB = decompress(piecewiseLinearColGroup);
	}

	private MatrixBlock decompress(AColGroup cg) {
		MatrixBlock mb = new MatrixBlock(numRows, numCols, false);
		mb.allocateDenseBlock();
		cg.decompressToDenseBlock(mb.getDenseBlock(), 0, numRows, 0, 0);
		return mb;
	}

	/// check elementwise to compare results from compressed and decompressed matrixblock
	private void checkMatrixEquals(String msg, MatrixBlock mb1, MatrixBlock mb2) {
		if(mb1.getNumRows() != mb2.getNumRows() || mb1.getNumColumns() != mb2.getNumColumns())
			fail(msg + " dimension mismatch");
		for(int r = 0; r < numRows; r++)
			for(int c = 0; c < numCols; c++)
				assertEquals(msg + "[" + r + "," + c + "]", mb1.get(r, c), mb2.get(r, c), DELTA);
	}

	/// compute column sum to validate
	private double[] computeSums(MatrixBlock mb) {
		double[] sums = new double[numCols];
		for(int c = 0; c < numCols; c++)
			for(int r = 0; r < numRows; r++)
				sums[c] += mb.get(r, c);
		return sums;
	}

	/// create row vector
	private double[] buildRowVector() {
		double[] v = new double[numCols];
		for(int i = 0; i < numCols; i++)
			v[i] = 0.5 * (i + 1);
		return v;
	}

	private int[] buildColArray(int n) {
		int[] cols = new int[n];
		for(int i = 0; i < n; i++)
			cols[i] = i;
		return cols;
	}

	private MatrixBlock applyBinaryRowOpLeft(MatrixBlock mb, BinaryOperator op, double[] v) {
		MatrixBlock result = new MatrixBlock(numRows, numCols, false);
		result.allocateDenseBlock();
		for(int r = 0; r < numRows; r++)
			for(int c = 0; c < numCols; c++)
				result.getDenseBlock().set(r, c, op.fn.execute(v[c], mb.get(r, c)));
		return result;
	}

	private MatrixBlock applyBinaryRowOpRight(MatrixBlock mb, BinaryOperator op, double[] v) {
		MatrixBlock result = new MatrixBlock(numRows, numCols, false);
		result.allocateDenseBlock();
		for(int r = 0; r < numRows; r++)
			for(int c = 0; c < numCols; c++)
				result.getDenseBlock().set(r, c, op.fn.execute(mb.get(r, c), v[c]));
		return result;
	}

	@Test
	public void testComputeSum() {
		double[] sumsComp = new double[1];
		piecewiseLinearColGroup.computeSum(sumsComp, numRows);
		double expectedTotal = 0;
		for(double s : computeSums(decompressedMB))
			expectedTotal += s;
		assertEquals(expectedTotal, sumsComp[0], DELTA);
	}

	@Test
	public void testComputeColSums() {
		double[] sumsComp = new double[numCols];
		piecewiseLinearColGroup.computeColSums(sumsComp, numRows);
		assertArrayEquals(sumsComp, computeSums(decompressedMB), DELTA);
	}

	private void testScalarOp(ScalarOperator op, double scalar) {
		MatrixBlock expected = new MatrixBlock(numRows, numCols, false);
		expected.allocateDenseBlock();
		for(int r = 0; r < numRows; r++)
			for(int c = 0; c < numCols; c++)
				expected.getDenseBlock().set(r, c, op.fn.execute(decompressedMB.get(r, c), scalar));

		checkMatrixEquals("scalarOp " + op.fn.getClass().getSimpleName(), expected,
			decompress(piecewiseLinearColGroup.scalarOperation(op)));
	}

	@Test
	public void testScalarPlus() {
		testScalarOp(new RightScalarOperator(Plus.getPlusFnObject(), 3.7), 3.7);
	}

	@Test
	public void testScalarMinus() {
		testScalarOp(new RightScalarOperator(Minus.getMinusFnObject(), 1.5), 1.5);
	}

	@Test
	public void testScalarMultiply() {
		testScalarOp(new RightScalarOperator(Multiply.getMultiplyFnObject(), 2.0), 2.0);
	}

	@Test
	public void testScalarDivide() {
		testScalarOp(new RightScalarOperator(Divide.getDivideFnObject(), 4.0), 4.0);
	}

	@Test
	public void testBinaryRowOpLeftPlus() {
		BinaryOperator op = new BinaryOperator(Plus.getPlusFnObject());
		double[] v = buildRowVector();
		checkMatrixEquals("binaryRowOpLeft Plus", applyBinaryRowOpLeft(decompressedMB, op, v),
			decompress(piecewiseLinearColGroup.binaryRowOpLeft(op, v, false)));
	}

	@Test
	public void testBinaryRowOpLeftMultiply() {
		BinaryOperator op = new BinaryOperator(Multiply.getMultiplyFnObject());
		double[] v = buildRowVector();
		checkMatrixEquals("binaryRowOpLeft Multiply", applyBinaryRowOpLeft(decompressedMB, op, v),
			decompress(piecewiseLinearColGroup.binaryRowOpLeft(op, v, false)));
	}

	@Test
	public void testBinaryRowOpRightMinus() {
		BinaryOperator op = new BinaryOperator(Minus.getMinusFnObject());
		double[] v = buildRowVector();
		checkMatrixEquals("binaryRowOpRight Minus", applyBinaryRowOpRight(decompressedMB, op, v),
			decompress(piecewiseLinearColGroup.binaryRowOpRight(op, v, false)));
	}

	@Test
	public void testBinaryRowOpRightDivide() {
		BinaryOperator op = new BinaryOperator(Divide.getDivideFnObject());
		double[] v = buildRowVector();
		checkMatrixEquals("binaryRowOpRight Divide", applyBinaryRowOpRight(decompressedMB, op, v),
			decompress(piecewiseLinearColGroup.binaryRowOpRight(op, v, false)));
	}

	@Test
	public void testContainsValueIntercept() {
		double pattern = piecewiseLinearColGroup.getInterceptsPerCol()[0][0];
		assertTrue("intercept of col 0 seg 0 should exist", piecewiseLinearColGroup.containsValue(pattern));
	}

	@Test
	public void testContainsValueEndpoint() {
		int[] breakpoints = piecewiseLinearColGroup.getBreakpointsPerCol()[0];
		double[] intercepts = piecewiseLinearColGroup.getInterceptsPerCol()[0];
		double[] slopes = piecewiseLinearColGroup.getSlopesPerCol()[0];
		if(breakpoints.length > 1) {
			double pattern = intercepts[0] + slopes[0] * (breakpoints[1] - breakpoints[0] - 1);
			assertTrue("endpoint of col 0 seg 0 should exist", piecewiseLinearColGroup.containsValue(pattern));
		}
	}

	@Test
	public void testContainsValueConstantSegment() {
		ColGroupPiecewiseLinearCompressed cg = (ColGroupPiecewiseLinearCompressed) ColGroupPiecewiseLinearCompressed
			.create(ColIndexFactory.create(new int[] {0}), new int[][] {{0, numRows}}, new double[][] {{0.0}},
				new double[][] {{1.23}}, numRows);

		assertTrue("constant value 1.23 should exist", cg.containsValue(1.23));
		assertFalse("value 2.0 should not exist", cg.containsValue(2.0));
	}

	@Test
	public void testContainsValueOutsideRange() {
		assertFalse("value -10 outside data range", piecewiseLinearColGroup.containsValue(-10.0));
		assertFalse("value +10 outside data range", piecewiseLinearColGroup.containsValue(10.0));
	}

	@Test
	public void testGetIdxMatchesDecompress() {
		for(int c = 0; c < numCols; c++)
			for(int r = 0; r < numRows; r++)
				assertEquals("getIdx(" + r + "," + c + ")", decompressedMB.get(r, c),
					piecewiseLinearColGroup.getIdx(r, c), 1e-10);
	}

	@Test
	public void testGetIdxInvalidBounds() {
		assertEquals("row < 0", 0.0, piecewiseLinearColGroup.getIdx(-1, 0), DELTA);
		assertEquals("row >= numRows", 0.0, piecewiseLinearColGroup.getIdx(numRows, 0), DELTA);
		assertEquals("col < 0", 0.0, piecewiseLinearColGroup.getIdx(0, -1), DELTA);
		assertEquals("col >= ncols", 0.0, piecewiseLinearColGroup.getIdx(0, numCols), DELTA);
	}

	@Test
	public void testGetNumValues() {
		int expected = 0;
		for(int c = 0; c < numCols; c++) {
			int breakpointsLen = piecewiseLinearColGroup.getBreakpointsPerCol()[c].length;
			int slopesLen = piecewiseLinearColGroup.getSlopesPerCol()[c].length;
			int interceptsLen = piecewiseLinearColGroup.getInterceptsPerCol()[c].length;
			assertEquals("breakpoints != slopes+1 for col " + c, breakpointsLen, slopesLen + 1);
			assertEquals("slopes != intercepts for col " + c, slopesLen, interceptsLen);
			expected += breakpointsLen + slopesLen + interceptsLen;
		}
		assertEquals("getNumValues() mismatch", expected, piecewiseLinearColGroup.getNumValues());
	}

	@Test
	public void testGetExactSizeOnDisk() {
		Random rng = new Random(SEED);
		int rows = 80 + rng.nextInt(40);
		int numSegs = 1 + rng.nextInt(3);

		int[] breakpoints = new int[numSegs + 1];
		breakpoints[0] = 0;
		breakpoints[numSegs] = rows;
		for(int s = 1; s < numSegs; s++)
			breakpoints[s] = rng.nextInt(rows * 2 / 3) + rows / 10;

		double[] slopes = new double[numSegs];
		double[] intercepts = new double[numSegs];
		for(int s = 0; s < numSegs; s++) {
			slopes[s] = rng.nextDouble() * 4 - 2;
			intercepts[s] = rng.nextDouble() * 4 - 2;
		}
		/// PLC Piecewise Linear Compressed
		AColGroup colGroupPLC = ColGroupPiecewiseLinearCompressed.create(
			ColIndexFactory.create(new int[] {rng.nextInt(20)}), new int[][] {breakpoints}, new double[][] {slopes},
			new double[][] {intercepts}, rows);

		assertTrue("disk size should be positive", colGroupPLC.getExactSizeOnDisk() > 0);
		assertTrue("num values should be positive", colGroupPLC.getNumValues() > 0);
	}

	@Test
	public void testCreate() {
		ColGroupPiecewiseLinearCompressed plc = (ColGroupPiecewiseLinearCompressed) piecewiseLinearColGroup;

		AColGroup result = ColGroupPiecewiseLinearCompressed.create(plc.getColIndices(), plc.getBreakpointsPerCol(),
			plc.getSlopesPerCol(), plc.getInterceptsPerCol(), NROWS);
		assertTrue(result instanceof ColGroupPiecewiseLinearCompressed);

		// equal to piecewiseLinearColGroup instance
		assertArrayEquals(((ColGroupPiecewiseLinearCompressed) result).getBreakpointsPerCol(),
			plc.getBreakpointsPerCol());
		assertArrayEquals(((ColGroupPiecewiseLinearCompressed) result).getSlopesPerCol(), plc.getSlopesPerCol());
		assertArrayEquals(((ColGroupPiecewiseLinearCompressed) result).getInterceptsPerCol(),
			plc.getInterceptsPerCol());
	}

	@Test
	public void testDecompressToDenseBlock() {
		MatrixBlock mb_compare = new MatrixBlock(originalMB);
		mb_compare.recomputeNonZeros();
		DenseBlock db_compare = mb_compare.getDenseBlock();

		MatrixBlock mb_result = new MatrixBlock(NROWS, NCOLS, false);
		mb_result.allocateDenseBlock();
		mb_result.recomputeNonZeros();
		piecewiseLinearColGroup.decompressToDenseBlock(mb_result.getDenseBlock(), 0, 3, 0, 0);
		DenseBlock db_result = mb_result.getDenseBlock();

		// DenseBlockFP64 is just one large 1 dim array
		assertTrue(db_result instanceof DenseBlockFP64);
		assertTrue(db_compare instanceof DenseBlockFP64);

		assertArrayEquals(db_result.values(NCOLS), db_compare.values(NCOLS), TARGET_LOSS);
	}

	private double highestLoss(MatrixBlock result, MatrixBlock compare) {
		// recompute non zeros
		result.recomputeNonZeros();
		compare.recomputeNonZeros();

		// asserEquals size correct
		assertEquals(result.getNumRows(), compare.getNumRows());
		assertEquals(result.getNumColumns(), compare.getNumColumns());

		// MatrixBlock diff
		MatrixBlock diff = new MatrixBlock(NCOLS, NROWS, false);

		// binary Operation Minus
		ValueFunction fn = Minus.getMinusFnObject();
		BinaryOperator op = new BinaryOperator(fn);
		result.binaryOperations(op, compare, diff);

		// get max and min
		double max = diff.max();
		double min = diff.min();

		// choose max absolute value
		return Math.max(Math.abs(max), Math.abs(min));
	}

	@Test
	public void testUnaryOperationMultiply2() {
		MatrixBlock compare = new MatrixBlock(originalMB);
		ValueFunction fn = Multiply2.getMultiply2FnObject();
		AColGroup result = piecewiseLinearColGroup.unaryOperation(new UnaryOperator(fn));
		assertTrue(result instanceof ColGroupUncompressed);

		MatrixBlock resultMB = ((ColGroupUncompressed) result).getData();
		MatrixBlock compareMB = compare;

		// do unaryOperation on compare
		MatrixBlock compare_final = compare.unaryOperations(new UnaryOperator(fn));

		// check if highestLoss smaller than worst case expected loss
		double biggest_loss = highestLoss(resultMB, compareMB);
		assertEquals(TARGET_LOSS * 2, Math.max(biggest_loss, TARGET_LOSS * 2), 0.0);
	}

	@Test
	public void testUnaryOperationPower2() {
		MatrixBlock compare = new MatrixBlock(originalMB);
		ValueFunction fn = Power2.getPower2FnObject();
		AColGroup result = piecewiseLinearColGroup.unaryOperation(new UnaryOperator(fn));
		assertTrue(result instanceof ColGroupUncompressed);

		MatrixBlock resultMB = ((ColGroupUncompressed) result).getData();
		MatrixBlock compareMB = compare;

		// do unaryOperation on compare
		MatrixBlock compare_final = compare.unaryOperations(new UnaryOperator(fn));

		// check if highestLoss smaller than worst case expected loss
		double biggest_loss = highestLoss(resultMB, compareMB);
		assertEquals(TARGET_LOSS * TARGET_LOSS, Math.max(biggest_loss, TARGET_LOSS * TARGET_LOSS), 0.0);
	}

	@Test
	public void testReplace() {
		// correct Data Type
		AColGroup result = piecewiseLinearColGroup.replace(5.0, 1.0);
		assertTrue(result instanceof ColGroupUncompressed);
	}

	@Test
	public void testread2DIntegerArray(){}
//public static int[][] read2DIntegerArray(DataInput in, int numRows) throws IOException

	@Test
	public void testread2DDoubleArray(){}
//public static double[][] read2DDoubleArray(DataInput in, int numRows) throws IOException

	@Test
	public void testread(){}
//public static ColGroupPiecewiseLinearCompressed read(DataInput in) throws IOException

	@Test
	public void testwrite(){}
//public void write(DataOutput out) throws IOException

	@Test
	public void testcomputeMxx(){}
//protected double computeMxx(double c, Builtin builtin)

	@Test
	public void testcomputeColMxx(){}
//protected void computeColMxx(double[] c, Builtin builtin)

	@Test
	public void testcomputeSumSq(){}
//protected void computeSumSq(double[] c, int nRows)

	@Test
	public void testcomputeColSumsSq(){}
//protected void computeColSumsSq(double[] c, int nRows)

	@Test
	public void testsegmentSumSq(){}
//private double segmentSumSq(int col)

	@Test
	public void testsumOfSquares(){}
//private static double sumOfSquares(int start, int end)

	@Test
	public void testcomputeRowSums(){}
//protected void computeRowSums(double[] c, int rl, int ru, double[] preAgg)

	@Test
	public void testcomputeRowMxx(){}
//protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] preAgg)

	@Test
	public void testcomputeProduct(){}
//protected void computeProduct(double[] c, int nRows)

	@Test
	public void testcomputeRowProduct(){}
//protected void computeRowProduct(double[] c, int rl, int ru, double[] preAgg)

	@Test
	public void testcomputeColProduct(){}
//protected void computeColProduct(double[] c, int nRows)

	@Test
	public void testpreAggSumRows(){}
//protected double[] preAggSumRows()

	@Test
	public void testpreAggSumSqRows(){}
//protected double[] preAggSumSqRows()

	@Test
	public void testpreAggProductRows(){}
//protected double[] preAggProductRows()

	@Test
	public void testpreAggBuiltinRows(){}
//protected double[] preAggBuiltinRows(Builtin builtin)

	@Test
	public void testsameIndexStructure(){}
//public boolean sameIndexStructure(AColGroupCompressed that)

	@Test
	public void testtsmm(){}
//protected void tsmm(double[] result, int numColumns, int nRows)

	@Test
	public void testcrossColDotProduct(){}
//private double crossColDotProduct(int i, int j)

	@Test
	public void testcopyAndSet(){}
//public AColGroup copyAndSet(IColIndex colIndexes)

	@Test
	public void testdecompressToDenseBlockTransposed(){}
//public void decompressToDenseBlockTransposed(DenseBlock db, int rl, int ru)

	@Test
	public void testdecompressToSparseBlockTransposed(){}
//public void decompressToSparseBlockTransposed(SparseBlockMCSR sb, int nColOut)

	@Test
	public void testdecompressToSparseBlock(){}
//public void decompressToSparseBlock(SparseBlock sb, int rl, int ru, int offR, int offC)

	@Test
	public void testrightMultByMatrix(){}
//public AColGroup rightMultByMatrix(MatrixBlock right, IColIndex allCols, int k)

	@Test
	public void testleftMultByMatrixNoPreAgg(){}
//public void leftMultByMatrixNoPreAgg(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu)

	@Test
	public void testleftMultByAColGroup(){}
//public void leftMultByAColGroup(AColGroup lhs, MatrixBlock result, int nRows)

	@Test
	public void testtsmmAColGroup(){}
//public void tsmmAColGroup(AColGroup other, MatrixBlock result)

	@Test
	public void testsliceSingleColumn(){}
//protected AColGroup sliceSingleColumn(int idx)

	@Test
	public void testsliceMultiColumns(){}
//protected AColGroup sliceMultiColumns(int idStart, int idEnd, IColIndex outputCols)

	@Test
	public void testsliceRows(){}
//public AColGroup sliceRows(int rl, int ru)

	@Test
	public void testgetNumberNonZeros(){}
//public long getNumberNonZeros(int nRows)

	@Test
	public void testcentralMoment(){}
//public CmCovObject centralMoment(CMOperator op, int nRows)

	@Test
	public void testrexpandCols(){}
//public AColGroup rexpandCols(int max, boolean ignore, boolean cast, int nRows)

	@Test
	public void testgetCost(){}
//public double getCost(ComputationCostEstimator e, int nRows)

	@Test
	public void testappend(){}
//public AColGroup append(AColGroup g)

	@Test
	public void testappendNInternal(){}
//protected AColGroup appendNInternal(AColGroup[] groups, int blen, int rlen)

	@Test
	public void testgetCompressionScheme(){}
//public ICLAScheme getCompressionScheme()

	@Test
	public void testrecompress(){}
//public AColGroup recompress()

	@Test
	public void testgetCompressionInfo(){}
//public CompressedSizeInfoColGroup getCompressionInfo(int nRow)

	@Test
	public void testfixColIndexes(){}
//protected AColGroup fixColIndexes(IColIndex newColIndex, int[] reordering)

	@Test
	public void testremoveEmptyColsSubset(){}
//public AColGroup removeEmptyColsSubset(IColIndex indexes, IntArrayList emptyCols)

	@Test
	public void testremoveEmptyRows(){}
//public AColGroup removeEmptyRows(boolean[] emptyRows, int newNumRows)

	@Test
	public void testsort(){}
//public AColGroup sort()

	@Test
	public void testreduceCols(){}
//public AColGroup reduceCols()

	@Test
	public void testgetSparsity(){}
//public double getSparsity()

	@Test
	public void testsparseSelection(){}
//protected void sparseSelection(MatrixBlock selection, ColGroupUtils.P[] points, MatrixBlock ret, int rl, int ru)

	@Test
	public void testdenseSelection(){}
//protected void denseSelection(MatrixBlock selection, ColGroupUtils.P[] points, MatrixBlock ret, int rl, int ru)

	@Test
	public void testsplitReshape(){}
//public AColGroup[] splitReshape(int multiplier, int nRow, int nColOrg)


}
