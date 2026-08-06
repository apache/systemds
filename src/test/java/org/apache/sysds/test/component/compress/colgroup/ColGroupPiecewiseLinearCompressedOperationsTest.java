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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUtils;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.colgroup.ColGroupPiecewiseLinearCompressed;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.cost.CostEstimatorFactory;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.DenseBlockFP64;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.KahanPlusSq;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Multiply2;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.Power2;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.functionobjects.CM;
import org.apache.sysds.runtime.instructions.cp.CmCovObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.junit.Before;
import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests for ColGroupPiecewiseLinearCompressed operations.
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
		double[][] data = getRandomMatrix(numRows, numCols, -30, 30, 1.0, SEED);
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

	@Test
	public void testGetCompType() {
		assertEquals(AColGroup.CompressionType.PiecewiseLinearCompressed, piecewiseLinearColGroup.getCompType());
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

	@Override
	public double[][] getRandomMatrix(int rows, int cols, double min, double max, double sparsity, long seed) {
		Random rng = new Random(seed);
		double[][] data = new double[rows][cols];
		for(int r = 0; r < rows; r++)
			for(int c = 0; c < cols; c++)
				data[r][c] = min + rng.nextDouble() * (max - min);
		return data;
	}

	@Test
	public void testCreate() {
		ColGroupPiecewiseLinearCompressed plc = piecewiseLinearColGroup;

		AColGroup result = ColGroupPiecewiseLinearCompressed.create(plc.getColIndices(), plc.getBreakpointsPerCol(),
			plc.getSlopesPerCol(), plc.getInterceptsPerCol(), NROWS);
		assertTrue(result instanceof ColGroupPiecewiseLinearCompressed);

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

		assertTrue(db_result instanceof DenseBlockFP64);
		assertTrue(db_compare instanceof DenseBlockFP64);

		assertArrayEquals(db_result.values(NCOLS), db_compare.values(NCOLS), TARGET_LOSS);
	}

	private double highest_loss(MatrixBlock result, MatrixBlock compare) {
		result.recomputeNonZeros();
		compare.recomputeNonZeros();

		assertEquals(result.getNumRows(), compare.getNumRows());
		assertEquals(result.getNumColumns(), compare.getNumColumns());

		MatrixBlock diff = new MatrixBlock(NCOLS, NROWS, false);

		ValueFunction fn = Minus.getMinusFnObject();
		BinaryOperator op = new BinaryOperator(fn);
		result.binaryOperations(op, compare, diff);

		double max = diff.max();
		double min = diff.min();

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

		double biggest_loss = highest_loss(resultMB, compareMB);
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

		double biggest_loss = highest_loss(resultMB, compareMB);
		assertEquals(TARGET_LOSS * TARGET_LOSS, Math.max(biggest_loss, TARGET_LOSS * TARGET_LOSS), 0.0);
	}

	@Test
	public void testReplace() {
		AColGroup result = piecewiseLinearColGroup.replace(5.0, 1.0);
		assertTrue(result instanceof ColGroupUncompressed);
	}

	@Test
	public void testWrite() throws IOException {
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		DataOutputStream out = new DataOutputStream(baos);

		piecewiseLinearColGroup.write(out);
		out.flush();

		byte[] bytes = baos.toByteArray();
		assertTrue(bytes.length > 0);
	}

	@Test
	public void testWriteRead() throws IOException {
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		DataOutputStream dos = new DataOutputStream(bos);

		piecewiseLinearColGroup.write(dos);
		dos.flush();

		ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
		DataInputStream dis = new DataInputStream(bis);

		ColGroupPiecewiseLinearCompressed copy = ColGroupPiecewiseLinearCompressed.read(dis);

		for(int i = 0; i < piecewiseLinearColGroup.getBreakpointsPerCol().length; i++)
			assertArrayEquals(piecewiseLinearColGroup.getBreakpointsPerCol()[i], copy.getBreakpointsPerCol()[i]);

		for(int i = 0; i < piecewiseLinearColGroup.getSlopesPerCol().length; i++)
			assertArrayEquals(piecewiseLinearColGroup.getSlopesPerCol()[i], copy.getSlopesPerCol()[i], DELTA);

		for(int i = 0; i < piecewiseLinearColGroup.getInterceptsPerCol().length; i++)
			assertArrayEquals(piecewiseLinearColGroup.getInterceptsPerCol()[i], copy.getInterceptsPerCol()[i], DELTA);
	}

	@Test
	public void testcomputeMxx() {
		Builtin maxBuiltin = Builtin.getBuiltinFnObject(Builtin.BuiltinCode.MAX);

		AggregateUnaryOperator op = new AggregateUnaryOperator(new AggregateOperator(1, maxBuiltin),
			ReduceAll.getReduceAllFnObject());

		double[] c = new double[1];
		c[0] = Double.NEGATIVE_INFINITY;

		piecewiseLinearColGroup.unaryAggregateOperations(op, c, numRows, 0, numRows, null);

		double expected = Double.NEGATIVE_INFINITY;
		for(int r = 0; r < numRows; r++)
			for(int col = 0; col < numCols; col++)
				expected = Math.max(expected, decompressedMB.get(r, col));

		assertEquals(expected, c[0], DELTA);
	}

	@Test
	public void testcomputeColMxx() {
		Builtin maxBuiltin = Builtin.getBuiltinFnObject(Builtin.BuiltinCode.MAX);

		AggregateUnaryOperator op = new AggregateUnaryOperator(new AggregateOperator(1, maxBuiltin),
			ReduceRow.getReduceRowFnObject());

		double[] c = new double[numCols];
		Arrays.fill(c, Double.NEGATIVE_INFINITY);

		piecewiseLinearColGroup.unaryAggregateOperations(op, c, numRows, 0, numRows, null);

		for(int col = 0; col < numCols; col++) {
			double expected = Double.NEGATIVE_INFINITY;
			for(int r = 0; r < numRows; r++)
				expected = Math.max(expected, decompressedMB.get(r, col));

			assertEquals("column " + col, expected, c[col], DELTA);
		}
	}

	@Test
	public void testcomputeSumSq() {
		AggregateUnaryOperator op = new AggregateUnaryOperator(
			new AggregateOperator(0, KahanPlusSq.getKahanPlusSqFnObject()), ReduceAll.getReduceAllFnObject());

		double[] c = new double[1];

		piecewiseLinearColGroup.unaryAggregateOperations(op, c, numRows, 0, numRows, null);

		double expected = 0.0;
		for(int r = 0; r < numRows; r++)
			for(int col = 0; col < numCols; col++) {
				double v = decompressedMB.get(r, col);
				expected += v * v;
			}

		assertEquals(expected, c[0], DELTA * 10);
	}

	@Test
	public void testcomputeColSumsSq() {
		AggregateUnaryOperator op = new AggregateUnaryOperator(
			new AggregateOperator(0, KahanPlusSq.getKahanPlusSqFnObject()), ReduceRow.getReduceRowFnObject());

		double[] sumsSqComp = new double[numCols];
		piecewiseLinearColGroup.unaryAggregateOperations(op, sumsSqComp, numRows, 0, numRows, null);

		double[] expectedSumsSq = new double[numCols];
		for(int c = 0; c < numCols; c++) {
			for(int r = 0; r < numRows; r++) {
				double v = decompressedMB.get(r, c);
				expectedSumsSq[c] += v * v;
			}
		}

		assertArrayEquals(expectedSumsSq, sumsSqComp, DELTA * 10);
	}

	@Test
	public void testcomputeRowSums() {
		double[] preAgg = piecewiseLinearColGroup.preAggRows(Plus.getPlusFnObject());

		AggregateUnaryOperator op = new AggregateUnaryOperator(new AggregateOperator(0, Plus.getPlusFnObject()),
			ReduceCol.getReduceColFnObject());

		double[] c = new double[numRows];

		piecewiseLinearColGroup.unaryAggregateOperations(op, c, numRows, 0, numRows, preAgg);

		for(int r = 0; r < numRows; r++)
			assertEquals("row " + r, preAgg[r], c[r], DELTA);
	}

	@Test
	public void testcomputeRowMxx() {
		Builtin maxBuiltin = Builtin.getBuiltinFnObject(Builtin.BuiltinCode.MAX);

		double[] preAgg = piecewiseLinearColGroup.preAggRows(maxBuiltin);

		AggregateUnaryOperator op = new AggregateUnaryOperator(new AggregateOperator(1, maxBuiltin),
			ReduceCol.getReduceColFnObject());

		double[] c = new double[numRows];
		Arrays.fill(c, Double.NEGATIVE_INFINITY);

		piecewiseLinearColGroup.unaryAggregateOperations(op, c, numRows, 0, numRows, preAgg);

		for(int r = 0; r < numRows; r++)
			assertEquals("row " + r, preAgg[r], c[r], DELTA);
	}

	@Test
	public void testcomputeProduct() {
		AggregateUnaryOperator op = new AggregateUnaryOperator(new AggregateOperator(1, Multiply.getMultiplyFnObject()),
			ReduceAll.getReduceAllFnObject());

		double[] result = new double[] {1.0};
		piecewiseLinearColGroup.unaryAggregateOperations(op, result, numRows, 0, numRows, null);

		double expectedProd = 1.0;
		for(int c = 0; c < numCols; c++) {
			for(int r = 0; r < numRows; r++) {
				double v = decompressedMB.get(r, c);
				if(v == 0.0) {
					expectedProd = 0.0;
					break;
				}
				expectedProd *= v;
			}
			if(expectedProd == 0.0)
				break;
		}

		assertEquals(expectedProd, result[0], DELTA);
	}

	@Test
	public void testcomputeRowProduct() {
		double[] preAgg = piecewiseLinearColGroup.preAggRows(Multiply.getMultiplyFnObject());

		AggregateUnaryOperator op = new AggregateUnaryOperator(new AggregateOperator(1, Multiply.getMultiplyFnObject()),
			ReduceCol.getReduceColFnObject());

		double[] c = new double[numRows];
		Arrays.fill(c, 1.0);

		piecewiseLinearColGroup.unaryAggregateOperations(op, c, numRows, 0, numRows, preAgg);

		for(int r = 0; r < numRows; r++)
			assertEquals("row " + r, preAgg[r], c[r], DELTA);
	}

	@Test
	public void testcomputeColProduct() {
		AggregateUnaryOperator op = new AggregateUnaryOperator(new AggregateOperator(1, Multiply.getMultiplyFnObject()),
			ReduceRow.getReduceRowFnObject());

		double[] c = new double[numCols];
		Arrays.fill(c, 1.0);

		piecewiseLinearColGroup.unaryAggregateOperations(op, c, numRows, 0, numRows, null);

		for(int col = 0; col < numCols; col++) {
			double expected = 1.0;
			for(int r = 0; r < numRows; r++)
				expected *= decompressedMB.get(r, col);

			assertEquals("column " + col, expected, c[col], DELTA);
		}
	}

	@Test
	public void testpreAggSumRows() {
		double[] agg = piecewiseLinearColGroup.preAggRows(Plus.getPlusFnObject());
		assertEquals(numRows, agg.length);
		for(int r = 0; r < numRows; r++) {
			double expected = 0;
			for(int c = 0; c < numCols; c++)
				expected += decompressedMB.get(r, c);
			assertEquals("row " + r, expected, agg[r], DELTA);
		}
	}

	@Test
	public void testpreAggSumSqRows() {
		double[] agg = piecewiseLinearColGroup.preAggRows(KahanPlusSq.getKahanPlusSqFnObject());
		assertEquals(numRows, agg.length);
		for(int r = 0; r < numRows; r++) {
			double expected = 0;
			for(int c = 0; c < numCols; c++) {
				double v = decompressedMB.get(r, c);
				expected += v * v;
			}
			assertEquals("row " + r, expected, agg[r], DELTA);
		}
	}

	@Test
	public void testpreAggProductRows() {
		double[] agg = piecewiseLinearColGroup.preAggRows(Multiply.getMultiplyFnObject());
		assertEquals(numRows, agg.length);
		for(int r = 0; r < numRows; r++) {
			double expected = 1.0;
			for(int c = 0; c < numCols; c++)
				expected *= decompressedMB.get(r, c);
			assertEquals("row " + r, expected, agg[r], DELTA);
		}
	}

	@Test
	public void testpreAggBuiltinRows() {
		Builtin maxBuiltin = Builtin.getBuiltinFnObject(Builtin.BuiltinCode.MAX);
		double[] agg = piecewiseLinearColGroup.preAggRows(maxBuiltin);
		assertEquals(numRows, agg.length);
		for(int r = 0; r < numRows; r++) {
			double expected = Double.NEGATIVE_INFINITY;
			for(int c = 0; c < numCols; c++)
				expected = Math.max(expected, decompressedMB.get(r, c));
			assertEquals("row " + r, expected, agg[r], DELTA);
		}
	}

	@Test
	public void testsameIndexStructure() {
		assertTrue(piecewiseLinearColGroup.sameIndexStructure(piecewiseLinearColGroup));
		ColGroupPiecewiseLinearCompressed other = (ColGroupPiecewiseLinearCompressed) ColGroupPiecewiseLinearCompressed
			.create(ColIndexFactory.create(new int[] {0}), new int[][] {{0, numRows}}, new double[][] {{0.5}},
				new double[][] {{1.0}}, numRows);
		assertTrue(piecewiseLinearColGroup.sameIndexStructure(other));
	}

	@Test
	public void testtsmm() {
		MatrixBlock result = new MatrixBlock(numCols, numCols, false);
		result.allocateDenseBlock();
		piecewiseLinearColGroup.tsmm(result, numRows);

		double[] expected = new double[numCols * numCols];
		for(int r = 0; r < numRows; r++)
			for(int i = 0; i < numCols; i++)
				for(int j = i; j < numCols; j++)
					expected[i * numCols + j] += decompressedMB.get(r, i) * decompressedMB.get(r, j);

		for(int i = 0; i < numCols; i++)
			for(int j = i; j < numCols; j++)
				assertEquals("[" + i + "," + j + "]", expected[i * numCols + j], result.get(i, j), 1e-6);
	}

	@Test
	public void testcrossColDotProduct() {
		ColGroupPiecewiseLinearCompressed single = (ColGroupPiecewiseLinearCompressed) ColGroupPiecewiseLinearCompressed
			.create(ColIndexFactory.create(new int[] {0}),
				new int[][] {piecewiseLinearColGroup.getBreakpointsPerCol()[0]},
				new double[][] {piecewiseLinearColGroup.getSlopesPerCol()[0]},
				new double[][] {piecewiseLinearColGroup.getInterceptsPerCol()[0]}, numRows);
		MatrixBlock result = new MatrixBlock(1, 1, false);
		result.allocateDenseBlock();
		single.tsmm(result, numRows);
		double expected = 0;
		for(int r = 0; r < numRows; r++) {
			double v = decompressedMB.get(r, 0);
			expected += v * v;
		}
		assertEquals(expected, result.get(0, 0), 1e-6);
	}

	@Test
	public void testcopyAndSet() {
		IColIndex newCols = ColIndexFactory.create(new int[] {5, 6, 7});
		AColGroup copy = piecewiseLinearColGroup.copyAndSet(newCols);
		assertTrue(copy instanceof ColGroupPiecewiseLinearCompressed);
		assertEquals(newCols, copy.getColIndices());
		ColGroupPiecewiseLinearCompressed plcCopy = (ColGroupPiecewiseLinearCompressed) copy;
		assertArrayEquals(piecewiseLinearColGroup.getBreakpointsPerCol(), plcCopy.getBreakpointsPerCol());
		assertArrayEquals(piecewiseLinearColGroup.getSlopesPerCol(), plcCopy.getSlopesPerCol());
	}

	@Test
	public void testdecompressToDenseBlockTransposed() {
		MatrixBlock transposed = new MatrixBlock(numCols, numRows, false);
		transposed.allocateDenseBlock();
		piecewiseLinearColGroup.decompressToDenseBlockTransposed(transposed.getDenseBlock(), 0, numRows);
		for(int r = 0; r < numRows; r++)
			for(int c = 0; c < numCols; c++)
				assertEquals("[" + r + "," + c + "]", decompressedMB.get(r, c), transposed.get(c, r), DELTA);
	}

	@Test(expected = NotImplementedException.class)
	public void testdecompressToSparseBlockTransposed() {
		SparseBlockMCSR sb = new SparseBlockMCSR(numCols);
		piecewiseLinearColGroup.decompressToSparseBlockTransposed(sb, numCols);
	}

	@Test
	public void testdecompressToSparseBlock() {
		MatrixBlock sparse = new MatrixBlock(numRows, numCols, true);
		sparse.allocateSparseRowsBlock();
		piecewiseLinearColGroup.decompressToSparseBlock(sparse.getSparseBlock(), 0, numRows, 0, 0);
		for(int r = 0; r < numRows; r++)
			for(int c = 0; c < numCols; c++)
				assertEquals("[" + r + "," + c + "]", decompressedMB.get(r, c), sparse.get(r, c), DELTA);
	}

	@Test
	public void testrightMultByMatrix() {
		MatrixBlock identity = new MatrixBlock(numCols, numCols, false);
		identity.allocateDenseBlock();
		for(int i = 0; i < numCols; i++)
			identity.set(i, i, 1.0);
		AColGroup result = piecewiseLinearColGroup.rightMultByMatrix(identity, null, 1);
		assertTrue(result instanceof ColGroupUncompressed);
		MatrixBlock resultMB = ((ColGroupUncompressed) result).getData();
		for(int r = 0; r < numRows; r++)
			for(int c = 0; c < numCols; c++)
				assertEquals("[" + r + "," + c + "]", decompressedMB.get(r, c), resultMB.get(r, c), DELTA);
	}

	@Test
	public void testleftMultByMatrixNoPreAgg() {
		MatrixBlock identity = new MatrixBlock(numRows, numRows, false);
		identity.allocateDenseBlock();
		for(int i = 0; i < numRows; i++)
			identity.set(i, i, 1.0);
		MatrixBlock result = new MatrixBlock(numRows, numCols, false);
		result.allocateDenseBlock();
		piecewiseLinearColGroup.leftMultByMatrixNoPreAgg(identity, result, 0, numRows, 0, numRows);
		for(int r = 0; r < numRows; r++)
			for(int c = 0; c < numCols; c++)
				assertEquals("[" + r + "," + c + "]", decompressedMB.get(r, c), result.get(r, c), DELTA);
	}

	@Test
	public void testleftMultByAColGroup() {
		MatrixBlock result = new MatrixBlock(numCols, numCols, false);
		result.allocateDenseBlock();
		piecewiseLinearColGroup.leftMultByAColGroup(piecewiseLinearColGroup, result, numRows);
		double[] expected = new double[numCols * numCols];
		for(int r = 0; r < numRows; r++)
			for(int i = 0; i < numCols; i++)
				for(int j = 0; j < numCols; j++)
					expected[i * numCols + j] += decompressedMB.get(r, i) * decompressedMB.get(r, j);
		for(int i = 0; i < numCols; i++)
			for(int j = 0; j < numCols; j++)
				assertEquals("[" + i + "," + j + "]", expected[i * numCols + j], result.get(i, j), 1e-6);
	}

	@Test(expected = DMLCompressionException.class)
	public void testtsmmAColGroup() {
		MatrixBlock result = new MatrixBlock(numCols, numCols, false);
		result.allocateDenseBlock();
		piecewiseLinearColGroup.tsmmAColGroup(piecewiseLinearColGroup, result);
	}

	@Test
	public void testsliceSingleColumn() throws Exception {
		int colToSlice = 0;

		Method method = ColGroupPiecewiseLinearCompressed.class.getDeclaredMethod("sliceSingleColumn", int.class);
		method.setAccessible(true);

		AColGroup slice = (AColGroup) method.invoke(piecewiseLinearColGroup, colToSlice);

		assertNotNull(slice);
		assertTrue(slice instanceof ColGroupPiecewiseLinearCompressed);
		assertEquals(1, slice.getNumCols());

		for(int r = 0; r < numRows; r++) {
			assertEquals("Mismatch at row " + r, decompressedMB.get(r, colToSlice), slice.getIdx(r, 0), DELTA);
		}
	}

	@Test
	public void testsliceMultiColumns() throws Exception {
		int startCol = 0;
		int stopCol = 2;
		IColIndex outputCols = ColIndexFactory.create(stopCol - startCol);

		Method method = ColGroupPiecewiseLinearCompressed.class.getDeclaredMethod("sliceMultiColumns", int.class,
			int.class, IColIndex.class);
		method.setAccessible(true);

		AColGroup slice = (AColGroup) method.invoke(piecewiseLinearColGroup, startCol, stopCol, outputCols);

		assertNotNull(slice);
		assertTrue(slice instanceof ColGroupPiecewiseLinearCompressed);
		assertEquals(stopCol - startCol, slice.getNumCols());

		for(int r = 0; r < numRows; r++) {
			for(int c = 0; c < (stopCol - startCol); c++) {
				assertEquals("Mismatch at row " + r + " col " + c, decompressedMB.get(r, startCol + c),
					slice.getIdx(r, c), DELTA);
			}
		}
	}

	@Test
	public void testsliceRows() {
		AColGroup slice = piecewiseLinearColGroup.sliceRows(0, numRows);
		assertTrue(slice instanceof ColGroupPiecewiseLinearCompressed);
		for(int r = 0; r < numRows; r++)
			for(int c = 0; c < numCols; c++)
				assertEquals("row " + r + ", col " + c, decompressedMB.get(r, c), slice.getIdx(r, c), DELTA);
	}

	@Test
	public void testgetNumberNonZeros() {
		decompressedMB.recomputeNonZeros();
		long numNonZeros = piecewiseLinearColGroup.getNumberNonZeros(numRows);
		assertEquals("Number of non-zeros", decompressedMB.getNonZeros(), numNonZeros);
	}

	@Test
	public void testcentralMoment() {
		CMOperator op = new CMOperator(CM.getCMFnObject(CMOperator.AggregateOperationTypes.VARIANCE),
			CMOperator.AggregateOperationTypes.VARIANCE);
		CmCovObject result = piecewiseLinearColGroup.centralMoment(op, numRows);
		assertNotNull(result);
	}

	@Test
	public void testrexpandCols() {
		assertThrows(NotImplementedException.class, () -> {
			piecewiseLinearColGroup.rexpandCols(10, true, false, numRows);
		});
	}

	@Test
	public void testgetCost() {
		ComputationCostEstimator costEstimator = new ComputationCostEstimator(1, 1, 1, 1, 1, 1, 1, 1, false);
		double cost = piecewiseLinearColGroup.getCost(costEstimator, numRows);
		assertTrue("Cost should be non-negative", cost >= 0.0);
		assertFalse("Cost should not be infinite or NaN", Double.isInfinite(cost) || Double.isNaN(cost));
	}

	@Test
	public void testappend() {
		assertNull(piecewiseLinearColGroup.append(null));
	}

	@Test
	public void testappendNInternal() {
		ColGroupPiecewiseLinearCompressed g1 = piecewiseLinearColGroup;
		ColGroupPiecewiseLinearCompressed g2 = piecewiseLinearColGroup;
		AColGroup[] groups = new AColGroup[] {g1, g2};
		int totalRows = numRows * 2;
		AColGroup merged = g1.appendN(groups, totalRows, totalRows);
		assertTrue(merged instanceof ColGroupPiecewiseLinearCompressed);

		for(int r = 0; r < totalRows; r++) {
			double expected = (r < numRows) ? g1.getIdx(r, 0) : g2.getIdx(r - numRows, 0);
			assertEquals(expected, merged.getIdx(r, 0), DELTA);
		}
	}

	@Test
	public void testAppendNInternalIncompatibleGroup() {
		AColGroup dummyGroup = new ColGroupEmpty(colIndexes);
		AColGroup[] groups = new AColGroup[] {piecewiseLinearColGroup, dummyGroup};
		int totalRows = numRows * 2;

		assertThrows(NotImplementedException.class, () -> {
			piecewiseLinearColGroup.appendN(groups, totalRows, totalRows);
		});
	}

	@Test
	public void testgetCompressionScheme() {
		assertNull(piecewiseLinearColGroup.getCompressionScheme());
	}

	@Test
	public void testrecompress() {
		assertEquals(piecewiseLinearColGroup, piecewiseLinearColGroup.recompress());
	}

	@Test
	public void testgetCompressionInfo() {
		assertThrows(NotImplementedException.class, () -> {
			piecewiseLinearColGroup.getCompressionInfo(numRows);
		});
	}

	@Test
	public void testfixColIndexes() throws Exception {
		IColIndex newColIndexes = ColIndexFactory.create(new int[] {0, 1, 2});
		int[] reordering = new int[] {2, 1, 0};

		Method method = ColGroupPiecewiseLinearCompressed.class.getDeclaredMethod("fixColIndexes", IColIndex.class,
			int[].class);
		method.setAccessible(true);

		AColGroup reorderedGroup = (AColGroup) method.invoke(piecewiseLinearColGroup, newColIndexes, reordering);

		assertNotNull(reorderedGroup);
		assertTrue(reorderedGroup instanceof ColGroupPiecewiseLinearCompressed);

		for(int r = 0; r < numRows; r++) {
			for(int i = 0; i < numCols; i++) {
				int originalColIdx = reordering[i];
				assertEquals("Mismatch at row " + r + " col " + i, decompressedMB.get(r, originalColIdx),
					reorderedGroup.getIdx(r, i), DELTA);
			}
		}
	}

	@Test
	public void testremoveEmptyColsSubset() {
		assertNull(piecewiseLinearColGroup.removeEmptyColsSubset(null, null));
	}

	@Test
	public void testremoveEmptyRows() {
		assertNull(piecewiseLinearColGroup.removeEmptyRows(null, 0));
	}

	@Test
	public void testsort() {
		assertEquals(piecewiseLinearColGroup, piecewiseLinearColGroup.sort());
	}

	@Test
	public void testreduceCols() {
		assertNull(piecewiseLinearColGroup.reduceCols());
	}

	@Test
	public void testgetSparsity() {
		assertEquals(1.0, piecewiseLinearColGroup.getSparsity(), DELTA);
	}

	@Test
	public void testsparseSelection() throws Exception {
		int rl = 0;
		int ru = 5;

		MatrixBlock selection = new MatrixBlock(ru, numRows, true);
		for(int r = rl; r < ru; r++) {
			selection.appendValue(r, r, 1.0);
		}
		selection.recomputeNonZeros();

		MatrixBlock ret = new MatrixBlock(ru, numCols, true);
		ret.allocateSparseRowsBlock();

		Method method = ColGroupPiecewiseLinearCompressed.class.getDeclaredMethod("sparseSelection", MatrixBlock.class,
			ColGroupUtils.P[].class, MatrixBlock.class, int.class, int.class);
		method.setAccessible(true);

		method.invoke(piecewiseLinearColGroup, selection, null, ret, rl, ru);

		for(int r = rl; r < ru; r++) {
			for(int c = 0; c < numCols; c++) {
				assertEquals("Mismatch at row " + r + " col " + c, decompressedMB.get(r, c), ret.get(r, c), DELTA);
			}
		}
	}

	@Test
	public void testdenseSelection() throws Exception {
		int rl = 0;
		int ru = 5;

		MatrixBlock selection = new MatrixBlock(ru, numRows, true);
		for(int r = rl; r < ru; r++) {
			selection.appendValue(r, r, 1.0);
		}
		selection.recomputeNonZeros();

		MatrixBlock ret = new MatrixBlock(ru, numCols, false);
		ret.allocateDenseBlock();

		Method method = ColGroupPiecewiseLinearCompressed.class.getDeclaredMethod("denseSelection", MatrixBlock.class,
			ColGroupUtils.P[].class, MatrixBlock.class, int.class, int.class);
		method.setAccessible(true);

		method.invoke(piecewiseLinearColGroup, selection, null, ret, rl, ru);

		for(int r = rl; r < ru; r++) {
			for(int c = 0; c < numCols; c++) {
				assertEquals("Mismatch at row " + r + " col " + c, decompressedMB.get(r, c), ret.get(r, c), DELTA);
			}
		}
	}

	@Test
	public void testsplitReshape() {
		int multiplier = 2;
		int nRow = numRows;
		int nColOrg = numCols;

		AColGroup[] resultGroups = piecewiseLinearColGroup.splitReshape(multiplier, nRow, nColOrg);

		assertNotNull(resultGroups);
		assertEquals(1, resultGroups.length);

		AColGroup reshaped = resultGroups[0];
		assertTrue(reshaped instanceof ColGroupPiecewiseLinearCompressed);

		int expectedNewNRow = nRow / multiplier;
		int expectedTotalNewCols = numCols * multiplier;
		assertEquals(expectedTotalNewCols, reshaped.getNumCols());

		for(int i = 0; i < multiplier; i++) {
			int rowOffset = i * expectedNewNRow;
			for(int r = 0; r < expectedNewNRow; r++) {
				for(int c = 0; c < numCols; c++) {
					double expected = decompressedMB.get(rowOffset + r, c);
					int reshapedColIndex = i * numCols + c;
					double actual = reshaped.getIdx(r, reshapedColIndex);
					assertEquals(expected, actual, DELTA);
				}
			}
		}
	}
}
