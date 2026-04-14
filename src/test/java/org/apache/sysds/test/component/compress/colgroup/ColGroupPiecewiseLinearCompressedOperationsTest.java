package org.apache.sysds.test.component.compress.colgroup;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.colgroup.ColGroupPiecewiseLinearCompressed;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.*;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Tests for ColGroupPiecewiseLinearCompressed operations containing: scalarOperation, binaryRowOps, computeSum,
 * containsValue, getIdx, getExactSizeOnDisk.
 */
public class ColGroupPiecewiseLinearCompressedOperationsTest extends AutomatedTestBase {

	private static final long SEED = 42L;
	private static final int NROWS = 50;
	private static final int NCOLS = 3;
	private static final double TARGET_LOSS = 1e-8;
	private static final double DELTA = 1e-9;

	private ColGroupPiecewiseLinearCompressed piecewiseLinearColGroup;
	private MatrixBlock orignalMB;
	private MatrixBlock decompressedMB;
	private IColIndex colIndexes;
	private int numRows;
	private int numCols;

	@Before
	public void setUp() {
		numRows = NROWS;
		numCols = NCOLS;

		///  generate random matrix
		double[][] data = getRandomMatrix(numRows, numCols, -3, 3, 1.0, SEED);
		orignalMB = DataConverter.convertToMatrixBlock(data);
		orignalMB.allocateDenseBlock();

		colIndexes = ColIndexFactory.create(buildColArray(numCols));

		CompressionSettings cs = new CompressionSettingsBuilder().create();
		cs.setPiecewiseTargetLoss(TARGET_LOSS);

		///  create ColGroupPiecewiseLinearCompressed instance
		AColGroup result = ColGroupFactory.compressPiecewiseLinearFunctional(colIndexes, orignalMB, cs);
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
		double[] sumsComp = new double[numCols];
		piecewiseLinearColGroup.computeSum(sumsComp, numRows);
		assertArrayEquals(sumsComp, computeSums(decompressedMB), DELTA);
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
		ColGroupPiecewiseLinearCompressed cg = (ColGroupPiecewiseLinearCompressed) ColGroupPiecewiseLinearCompressed.create(
			ColIndexFactory.create(new int[] {0}), new int[][] {{0, numRows}}, new double[][] {{0.0}},
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
		///  PLC Piecewise Linear Compressed
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
}
