package org.apache.sysds.test.component.compress.colgroup;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.colgroup.ColGroupPiecewiseLinearCompressed;
import org.apache.sysds.runtime.compress.colgroup.functional.PiecewiseLinearUtils;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.apache.sysds.runtime.compress.colgroup.functional.PiecewiseLinearUtils.*;
import static org.junit.Assert.*;

public class ColGroupPiecewiseLinearCompressedTest extends AutomatedTestBase {
	@Override
	public void setUp() {

	}

	private static final long SEED = 42L;

	@Test
	public void testCompressPiecewiseLinearFunctionalRandom() {
		// Generate random data
		final int nrows = 50, ncols = 3;
		double[][] data = getRandomMatrix(nrows, ncols, -3, 3, 1.0, SEED);
		MatrixBlock in = DataConverter.convertToMatrixBlock(data);
		in.allocateDenseBlock();

		// extract columns
		double[][] columns = new double[ncols][nrows];
		for(int c = 0; c < ncols; c++)
			for(int r = 0; r < nrows; r++)
				columns[c][r] = data[r][c];

		// create ColIndexes
		int[] colArray = {0, 1, 2};
		IColIndex colIndexes = ColIndexFactory.create(colArray);

		// set targetloss
		CompressionSettings cs = new CompressionSettingsBuilder().create();
		cs.setPiecewiseTargetLoss(25.0);

		// compress
		AColGroup result = ColGroupFactory.compressPiecewiseLinearFunctional(colIndexes, in, cs);
		assertTrue(result instanceof ColGroupPiecewiseLinearCompressed);
		ColGroupPiecewiseLinearCompressed plGroup = (ColGroupPiecewiseLinearCompressed) result;

		// check structure
		int[][] bp = plGroup.getBreakpointsPerCol();
		assertEquals(3, bp.length);  // 3 Spalten
		assertEquals(3, colIndexes.size());

		for(int c = 0; c < ncols; c++) {
			assertEquals(0, bp[c][0]);                    // start with 0
			assertEquals(nrows, bp[c][bp[c].length - 1]);
			assertTrue(bp[c].length >= 2);                // Mind. 1 Segment
		}

		double[][] slopes = plGroup.getSlopesPerCol();
		double[][] intercepts = plGroup.getInterceptsPerCol();
		assertEquals(3, slopes.length);
		for(int c = 0; c < ncols; c++) {
			assertEquals(bp[c].length - 1, slopes[c].length);
			assertEquals(bp[c].length - 1, intercepts[c].length);
		}

		// check col indexes shouldnt change
		assertEquals(3, plGroup.getColIndices().size());

		// decompress
		MatrixBlock recon = new MatrixBlock(nrows, ncols, false);
		recon.allocateDenseBlock();
		plGroup.decompressToDenseBlock(recon.getDenseBlock(), 0, nrows, 0, ncols - 1);
		assertFalse(Double.isNaN(recon.get(0, 0)));
	}

	private void testCompressStructure(double[][] data) {
		final int nrows = data.length, ncols = data[0].length;
		MatrixBlock in = DataConverter.convertToMatrixBlock(data);
		in.allocateDenseBlock();

		int[] colArray = new int[ncols];
		for(int i = 0; i < ncols; i++)
			colArray[i] = i;
		IColIndex colIndexes = ColIndexFactory.create(colArray);
		CompressionSettings cs = new CompressionSettingsBuilder().create();
		cs.setPiecewiseTargetLoss(100.0);

		AColGroup result = ColGroupFactory.compressPiecewiseLinearFunctional(colIndexes, in, cs);
		assertTrue(result instanceof ColGroupPiecewiseLinearCompressed);
		ColGroupPiecewiseLinearCompressed plGroup = (ColGroupPiecewiseLinearCompressed) result;

		int[][] bp = plGroup.getBreakpointsPerCol();
		assertEquals(ncols, bp.length);
		for(int c = 0; c < ncols; c++) {
			assertEquals(0, bp[c][0]);
			assertEquals(nrows, bp[c][bp[c].length - 1]);
		}
		double[][] slopes = plGroup.getSlopesPerCol();
		assertEquals(ncols, slopes.length);
		for(int c = 0; c < ncols; c++) {
			assertEquals(bp[c].length - 1, slopes[c].length);
		}
		assertEquals(ncols, plGroup.getColIndices().size());

		MatrixBlock recon = new MatrixBlock(nrows, ncols, false);
		recon.allocateDenseBlock();
		plGroup.decompressToDenseBlock(recon.getDenseBlock(), 0, nrows, 0, ncols - 1);
	}

	@Test
	public void testCompressTrendNoise() {
		final int nrows = 100, ncols = 2;
		Random rng = new Random(SEED);
		double[][] data = new double[nrows][ncols];

		for(int r = 0; r < nrows; r++) {
			double trend = 0.05 * r;
			for(int c = 0; c < ncols; c++) {
				data[r][c] = trend + rng.nextGaussian() * 1.5 + c * 2.0;
			}
		}

		testCompressStructure(data);
	}

	@Test
	public void testCompressJumps() {
		final int nrows = 80, ncols = 3;
		double[][] data = getRandomMatrix(nrows, ncols, -2, 2, 1.0, SEED);
		for(int c = 0; c < ncols; c++) {
			for(int r = 25; r < 55; r++)
				data[r][c] += 8.0;
			for(int r = 55; r < nrows; r++)
				data[r][c] += 15.0;
		}
		testCompressStructure(data);
	}

	@Test
	public void testCompressHighFreq() {
		final int nrows = 100, ncols = 50;
		Random rng = new Random(SEED);
		double[][] data = new double[nrows][ncols];
		for(int r = 0; r < nrows; r++) {
			double sine = Math.sin(r * 0.4) * 4.0;
			for(int c = 0; c < ncols; c++) {
				data[r][c] = sine + rng.nextGaussian() * 0.8 + Math.sin(r * 0.2 + c) * 2.0;
			}
		}
		testCompressStructure(data);
	}

	@Test
	public void testCompressSingleLowVariance() {
		final int nrows = 50, ncols = 1;
		double[][] data = getRandomMatrix(nrows, ncols, -1, 1, 1.0, SEED);
		testCompressStructure(data);
	}

	@Test
	public void testCompressSingleColumnStructure() {
		double[][] data = getRandomMatrix(50, 1, -1, 1, 1.0, SEED);
		testCompressStructure(data);
	}

	@Test(expected = NullPointerException.class)  // ← Dein realer Crash!
	public void testCreateNullBreakpoints() {
		IColIndex cols = ColIndexFactory.create(new int[] {0});
		int[][] nullBp = {null};
		ColGroupPiecewiseLinearCompressed.create(cols, nullBp, new double[][] {{1.0}}, new double[][] {{0.0}}, 10);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testCreateTooFewBreakpoints() {
		int[][] singleBp = {new int[] {0}};
		ColGroupPiecewiseLinearCompressed.create(ColIndexFactory.create(new int[] {0}), singleBp,
			new double[][] {new double[] {1.0}}, new double[][] {new double[] {0.0}}, 10);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testCreateInconsistentSlopes() {
		int[] bp = {0, 5, 10};
		ColGroupPiecewiseLinearCompressed.create(ColIndexFactory.create(new int[] {0}), new int[][] {bp},
			new double[][] {new double[] {1.0, 2.0, 3.0}}, new double[][] {new double[] {0.0, 1.0}}, 10);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testCreateInconsistentIntercepts() {
		int[] bp = {0, 5, 10};
		ColGroupPiecewiseLinearCompressed.create(ColIndexFactory.create(new int[] {0}), new int[][] {bp},
			new double[][] {new double[] {1.0, 2.0}}, new double[][] {new double[] {0.0}}, 10);
	}

	private int findSegment(int[] bps, int r) {
		for(int s = 0; s < bps.length - 1; s++) {
			if(r < bps[s + 1])
				return s;
		}
		return bps.length - 2;
	}

	@Test
	public void testCreateValidMultiSegmentRandom() {
		Random rng = new Random(SEED);
		final int nrows = 20;

		int[][] bp = {{0, rng.nextInt(5) + 3, rng.nextInt(10) + 8, nrows}, {0, rng.nextInt(8) + 2, nrows}};
		double[][] slopes = {{rng.nextDouble() * 3 - 1.5, rng.nextDouble() * 3 - 1.5, rng.nextDouble() * 3 - 1.5},
			{rng.nextDouble() * 3 - 1.5, rng.nextDouble() * 3 - 1.5}};
		double[][] intercepts = {{rng.nextDouble() * 2 - 1, rng.nextDouble() * 2 - 1, rng.nextDouble() * 2 - 1},
			{rng.nextDouble() * 2 - 1, rng.nextDouble() * 2 - 1}};

		IColIndex cols = ColIndexFactory.create(new int[] {0, 1});
		AColGroup cg = ColGroupPiecewiseLinearCompressed.create(cols, bp, slopes, intercepts, nrows);

		assertTrue(cg instanceof ColGroupPiecewiseLinearCompressed);
		ColGroupPiecewiseLinearCompressed pl = (ColGroupPiecewiseLinearCompressed) cg;
		assertNotSame(bp, pl.getBreakpointsPerCol());
		assertEquals(2, pl.getBreakpointsPerCol().length);

		for(int c = 0; c < 2; c++) {
			for(int r = 0; r < nrows; r++) {
				int seg = findSegment(bp[c], r);
				double expected = slopes[c][seg] * r + intercepts[c][seg];
				assertEquals(expected, cg.getIdx(r, c), 1e-8);
			}
		}
	}

	@Test
	public void testCreateMultiColumnRandom() {
		Random rng = new Random(SEED);
		final int nrows = 80, numGlobalCols = 5;
		int[] globalCols = {2, 7, 12, 25, 42};

		int numSegs = rng.nextInt(3) + 1;
		int[][] bp = new int[numGlobalCols][numSegs + 1];
		double[][] slopes = new double[numGlobalCols][numSegs];
		double[][] intercepts = new double[numGlobalCols][numSegs];

		double slope = rng.nextDouble() * 4 - 2;
		double intercept = rng.nextDouble() * 4 - 2;
		for(int c = 0; c < numGlobalCols; c++) {
			bp[c][0] = 0;
			bp[c][numSegs] = nrows;
			for(int s = 1; s < numSegs; s++)
				bp[c][s] = rng.nextInt(nrows - 10) + 5;
			Arrays.fill(slopes[c], slope);
			Arrays.fill(intercepts[c], intercept);
		}

		IColIndex cols = ColIndexFactory.create(globalCols);
		AColGroup cg = ColGroupPiecewiseLinearCompressed.create(cols, bp, slopes, intercepts, nrows);

		assertTrue(cg.getNumValues() > 0);
		assertEquals(numGlobalCols, cols.size());

		for(int r = 0; r < nrows; r++) {
			double expected = slope * r + intercept;
			for(int localC = 0; localC < numGlobalCols; localC++) {
				assertEquals(expected, cg.getIdx(r, localC), 1e-8);
			}
		}
	}

	@Test
	public void testCreateSingleColumnRandom() {
		Random rng = new Random(SEED);
		final int nrows = rng.nextInt(30) + 20;
		int numSegs = rng.nextInt(3) + 1;

		int[] bp = new int[numSegs + 1];
		bp[0] = 0;
		bp[numSegs] = nrows;
		for(int s = 1; s < numSegs; s++)
			bp[s] = rng.nextInt(nrows / 2) + 5;

		double[] slopes = new double[numSegs];
		double[] intercepts = new double[numSegs];
		for(int s = 0; s < numSegs; s++) {
			slopes[s] = rng.nextDouble() * 4 - 2;
			intercepts[s] = rng.nextDouble() * 4 - 2;
		}

		IColIndex cols = ColIndexFactory.create(new int[] {rng.nextInt(50)});
		int[][] bp2d = {bp};
		double[][] slopes2d = {slopes};
		double[][] ints2d = {intercepts};

		AColGroup cg = ColGroupPiecewiseLinearCompressed.create(cols, bp2d, slopes2d, ints2d, nrows);

		for(int r = 0; r < nrows; r++) {
			int seg = findSegment(bp, r);
			double expected = slopes[seg] * r + intercepts[seg];
			assertEquals(expected, cg.getIdx(r, 0), 1e-8);
		}
	}

	@Test
	public void testDecompressToDenseBlock() {
		int[][] bp = {{0, 5, 10}};
		double[][] slopes = {{1.0, 2.0}};
		double[][] intercepts = {{0.0, 1.0}};
		int numRows = 10;

		IColIndex cols = ColIndexFactory.create(new int[] {0});
		AColGroup cg = ColGroupPiecewiseLinearCompressed.create(cols, bp, slopes, intercepts, numRows);

		MatrixBlock target = new MatrixBlock(numRows, 1, false);
		target.allocateDenseBlock();

		DenseBlock db = target.getDenseBlock();
		assertNotNull("DenseBlock null?", db);

		cg.decompressToDenseBlock(db, 0, numRows, 0, 0);

		for(int r = 0; r < numRows; r++) {
			double expected = (r < 5) ? (1.0 * r + 0.0) : (2.0 * r + 1.0);
			assertEquals("Row " + r + " mismatch", expected, db.get(r, 0), 1e-9);
		}

		assertEquals(0.0, db.get(0, 0), 1e-9);
		assertEquals(4.0, db.get(4, 0), 1e-9);
		assertEquals(11.0, db.get(5, 0), 1e-9);
		assertEquals(19.0, db.get(9, 0), 1e-9);
	}

	private ColGroupPiecewiseLinearCompressed createTestGroup(int numRows) {
		int[][] bp = {{0, 5, numRows}};
		double[][] slopes = {{1.0, 3.0}};
		double[][] intercepts = {{0.0, 2.0}};
		return (ColGroupPiecewiseLinearCompressed) ColGroupPiecewiseLinearCompressed.create(
			ColIndexFactory.create(new int[] {0}), bp, slopes, intercepts, numRows);
	}

	private double computeMSE(MatrixBlock orig, MatrixBlock recon) {
		double sumSqErr = 0.0;
		final int rows = orig.getNumRows(), cols = orig.getNumColumns();
		DenseBlock origDb = orig.getDenseBlock();
		DenseBlock reconDb = recon.getDenseBlock();

		for(int r = 0; r < rows; r++)
			for(int c = 0; c < cols; c++) {
				double diff = origDb.get(r, c) - reconDb.get(r, c);
				sumSqErr += diff * diff;
			}
		return sumSqErr / (rows * cols);
	}

	@Test
	public void testDecompressRandomMultiCol() {
		final int nrows = 50, ncols = 3;
		double[][] origData = getRandomMatrix(nrows, ncols, -3, 3, 1.0, SEED);

		int[] colArray = {0, 1, 2};
		IColIndex cols = ColIndexFactory.create(colArray);
		CompressionSettings cs = new CompressionSettingsBuilder().create();
		cs.setPiecewiseTargetLoss(10.0);

		MatrixBlock orig = DataConverter.convertToMatrixBlock(origData);
		orig.allocateDenseBlock();

		AColGroup cg = ColGroupFactory.compressPiecewiseLinearFunctional(cols, orig, cs);
		ColGroupPiecewiseLinearCompressed pl = (ColGroupPiecewiseLinearCompressed) cg;

		MatrixBlock recon = new MatrixBlock(nrows, ncols, false);
		recon.allocateDenseBlock();
		pl.decompressToDenseBlock(recon.getDenseBlock(), 0, nrows, 0, ncols - 1);

		double mse = computeMSE(orig, recon);
		assertTrue("MSE=" + mse + " > bound 20.0", mse <= 20.0);
	}

	@Test
	public void testDecompressRandomSingleCol() {
		final int nrows = 40, ncols = 1;
		double[][] origData = getRandomMatrix(nrows, ncols, -2, 2, 1.0, SEED);

		IColIndex cols = ColIndexFactory.create(new int[] {0});
		CompressionSettings cs = new CompressionSettingsBuilder().create();
		cs.setPiecewiseTargetLoss(5.0);

		MatrixBlock orig = DataConverter.convertToMatrixBlock(origData);
		orig.allocateDenseBlock();

		AColGroup cg = ColGroupFactory.compressPiecewiseLinearFunctional(cols, orig, cs);
		ColGroupPiecewiseLinearCompressed pl = (ColGroupPiecewiseLinearCompressed) cg;

		MatrixBlock recon = new MatrixBlock(nrows, 1, false);
		recon.allocateDenseBlock();
		pl.decompressToDenseBlock(recon.getDenseBlock(), 0, nrows, 0, 0);

		double mse = computeMSE(orig, recon);
		assertTrue("Single-Col MSE=" + mse + " > 8.0", mse <= 8.0);
	}

	@Test
	public void testDecompressRandomTrend() {
		final int nrows = 60, ncols = 2;
		Random rng = new Random(SEED);
		double[][] origData = new double[nrows][ncols];

		for(int r = 0; r < nrows; r++) {
			double trend = 0.03 * r;
			for(int c = 0; c < ncols; c++) {
				origData[r][c] = trend + rng.nextGaussian() * 1.2 + c * 1.5;
			}
		}

		int[] colArray = {0, 1};
		IColIndex cols = ColIndexFactory.create(colArray);

		CompressionSettings cs = new CompressionSettingsBuilder().create();
		cs.setPiecewiseTargetLoss(8.0);

		MatrixBlock orig = DataConverter.convertToMatrixBlock(origData);
		orig.allocateDenseBlock();

		AColGroup cg = ColGroupFactory.compressPiecewiseLinearFunctional(cols, orig, cs);
		assertTrue(cg instanceof ColGroupPiecewiseLinearCompressed);
		ColGroupPiecewiseLinearCompressed pl = (ColGroupPiecewiseLinearCompressed) cg;

		MatrixBlock recon = new MatrixBlock(nrows, ncols, false);
		recon.allocateDenseBlock();
		pl.decompressToDenseBlock(recon.getDenseBlock(), 0, nrows, 0, ncols - 1);

		double mse = computeMSE(orig, recon);
		assertTrue("Trend MSE=" + String.format("%.4f", mse) + " > bound 12.0", mse <= 12.0);

		int[][] bp = pl.getBreakpointsPerCol();
		assertEquals(2, bp.length);
		for(int c = 0; c < 2; c++) {
			assertEquals(0, bp[c][0]);
			assertEquals(nrows, bp[c][bp[c].length - 1]);
			assertTrue(bp[c].length >= 2);
		}
	}

	@Test
	public void testDecompressRandomJumps() {
		final int nrows = 50, ncols = 2;
		double[][] origData = getRandomMatrix(nrows, ncols, -2, 2, 1.0, SEED);

		for(int c = 0; c < ncols; c++) {
			for(int r = 20; r < 30; r++)
				origData[r][c] += 2.0;
			for(int r = 35; r < nrows; r++)
				origData[r][c] += 7.0;
		}

		int[] colArray = {0, 1};
		IColIndex cols = ColIndexFactory.create(colArray);

		CompressionSettings cs = new CompressionSettingsBuilder().create();
		cs.setPiecewiseTargetLoss(12.0);

		MatrixBlock orig = DataConverter.convertToMatrixBlock(origData);
		orig.allocateDenseBlock();

		AColGroup cg = ColGroupFactory.compressPiecewiseLinearFunctional(cols, orig, cs);
		assertTrue(cg instanceof ColGroupPiecewiseLinearCompressed);
		ColGroupPiecewiseLinearCompressed pl = (ColGroupPiecewiseLinearCompressed) cg;

		MatrixBlock recon = new MatrixBlock(nrows, ncols, false);
		recon.allocateDenseBlock();
		pl.decompressToDenseBlock(recon.getDenseBlock(), 0, nrows, 0, ncols - 1);

		double mse = computeMSE(orig, recon);
		assertTrue("Jumps MSE=" + String.format("%.4f", mse) + " > bound 18.0", mse <= 18.0);

		int[][] bp = pl.getBreakpointsPerCol();
		assertEquals(2, bp.length);
		for(int c = 0; c < 2; c++) {
			assertEquals(0, bp[c][0]);
			assertEquals(nrows, bp[c][bp[c].length - 1]);
			assertTrue(bp[c].length >= 3);
		}
	}

	private CompressedSizeInfo createTestCompressedSizeInfo() {
		IColIndex cols = ColIndexFactory.create(new int[] {0});
		EstimationFactors facts = new EstimationFactors(2, 10);

		CompressedSizeInfoColGroup info = new CompressedSizeInfoColGroup(cols, facts,
			AColGroup.CompressionType.PiecewiseLinear);

		List<CompressedSizeInfoColGroup> infos = Arrays.asList(info);
		CompressedSizeInfo csi = new CompressedSizeInfo(infos);

		return csi;
	}

	@Test
	public void testCompressPiecewiseLinearViaRealAPI() {

		MatrixBlock in = new MatrixBlock(10, 1, false);
		in.allocateDenseBlock();
		for(int r = 0; r < 10; r++) {
			in.set(r, 0, r * 0.5);
		}

		CompressionSettings cs = new CompressionSettingsBuilder().addValidCompression(
			AColGroup.CompressionType.PiecewiseLinear).create();

		CompressedSizeInfo csi = createTestCompressedSizeInfo();

		List<AColGroup> colGroups = ColGroupFactory.compressColGroups(in, csi, cs);

		boolean hasPiecewise = colGroups.stream().anyMatch(cg -> cg instanceof ColGroupPiecewiseLinearCompressed);
		assertTrue(hasPiecewise);
	}

	private double computeColumnMSE(MatrixBlock orig, MatrixBlock target, int col) {
		final int numRows = orig.getNumRows();
		double totalSSE = 0.0;
		final int origStride = orig.getNumColumns();
		final int tgtStride = target.getNumColumns();

		for(int r = 0; r < numRows; r++) {
			double origVal = orig.getDenseBlock().pos(r * origStride + col);
			double tgtVal = target.getDenseBlock().pos(r * tgtStride + col);
			totalSSE += (origVal - tgtVal) * (origVal - tgtVal);
		}
		return totalSSE / numRows;
	}

	@Test
	public void testSukzessiveLinearColumnSingleSegment() {
		double[] linearCol = {1.0, 2.0, 3.0, 4.0, 5.0};
		CompressionSettings cs = new CompressionSettingsBuilder().create();
		cs.setPiecewiseTargetLoss(1e-6);

		List<Integer> breaks = PiecewiseLinearUtils.computeBreakpointSukzessive(linearCol, cs);
		assertEquals("[0, 5]", breaks.toString());
	}

	@Test
	public void testSukzessiveNoisyColumnMultipleSegments() {
		double[] noisyCol = {1.1, 1.9, 2.2, 10.1, 10.8, 11.3};
		CompressionSettings cs = new CompressionSettingsBuilder().create();
		cs.setPiecewiseTargetLoss(1.0);

		List<Integer> breaks = PiecewiseLinearUtils.computeBreakpointSukzessive(noisyCol, cs);
		assertTrue(breaks.size() >= 3);
	}

	@Test
	public void testSukzessiveTargetLossIncreasesSegments() {
		double[] colWithJumps = {1, 2, 3, 10, 11, 12, 20, 21, 22};
		CompressionSettings csStrict = new CompressionSettingsBuilder().create();
		csStrict.setPiecewiseTargetLoss(0.01);

		CompressionSettings csLoose = new CompressionSettingsBuilder().create();
		csLoose.setPiecewiseTargetLoss(10.0);

		List<Integer> strictBreaks = PiecewiseLinearUtils.computeBreakpointSukzessive(colWithJumps, csStrict);
		List<Integer> looseBreaks = PiecewiseLinearUtils.computeBreakpointSukzessive(colWithJumps, csLoose);

		assertTrue(strictBreaks.size() > looseBreaks.size());
	}

	@Test
	public void testMultiColumnTargetLossRespected() {
		final int rows = 50, cols = 2;
		double[][] data = getRandomMatrix(rows, cols, 0, 10, 1.0, 42L);
		MatrixBlock orig = DataConverter.convertToMatrixBlock(data);
		orig.allocateDenseBlock();

		IColIndex colIdx = ColIndexFactory.create(0, cols - 1);
		CompressionSettings cs = new CompressionSettingsBuilder().create();
		cs.setPiecewiseTargetLoss(1.0);

		AColGroup cg = ColGroupFactory.compressPiecewiseLinearFunctionalSukzessive(colIdx, orig, cs);

		MatrixBlock target = new MatrixBlock(rows, cols, false);
		target.allocateDenseBlock();

		cg.decompressToDenseBlock(target.getDenseBlock(), 0, rows - 1, 0, cols - 1);

		for(int c = 0; c < cols; c++) {
			double mse = computeColumnMSE(orig, target, c);
			System.out.println("Col " + c + " MSE = " + mse);
			assertTrue("Col " + c + " MSE=" + mse + " > target=1.0", mse <= 1.0 + 1e-10);
		}
	}

	@Test
	public void testMultiColumnRandomDecompressLoss() {
		final int rows = 60, cols = 3;
		double[][] origData = getRandomMatrix(rows, cols, -5, 5, 1.0, SEED);

		IColIndex colIdx = ColIndexFactory.create(0, cols - 1);
		CompressionSettings cs = new CompressionSettingsBuilder().create();
		cs.setPiecewiseTargetLoss(8.0);

		MatrixBlock orig = DataConverter.convertToMatrixBlock(origData);
		orig.allocateDenseBlock();

		AColGroup cg = ColGroupFactory.compressPiecewiseLinearFunctionalSukzessive(colIdx, orig, cs);

		MatrixBlock target = new MatrixBlock(rows, cols, false);
		target.allocateDenseBlock();
		cg.decompressToDenseBlock(target.getDenseBlock(), 0, rows, 0, cols - 1);

		for(int c = 0; c < cols; c++) {
			double mse = computeColumnMSE(orig, target, c);
			assertTrue("Col " + c + " MSE=" + mse + " > bound 15.0", mse <= 15.0);
		}
	}

	@Test
	public void testDecompressRandomTrendJumps() {
		final int rows = 80, cols = 2;
		Random rng = new Random(42L);
		double[][] origData = new double[rows][cols];

		for(int r = 0; r < rows; r++) {
			double trend = 0.04 * r;
			for(int c = 0; c < cols; c++) {
				origData[r][c] = trend + rng.nextGaussian() * 1.5;
				if(r >= 25 && r < 45)
					origData[r][c] += 6.0;
				if(r >= 60)
					origData[r][c] += 10.0;
			}
		}

		IColIndex colIdx = ColIndexFactory.create(0, cols - 1);
		CompressionSettings cs = new CompressionSettingsBuilder().create();
		cs.setPiecewiseTargetLoss(10.0);

		MatrixBlock orig = DataConverter.convertToMatrixBlock(origData);
		orig.allocateDenseBlock();

		AColGroup cg = ColGroupFactory.compressPiecewiseLinearFunctionalSukzessive(colIdx, orig, cs);
		MatrixBlock target = new MatrixBlock(rows, cols, false);
		target.allocateDenseBlock();
		cg.decompressToDenseBlock(target.getDenseBlock(), 0, rows, 0, cols - 1);

		for(int c = 0; c < cols; c++) {
			double mse = computeColumnMSE(orig, target, c);
			assertTrue("Trend+Jumps Col " + c + ": MSE=" + mse + " > 20.0", mse <= 20.0);
		}
	}

	@Test
	public void testDecompressRandomSingleColSukzessive() {
		final int rows = 40;
		Random rng = new Random(SEED);
		double[] origCol = new double[rows];

		for(int r = 0; r < rows; r++) {
			origCol[r] = 0.02 * r + rng.nextGaussian() * 0.8;
		}

		double[][] origData = new double[rows][1];
		for(int r = 0; r < rows; r++)
			origData[r][0] = origCol[r];

		IColIndex colIdx = ColIndexFactory.create(new int[] {0});

		CompressionSettings cs = new CompressionSettingsBuilder().create();
		cs.setPiecewiseTargetLoss(1.0);

		MatrixBlock orig = DataConverter.convertToMatrixBlock(origData);
		orig.allocateDenseBlock();

		AColGroup cg = ColGroupFactory.compressPiecewiseLinearFunctionalSukzessive(colIdx, orig, cs);
		ColGroupPiecewiseLinearCompressed pl = (ColGroupPiecewiseLinearCompressed) cg;

		MatrixBlock target = new MatrixBlock(rows, 1, false);
		target.allocateDenseBlock();
		pl.decompressToDenseBlock(target.getDenseBlock(), 0, rows, 0, 0);

		double mse = computeColumnMSE(orig, target, 0);
		assertTrue("Single-Col MSE=" + mse + " > 3.0", mse <= 3.0);

		int[][] bp = pl.getBreakpointsPerCol();
		assertEquals(1, bp.length);
		assertEquals(0, bp[0][0]);
		assertEquals(rows, bp[0][bp[0].length - 1]);
	}

	private boolean hasBreakInRange(int[] bps, int min, int max) {
		for(int i = 1; i < bps.length - 1; i++) {
			if(bps[i] >= min && bps[i] <= max)
				return true;
		}
		return false;
	}

	@Test
	public void testBreakpointsRandomJump() {
		final int len = 30;
		double[] col = getRandomColumn(len, SEED);

		for(int r = 10; r < 20; r++)
			col[r] += 8.0;

		CompressionSettings cs = new CompressionSettingsBuilder().create();
		cs.setPiecewiseTargetLoss(2.0);

		List<Integer> bps = computeBreakpointSukzessive(col, cs);
		int[] bpsArray = bps.stream().mapToInt(Integer::intValue).toArray();

		assertTrue(" (Segs=" + bps.size() + ")", bps.size() >= 3);
		assertTrue("No Break in Jump", hasBreakInRange(bpsArray, 8, 22));
	}

	@Test
	public void testGlobalMSE_random() {
		final int len = 40;
		double[] col = getRandomColumn(len, SEED + 1);

		CompressionSettings cs = new CompressionSettingsBuilder().create();
		cs.setPiecewiseTargetLoss(1.5);

		List<Integer> bps = computeBreakpointSukzessive(col, cs);
		double totalSSE = 0.0;
		for(int i = 0; i < bps.size() - 1; i++) {
			totalSSE += computeSegmentCost(col, bps.get(i), bps.get(i + 1));
		}
		double mse = totalSSE / col.length;

		assertTrue("Global MSE=" + mse + " > target=" + cs.getPiecewiseTargetLoss(),
			mse <= cs.getPiecewiseTargetLoss() + 1e-10);
	}

	private double[] getRandomColumn(int len, long seed) {
		Random rng = new Random(seed);
		double[] col = new double[len];
		for(int i = 0; i < len; i++)
			col[i] = rng.nextGaussian() * 2 + i * 0.01;
		return col;
	}

	@Test
	public void testGetExactSizeOnDiskRandom() {
		Random rng = new Random(SEED);
		final int nrows = 80 + rng.nextInt(40);

		int numSegs = 1 + rng.nextInt(3);
		int[] bp = new int[numSegs + 1];
		bp[0] = 0;
		bp[numSegs] = nrows;
		for(int s = 1; s < numSegs; s++)
			bp[s] = rng.nextInt(nrows * 2 / 3) + nrows / 10;

		double[] slopes = new double[numSegs];
		double[] intercepts = new double[numSegs];
		for(int s = 0; s < numSegs; s++) {
			slopes[s] = rng.nextDouble() * 4 - 2;
			intercepts[s] = rng.nextDouble() * 4 - 2;
		}

		IColIndex cols = ColIndexFactory.create(new int[] {rng.nextInt(20)});
		int[][] bp2d = {bp};
		double[][] slopes2d = {slopes};
		double[][] ints2d = {intercepts};

		AColGroup cg = ColGroupPiecewiseLinearCompressed.create(cols, bp2d, slopes2d, ints2d, nrows);

		long diskSize = cg.getExactSizeOnDisk();
		System.out.println("Single Random: nrows=" + nrows + ", segs=" + numSegs + ", size=" + diskSize);

		assertTrue(diskSize > 0);
		assertTrue(cg.getNumValues() > 0);
	}

	@Test
	public void testMultiColSizeRandom() {
		Random rng = new Random(SEED + 1);
		final int nrows = 100;
		final int numGlobalCols = 3 + rng.nextInt(3);
		int[] globalCols = new int[numGlobalCols];
		for(int i = 0; i < numGlobalCols; i++)
			globalCols[i] = rng.nextInt(50) + i * 5;

		int[][] bp = new int[numGlobalCols][];
		double[][] slopes = new double[numGlobalCols][];
		double[][] intercepts = new double[numGlobalCols][];

		for(int c = 0; c < numGlobalCols; c++) {
			int numSegs = 1 + rng.nextInt(4);
			bp[c] = new int[numSegs + 1];
			bp[c][0] = 0;
			bp[c][numSegs] = nrows;
			for(int s = 1; s < numSegs; s++)
				bp[c][s] = rng.nextInt(nrows * 3 / 4) + nrows / 8;

			slopes[c] = new double[numSegs];
			intercepts[c] = new double[numSegs];
			for(int s = 0; s < numSegs; s++) {
				slopes[c][s] = rng.nextDouble() * 3 - 1.5;
				intercepts[c][s] = rng.nextDouble() * 3 - 1.5;
			}
		}

		IColIndex cols = ColIndexFactory.create(globalCols);
		AColGroup cg = ColGroupPiecewiseLinearCompressed.create(cols, bp, slopes, intercepts, nrows);

		if(cg instanceof ColGroupPiecewiseLinearCompressed) {
			ColGroupPiecewiseLinearCompressed pl = (ColGroupPiecewiseLinearCompressed) cg;

			long diskSize = cg.getExactSizeOnDisk();
			System.out.println("Multi Random: cols=" + numGlobalCols + ", size=" + diskSize);

			assertEquals(numGlobalCols, cols.size());
			assertEquals(numGlobalCols, pl.getBreakpointsPerCol().length);
			for(int c = 0; c < numGlobalCols; c++) {
				assertEquals(nrows, pl.getBreakpointsPerCol()[c][pl.getBreakpointsPerCol()[c].length - 1]);
			}
			assertTrue(diskSize > 0);
		}

	}

	private ColGroupPiecewiseLinearCompressed createTestColGroup() {
		int[][] bps = {{0, 2, 6},  // Col 0: Seg1(len=2), Seg2(len=4)
			{0, 3, 6}   // Col 1: Seg1(len=3), Seg2(len=3)
		};
		double[][] ints = {{1.0, 3.0},    // Col 0 intercepts
			{2.0, 4.0}     // Col 1 intercepts
		};
		double[][] slps = {{0.5, 1.0},    // Col 0 slopes
			{0.0, 2.0}     // Col 1 slopes
		};
		return new ColGroupPiecewiseLinearCompressed(ColIndexFactory.create(0, 2), bps, slps, ints, 6);
	}

	@Test
	public void testComputeSum() {
		ColGroupPiecewiseLinearCompressed cg = createTestColGroup();
		double[] c = new double[2];
		cg.computeSum(c, 6);
		assertEquals(20.5, c[0], 1e-8);
		assertEquals(24.0, c[1], 1e-8);
	}

	@Test
	public void testComputeColSums() {
		ColGroupPiecewiseLinearCompressed cg = createTestColGroup();
		double[] c = new double[2];

		cg.computeColSums(c, 6);
		assertEquals(20.5, c[0], 1e-8);
		assertEquals(24.0, c[1], 1e-8);
	}

	@Test
	public void testSingleColumn() {
		int[][] bps1 = {{0, 3}};
		double[][] ints1 = {{1.0}};
		double[][] slps1 = {{2.0}};
		ColGroupPiecewiseLinearCompressed cg1 = new ColGroupPiecewiseLinearCompressed(ColIndexFactory.create(0, 1),
			bps1, slps1, ints1, 3);

		RightScalarOperator plus5 = new RightScalarOperator(Plus.getPlusFnObject(), 5.0);
		AColGroup result = cg1.scalarOperation(plus5);

		ColGroupPiecewiseLinearCompressed plResult = (ColGroupPiecewiseLinearCompressed) result;
		assertEquals(6.0, plResult.getInterceptsPerCol()[0][0], 1e-8);
		double[] origSum = new double[1];
		cg1.computeSum(origSum, 3);
		double[] newSum = new double[1];
		((ColGroupPiecewiseLinearCompressed) result).computeSum(newSum, 3);
		assertEquals(origSum[0] + 5.0 * 3, newSum[0], 1e-8);
	}

	@Test
	public void testScalarPlus() {
		ColGroupPiecewiseLinearCompressed cg = createTestColGroup();
		RightScalarOperator plus2 = new RightScalarOperator(Plus.getPlusFnObject(), 2.0);
		ColGroupPiecewiseLinearCompressed result = (ColGroupPiecewiseLinearCompressed) cg.scalarOperation(plus2);
		ColGroupPiecewiseLinearCompressed plResult = (ColGroupPiecewiseLinearCompressed) result;

		assertArrayEquals(new double[] {0.5, 1.0}, plResult.getSlopesPerCol()[0], 1e-8);
		assertArrayEquals(new double[] {0.0, 2.0}, plResult.getSlopesPerCol()[1], 1e-8);

		assertArrayEquals(new double[] {3.0, 5.0}, plResult.getInterceptsPerCol()[0], 1e-8);
		assertArrayEquals(new double[] {4.0, 6.0}, plResult.getInterceptsPerCol()[1], 1e-8);

		double[] origSums = new double[2];
		cg.computeSum(origSums, 6);
		double[] newSums = new double[2];
		result.computeSum(newSums, 6);
		assertEquals(origSums[0] + 12.0, newSums[0], 1e-8);
		assertEquals(origSums[1] + 12.0, newSums[1], 1e-8);
	}

	@Test
	public void testBinaryRowOpLeftMultiply() {
		ColGroupPiecewiseLinearCompressed cg = createTestColGroup();
		double[] v = {3.0, 4.0};
		BinaryOperator mult = new BinaryOperator(Multiply.getMultiplyFnObject());

		AColGroup result = cg.binaryRowOpLeft(mult, v, false);

		double[] sums = new double[2];
		result.computeColSums(sums, 6);

		assertEquals(61.5, sums[0], 1e-8);
		assertEquals(96.0, sums[1], 1e-8);
	}

	@Test
	public void testBinaryRowOpRightPlus() {
		ColGroupPiecewiseLinearCompressed cg = createTestColGroup();
		double[] v = {1.0, 2.0};
		BinaryOperator plus = new BinaryOperator(Plus.getPlusFnObject());

		AColGroup result = cg.binaryRowOpRight(plus, v, false);

		double[] sums = new double[2];
		result.computeColSums(sums, 6);
		assertEquals(26.5, sums[0], 1e-8);
		assertEquals(36.0, sums[1], 1e-8);
	}

	@Test
	public void testContainsValue() {
		ColGroupPiecewiseLinearCompressed cg = createTestColGroup();
		assertTrue(cg.containsValue(1.0));
		assertTrue(cg.containsValue(2.0));
		assertTrue(cg.containsValue(1.5));
		assertFalse(cg.containsValue(999.0));
		assertFalse(cg.containsValue(0.0));
	}

	@Test
	public void testEdgeCases() {
		ColGroupPiecewiseLinearCompressed cg = createTestColGroup();
		double[] c = new double[2];
		cg.computeSum(c, 6);
		assertNotNull(cg.binaryRowOpLeft(new BinaryOperator(Plus.getPlusFnObject()), new double[] {0, 0}, true));
	}

}


