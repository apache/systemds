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
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.*;

/**
 * Unit tests of ColGroupPiecewiseLinearCompression Covers Validation, Compression and decompression
 */
public class ColGroupPiecewiseLinearCompressedTest extends AutomatedTestBase {

	private static final long SEED = 42L;

	@Override
	public void setUp() {

	}

	@Test(expected = NullPointerException.class)
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

	@Test
	public void testCompressAndDecompressDP() {

		// create random matrix
		final int nrows = 50, ncols = 3;
		double[][] data = getRandomMatrix(nrows, ncols, -3, 3, 1.0, SEED);
		MatrixBlock in = DataConverter.convertToMatrixBlock(data);
		in.allocateDenseBlock();

		IColIndex colIndexes = ColIndexFactory.create(new int[] {0, 1, 2});
		CompressionSettings cs = new CompressionSettingsBuilder().create();
		cs.setPiecewiseTargetLoss(1e-8);

		AColGroup result = ColGroupFactory.compressPiecewiseLinearFunctional(colIndexes, in, cs);
		assertTrue(result instanceof ColGroupPiecewiseLinearCompressed);
		ColGroupPiecewiseLinearCompressed plGroup = (ColGroupPiecewiseLinearCompressed) result;

		// check the structure
		int[][] breakpoints = plGroup.getBreakpointsPerCol();
		double[][] slopes = plGroup.getSlopesPerCol();
		double[][] intercepts = plGroup.getInterceptsPerCol();

		assertEquals("wrong number of columns in breakpoints", ncols, breakpoints.length);
		for(int c = 0; c < ncols; c++) {
			assertTrue("breakpoints[" + c + "] needs at least 2 entries", breakpoints[c].length >= 2);
			assertEquals("breakpoints[" + c + "] must start at 0", 0, breakpoints[c][0]);
			assertEquals("breakpoints[" + c + "] must end at nrows", nrows, breakpoints[c][breakpoints[c].length - 1]);
			int numSegs = breakpoints[c].length - 1;
			assertEquals("slopes[" + c + "] length mismatch", numSegs, slopes[c].length);
			assertEquals("intercepts[" + c + "] length mismatch", numSegs, intercepts[c].length);
		}

		// decompress and check reconstruction of column group
		MatrixBlock recon = new MatrixBlock(nrows, ncols, false);
		recon.allocateDenseBlock();
		plGroup.decompressToDenseBlock(recon.getDenseBlock(), 0, nrows, 0, 0);
		DenseBlock db = recon.getDenseBlock();

		for(int r = 0; r < nrows; r++) {
			for(int c = 0; c < ncols; c++) {
				double val = db.get(r, c);
				assertFalse("NaN at [" + r + "," + c + "]", Double.isNaN(val));
				assertFalse("Infinite at [" + r + "," + c + "]", Double.isInfinite(val));
				assertEquals("reconstruction error too large at [" + r + "," + c + "]", data[r][c], val, 1e-6);
			}
		}
	}

	@Test
	public void testCompressAndDecompressSuccessive() {

		//create random matrix
		final int nrows = 50, ncols = 3;
		double[][] data = getRandomMatrix(nrows, ncols, -3, 3, 1.0, SEED);
		MatrixBlock in = DataConverter.convertToMatrixBlock(data);
		in.allocateDenseBlock();

		IColIndex colIndexes = ColIndexFactory.create(new int[] {0, 1, 2});
		CompressionSettings cs = new CompressionSettingsBuilder().create();
		cs.setPiecewiseTargetLoss(1e-8);

		// create ColGroupPiecewiseLinearCompressed with successive compression
		AColGroup result = ColGroupFactory.compressPiecewiseLinearFunctionalSuccessive(colIndexes, in, cs);
		assertTrue(result instanceof ColGroupPiecewiseLinearCompressed);
		ColGroupPiecewiseLinearCompressed plGroup = (ColGroupPiecewiseLinearCompressed) result;

		// structure checks
		int[][] bp = plGroup.getBreakpointsPerCol();
		double[][] slopes = plGroup.getSlopesPerCol();
		double[][] intercepts = plGroup.getInterceptsPerCol();

		assertEquals("wrong number of columns in bp", ncols, bp.length);
		for(int c = 0; c < ncols; c++) {
			assertTrue("bp[" + c + "] needs at least 2 entries", bp[c].length >= 2);
			assertEquals("bp[" + c + "] must start at 0", 0, bp[c][0]);
			assertEquals("bp[" + c + "] must end at nrows", nrows, bp[c][bp[c].length - 1]);
			int numSegs = bp[c].length - 1;
			assertEquals("slopes[" + c + "] length mismatch", numSegs, slopes[c].length);
			assertEquals("intercepts[" + c + "] length mismatch", numSegs, intercepts[c].length);
		}

		// validate decompression
		MatrixBlock recon = new MatrixBlock(nrows, ncols, false);
		recon.allocateDenseBlock();
		plGroup.decompressToDenseBlock(recon.getDenseBlock(), 0, nrows, 0, 0);
		DenseBlock db = recon.getDenseBlock();

		for(int r = 0; r < nrows; r++) {
			for(int c = 0; c < ncols; c++) {
				double val = db.get(r, c);
				assertFalse("NaN at [" + r + "," + c + "]", Double.isNaN(val));
				assertFalse("Infinite at [" + r + "," + c + "]", Double.isInfinite(val));
				assertEquals("reconstruction error too large at [" + r + "," + c + "]", data[r][c], val, 1e-6);
			}
		}
	}

	/// Wrapper-Classes: Test setup for DP and successive compression

	private void testRoundtripDP(double[][] data, int nrows, int ncols, double targetLoss, double tolerance,
		int maxFailures) {
		testRoundtrip(data, nrows, ncols, targetLoss, tolerance, maxFailures, false);
	}

	private void testRoundtripSuccessive(double[][] data, int nrows, int ncols, double targetLoss, double tolerance,
		int maxFailures) {
		testRoundtrip(data, nrows, ncols, targetLoss, tolerance, maxFailures, true);
	}

	/**
	 * Set test setup: converting data in matrix block, set compression setting does compression, decompression,
	 * validation
	 */
	private void testRoundtrip(double[][] data, int nrows, int ncols, double targetLoss, double tolerance,
		int maxFailures, boolean successive) {

		///create a matrix
		MatrixBlock orig = DataConverter.convertToMatrixBlock(data);
		orig.allocateDenseBlock();

		IColIndex colIndexes = ColIndexFactory.create(buildColArray(ncols));
		CompressionSettings cs = new CompressionSettingsBuilder().create();
		cs.setPiecewiseTargetLoss(targetLoss);

		/// choose compression
		AColGroup result = successive ? ColGroupFactory.compressPiecewiseLinearFunctionalSuccessive(colIndexes, orig,
			cs) : ColGroupFactory.compressPiecewiseLinearFunctional(colIndexes, orig, cs);

		assertTrue(result instanceof ColGroupPiecewiseLinearCompressed);
		ColGroupPiecewiseLinearCompressed plGroup = (ColGroupPiecewiseLinearCompressed) result;

		/// structure checks
		checkStructure(plGroup, nrows, ncols);

		/// decompression check
		MatrixBlock recon = new MatrixBlock(nrows, ncols, false);
		recon.allocateDenseBlock();
		plGroup.decompressToDenseBlock(recon.getDenseBlock(), 0, nrows, 0, 0);
		DenseBlock db = recon.getDenseBlock();

		int failures = 0;
		for(int r = 0; r < nrows; r++) {
			for(int c = 0; c < ncols; c++) {
				double val = db.get(r, c);
				assertFalse("NaN at [" + r + "," + c + "]", Double.isNaN(val));
				assertFalse("Infinite at [" + r + "," + c + "]", Double.isInfinite(val));
				if(Math.abs(data[r][c] - val) > tolerance)
					failures++;
			}
		}
		assertTrue("too many reconstruction failures: " + failures, failures <= maxFailures);
	}

	private void checkStructure(ColGroupPiecewiseLinearCompressed plGroup, int nrows, int ncols) {
		int[][] breakpoints = plGroup.getBreakpointsPerCol();
		double[][] slopes = plGroup.getSlopesPerCol();
		double[][] intercepts = plGroup.getInterceptsPerCol();

		assertEquals("wrong number of columns in breakpoints", ncols, breakpoints.length);
		assertEquals("wrong number of col indices", ncols, plGroup.getColIndices().size());

		for(int c = 0; c < ncols; c++) {
			assertTrue("breakpoints[" + c + "] needs at least 2 entries", breakpoints[c].length >= 2);
			assertEquals("breakpoints[" + c + "] must start at 0", 0, breakpoints[c][0]);
			assertEquals("breakpoints[" + c + "] must end at nrows", nrows, breakpoints[c][breakpoints[c].length - 1]);
			int numSegs = breakpoints[c].length - 1;
			assertEquals("slopes[" + c + "] length mismatch", numSegs, slopes[c].length);
			assertEquals("intercepts[" + c + "] length mismatch", numSegs, intercepts[c].length);
		}
	}

	private double[][] buildMultiSegmentData(int nrows, int ncols) {
		Random rng = new Random(SEED);
		double[][] data = new double[nrows][ncols];
		int[] segStarts = {0, 15, 30, 45, 60};
		double[] slopes = {0.5, -1.2, 2.0, -0.8};

		for(int c = 0; c < ncols; c++) {
			double offset = c;
			for(int r = 0; r < nrows; r++) {
				int seg = 0;
				while(seg < segStarts.length - 1 && r >= segStarts[seg + 1])
					seg++;
				data[r][c] = slopes[seg] * (r - segStarts[seg]) + offset + rng.nextGaussian() * 0.8;
				offset += 0.01;
			}
		}
		return data;
	}

	private int[] buildColArray(int ncols) {
		int[] cols = new int[ncols];
		for(int i = 0; i < ncols; i++)
			cols[i] = i;
		return cols;
	}

	@Test
	public void testTrendWithNoise() {
		final int nrows = 100, ncols = 2;
		Random rng = new Random(SEED);
		double[][] data = new double[nrows][ncols];
		for(int r = 0; r < nrows; r++) {
			double trend = 0.05 * r;
			for(int c = 0; c < ncols; c++)
				data[r][c] = trend + rng.nextGaussian() * 1.5 + c * 2.0;
		}
		testRoundtripDP(data, nrows, ncols, 1.0, 4.0, 45);
		testRoundtripSuccessive(data, nrows, ncols, 1.0, 4.0, 45);
	}

	@Test
	public void testAbruptJumps() {
		final int nrows = 80, ncols = 3;
		double[][] data = getRandomMatrix(nrows, ncols, -2, 2, 1.0, SEED);
		for(int c = 0; c < ncols; c++) {
			for(int r = 25; r < 55; r++)
				data[r][c] += 8.0;
			for(int r = 55; r < nrows; r++)
				data[r][c] += 15.0;
		}
		// successive needs looser tolerance on jumps
		testRoundtripDP(data, nrows, ncols, 5.0, 10.0, 50);
		testRoundtripSuccessive(data, nrows, ncols, 25.0, 18.0, 55);
	}

	@Test
	public void testHighFrequency() {
		final int nrows = 100, ncols = 50;
		Random rng = new Random(SEED);
		double[][] data = new double[nrows][ncols];
		for(int r = 0; r < nrows; r++) {
			double sine = Math.sin(r * 0.4) * 4.0;
			for(int c = 0; c < ncols; c++)
				data[r][c] = sine + rng.nextGaussian() * 0.8 + Math.sin(r * 0.2 + c) * 2.0;
		}
		// both struggle with high frequency; successive slightly worse
		testRoundtripDP(data, nrows, ncols, 2.0, 2.0, 3500);
		testRoundtripSuccessive(data, nrows, ncols, 2.0, 2.5, 2500);
	}

	@Test
	public void testLowVarianceSingleColumn() {
		double[][] data = getRandomMatrix(50, 1, -1, 1, 0.3, SEED);
		testRoundtripDP(data, 50, 1, 0.1, 0.5, 5);
		testRoundtripSuccessive(data, 50, 1, 0.05, 0.4, 3);
	}

	@Test
	public void testSingleColumn() {
		double[][] data = getRandomMatrix(50, 1, -1, 1, 1.0, SEED);
		testRoundtripDP(data, 50, 1, 0.5, 1.0, 8);
		testRoundtripSuccessive(data, 50, 1, 0.5, 1.0, 8);
	}

	@Test
	public void testKnownSegmentBoundaries() {
		final int nrows = 60, ncols = 2;
		double[][] data = buildMultiSegmentData(nrows, ncols);
		// successive needs slightly higher targetLoss for same data
		testRoundtripDP(data, nrows, ncols, 0.8, 5.0, 35);
		testRoundtripSuccessive(data, nrows, ncols, 1.0, 5.0, 35);
	}

	@Test
	public void testMultipleColumns() {
		double[][] data = getRandomMatrix(80, 5, -5, 5, 1.5, SEED);
		testRoundtripDP(data, 80, 5, 3.0, 4.0, 120);
		testRoundtripSuccessive(data, 80, 5, 3.0, 4.0, 120);
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

	@Test
	public void testSuccessiveLinearColumnSingleSegment() {
		double[] col = {1.0, 2.0, 3.0, 4.0, 5.0};
		CompressionSettings cs = new CompressionSettingsBuilder().create();
		cs.setPiecewiseTargetLoss(1e-6);

		List<Integer> breaks = PiecewiseLinearUtils.computeBreakpointSuccessive(col, cs);
		assertEquals("[0, 5]", breaks.toString());
	}

	@Test
	public void testSuccessiveNoisyColumnMultipleSegments() {
		double[] col = {1.1, 1.9, 2.2, 10.1, 10.8, 11.3};
		CompressionSettings cs = new CompressionSettingsBuilder().create();
		cs.setPiecewiseTargetLoss(1.0);

		List<Integer> breaks = PiecewiseLinearUtils.computeBreakpointSuccessive(col, cs);
		assertTrue("expected at least 3 breakpoints", breaks.size() >= 3);
	}

	@Test
	public void testSuccessiveStrictLossProducesMoreSegments() {
		double[] col = {1, 2, 3, 10, 11, 12, 20, 21, 22};

		CompressionSettings strict = new CompressionSettingsBuilder().create();
		strict.setPiecewiseTargetLoss(0.01);

		CompressionSettings loose = new CompressionSettingsBuilder().create();
		loose.setPiecewiseTargetLoss(10.0);

		List<Integer> strictBreaks = PiecewiseLinearUtils.computeBreakpointSuccessive(col, strict);
		List<Integer> looseBreaks = PiecewiseLinearUtils.computeBreakpointSuccessive(col, loose);

		assertTrue("strict loss should produce more segments", strictBreaks.size() > looseBreaks.size());
	}

	@Test
	public void testSuccessiveBreakpointDetectedAtJump() {
		double[] col = getRandomColumn(30, SEED);
		for(int r = 10; r < 20; r++)
			col[r] += 8.0;

		CompressionSettings cs = new CompressionSettingsBuilder().create();
		cs.setPiecewiseTargetLoss(2.0);

		int[] bps = PiecewiseLinearUtils.computeBreakpointSuccessive(col, cs).stream().mapToInt(Integer::intValue)
			.toArray();

		assertTrue("expected at least 3 segments", bps.length >= 3);
		assertTrue("expected breakpoint near jump [10,20]", hasBreakInRange(bps, 8, 22));
	}

	@Test
	public void testSuccessiveGlobalMSEWithinTarget() {
		double[] col = getRandomColumn(40, SEED + 1);
		CompressionSettings cs = new CompressionSettingsBuilder().create();
		cs.setPiecewiseTargetLoss(1.5);

		List<Integer> bps = PiecewiseLinearUtils.computeBreakpointSuccessive(col, cs);
		double sse = 0.0;
		for(int i = 0; i < bps.size() - 1; i++)
			sse += PiecewiseLinearUtils.computeSegmentCost(col, bps.get(i), bps.get(i + 1));

		double mse = sse / col.length;
		assertTrue("global MSE=" + mse + " exceeds target=" + cs.getPiecewiseTargetLoss(),
			mse <= cs.getPiecewiseTargetLoss() + 1e-10);
	}

	private boolean hasBreakInRange(int[] bps, int min, int max) {
		for(int i = 1; i < bps.length - 1; i++)
			if(bps[i] >= min && bps[i] <= max)
				return true;
		return false;
	}

	private double[] getRandomColumn(int len, long seed) {
		Random rng = new Random(seed);
		double[] col = new double[len];
		for(int i = 0; i < len; i++)
			col[i] = rng.nextGaussian() * 2 + i * 0.01;
		return col;
	}

}


