package org.apache.sysds.performance;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.colgroup.ColGroupPiecewiseLinearCompressed;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.utils.stats.Timing;
import java.util.Random;

/**
 * Performance benchmark for piecewise linear compression.
 * Successive is benchmarked across large matrices to show scalability.
 * DP is only used as a quality reference on small matrices due to quadratic complexity

 */
public class PiecewiseLinearCompressionPerformanceTest {

	//different target losses : loose, avg, strict
	private static final double[] LOSSES = {1e-1, 1e-2, 1e-4};
	// how often compressed
	private static final int      REPS   = 3;

	/**
	 * generate of a time series matrix to have a realistic test set up
	 * @param nr number of rows
	 * @param nc number of columns
	 * @return matrix with random generated data
	 */
	private static MatrixBlock generateTestMatrix(int nr, int nc) {
		MatrixBlock mb = new MatrixBlock(nr, nc, true);
		mb.allocateDenseBlock();
		Random rng = new Random(42);
		for(int c = 0; c < nc; c++) {
			double trend      = 0.001 * c;
			double level      = rng.nextDouble() * 5.0;
			double volatility = 0.1 + 0.01 * c;
			double residual   = 0.0;

			for(int row = 0; row < nr; row++) {
				// random level shift every 75-150 rows
				if(row % (75 + (int)(75 * rng.nextDouble())) == 0) {
					level += (rng.nextDouble() - 0.5) * 2.0;
					trend += (rng.nextDouble() - 0.5) * 0.0005;
				}
				// noise: residual = 0.7 * prev + random
				residual = 0.7 * residual + rng.nextGaussian() * volatility;
				mb.set(row, c, Math.max(0, trend * row + level + residual));
			}
		}
		return mb;
	}
	///  returns a average number of segments per column
	private static double avgSegments(AColGroup cg) {
		int[][] breakpoints = ((ColGroupPiecewiseLinearCompressed) cg).getBreakpointsPerCol();
		int total = 0;
		for(int[] bp : breakpoints) total += bp.length - 1;
		return total / (double) breakpoints.length;
	}

	/**
	 * computes MSE between the compression, the original data and decompression
	 * @param orig original matrix
	 * @param cg piecewise linear compressed column group
	 * @return MSE
	 */
	private static double reconstructionMSE(MatrixBlock orig, AColGroup cg) {
		int nr = orig.getNumRows(), nc = orig.getNumColumns();
		MatrixBlock recon = new MatrixBlock(nr, nc, false);
		recon.allocateDenseBlock();
		cg.decompressToDenseBlock(recon.getDenseBlock(), 0, nr, 0, 0);
		double sse = 0;
		for(int r = 0; r < nr; r++)
			for(int c = 0; c < nc; c++) {
				double diff = orig.get(r, c) - recon.get(r, c);
				sse += diff * diff;
			}
		return sse / (nr * nc);
	}

	/**
	 * benchmarks successive compression for a given matrix and target loss
	 * reports segments, compressed data size, runtime and reconstruction
	 * @param mb original matrix to compress
	 * @param loss target loss param
	 */
	private static void benchmarkSuccessive(MatrixBlock mb, double loss) {
		long origSize = mb.getInMemorySize();
		int numRows = mb.getNumRows(), numCol = mb.getNumColumns();
		CompressionSettings cs = new CompressionSettingsBuilder().create();
		cs.setPiecewiseTargetLoss(loss);
		IColIndex colIndexes = ColIndexFactory.create(numCol);

		ColGroupFactory.compressPiecewiseLinearFunctionalSuccessive(colIndexes, mb, cs);

		Timing t = new Timing();
		AColGroup cg = null;
		t.start();
		for(int i = 0; i < REPS; i++)
			cg = ColGroupFactory.compressPiecewiseLinearFunctionalSuccessive(colIndexes, mb, cs);
		double time = t.stop() / REPS;

		long size = cg.getExactSizeOnDisk();
		String saving = size < origSize
			? String.format("saved %3.0f%%", 100.0 - 100.0 * size / origSize)
			: String.format("larger +%.0f%%", 100.0 * size / origSize - 100);

		System.out.printf("  successive  loss=%.0e  %5.1f segs  %6.2f MB (%s)  %6.1f ms  MSE=%.2e%n",
			loss, avgSegments(cg), size / 1e6, saving, time, reconstructionMSE(mb, cg));
	}

	/**
	 * benchmarks dynamic programming compression for a given matrix and target loss
	 * no repetition, because DP is too slow due complexity
	 * reports segments, compressed data size, runtime and reconstruction
	 * @param mb original matrix to compress
	 * @param loss target loss param
	 */
	private static void benchmarkDP(MatrixBlock mb, double loss) {
		long origSize = mb.getInMemorySize();
		int numColumns = mb.getNumColumns();
		CompressionSettings cs = new CompressionSettingsBuilder().create();
		cs.setPiecewiseTargetLoss(loss);
		IColIndex colIndexes = ColIndexFactory.create(numColumns);

		Timing t = new Timing();
		t.start();
		AColGroup cg = ColGroupFactory.compressPiecewiseLinearFunctional(colIndexes, mb, cs);
		double time = t.stop();

		long size = cg.getExactSizeOnDisk();
		String saving = size < origSize
			? String.format("saved %3.0f%%", 100.0 - 100.0 * size / origSize)
			: String.format("LARGER +%.0f%%", 100.0 * size / origSize - 100);

		System.out.printf("  DP          loss=%.0e  %5.1f segs  %6.2f MB (%s)  %6.1f ms  MSE=%.2e%n",
			loss, avgSegments(cg), size / 1e6, saving, time, reconstructionMSE(mb, cg));
	}

	public static void main(String[] args) {
		System.out.println("=== Piecewise Linear Compression Benchmark ===\n");

		// Successive scalability across large matrices
		System.out.println("=== Successive: scalability ===");
		int[][] configs = {{1000, 10}, {1000, 100}, {1000, 500},
			{5000, 10}, {5000, 100}, {5000, 500},
			{10000, 10}, {10000, 100}, {10000, 500}};

		for(int[] cfg : configs) {
			int nr = cfg[0], nc = cfg[1];
			MatrixBlock mb = generateTestMatrix(nr, nc);
			System.out.printf("%nnrows=%d  ncols=%d  original=%.2f MB%n",
				nr, nc, mb.getInMemorySize() / 1e6);
			for(double loss : LOSSES)
				benchmarkSuccessive(mb, loss);
		}

		// DP quality reference on small matrix
		System.out.println("\n=== DP: quality reference (nrows=1000, ncols=10) ===");
		MatrixBlock mbSmall = generateTestMatrix(1000, 10);
		System.out.printf("original=%.2f MB%n", mbSmall.getInMemorySize() / 1e6);
		for(double loss : LOSSES)
			benchmarkDP(mbSmall, loss);
	}
}
