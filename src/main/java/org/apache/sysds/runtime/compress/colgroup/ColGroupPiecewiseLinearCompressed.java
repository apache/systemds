package org.apache.sysds.runtime.compress.colgroup;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.scheme.ICLAScheme;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.functionobjects.*;
import org.apache.sysds.runtime.instructions.cp.CmCovObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.utils.MemoryEstimates;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * This class represents a new ColGroup which is compresses column into segments (piecewise linear) to represent the
 * original Data each column is approximate by a set of linear segments defined by breakpoints, slopes and intercepts
 */

public class ColGroupPiecewiseLinearCompressed extends AColGroupCompressed {
	/**
	 * breakpoints indices per column to define the segment boundaries slopes of the regression line per segment per
	 * column intercepts of the regression line per segment per column
	 */
	int[][] breakpointsPerCol;
	double[][] slopesPerCol;
	double[][] interceptsPerCol;
	int numRows;

	protected ColGroupPiecewiseLinearCompressed(IColIndex colIndices) {
		super(colIndices);
	}

	public ColGroupPiecewiseLinearCompressed(IColIndex colIndices, int[][] breakpoints, double[][] slopes,
		double[][] intercepts, int numRows) {
		super(colIndices);
		this.breakpointsPerCol = breakpoints;
		this.slopesPerCol = slopes.clone();
		this.interceptsPerCol = intercepts.clone();
		this.numRows = numRows;
	}

	/**
	 * creates a new piecewise linear compress column group validates inputs and copies all arrays before storing
	 *
	 * @param colIndices        the column indices this group represents
	 * @param breakpointsPerCol breakpoint indices per column
	 * @param slopesPerCol      slope of each segment per column
	 * @param interceptsPerCol  intercept of each segment per column
	 * @param numRows           number of rows in the original matrix
	 * @return a new ColGroupPiecewiseLinearCompressed instance
	 * @throws IllegalArgumentException if breakpoints are invalid or arrays are inconsistent
	 */

	public static AColGroup create(IColIndex colIndices, int[][] breakpointsPerCol, double[][] slopesPerCol,
		double[][] interceptsPerCol, int numRows) {
		final int numCols = colIndices.size();
		if(breakpointsPerCol.length != numCols)
			throw new IllegalArgumentException(
				"bp.length=" + breakpointsPerCol.length + " != colIndices.size()=" + numCols);

		for(int c = 0; c < numCols; c++) {
			if(breakpointsPerCol[c].length < 1 || breakpointsPerCol[c][0] != 0 ||
				breakpointsPerCol[c][breakpointsPerCol[c].length - 1] != numRows)
				throw new IllegalArgumentException(
					"Invalid breakpoints for col " + c + ": must start=0, end=numRows, >=1 pts");

			if(slopesPerCol[c].length != interceptsPerCol[c].length ||
				slopesPerCol[c].length != breakpointsPerCol[c].length - 1)
				throw new IllegalArgumentException("Inconsistent array lengths col " + c);
		}

		int[][] bpCopy = new int[numCols][];
		double[][] slopeCopy = new double[numCols][];
		double[][] interceptCopy = new double[numCols][];
		// defensive copy to prevent external modification
		for(int c = 0; c < numCols; c++) {
			bpCopy[c] = Arrays.copyOf(breakpointsPerCol[c], breakpointsPerCol[c].length);
			slopeCopy[c] = Arrays.copyOf(slopesPerCol[c], slopesPerCol[c].length);
			interceptCopy[c] = Arrays.copyOf(interceptsPerCol[c], interceptsPerCol[c].length);
		}

		return new ColGroupPiecewiseLinearCompressed(colIndices, bpCopy, slopeCopy, interceptCopy, numRows);

	}

	/**
	 * Decompresses a ColGroupPiecewiseLinearCompress into a DenseBlock Each value is reconstructed via slopes[seg]*row
	 * + intercept[seg]
	 *
	 * @param db   Target DenseBlock
	 * @param rl   Row to start decompression from
	 * @param ru   Row to end decompression at (not inclusive)
	 * @param offR Row offset into the target to decompress
	 * @param offC Column offset into the target to decompress
	 */
	@Override
	public void decompressToDenseBlock(DenseBlock db, int rl, int ru, int offR, int offC) {
		if(db == null || _colIndexes == null || _colIndexes.size() == 0 || breakpointsPerCol == null ||
			slopesPerCol == null || interceptsPerCol == null) {
			return;
		}
		for(int col = 0; col < _colIndexes.size(); col++) {
			final int colIndex = _colIndexes.get(col);
			int[] breakpoints = breakpointsPerCol[col];
			double[] slopes = slopesPerCol[col];
			double[] intercepts = interceptsPerCol[col];
			// per segment in this column
			for(int seg = 0; seg + 1 < breakpoints.length; seg++) {
				int segStart = breakpoints[seg];
				int segEnd = breakpoints[seg + 1];
				if(segStart >= segEnd)
					continue;

				double currentSlopeInSegment = slopes[seg];
				double currentInterceptInSegment = intercepts[seg];
				// intersect segment with requested row range [rl, ru)

				int rowStart = Math.max(segStart, rl);
				int rowEnd = Math.min(segEnd, ru);
				if(rowStart >= rowEnd)
					continue;

				//Fill DenseBlock für this column and Segment
				for(int row = rowStart; row < rowEnd; row++) {
					double yhat = currentSlopeInSegment * row + currentInterceptInSegment;
					int dbRow = offR + row;
					int dbCol = offC + colIndex;

					if(dbRow >= 0 && dbRow < db.numRows() && dbCol >= 0 && dbCol < db.numCols()) {
						db.set(dbRow, dbCol, yhat);
					}
				}

			}

		}
	}

	public int[][] getBreakpointsPerCol() {
		return breakpointsPerCol;
	}

	public double[][] getSlopesPerCol() {
		return slopesPerCol;
	}

	public double[][] getInterceptsPerCol() {
		return interceptsPerCol;
	}

	/**
	 * Return a decompressed value at row r and column colIdx uses binary search to find the correct segment
	 *
	 * @param r      row
	 * @param colIdx column index in the _colIndexes.
	 * @return reconstructed value with slope[segment]*r+intercepts[segment]
	 */
	@Override
	public double getIdx(int r, int colIdx) {
		//safety check
		if(r < 0 || r >= numRows || colIdx < 0 || colIdx >= _colIndexes.size()) {
			return 0.0;
		}
		int[] breakpoints = breakpointsPerCol[colIdx];
		double[] slopes = slopesPerCol[colIdx];
		double[] intercepts = interceptsPerCol[colIdx];
		// binary search for the segment containing row r
		int lowerBound = 0;
		int higherBound = breakpoints.length - 2;
		while(lowerBound <= higherBound) {
			int mid = (lowerBound + higherBound) / 2;
			if(r < breakpoints[mid + 1]) {
				higherBound = mid - 1;
			}
			else
				lowerBound = mid + 1;
		}
		int segment = Math.min(lowerBound, breakpoints.length - 2);
		return slopes[segment] * (double) r + intercepts[segment];
	}

	/**
	 * Returns a total number of stored values remaining all columns counting breakpoints, slopes and intercepts per
	 * column
	 *
	 * @return total number of stored compression values
	 */
	@Override
	public int getNumValues() {
		int total = 0;
		for(int c = 0; c < _colIndexes.size(); c++) {
			total += breakpointsPerCol[c].length + slopesPerCol[c].length + interceptsPerCol[c].length;
		}
		return total;
	}

	/**
	 * Returns the exact size on disk in bytes includes per column arrays for breakpoints, slopes, intercepts
	 *
	 * @return size in bytes
	 */
	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		int numCols = _colIndexes.size();
		ret += 8L * numCols * 3; //array reference pointers
		ret += 24L * 3; // outer array headers
		ret += 4L; //numRows field

		for(int c = 0; c < numCols; c++) {
			ret += (long) MemoryEstimates.intArrayCost(breakpointsPerCol[c].length);
			ret += (long) MemoryEstimates.doubleArrayCost(slopesPerCol[c].length);
			ret += (long) MemoryEstimates.doubleArrayCost(interceptsPerCol[c].length);
		}

		return ret;

	}

	/**
	 * Computes the column sums of the decompressed matrix using sum of arithmetic series Where sumX = len * (2*start +
	 * len - 1) / 2
	 *
	 * @param c     output array to accumulate column sums into
	 * @param nRows number of rows, which is used because it is covered by the breakpoints
	 */
	/** Accumulates the sum of all decompressed values across all columns into c[0]. */
	@Override
	public void computeSum(double[] c, int nRows) {
		for(int col = 0; col < _colIndexes.size(); col++) {
			int[] breakpoints = breakpointsPerCol[col];
			double[] intercepts = interceptsPerCol[col];
			double[] slopes = slopesPerCol[col];

			for(int seg = 0; seg < slopes.length; seg++) {
				int start = breakpoints[seg];
				int end = breakpoints[seg + 1];
				int len = end - start;
				if(len <= 0)
					continue;

				double sumX = (double) len * (2.0 * start + (len - 1)) / 2.0;
				c[0] += slopes[seg] * sumX + intercepts[seg] * len;
			}
		}
	}

	/** Accumulates the sum for each column into c[_colIndexes.get(col)]. */
	@Override
	public void computeColSums(double[] c, int nRows) {
		for(int col = 0; col < _colIndexes.size(); col++) {
			int gcol = _colIndexes.get(col);
			int[] breakpoints = breakpointsPerCol[col];
			double[] intercepts = interceptsPerCol[col];
			double[] slopes = slopesPerCol[col];

			for(int seg = 0; seg < slopes.length; seg++) {
				int start = breakpoints[seg];
				int end = breakpoints[seg + 1];
				int len = end - start;
				if(len <= 0)
					continue;

				double sumX = (double) len * (2.0 * start + (len - 1)) / 2.0;
				c[gcol] += slopes[seg] * sumX + intercepts[seg] * len;
			}
		}
	}

	@Override
	public CompressionType getCompType() {
		return CompressionType.PiecewiseLinear;
	}

	@Override
	protected ColGroupType getColGroupType() {
		return ColGroupType.PiecewiseLinear;
	}

	/**
	 * Applies a scalar operation to all segments of this column group For plus/minus operation are only the intercepts
	 * modified For Multiply/Divide slopes and intercepts are scaled
	 *
	 * @param op operation to perform
	 * @return a new ColGroupPiecewiseLinearCompressed with updated coefficients
	 * @throws NotImplementedException if the operator is not plus, minus, multiply or divide
	 */
	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		final int numCols = _colIndexes.size();

		if(!(op.fn instanceof Plus || op.fn instanceof Minus || op.fn instanceof Multiply || op.fn instanceof Divide)) {
			throw new NotImplementedException("Unsupported scalar op: " + op.fn.getClass().getSimpleName());
		}

		double[][] newIntercepts = new double[numCols][];
		double[][] newSlopes = new double[numCols][];

		for(int col = 0; col < numCols; col++) {
			final int numSegments = interceptsPerCol[col].length;
			newIntercepts[col] = new double[numSegments];
			newSlopes[col] = new double[numSegments];

			for(int seg = 0; seg < numSegments; seg++) {
				if(op.fn instanceof Plus || op.fn instanceof Minus) {
					// only intercepts changes
					newSlopes[col][seg] = slopesPerCol[col][seg];
					newIntercepts[col][seg] = op.executeScalar(interceptsPerCol[col][seg]);
				}
				else {  // Multiply/Divide
					newSlopes[col][seg] = op.executeScalar(slopesPerCol[col][seg]);
					newIntercepts[col][seg] = op.executeScalar(interceptsPerCol[col][seg]);
				}
			}
		}

		return new ColGroupPiecewiseLinearCompressed(_colIndexes, breakpointsPerCol, newSlopes, newIntercepts, numRows);
	}

	/**
	 * Applies a row vector operation from the left For plus/minus are the intercepts shifted For multiply/divide slopes
	 * and intercepts are scaled
	 *
	 * @param op        The operation to execute
	 * @param v         The vector of values to apply the values contained should be at least the length of the highest
	 *                  value in the column index
	 * @param isRowSafe True if the binary op is applied to an entire zero row and all results are zero
	 * @return a new ColGroupPiecewiseLinearCompressed with updated coefficients
	 */

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		final int numCols = _colIndexes.size();
		double[][] newIntercepts = new double[numCols][];
		double[][] newSlopes = new double[numCols][];
		final boolean isAddSub = op.fn instanceof Plus || op.fn instanceof Minus;

		if(!isAddSub && !(op.fn instanceof Multiply || op.fn instanceof Divide))
			throw new NotImplementedException("Unsupported binary op: " + op.fn.getClass().getSimpleName());

		for(int col = 0; col < numCols; col++) {
			double rowValue = v[_colIndexes.get(col)];
			int numSegs = interceptsPerCol[col].length;
			newIntercepts[col] = new double[numSegs];

			// Plus/Minus: slope is translation-invariant, only intercept shifts
			newSlopes[col] = isAddSub ? slopesPerCol[col].clone() : new double[numSegs];

			for(int seg = 0; seg < numSegs; seg++) {
				newIntercepts[col][seg] = op.fn.execute(rowValue, interceptsPerCol[col][seg]);
				if(!isAddSub)
					newSlopes[col][seg] = op.fn.execute(rowValue, slopesPerCol[col][seg]);
			}
		}
		return new ColGroupPiecewiseLinearCompressed(_colIndexes, breakpointsPerCol, newSlopes, newIntercepts, numRows);
	}

	/**
	 * Applies a row vector operation from the right For plus/minus are the intercepts shifted For multiply/divide
	 * slopes and intercepts are scaled
	 *
	 * @param op        The operation to execute
	 * @param v         The vector of values to apply the values contained should be at least the length of the highest
	 *                  value in the column index
	 * @param isRowSafe True if the binary op is applied to an entire zero row and all results are zero
	 * @return a new ColGroupPiecewiseLinearCompressed with updated coefficients
	 */
	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		final int numCols = _colIndexes.size();
		final boolean isAddSub = op.fn instanceof Plus || op.fn instanceof Minus;

		if(!isAddSub && !(op.fn instanceof Multiply || op.fn instanceof Divide))
			throw new NotImplementedException("Unsupported scalar op: " + op.fn.getClass().getSimpleName());

		double[][] newSlopes = new double[numCols][];
		double[][] newIntercepts = new double[numCols][];

		for(int col = 0; col < numCols; col++) {
			double val = v[_colIndexes.get(col)];
			int numSegs = interceptsPerCol[col].length;
			// Plus/Minus shifts intercept only, slopes are unchanged
			newSlopes[col] = isAddSub ? slopesPerCol[col].clone() : new double[numSegs];
			newIntercepts[col] = new double[numSegs];

			for(int seg = 0; seg < numSegs; seg++) {
				newIntercepts[col][seg] = op.fn.execute(interceptsPerCol[col][seg], val);
				if(!isAddSub)
					newSlopes[col][seg] = op.fn.execute(slopesPerCol[col][seg], val);
			}
		}
		return new ColGroupPiecewiseLinearCompressed(_colIndexes, breakpointsPerCol, newSlopes, newIntercepts, numRows);
	}

	/**
	 * Returns true if any decompressed value in this column group equals the given pattern
	 *
	 * @param pattern The value to look for.
	 * @return true if pattern is found, else false
	 */
	@Override
	public boolean containsValue(double pattern) {
		for(int col = 0; col < _colIndexes.size(); col++) {
			if(colContainsValue(col, pattern))
				return true;
		}
		return false;
	}

	/**
	 * checks if any reconstructed value in column col equals the pattern for each segment, solves the m * x + b =
	 * pattern instead of scanning all rows
	 *
	 * @param col     column index
	 * @param pattern the value to search for
	 * @return true if the pattern is found
	 */

	private boolean colContainsValue(int col, double pattern) {
		int[] breakpoints = breakpointsPerCol[col];
		double[] intercepts = interceptsPerCol[col];
		double[] slopes = slopesPerCol[col];
		for(int seg = 0; seg < breakpoints.length - 1; seg++) {
			int start = breakpoints[seg];
			int len = breakpoints[seg + 1] - start;
			if(len <= 0)
				continue;

			double b = intercepts[seg];
			double m = slopes[seg];

			if(m == 0.0) {
				// constant segment: all values equal b
				if(Double.compare(b, pattern) == 0)
					return true;
				continue;
			}

			// check if pattern lies on the line: solve m*x + b = pattern for x
			double x = (pattern - b) / m;
			int xi = (int) x;
			if(xi >= start && xi < start + len && Double.compare(m * xi + b, pattern) == 0)
				return true;
		}
		return false;
	}

	@Override
	public AColGroup unaryOperation(UnaryOperator op) {
		throw new NotImplementedException("unaryOperation not supported for PiecewiseLinear");
	}

	@Override
	public AColGroup replace(double pattern, double replace) {
		throw new NotImplementedException("replace not supported for PiecewiseLinear");
	}

	/**
	 * Computes global min or max over all decompressed values. For each linear segment the extreme is at one endpoint.
	 */
	@Override
	protected double computeMxx(double c, Builtin builtin) {
		for(int col = 0; col < _colIndexes.size(); col++) {
			int[] bp = breakpointsPerCol[col];
			double[] slopes = slopesPerCol[col];
			double[] intercepts = interceptsPerCol[col];
			for(int seg = 0; seg + 1 < bp.length; seg++) {
				int start = bp[seg];
				int end = bp[seg + 1] - 1; // last row index in this segment
				if(start > end)
					continue;
				double valStart = slopes[seg] * start + intercepts[seg];
				double valEnd = slopes[seg] * end + intercepts[seg];
				c = builtin.execute(c, valStart);
				c = builtin.execute(c, valEnd);
			}
		}
		return c;
	}

	/** Computes per-column min or max, storing in c[_colIndexes.get(col)]. */
	@Override
	protected void computeColMxx(double[] c, Builtin builtin) {
		for(int col = 0; col < _colIndexes.size(); col++) {
			int gcol = _colIndexes.get(col);
			int[] bp = breakpointsPerCol[col];
			double[] slopes = slopesPerCol[col];
			double[] intercepts = interceptsPerCol[col];
			for(int seg = 0; seg + 1 < bp.length; seg++) {
				int start = bp[seg];
				int end = bp[seg + 1] - 1;
				if(start > end)
					continue;
				double valStart = slopes[seg] * start + intercepts[seg];
				double valEnd = slopes[seg] * end + intercepts[seg];
				c[gcol] = builtin.execute(c[gcol], valStart);
				c[gcol] = builtin.execute(c[gcol], valEnd);
			}
		}
	}

	/**
	 * Computes sum of squares of all decompressed values using the closed-form formula:
	 * sum_{i=start}^{end-1} (m*i + b)^2 = m^2*sumI2 + 2*m*b*sumI + b^2*len
	 */
	@Override
	protected void computeSumSq(double[] c, int nRows) {
		double total = 0.0;
		for(int col = 0; col < _colIndexes.size(); col++)
			total += segmentSumSq(col);
		c[0] += total;
	}

	/** Computes per-column sum of squares. */
	@Override
	protected void computeColSumsSq(double[] c, int nRows) {
		for(int col = 0; col < _colIndexes.size(); col++)
			c[_colIndexes.get(col)] += segmentSumSq(col);
	}

	private double segmentSumSq(int col) {
		double total = 0.0;
		int[] bp = breakpointsPerCol[col];
		double[] slopes = slopesPerCol[col];
		double[] intercepts = interceptsPerCol[col];
		for(int seg = 0; seg + 1 < bp.length; seg++) {
			int start = bp[seg];
			int end = bp[seg + 1];
			int len = end - start;
			if(len <= 0)
				continue;
			double m = slopes[seg];
			double b = intercepts[seg];
			double sumI = (double) len * (2.0 * start + (len - 1)) / 2.0;
			double sumI2 = sumOfSquares(start, end);
			total += m * m * sumI2 + 2.0 * m * b * sumI + b * b * len;
		}
		return total;
	}

	/** sum_{i=start}^{end-1} i^2, using the closed form end*(end-1)*(2*end-1)/6 - start*(start-1)*(2*start-1)/6 */
	private static double sumOfSquares(int start, int end) {
		double s = 0;
		if(end > 0)
			s += (double) end * (end - 1) * (2 * end - 1) / 6.0;
		if(start > 0)
			s -= (double) start * (start - 1) * (2 * start - 1) / 6.0;
		return s;
	}

	/** Adds preAgg[rix] to c[rix] for each row in [rl, ru). preAgg is the row-sum across all columns. */
	@Override
	protected void computeRowSums(double[] c, int rl, int ru, double[] preAgg) {
		for(int rix = rl; rix < ru; rix++)
			c[rix] += preAgg[rix];
	}

	/** Applies builtin(c[rix], preAgg[rix]) for each row in [rl, ru). preAgg is the row min/max across columns. */
	@Override
	protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] preAgg) {
		for(int rix = rl; rix < ru; rix++)
			c[rix] = builtin.execute(c[rix], preAgg[rix]);
	}

	/** Computes the product of all decompressed values, accumulated into c[0]. */
	@Override
	protected void computeProduct(double[] c, int nRows) {
		for(int col = 0; col < _colIndexes.size(); col++) {
			int[] bp = breakpointsPerCol[col];
			double[] slopes = slopesPerCol[col];
			double[] intercepts = interceptsPerCol[col];
			for(int seg = 0; seg + 1 < bp.length; seg++) {
				double m = slopes[seg];
				double b = intercepts[seg];
				for(int r = bp[seg]; r < bp[seg + 1]; r++) {
					double v = m * r + b;
					if(v == 0) {
						c[0] = 0;
						return;
					}
					c[0] *= v;
				}
			}
		}
	}

	/** Multiplies c[rix] by the product of all column values at row rix (from preAgg). */
	@Override
	protected void computeRowProduct(double[] c, int rl, int ru, double[] preAgg) {
		for(int rix = rl; rix < ru; rix++)
			c[rix] *= preAgg[rix];
	}

	/** Computes per-column product of all decompressed values. */
	@Override
	protected void computeColProduct(double[] c, int nRows) {
		for(int col = 0; col < _colIndexes.size(); col++) {
			int gcol = _colIndexes.get(col);
			int[] bp = breakpointsPerCol[col];
			double[] slopes = slopesPerCol[col];
			double[] intercepts = interceptsPerCol[col];
			for(int seg = 0; seg + 1 < bp.length; seg++) {
				double m = slopes[seg];
				double b = intercepts[seg];
				for(int r = bp[seg]; r < bp[seg + 1]; r++) {
					double v = m * r + b;
					if(v == 0) {
						c[gcol] = 0;
						break;
					}
					c[gcol] *= v;
				}
				if(c[gcol] == 0)
					break;
			}
		}
	}

	/** Returns array[r] = sum of all column values at row r (used by computeRowSums). */
	@Override
	protected double[] preAggSumRows() {
		double[] agg = new double[numRows];
		for(int col = 0; col < _colIndexes.size(); col++) {
			int[] bp = breakpointsPerCol[col];
			double[] slopes = slopesPerCol[col];
			double[] intercepts = interceptsPerCol[col];
			for(int seg = 0; seg + 1 < bp.length; seg++) {
				double m = slopes[seg];
				double b = intercepts[seg];
				for(int r = bp[seg]; r < bp[seg + 1]; r++)
					agg[r] += m * r + b;
			}
		}
		return agg;
	}

	/** Returns array[r] = sum of squared column values at row r (used by computeRowSums for SumSq). */
	@Override
	protected double[] preAggSumSqRows() {
		double[] agg = new double[numRows];
		for(int col = 0; col < _colIndexes.size(); col++) {
			int[] bp = breakpointsPerCol[col];
			double[] slopes = slopesPerCol[col];
			double[] intercepts = interceptsPerCol[col];
			for(int seg = 0; seg + 1 < bp.length; seg++) {
				double m = slopes[seg];
				double b = intercepts[seg];
				for(int r = bp[seg]; r < bp[seg + 1]; r++) {
					double v = m * r + b;
					agg[r] += v * v;
				}
			}
		}
		return agg;
	}

	/** Returns array[r] = product of all column values at row r (used by computeRowProduct). */
	@Override
	protected double[] preAggProductRows() {
		double[] agg = new double[numRows];
		Arrays.fill(agg, 1.0);
		for(int col = 0; col < _colIndexes.size(); col++) {
			int[] bp = breakpointsPerCol[col];
			double[] slopes = slopesPerCol[col];
			double[] intercepts = interceptsPerCol[col];
			for(int seg = 0; seg + 1 < bp.length; seg++) {
				double m = slopes[seg];
				double b = intercepts[seg];
				for(int r = bp[seg]; r < bp[seg + 1]; r++)
					agg[r] *= m * r + b;
			}
		}
		return agg;
	}

	/** Returns array[r] = builtin applied across all column values at row r (used by computeRowMxx). */
	@Override
	protected double[] preAggBuiltinRows(Builtin builtin) {
		double init = builtin.getBuiltinCode() == Builtin.BuiltinCode.MAX ? Double.NEGATIVE_INFINITY
			: Double.POSITIVE_INFINITY;
		double[] agg = new double[numRows];
		Arrays.fill(agg, init);
		for(int col = 0; col < _colIndexes.size(); col++) {
			int[] bp = breakpointsPerCol[col];
			double[] slopes = slopesPerCol[col];
			double[] intercepts = interceptsPerCol[col];
			for(int seg = 0; seg + 1 < bp.length; seg++) {
				double m = slopes[seg];
				double b = intercepts[seg];
				for(int r = bp[seg]; r < bp[seg + 1]; r++)
					agg[r] = builtin.execute(agg[r], m * r + b);
			}
		}
		return agg;
	}

	/** Two piecewise linear groups have the same index structure if they are both piecewise linear. */
	@Override
	public boolean sameIndexStructure(AColGroupCompressed that) {
		return that instanceof ColGroupPiecewiseLinearCompressed;
	}

	/**
	 * Computes the transpose self-matrix multiplication (t(A) %*% A) using closed-form arithmetic series.
	 * For each pair of columns i, j, merges their breakpoint sequences and sums segment cross-products analytically.
	 */
	@Override
	protected void tsmm(double[] result, int numColumns, int nRows) {
		final int numCols = _colIndexes.size();
		for(int i = 0; i < numCols; i++) {
			final int gcol_i = _colIndexes.get(i);
			for(int j = i; j < numCols; j++) {
				final int gcol_j = _colIndexes.get(j);
				double dotProduct = crossColDotProduct(i, j);
				result[gcol_i * numColumns + gcol_j] += dotProduct;
			}
		}
	}

	/**
	 * Computes sum_r val_i(r) * val_j(r) by merging breakpoints of columns i and j.
	 * Within each merged interval, the product of two linear functions has a closed form.
	 */
	private double crossColDotProduct(int i, int j) {
		int[] bp_i = breakpointsPerCol[i];
		int[] bp_j = breakpointsPerCol[j];
		double[] slopes_i = slopesPerCol[i];
		double[] intercepts_i = interceptsPerCol[i];
		double[] slopes_j = slopesPerCol[j];
		double[] intercepts_j = interceptsPerCol[j];

		double dot = 0.0;
		int si = 0, sj = 0;
		int a = 0;

		while(si < slopes_i.length && sj < slopes_j.length) {
			int end_i = bp_i[si + 1];
			int end_j = bp_j[sj + 1];
			int b = Math.min(end_i, end_j);

			double m_i = slopes_i[si];
			double b_i = intercepts_i[si];
			double m_j = slopes_j[sj];
			double b_j = intercepts_j[sj];

			int len = b - a;
			if(len > 0) {
				double sumI = (double) len * (2.0 * a + (len - 1)) / 2.0;
				double sumI2 = sumOfSquares(a, b);
				dot += m_i * m_j * sumI2 + (m_i * b_j + m_j * b_i) * sumI + b_i * b_j * len;
			}

			a = b;
			if(b >= end_i)
				si++;
			if(b >= end_j)
				sj++;
		}
		return dot;
	}

	/** Returns a copy of this group with new column indices. */
	@Override
	public AColGroup copyAndSet(IColIndex colIndexes) {
		return new ColGroupPiecewiseLinearCompressed(colIndexes, breakpointsPerCol, slopesPerCol, interceptsPerCol,
			numRows);
	}

	/**
	 * Decompresses rows [rl, ru) into the DenseBlock in transposed form:
	 * db row = _colIndexes.get(col), db column = original row index.
	 */
	@Override
	public void decompressToDenseBlockTransposed(DenseBlock db, int rl, int ru) {
		for(int col = 0; col < _colIndexes.size(); col++) {
			final int gcol = _colIndexes.get(col);
			final double[] c = db.values(gcol);
			final int off = db.pos(gcol);
			int[] bp = breakpointsPerCol[col];
			double[] slopes = slopesPerCol[col];
			double[] intercepts = interceptsPerCol[col];
			for(int seg = 0; seg + 1 < bp.length; seg++) {
				int segStart = Math.max(bp[seg], rl);
				int segEnd = Math.min(bp[seg + 1], ru);
				double m = slopes[seg];
				double b = intercepts[seg];
				for(int r = segStart; r < segEnd; r++)
					c[off + r] += m * r + b;
			}
		}
	}

	@Override
	public void decompressToSparseBlockTransposed(SparseBlockMCSR sb, int nColOut) {
		throw new NotImplementedException("decompressToSparseBlockTransposed not supported for PiecewiseLinear");
	}

	/** Decompresses rows [rl, ru) into a SparseBlock, iterating row-first to satisfy column-order append requirement. */
	@Override
	public void decompressToSparseBlock(SparseBlock sb, int rl, int ru, int offR, int offC) {
		final int numCols = _colIndexes.size();
		for(int row = rl; row < ru; row++) {
			for(int col = 0; col < numCols; col++) {
				double v = getIdx(row, col);
				if(v != 0)
					sb.append(row + offR, _colIndexes.get(col) + offC, v);
			}
		}
	}

	/**
	 * Right-multiplies this column group by the given matrix, returning an uncompressed column group.
	 * For each output column j and each input segment, accumulates the weighted sum row-by-row.
	 */
	@Override
	public AColGroup rightMultByMatrix(MatrixBlock right, IColIndex allCols, int k) {
		final int nColR = right.getNumColumns();
		final IColIndex outputCols = allCols != null ? allCols : ColIndexFactory.create(nColR);
		final MatrixBlock result = new MatrixBlock(numRows, nColR, false);
		result.allocateDenseBlock();
		final double[] resultValues = result.getDenseBlockValues();

		for(int col = 0; col < _colIndexes.size(); col++) {
			final int gcol = _colIndexes.get(col);
			int[] bp = breakpointsPerCol[col];
			double[] slopes = slopesPerCol[col];
			double[] intercepts = interceptsPerCol[col];

			for(int seg = 0; seg + 1 < bp.length; seg++) {
				double m = slopes[seg];
				double b = intercepts[seg];
				for(int j = 0; j < nColR; j++) {
					double w = right.get(gcol, j);
					if(w == 0)
						continue;
					for(int r = bp[seg]; r < bp[seg + 1]; r++)
						resultValues[r * nColR + j] += w * (m * r + b);
				}
			}
		}
		result.recomputeNonZeros();
		return ColGroupUncompressed.create(result, outputCols);
	}

	/**
	 * Left-multiplies a sub-range of the given matrix by this column group.
	 * Rows [rl, ru) of matrix, columns [cl, cu) are multiplied against this group's rows [cl, cu).
	 */
	@Override
	public void leftMultByMatrixNoPreAgg(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {
		final int numCols = _colIndexes.size();
		for(int col = 0; col < numCols; col++) {
			final int gcol = _colIndexes.get(col);
			int[] bp = breakpointsPerCol[col];
			double[] slopes = slopesPerCol[col];
			double[] intercepts = interceptsPerCol[col];

			for(int mRow = rl; mRow < ru; mRow++) {
				double sum = 0.0;
				for(int seg = 0; seg + 1 < bp.length; seg++) {
					int segStart = Math.max(bp[seg], cl);
					int segEnd = Math.min(bp[seg + 1], cu);
					if(segStart >= segEnd)
						continue;
					double m = slopes[seg];
					double b = intercepts[seg];
					for(int r = segStart; r < segEnd; r++)
						sum += matrix.get(mRow, r) * (m * r + b);
				}
				result.set(mRow, gcol, result.get(mRow, gcol) + sum);
			}
		}
	}

	/**
	 * Left-multiplies by another column group: computes t(lhs) %*% this and accumulates into result.
	 * Iterates over all rows, using getIdx to decompress both sides.
	 */
	@Override
	public void leftMultByAColGroup(AColGroup lhs, MatrixBlock result, int nRows) {
		if(lhs instanceof ColGroupEmpty)
			return;
		final int lhsNumCols = lhs.getNumCols();
		final int rhsNumCols = _colIndexes.size();
		final double[] resValues = result.getDenseBlockValues();
		final int resCols = result.getNumColumns();

		for(int col = 0; col < rhsNumCols; col++) {
			final int gcol = _colIndexes.get(col);
			int[] bp = breakpointsPerCol[col];
			double[] slopes = slopesPerCol[col];
			double[] intercepts = interceptsPerCol[col];

			for(int seg = 0; seg + 1 < bp.length; seg++) {
				double m = slopes[seg];
				double b = intercepts[seg];
				for(int r = bp[seg]; r < bp[seg + 1]; r++) {
					double rhsVal = m * r + b;
					if(rhsVal == 0)
						continue;
					for(int lhsCol = 0; lhsCol < lhsNumCols; lhsCol++) {
						double lhsVal = lhs.getIdx(r, lhsCol);
						if(lhsVal != 0)
							resValues[lhsCol * resCols + gcol] += lhsVal * rhsVal;
					}
				}
			}
		}
	}

	@Override
	public void tsmmAColGroup(AColGroup other, MatrixBlock result) {
		throw new DMLCompressionException("tsmmAColGroup should not be called on PiecewiseLinear");
	}

	/** Returns a new group with only column at index idx. */
	@Override
	protected AColGroup sliceSingleColumn(int idx) {
		IColIndex newCols = ColIndexFactory.create(1);
		return new ColGroupPiecewiseLinearCompressed(newCols,
			new int[][] {breakpointsPerCol[idx].clone()},
			new double[][] {slopesPerCol[idx].clone()},
			new double[][] {interceptsPerCol[idx].clone()},
			numRows);
	}

	/** Returns a new group with columns [idStart, idEnd), mapped to outputCols. */
	@Override
	protected AColGroup sliceMultiColumns(int idStart, int idEnd, IColIndex outputCols) {
		int numSelected = idEnd - idStart;
		int[][] newBp = new int[numSelected][];
		double[][] newSlopes = new double[numSelected][];
		double[][] newIntercepts = new double[numSelected][];
		for(int i = 0; i < numSelected; i++) {
			int src = idStart + i;
			newBp[i] = breakpointsPerCol[src].clone();
			newSlopes[i] = slopesPerCol[src].clone();
			newIntercepts[i] = interceptsPerCol[src].clone();
		}
		return new ColGroupPiecewiseLinearCompressed(outputCols, newBp, newSlopes, newIntercepts, numRows);
	}

	/**
	 * Returns a new group covering rows [rl, ru) only.
	 * Breakpoints are shifted to start at 0; intercepts are adjusted so that
	 * value at new row r' = value at original row r' + rl.
	 */
	@Override
	public AColGroup sliceRows(int rl, int ru) {
		int numCols = _colIndexes.size();
		int newNumRows = ru - rl;
		int[][] newBp = new int[numCols][];
		double[][] newSlopes = new double[numCols][];
		double[][] newIntercepts = new double[numCols][];

		for(int col = 0; col < numCols; col++) {
			int[] bp = breakpointsPerCol[col];
			double[] slopes = slopesPerCol[col];
			double[] intercepts = interceptsPerCol[col];

			List<Integer> bpList = new ArrayList<>();
			List<Double> slopeList = new ArrayList<>();
			List<Double> interceptList = new ArrayList<>();
			bpList.add(0);

			for(int seg = 0; seg + 1 < bp.length; seg++) {
				if(bp[seg + 1] <= rl)
					continue;
				if(bp[seg] >= ru)
					break;
				int newEnd = Math.min(bp[seg + 1], ru) - rl;
				slopeList.add(slopes[seg]);
				// adjust intercept: original value at r = m*r + b; at new index r'=r-rl: m*(r'+rl)+b = m*r' + (m*rl+b)
				interceptList.add(intercepts[seg] + slopes[seg] * rl);
				bpList.add(newEnd);
			}

			if(bpList.size() == 1) {
				slopeList.add(0.0);
				interceptList.add(0.0);
				bpList.add(newNumRows);
			}

			newBp[col] = bpList.stream().mapToInt(Integer::intValue).toArray();
			newSlopes[col] = slopeList.stream().mapToDouble(Double::doubleValue).toArray();
			newIntercepts[col] = interceptList.stream().mapToDouble(Double::doubleValue).toArray();
		}

		return new ColGroupPiecewiseLinearCompressed(_colIndexes, newBp, newSlopes, newIntercepts, newNumRows);
	}

	/**
	 * Counts non-zero decompressed values. For constant segments (slope=0) the answer is trivial.
	 * For linear segments, the zero crossing can occur at most once per segment.
	 */
	@Override
	public long getNumberNonZeros(int nRows) {
		long nnz = 0;
		for(int col = 0; col < _colIndexes.size(); col++) {
			int[] bp = breakpointsPerCol[col];
			double[] slopes = slopesPerCol[col];
			double[] intercepts = interceptsPerCol[col];
			for(int seg = 0; seg + 1 < bp.length; seg++) {
				int start = bp[seg];
				int end = bp[seg + 1];
				int len = end - start;
				if(len <= 0)
					continue;
				double m = slopes[seg];
				double b = intercepts[seg];
				if(m == 0) {
					if(b != 0)
						nnz += len;
				}
				else {
					// linear: zero at r = -b/m; at most one integer zero crossing
					double zeroAt = -b / m;
					int zi = (int) Math.round(zeroAt);
					if(zi >= start && zi < end && Math.abs(m * zi + b) < 1e-12)
						nnz += len - 1;
					else
						nnz += len;
				}
			}
		}
		return nnz;
	}

	/** Computes central moment by iterating all decompressed values row by row. */
	@Override
	public CmCovObject centralMoment(CMOperator op, int nRows) {
		CmCovObject ret = new CmCovObject();
		for(int col = 0; col < _colIndexes.size(); col++) {
			int[] bp = breakpointsPerCol[col];
			double[] slopes = slopesPerCol[col];
			double[] intercepts = interceptsPerCol[col];
			for(int seg = 0; seg + 1 < bp.length; seg++) {
				double m = slopes[seg];
				double b = intercepts[seg];
				for(int r = bp[seg]; r < bp[seg + 1]; r++)
					op.fn.execute(ret, m * r + b, 1);
			}
		}
		return ret;
	}

	@Override
	public AColGroup rexpandCols(int max, boolean ignore, boolean cast, int nRows) {
		throw new NotImplementedException("rexpandCols not supported for PiecewiseLinear");
	}

	@Override
	public double getCost(ComputationCostEstimator e, int nRows) {
		final int nCols = getNumCols();
		final int nVals = getNumValues();
		return e.getCost(nRows, nRows, nCols, nVals, 1.0);
	}

	/** Returns null to indicate the group cannot be merged; the framework will use the generic append path. */
	@Override
	public AColGroup append(AColGroup g) {
		return null;
	}

	/**
	 * Appends multiple piecewise linear blocks vertically. groups[0] == this.
	 * Each block i covers rows [i*blen, min((i+1)*blen, rlen)).
	 * Breakpoints are shifted by the block offset; intercepts are adjusted accordingly.
	 */
	@Override
	protected AColGroup appendNInternal(AColGroup[] groups, int blen, int rlen) {
		final int numCols = _colIndexes.size();
		int[][] mergedBp = new int[numCols][];
		double[][] mergedSlopes = new double[numCols][];
		double[][] mergedIntercepts = new double[numCols][];

		for(int col = 0; col < numCols; col++) {
			List<Integer> bpList = new ArrayList<>();
			List<Double> slopeList = new ArrayList<>();
			List<Double> interceptList = new ArrayList<>();
			bpList.add(0);

			int offset = 0;
			for(AColGroup g : groups) {
				if(g instanceof ColGroupPiecewiseLinearCompressed) {
					ColGroupPiecewiseLinearCompressed plg = (ColGroupPiecewiseLinearCompressed) g;
					int[] gbp = plg.breakpointsPerCol[col];
					double[] gSlopes = plg.slopesPerCol[col];
					double[] gIntercepts = plg.interceptsPerCol[col];
					for(int seg = 0; seg + 1 < gbp.length; seg++) {
						slopeList.add(gSlopes[seg]);
						// new intercept = original_intercept - slope * offset (so value at global row R=r+offset is
						// m*(R-offset)+b = m*R + (b - m*offset))
						interceptList.add(gIntercepts[seg] - gSlopes[seg] * offset);
						bpList.add(gbp[seg + 1] + offset);
					}
					offset += plg.numRows;
				}
				else {
					throw new NotImplementedException(
						"appendNInternal: cannot append " + g.getClass().getSimpleName() + " into PiecewiseLinear");
				}
			}

			mergedBp[col] = bpList.stream().mapToInt(Integer::intValue).toArray();
			mergedSlopes[col] = slopeList.stream().mapToDouble(Double::doubleValue).toArray();
			mergedIntercepts[col] = interceptList.stream().mapToDouble(Double::doubleValue).toArray();
		}

		return new ColGroupPiecewiseLinearCompressed(_colIndexes, mergedBp, mergedSlopes, mergedIntercepts, rlen);
	}

	/** No scheme available for piecewise linear compression; returns null. */
	@Override
	public ICLAScheme getCompressionScheme() {
		return null;
	}

	/** No recompression needed; returns this. */
	@Override
	public AColGroup recompress() {
		return this;
	}

	@Override
	public CompressedSizeInfoColGroup getCompressionInfo(int nRow) {
		throw new NotImplementedException("getCompressionInfo not implemented for PiecewiseLinear");
	}

	/** Returns a new group with columns reordered according to the reordering array. */
	@Override
	protected AColGroup fixColIndexes(IColIndex newColIndex, int[] reordering) {
		final int numCols = newColIndex.size();
		int[][] newBp = new int[numCols][];
		double[][] newSlopes = new double[numCols][];
		double[][] newIntercepts = new double[numCols][];
		for(int i = 0; i < numCols; i++) {
			int old = reordering[i];
			newBp[i] = breakpointsPerCol[old].clone();
			newSlopes[i] = slopesPerCol[old].clone();
			newIntercepts[i] = interceptsPerCol[old].clone();
		}
		return new ColGroupPiecewiseLinearCompressed(newColIndex, newBp, newSlopes, newIntercepts, numRows);
	}

	/** Piecewise linear column groups cannot be reduced to fewer columns; returns null. */
	@Override
	public AColGroup reduceCols() {
		return null;
	}

	/** All decompressed values are in general non-zero; returns 1.0. */
	@Override
	public double getSparsity() {
		return 1.0;
	}

	/**
	 * For each output row r in [rl, ru): finds the single "1" in row r of the sparse selection matrix,
	 * reads its column index as the compressed row to copy, then decompresses that row into output row r.
	 */
	@Override
	protected void sparseSelection(MatrixBlock selection, ColGroupUtils.P[] points, MatrixBlock ret, int rl, int ru) {
		final SparseBlock sb = selection.getSparseBlock();
		final SparseBlock retB = ret.getSparseBlock();
		for(int r = rl; r < ru; r++) {
			if(sb.isEmpty(r))
				continue;
			final int sPos = sb.pos(r);
			final int rowCompressed = sb.indexes(r)[sPos];
			decompressToSparseBlock(retB, rowCompressed, rowCompressed + 1, r - rowCompressed, 0);
		}
	}

	/**
	 * For each output row r in [rl, ru): finds the single "1" in row r of the sparse selection matrix,
	 * reads its column index as the compressed row to copy, then decompresses that row into the dense output at row r.
	 */
	@Override
	protected void denseSelection(MatrixBlock selection, ColGroupUtils.P[] points, MatrixBlock ret, int rl, int ru) {
		final SparseBlock sb = selection.getSparseBlock();
		final DenseBlock retB = ret.getDenseBlock();
		for(int r = rl; r < ru; r++) {
			if(sb.isEmpty(r))
				continue;
			final int sPos = sb.pos(r);
			final int rowCompressed = sb.indexes(r)[sPos];
			decompressToDenseBlock(retB, rowCompressed, rowCompressed + 1, r - rowCompressed, 0);
		}
	}

	/**
	 * Splits this column group for a row-wise reshape from (nRow x nColOrg) to (nRow/multiplier x nColOrg*multiplier).
	 * Each block of nRow/multiplier rows becomes new columns shifted by i*nColOrg.
	 * Returns a single ColGroupPiecewiseLinearCompressed covering all multiplier*numCols new columns.
	 */
	@Override
	public AColGroup[] splitReshape(int multiplier, int nRow, int nColOrg) {
		final int numCols = _colIndexes.size();
		final int newNRow = nRow / multiplier;
		final int totalNewCols = numCols * multiplier;

		final int[] newColIndices = new int[totalNewCols];
		for(int i = 0; i < multiplier; i++)
			for(int c = 0; c < numCols; c++)
				newColIndices[i * numCols + c] = _colIndexes.get(c) + i * nColOrg;

		final int[][] newBp = new int[totalNewCols][];
		final double[][] newSlopes = new double[totalNewCols][];
		final double[][] newIntercepts = new double[totalNewCols][];

		for(int i = 0; i < multiplier; i++) {
			final int rl = i * newNRow;
			final int ru = rl + newNRow;

			for(int c = 0; c < numCols; c++) {
				final int[] bp = breakpointsPerCol[c];
				final double[] slopes = slopesPerCol[c];
				final double[] intercepts = interceptsPerCol[c];

				final List<Integer> bpList = new ArrayList<>();
				final List<Double> slopeList = new ArrayList<>();
				final List<Double> interceptList = new ArrayList<>();
				bpList.add(0);

				for(int seg = 0; seg + 1 < bp.length; seg++) {
					if(bp[seg + 1] <= rl)
						continue;
					if(bp[seg] >= ru)
						break;
					final int newEnd = Math.min(bp[seg + 1], ru) - rl;
					slopeList.add(slopes[seg]);
					// adjust intercept: value at new row r' = m*(r'+rl)+b = m*r' + (b + m*rl)
					interceptList.add(intercepts[seg] + slopes[seg] * rl);
					bpList.add(newEnd);
				}

				if(bpList.size() == 1) {
					slopeList.add(0.0);
					interceptList.add(0.0);
					bpList.add(newNRow);
				}

				final int newColIdx = i * numCols + c;
				newBp[newColIdx] = bpList.stream().mapToInt(Integer::intValue).toArray();
				newSlopes[newColIdx] = slopeList.stream().mapToDouble(Double::doubleValue).toArray();
				newIntercepts[newColIdx] = interceptList.stream().mapToDouble(Double::doubleValue).toArray();
			}
		}

		return new AColGroup[] {new ColGroupPiecewiseLinearCompressed(ColIndexFactory.create(newColIndices), newBp,
			newSlopes, newIntercepts, newNRow)};
	}

}
