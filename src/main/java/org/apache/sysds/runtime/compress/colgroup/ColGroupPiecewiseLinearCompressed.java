package org.apache.sysds.runtime.compress.colgroup;

import org.apache.commons.lang3.NotImplementedException;
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

import java.util.Arrays;

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
	@Override
	public void computeSum(double[] c, int nRows) {
		for(int col = 0; col < _colIndexes.size(); col++) {
			double sum = 0.0;
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
				sum += slopes[seg] * sumX + intercepts[seg] * len;
			}
			c[col] += sum;
		}
	}

	/**
	 * Computes column sums by delegating to computeSum Methods are identical because every ColGroup just knows its own
	 * column
	 *
	 * @param c     The array to add the column sum to.
	 * @param nRows The number of rows in the column group.
	 */

	@Override
	public void computeColSums(double[] c, int nRows) {
		computeSum(c, nRows);
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
		throw new NotImplementedException();
	}

	@Override
	public AColGroup replace(double pattern, double replace) {
		throw new NotImplementedException();
	}

	@Override
	protected double computeMxx(double c, Builtin builtin) {
		throw new NotImplementedException();
	}

	@Override
	protected void computeColMxx(double[] c, Builtin builtin) {
		throw new NotImplementedException();
	}

	@Override
	protected void computeSumSq(double[] c, int nRows) {
		throw new NotImplementedException();

	}

	@Override
	protected void computeColSumsSq(double[] c, int nRows) {
		throw new NotImplementedException();

	}

	@Override
	protected void computeRowSums(double[] c, int rl, int ru, double[] preAgg) {
		throw new NotImplementedException();

	}

	@Override
	protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] preAgg) {
		throw new NotImplementedException();

	}

	@Override
	protected void computeProduct(double[] c, int nRows) {
		throw new NotImplementedException();

	}

	@Override
	protected void computeRowProduct(double[] c, int rl, int ru, double[] preAgg) {
		throw new NotImplementedException();

	}

	@Override
	protected void computeColProduct(double[] c, int nRows) {
		throw new NotImplementedException();

	}

	@Override
	protected double[] preAggSumRows() {
		throw new NotImplementedException();
	}

	@Override
	protected double[] preAggSumSqRows() {
		throw new NotImplementedException();
	}

	@Override
	protected double[] preAggProductRows() {
		throw new NotImplementedException();
	}

	@Override
	protected double[] preAggBuiltinRows(Builtin builtin) {
		throw new NotImplementedException();
	}

	@Override
	public boolean sameIndexStructure(AColGroupCompressed that) {
		throw new NotImplementedException();
	}

	@Override
	protected void tsmm(double[] result, int numColumns, int nRows) {
		throw new NotImplementedException();

	}

	@Override
	public AColGroup copyAndSet(IColIndex colIndexes) {
		throw new NotImplementedException();
	}

	@Override
	public void decompressToDenseBlockTransposed(DenseBlock db, int rl, int ru) {
		throw new NotImplementedException();

	}

	@Override
	public void decompressToSparseBlockTransposed(SparseBlockMCSR sb, int nColOut) {
		throw new NotImplementedException();

	}

	@Override
	public void decompressToSparseBlock(SparseBlock sb, int rl, int ru, int offR, int offC) {
		throw new NotImplementedException();

	}

	@Override
	public AColGroup rightMultByMatrix(MatrixBlock right, IColIndex allCols, int k) {
		throw new NotImplementedException();
	}

	@Override
	public void leftMultByMatrixNoPreAgg(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {
		throw new NotImplementedException();

	}

	@Override
	public void leftMultByAColGroup(AColGroup lhs, MatrixBlock result, int nRows) {
		throw new NotImplementedException();

	}

	@Override
	public void tsmmAColGroup(AColGroup other, MatrixBlock result) {
		throw new NotImplementedException();

	}

	@Override
	protected AColGroup sliceSingleColumn(int idx) {
		throw new NotImplementedException();
	}

	@Override
	protected AColGroup sliceMultiColumns(int idStart, int idEnd, IColIndex outputCols) {
		throw new NotImplementedException();
	}

	@Override
	public AColGroup sliceRows(int rl, int ru) {
		throw new NotImplementedException();
	}

	@Override
	public long getNumberNonZeros(int nRows) {
		throw new NotImplementedException();
	}

	@Override
	public CmCovObject centralMoment(CMOperator op, int nRows) {
		throw new NotImplementedException();
	}

	@Override
	public AColGroup rexpandCols(int max, boolean ignore, boolean cast, int nRows) {
		throw new NotImplementedException();
	}

	@Override
	public double getCost(ComputationCostEstimator e, int nRows) {
		throw new NotImplementedException();
	}

	@Override
	public AColGroup append(AColGroup g) {
		throw new NotImplementedException();
	}

	@Override
	protected AColGroup appendNInternal(AColGroup[] groups, int blen, int rlen) {
		throw new NotImplementedException();
	}

	@Override
	public ICLAScheme getCompressionScheme() {
		throw new NotImplementedException();
	}

	@Override
	public AColGroup recompress() {
		throw new NotImplementedException();
	}

	@Override
	public CompressedSizeInfoColGroup getCompressionInfo(int nRow) {
		throw new NotImplementedException();
	}

	@Override
	protected AColGroup fixColIndexes(IColIndex newColIndex, int[] reordering) {
		throw new NotImplementedException();
	}

	@Override
	public AColGroup reduceCols() {
		throw new NotImplementedException();
	}

	@Override
	public double getSparsity() {
		throw new NotImplementedException();
	}

	@Override
	protected void sparseSelection(MatrixBlock selection, ColGroupUtils.P[] points, MatrixBlock ret, int rl, int ru) {
		throw new NotImplementedException();
	}

	@Override
	protected void denseSelection(MatrixBlock selection, ColGroupUtils.P[] points, MatrixBlock ret, int rl, int ru) {
		throw new NotImplementedException();
	}

	@Override
	public AColGroup[] splitReshape(int multiplier, int nRow, int nColOrg) {
		throw new NotImplementedException();
	}

}

