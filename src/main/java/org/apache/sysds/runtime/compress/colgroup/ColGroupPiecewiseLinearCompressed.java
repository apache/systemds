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

public class ColGroupPiecewiseLinearCompressed extends AColGroupCompressed {

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
		this.slopesPerCol = slopes;
		this.interceptsPerCol = intercepts;
		this.numRows = numRows;
	}

	public static AColGroup create(IColIndex colIndices, int[][] breakpointsPerCol, double[][] slopesPerCol,
		double[][] interceptsPerCol, int numRows) {
		int expectedCols = colIndices.size();
		if(breakpointsPerCol.length != expectedCols)
			throw new IllegalArgumentException(
				"bp.length=" + breakpointsPerCol.length + " != colIndices.size()=" + expectedCols);
		if(breakpointsPerCol.length != colIndices.size())
			throw new IllegalArgumentException("Need at least one segment");

		for(int c = 0; c < colIndices.size(); c++) {
			if(breakpointsPerCol[c].length < 1 || breakpointsPerCol[c][0] != 0 ||
				breakpointsPerCol[c][breakpointsPerCol[c].length - 1] != numRows)
				throw new IllegalArgumentException(
					"Invalid breakpoints for col " + c + ": must start=0, end=numRows, >=1 pts");

			if(slopesPerCol[c].length != interceptsPerCol[c].length ||
				slopesPerCol[c].length != breakpointsPerCol[c].length - 1)
				throw new IllegalArgumentException("Inconsistent array lengths col " + c);
		}

		int numCols = colIndices.size();
		int[][] bpCopy = new int[numCols][];
		double[][] slopeCopy = new double[numCols][];
		double[][] interceptCopy = new double[numCols][];

		for(int c = 0; c < numCols; c++) {
			bpCopy[c] = Arrays.copyOf(breakpointsPerCol[c], breakpointsPerCol[c].length);
			slopeCopy[c] = Arrays.copyOf(slopesPerCol[c], slopesPerCol[c].length);
			interceptCopy[c] = Arrays.copyOf(interceptsPerCol[c], interceptsPerCol[c].length);
		}

		return new ColGroupPiecewiseLinearCompressed(colIndices, bpCopy, slopeCopy, interceptCopy, numRows);

	}

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
			for(int seg = 0; seg + 1 < breakpoints.length; seg++) {  // ← +1 statt length
				int segStart = breakpoints[seg];
				int segEnd = breakpoints[seg + 1];
				if(segStart >= segEnd)
					continue;

				double currentSlopeInSegment = slopes[seg];
				double currentInterceptInSegment = intercepts[seg];

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

	@Override
	public double getIdx(int r, int colIdx) {
		//Check if the rowIDx is valid (safety check)
		if(r < 0 || r >= numRows || colIdx < 0 || colIdx >= _colIndexes.size()) {
			return 0.0;
		}
		int[] bps = breakpointsPerCol[colIdx];
		double[] slps = slopesPerCol[colIdx];
		double[] ints = interceptsPerCol[colIdx];
		// Using Binary Search for efficient Search for the right Segment ( finding rowIdx r)
		// have to use int higherBound = breakpointsPerCol.length - 2 because it's the last valid segment
		int lowerBound = 0;
		int higherBound = bps.length - 2;
		while(lowerBound <= higherBound) {
			int mid = (lowerBound + higherBound) / 2;
			if(r < bps[mid + 1]) {
				higherBound = mid - 1;
			}
			else
				lowerBound = mid + 1;
		}
		int segment = Math.min(lowerBound, bps.length - 2);
		return slps[segment] * (double) r + ints[segment];
	}

	@Override
	public int getNumValues() {
		return breakpointsPerCol.length + slopesPerCol.length + interceptsPerCol.length;
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		int numCols = _colIndexes.size();
		ret += 8L * numCols * 3;
		ret += 24L * 3;

		for(int c = 0; c < numCols; c++) {
			ret += (long) MemoryEstimates.intArrayCost(breakpointsPerCol[c].length);
			ret += (long) MemoryEstimates.doubleArrayCost(slopesPerCol[c].length);
			ret += (long) MemoryEstimates.doubleArrayCost(interceptsPerCol[c].length);
		}

		ret += 4L;
		return ret;

	}

	@Override
	public void computeSum(double[] c, int nRows) {
		for(int col = 0; col < _colIndexes.size(); col++) {
			double colSum = 0.0;
			int[] breakpoints = breakpointsPerCol[col];
			double[] intercepts = interceptsPerCol[col];
			double[] slopes = slopesPerCol[col];
			for(int seg = 0; seg < breakpoints.length - 1; seg++) {
				int start = breakpoints[seg], end = breakpoints[seg + 1];
				int len = end - start;
				double b = intercepts[seg], m = slopes[seg];
				double sumR = (double) len * (len - 1) / 2.0;
				colSum += (double) len * b + m * sumR;
			}
			c[col] += colSum;
		}
	}

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

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		final int numCols = _colIndexes.size();
		double[][] newIntercepts = new double[numCols][];
		double[][] newSlopes = new double[numCols][];
		if(op.fn instanceof Plus || op.fn instanceof Minus) {
			for(int col = 0; col < numCols; col++) {
				int numSegments = interceptsPerCol[col].length;
				newIntercepts[col] = new double[numSegments];
				newSlopes[col] = slopesPerCol[col].clone();  // Unverändert
				for(int seg = 0; seg < numSegments; seg++)
					newIntercepts[col][seg] = op.executeScalar(interceptsPerCol[col][seg]);
			} // shift intercept
		}
		else if(op.fn instanceof Multiply || op.fn instanceof Divide) {
			for(int col = 0; col < numCols; col++) {
				int numSegments = interceptsPerCol[col].length;
				newIntercepts[col] = new double[numSegments];
				newSlopes[col] = new double[numSegments];
				for(int seg = 0; seg < numSegments; seg++) {
					newIntercepts[col][seg] = op.executeScalar(interceptsPerCol[col][seg]);
					newSlopes[col][seg] = op.executeScalar(slopesPerCol[col][seg]);
				}
			}//shift slope and intercept
		}
		else {
			throw new NotImplementedException("Unsupported scalar op");
		}
		// new ColGroup because of changed slopes, intercepts
		return new ColGroupPiecewiseLinearCompressed(_colIndexes, breakpointsPerCol, newSlopes, newIntercepts, numRows);
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		final int numCols = _colIndexes.size();
		double[][] newIntercepts = new double[numCols][];
		double[][] newSlopes = new double[numCols][];
		if(op.fn instanceof Plus || op.fn instanceof Minus) {
			for(int col = 0; col < numCols; col++) {
				double rowValue = v[_colIndexes.get(col)];
				int numSeg = interceptsPerCol[col].length;
				newIntercepts[col] = new double[numSeg];
				newSlopes[col] = slopesPerCol[col].clone();
				for(int seg = 0; seg < numSeg; seg++) {
					newIntercepts[col][seg] = op.fn.execute(rowValue, interceptsPerCol[col][seg]);
				}
			}
		}
		else if(op.fn instanceof Multiply || op.fn instanceof Divide) {
			for(int col = 0; col < numCols; col++) {
				double rowValue = v[_colIndexes.get(col)];
				int numSeg = interceptsPerCol[col].length;
				newIntercepts[col] = new double[numSeg];
				newSlopes[col] = new double[numSeg];
				for(int seg = 0; seg < numSeg; seg++) {
					newIntercepts[col][seg] = op.fn.execute(rowValue, interceptsPerCol[col][seg]);
					newSlopes[col][seg] = op.fn.execute(rowValue, slopesPerCol[col][seg]);
				}
			}
		}
		else {
			throw new NotImplementedException("Unsupported binary op");
		}
		return new ColGroupPiecewiseLinearCompressed(_colIndexes, breakpointsPerCol, newSlopes, newIntercepts, numRows);
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		final int numCols = _colIndexes.size();
		double[][] newIntercepts = new double[numCols][];
		double[][] newSlopes = new double[numCols][];
		if(op.fn instanceof Plus || op.fn instanceof Minus) {
			for(int col = 0; col < numCols; col++) {
				double rowValue = v[_colIndexes.get(col)];
				int numSeg = interceptsPerCol[col].length;
				newIntercepts[col] = new double[numSeg];
				newSlopes[col] = slopesPerCol[col].clone();
				for(int seg = 0; seg < numSeg; seg++) {
					newIntercepts[col][seg] = op.fn.execute(interceptsPerCol[col][seg], rowValue);
				}
			}
		}
		else if(op.fn instanceof Multiply || op.fn instanceof Divide) {
			for(int col = 0; col < numCols; col++) {
				double rowValue = v[_colIndexes.get(col)];
				int numSeg = interceptsPerCol[col].length;
				newIntercepts[col] = new double[numSeg];
				newSlopes[col] = new double[numSeg];
				for(int seg = 0; seg < numSeg; seg++) {
					newIntercepts[col][seg] = op.fn.execute(interceptsPerCol[col][seg], rowValue);
					newSlopes[col][seg] = op.fn.execute(slopesPerCol[col][seg], rowValue);
				}
			}
		}
		else {
			throw new NotImplementedException("Unsupported binary op");
		}
		return new ColGroupPiecewiseLinearCompressed(_colIndexes, breakpointsPerCol, newSlopes, newIntercepts, numRows);
	}

	@Override
	public boolean containsValue(double pattern) {
		for(int col = 0; col < _colIndexes.size(); col++) {
			if(colContainsValue(col, pattern))
				return true;
		}
		return false;
	}

	private boolean colContainsValue(int col, double pattern) {
		int[] breakpoints = breakpointsPerCol[col];
		double[] intercepts = interceptsPerCol[col];
		double[] slopes = slopesPerCol[col];
		int numSeg = breakpoints.length - 1;

		for(int seg = 0; seg < numSeg; seg++) {
			int start = breakpoints[seg];
			int end = breakpoints[seg + 1];
			int len = end - start;
			if(len <= 0)
				continue;

			double yIntercept = intercepts[seg];
			double slope = slopes[seg];

			if(slope == 0.0) {
				if(Double.compare(yIntercept, pattern) == 0)
					return true;
				continue;
			}

			if(Double.compare(yIntercept, pattern) == 0)
				return true;

			double endVal = yIntercept + slope * (len - 1);
			if(Double.compare(endVal, pattern) == 0)
				return true;

			double rowIndex = (pattern - yIntercept) / slope;
			if(rowIndex > 0 && rowIndex < (len - 1) && Double.compare(yIntercept + slope * rowIndex, pattern) == 0)
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

