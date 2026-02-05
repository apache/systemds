package org.apache.sysds.runtime.compress.colgroup;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.scheme.ICLAScheme;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.instructions.cp.CmCovObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

import java.util.Arrays;

public class ColGroupPiecewiseLinearCompressed extends AColGroupCompressed {

	IColIndex colIndexes;
	int[] breakpoints;
	double[] slopes;
	double[] intercepts;
	int numRows;

	protected ColGroupPiecewiseLinearCompressed(IColIndex colIndices) {
		super(colIndices);
	}

	public ColGroupPiecewiseLinearCompressed(IColIndex colIndexes, int[] breakpoints, double[] slopes,
		double[] intercepts, int numRows) {
		super(colIndexes);
		this.colIndexes = colIndexes;
		this.breakpoints = breakpoints;
		this.slopes = slopes;
		this.intercepts = intercepts;
		this.numRows = numRows;
	}

	public static AColGroup create(IColIndex colIndexes, int[] breakpoints, double[] slopes, double[] intercepts,
		int numRows) {
		if(breakpoints == null || breakpoints.length < 2)
			throw new IllegalArgumentException("Need at least one segment");

		int numSeg = breakpoints.length - 1;
		if(slopes.length != numSeg || intercepts.length != numSeg)
			throw new IllegalArgumentException("Inconsistent segment arrays");

		int[] bpCopy = Arrays.copyOf(breakpoints, breakpoints.length);
		double[] slopeCopy = Arrays.copyOf(slopes, slopes.length);
		double[] interceptCopy = Arrays.copyOf(intercepts, intercepts.length);

		return new ColGroupPiecewiseLinearCompressed(colIndexes, bpCopy, slopeCopy, interceptCopy, numRows);

	}

	@Override
	public void decompressToDenseBlock(DenseBlock db, int rl, int ru, int offR, int offC) {

		//Safety-Check:
		if(db == null || colIndexes == null || colIndexes.size() == 0 || breakpoints == null || slopes == null ||
			intercepts == null) {
			return;
		}
		//Validate Segments
		int sizeSegment = breakpoints.length - 1;
		if(sizeSegment <= 0 || rl >= ru) {
			return;
		}
		//Find every Segment
		final int column = _colIndexes.get(0);
		for(int currentSeg = 0; currentSeg < sizeSegment; currentSeg++) {
			int segStart = breakpoints[currentSeg];
			int segEnd = breakpoints[currentSeg + 1];
			if(segStart >= segEnd)
				continue;

			double currentSlope = slopes[currentSeg];
			double currentIntercepts = intercepts[currentSeg];

			int rowStart = Math.max(segStart, rl);
			int rowEnd = Math.min(segEnd, ru);
			if(rowStart >= rowEnd)
				continue;

			// Filling DenseBlock Matrix
			for(int r = rowStart; r < rowEnd; r++) {
				double yhat = currentSlope * r + currentIntercepts;
				int dbRow = offR + r;
				int dbColumn = offC + column;

				if(dbRow >= 0 && dbRow < db.numRows() && dbColumn >= 0 && dbColumn < db.numCols()) {
					db.set(dbRow, dbColumn, yhat);
				}
			}
		}
	}

	public int[] getBreakpoints() {
		return breakpoints;
	}

	public double[] getSlopes() {
		return slopes;
	}

	public double[] getIntercepts() {
		return intercepts;
	}

	@Override
	public double getIdx(int r, int colIdx) {
		//Check if the rowIDx is valid (safety check)
		if(r < 0 || r >= numRows || colIdx < 0 || colIdx >= colIndexes.size()) {
			return 0.0;
		}
		// Using Binary Search for efficient Search for the right Segment ( finding rowIdx r)
		// have to use int higherBound = breakpoints.length - 2 because it's the last valid segment
		int lowerBound = 0;
		int higherBound = breakpoints.length - 2;
		while(lowerBound <= higherBound) {
			int mid = (lowerBound + higherBound) / 2;
			if(r < breakpoints[mid] + 1) {
				higherBound = mid - 1;
			}
			else
				lowerBound = mid + 1;
		}
		int segment = Math.min(lowerBound, breakpoints.length - 2);

		return slopes[segment] * (double) r + intercepts[segment];
	}

	@Override
	public int getNumValues() {
		return breakpoints.length + slopes.length + intercepts.length;
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
	protected void computeSum(double[] c, int nRows) {
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
	public CompressionType getCompType() {
		throw new NotImplementedException();
	}

	@Override
	protected ColGroupType getColGroupType() {
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
	public AColGroup scalarOperation(ScalarOperator op) {
		throw new NotImplementedException();
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		throw new NotImplementedException();
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
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
	public boolean containsValue(double pattern) {
		throw new NotImplementedException();
	}

	@Override
	public long getNumberNonZeros(int nRows) {
		throw new NotImplementedException();
	}

	@Override
	public AColGroup replace(double pattern, double replace) {
		throw new NotImplementedException();
	}

	@Override
	public void computeColSums(double[] c, int nRows) {
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
	public AColGroup unaryOperation(UnaryOperator op) {
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

