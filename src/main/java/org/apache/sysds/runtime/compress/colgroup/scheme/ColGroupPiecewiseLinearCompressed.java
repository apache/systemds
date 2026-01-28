package org.apache.sysds.runtime.compress.colgroup.scheme;

import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroupCompressed;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUtils;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
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


    public ColGroupPiecewiseLinearCompressed(IColIndex colIndexes, int[] breakpoints, double[] slopes, double[] intercepts, int numRows) {
        super(colIndexes);
        this.breakpoints = breakpoints;
        this.slopes = slopes;
        this.intercepts = intercepts;
        this.numRows = numRows;
    }


    public static AColGroup create(IColIndex colIndexes, int[] breakpoints, double[] slopes, double[] intercepts, int numRows) {
        if (breakpoints == null || breakpoints.length < 2)
            throw new IllegalArgumentException("Need at least one segment");

        int numSeg = breakpoints.length - 1;
        if (slopes.length != numSeg || intercepts.length != numSeg)
            throw new IllegalArgumentException("Inconsistent segment arrays");

        int[] bpCopy = Arrays.copyOf(breakpoints, breakpoints.length);
        double[] slopeCopy = Arrays.copyOf(slopes, slopes.length);
        double[] interceptCopy = Arrays.copyOf(intercepts, intercepts.length);


        return new ColGroupPiecewiseLinearCompressed(
                colIndexes,
                bpCopy,
                slopeCopy,
                interceptCopy,
                numRows);

    }

    @Override
    public void decompressToDenseBlock(DenseBlock db, int rl, int ru, int offR, int offC) {
        // ✅ Vollständige Null-Safety
        if (db == null || _colIndexes == null || _colIndexes.size() == 0 ||
                breakpoints == null || slopes == null || intercepts == null) {
            return;
        }

        int numSeg = breakpoints.length - 1;
        if (numSeg <= 0 || rl >= ru) {
            return;
        }

        final int col = _colIndexes.get(0);

        for (int s = 0; s < numSeg; s++) {
            int segStart = breakpoints[s];
            int segEnd = breakpoints[s + 1];
            if (segStart >= segEnd) continue;  // Invalid Segment

            double a = slopes[s];
            double b = intercepts[s];

            int rs = Math.max(segStart, rl);
            int re = Math.min(segEnd, ru);
            if (rs >= re) continue;

            for (int r = rs; r < re; r++) {
                double yhat = a * r + b;
                int gr = offR + r;
                int gc = offC + col;

                // ✅ Bounds-Check vor set()
                if (gr >= 0 && gr < db.numRows() && gc >= 0 && gc < db.numCols()) {
                    db.set(gr, gc, yhat);
                }
            }
        }
    }

    @Override
    protected double computeMxx(double c, Builtin builtin) {
        return 0;
    }

    @Override
    protected void computeColMxx(double[] c, Builtin builtin) {

    }

    @Override
    protected void computeSum(double[] c, int nRows) {

    }

    @Override
    protected void computeSumSq(double[] c, int nRows) {

    }

    @Override
    protected void computeColSumsSq(double[] c, int nRows) {

    }

    @Override
    protected void computeRowSums(double[] c, int rl, int ru, double[] preAgg) {

    }

    @Override
    protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] preAgg) {

    }

    @Override
    protected void computeProduct(double[] c, int nRows) {

    }

    @Override
    protected void computeRowProduct(double[] c, int rl, int ru, double[] preAgg) {

    }

    @Override
    protected void computeColProduct(double[] c, int nRows) {

    }

    @Override
    protected double[] preAggSumRows() {
        return new double[0];
    }

    @Override
    protected double[] preAggSumSqRows() {
        return new double[0];
    }

    @Override
    protected double[] preAggProductRows() {
        return new double[0];
    }

    @Override
    protected double[] preAggBuiltinRows(Builtin builtin) {
        return new double[0];
    }

    @Override
    public boolean sameIndexStructure(AColGroupCompressed that) {
        return false;
    }

    @Override
    protected void tsmm(double[] result, int numColumns, int nRows) {

    }

    @Override
    public AColGroup copyAndSet(IColIndex colIndexes) {
        return null;
    }

    @Override
    public void decompressToDenseBlockTransposed(DenseBlock db, int rl, int ru) {

    }

    @Override
    public void decompressToSparseBlockTransposed(SparseBlockMCSR sb, int nColOut) {

    }

    @Override
    public double getIdx(int r, int colIdx) {
        // ✅ CRUCIAL: Bounds-Check für colIdx!
        if (r < 0 || r >= numRows || colIdx < 0 || colIdx >= _colIndexes.size()) {
            return 0.0;
        }

        // Segment-Suche (sicher jetzt)
        int seg = 0;
        for (int i = 1; i < breakpoints.length; i++) {
            if (r < breakpoints[i]) {
                break;
            }
            seg = i - 1;  // seg < numSeg immer!
        }

        return slopes[seg] * (double) r + intercepts[seg];
    }

    @Override
    public int getNumValues() {
        return breakpoints.length + slopes.length + intercepts.length;
    }

    @Override
    public CompressionType getCompType() {
        return null;
    }

    @Override
    protected ColGroupType getColGroupType() {
        return null;
    }



    @Override
    public void decompressToSparseBlock(SparseBlock sb, int rl, int ru, int offR, int offC) {

    }

    @Override
    public AColGroup rightMultByMatrix(MatrixBlock right, IColIndex allCols, int k) {
        return null;
    }

    @Override
    public void leftMultByMatrixNoPreAgg(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {

    }

    @Override
    public void leftMultByAColGroup(AColGroup lhs, MatrixBlock result, int nRows) {

    }

    @Override
    public void tsmmAColGroup(AColGroup other, MatrixBlock result) {

    }

    @Override
    public AColGroup scalarOperation(ScalarOperator op) {
        return null;
    }

    @Override
    public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
        return null;
    }

    @Override
    public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
        return null;
    }

    @Override
    protected AColGroup sliceSingleColumn(int idx) {
        return null;
    }

    @Override
    protected AColGroup sliceMultiColumns(int idStart, int idEnd, IColIndex outputCols) {
        return null;
    }

    @Override
    public AColGroup sliceRows(int rl, int ru) {
        return null;
    }

    @Override
    public boolean containsValue(double pattern) {
        return false;
    }

    @Override
    public long getNumberNonZeros(int nRows) {
        return 0;
    }

    @Override
    public AColGroup replace(double pattern, double replace) {
        return null;
    }

    @Override
    public void computeColSums(double[] c, int nRows) {

    }

    @Override
    public CmCovObject centralMoment(CMOperator op, int nRows) {
        return null;
    }


    @Override
    public AColGroup rexpandCols(int max, boolean ignore, boolean cast, int nRows) {
        return null;
    }

    @Override
    public double getCost(ComputationCostEstimator e, int nRows) {
        return 0;
    }

    @Override
    public AColGroup unaryOperation(UnaryOperator op) {
        return null;
    }

    @Override
    public AColGroup append(AColGroup g) {
        return null;
    }

    @Override
    protected AColGroup appendNInternal(AColGroup[] groups, int blen, int rlen) {
        return null;
    }

    @Override
    public ICLAScheme getCompressionScheme() {
        return null;
    }

    @Override
    public AColGroup recompress() {
        return null;
    }

    @Override
    public CompressedSizeInfoColGroup getCompressionInfo(int nRow) {
        return null;
    }

    @Override
    protected AColGroup fixColIndexes(IColIndex newColIndex, int[] reordering) {
        return null;
    }

    @Override
    public AColGroup reduceCols() {
        return null;
    }

    @Override
    public double getSparsity() {
        return 0;
    }

    @Override
    protected void sparseSelection(MatrixBlock selection, ColGroupUtils.P[] points, MatrixBlock ret, int rl, int ru) {

    }

    @Override
    protected void denseSelection(MatrixBlock selection, ColGroupUtils.P[] points, MatrixBlock ret, int rl, int ru) {

    }

    @Override
    public AColGroup[] splitReshape(int multiplier, int nRow, int nColOrg) {
        return new AColGroup[0];
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
}

